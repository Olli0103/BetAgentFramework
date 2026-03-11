"""Results Fetcher — retrieves final match scores and marks matches as FINISHED.

Queries the DB for matches that are past their scheduled time and still
NOT_STARTED or IN_PROGRESS, with PENDING bets waiting for settlement.
Uses a pluggable results backend (Cloudflare crawler, API, or manual)
to fetch final scores.

Part of the Auditor's morning audit pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Protocol

from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    BetStatus,
    Match,
    MatchState,
    PlacedBet,
)

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MatchResult:
    """Final result for a match, returned by a results backend."""

    home_team: str
    away_team: str
    home_score: int
    away_score: int
    is_finished: bool  # False if match was postponed/cancelled


@dataclass(frozen=True)
class FetchResult:
    """Summary of a results fetch run."""

    matches_checked: int
    matches_updated: int
    matches_not_found: int


# ── Results backend protocol ─────────────────────────────────────────


class ResultsBackend(Protocol):
    """Protocol for pluggable results data sources."""

    def fetch_result(self, match: Match) -> MatchResult | None:
        """Fetch the final score for a match. Returns None if not available."""
        ...


class ManualResultsBackend:
    """Backend that applies pre-loaded results from a dict.

    Useful for testing and for manual score entry.
    """

    def __init__(self, results: dict[str, MatchResult] | None = None):
        self._results = results or {}

    def add_result(self, match_key: str, result: MatchResult) -> None:
        """Add a result keyed by 'home_team vs away_team'."""
        self._results[match_key] = result

    def fetch_result(self, match: Match) -> MatchResult | None:
        key = f"{match.home_team} vs {match.away_team}"
        return self._results.get(key)


# ── Core logic ───────────────────────────────────────────────────────


def get_unsettled_matches(
    session: Session,
    before_date: date | None = None,
) -> list[Match]:
    """Find matches that are past scheduled time, not finished, with pending bets.

    Args:
        session: SQLAlchemy session.
        before_date: Only consider matches scheduled before this date.
                     Defaults to today (matches from yesterday or earlier).

    Returns:
        List of Match objects needing result updates.
    """
    if before_date is None:
        before_date = date.today()

    cutoff = datetime.combine(before_date, time.min, tzinfo=timezone.utc)

    # Matches that are past their scheduled time and not yet FINISHED
    # AND have at least one unsettled bet (PENDING or PLACED)
    has_unsettled_bets = (
        exists()
        .where(
            PlacedBet.match_id == Match.id,
            PlacedBet.status.in_([BetStatus.PENDING, BetStatus.PLACED]),
        )
    )

    query = (
        select(Match)
        .where(
            Match.match_state != MatchState.FINISHED,
            Match.scheduled_at < cutoff,
            has_unsettled_bets,
        )
        .order_by(Match.scheduled_at)
    )

    return list(session.execute(query).scalars().all())


def update_match_result(
    session: Session,
    match: Match,
    result: MatchResult,
) -> None:
    """Apply a MatchResult to a Match record in the database."""
    match.home_score = result.home_score
    match.away_score = result.away_score
    match.is_live = False

    if result.is_finished:
        match.match_state = MatchState.FINISHED
    # If not finished (e.g., postponed), leave state as-is for manual review


def fetch_and_update_results(
    session: Session,
    backend: ResultsBackend | None = None,
    before_date: date | None = None,
) -> FetchResult:
    """Fetch results for all unsettled matches and update the database.

    Args:
        session: SQLAlchemy session (caller manages transaction).
        backend: Pluggable results source. Defaults to ManualResultsBackend (no-op).
        before_date: Only process matches before this date.

    Returns:
        FetchResult summary of the run.
    """
    if backend is None:
        backend = ManualResultsBackend()

    matches = get_unsettled_matches(session, before_date)
    logger.info("Found %d unsettled matches needing results", len(matches))

    updated = 0
    not_found = 0

    for match in matches:
        try:
            result = backend.fetch_result(match)
            if result is not None:
                update_match_result(session, match, result)
                updated += 1
                logger.info(
                    "Updated %s vs %s: %d-%d (%s)",
                    match.home_team, match.away_team,
                    result.home_score, result.away_score,
                    "finished" if result.is_finished else "not finished",
                )
            else:
                not_found += 1
                logger.warning(
                    "No result found for %s vs %s (scheduled %s)",
                    match.home_team, match.away_team, match.scheduled_at,
                )
        except Exception as exc:
            not_found += 1
            logger.error(
                "Failed to fetch result for %s vs %s: %s",
                match.home_team, match.away_team, exc,
            )

    session.flush()
    return FetchResult(
        matches_checked=len(matches),
        matches_updated=updated,
        matches_not_found=not_found,
    )
