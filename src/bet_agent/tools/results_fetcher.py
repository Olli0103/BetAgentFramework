"""Results Fetcher — retrieves final match scores and marks matches as FINISHED.

Queries the DB for matches that are past their scheduled time and still
NOT_STARTED or IN_PROGRESS, with PENDING bets waiting for settlement.
Uses a pluggable results backend (Cloudflare crawler, API, or manual)
to fetch final scores.

Default backend: **TheOddsAPIResultsBackend** — calls the
``/v4/sports/{sport}/scores?daysFrom=3`` endpoint to pull completed
match scores.  Falls back to ManualResultsBackend (no-op) when
``THE_ODDS_API_KEY`` is not set.

Part of the Auditor's morning audit pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Protocol

import requests
import yaml
from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    BetStatus,
    Match,
    MatchState,
    PlacedBet,
    Sport,
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


# ── Sport → Odds API sport keys mapping ──────────────────────────────

_AGENTS_YAML = Path(__file__).resolve().parents[3] / "config" / "agents.yaml"

# Internal Sport enum → list of Odds API sport key prefixes
# Used to find the right /scores endpoint for each match.
_SPORT_TO_API_KEYS: dict[str, list[str]] = {
    "football": ["soccer_"],
    "basketball": ["basketball_"],
    "american_football": ["americanfootball_"],
    "ice_hockey": ["icehockey_"],
    "tennis": ["tennis_"],
    "darts": ["darts_"],
}


def _load_configured_sport_keys() -> dict[str, list[str]]:
    """Load explicit sport keys from agents.yaml sports_mapping."""
    if not _AGENTS_YAML.exists():
        return {}
    try:
        raw = yaml.safe_load(_AGENTS_YAML.read_text())
        mapping = (
            raw.get("agents", [{}])[1]
            .get("odds_api", {})
            .get("sports_mapping", {})
        )
        result: dict[str, list[str]] = {}
        for sport, keys_str in mapping.items():
            if isinstance(keys_str, str):
                result[sport] = [k.strip() for k in keys_str.split(",") if k.strip() and "*" not in k]
            elif isinstance(keys_str, list):
                result[sport] = [k for k in keys_str if "*" not in k]
        return result
    except Exception:
        return {}


class TheOddsAPIResultsBackend:
    """Fetch completed match scores from The Odds API /scores endpoint.

    Batches requests per sport key and caches within a single run to
    minimize API usage (each sport key = 2 credits with daysFrom=3).

    Matching strategy:
      1. Exact home_team + away_team match (case-insensitive)
      2. Fuzzy substring match as fallback (handles "FC Bayern München"
         vs "Bayern Munich" style mismatches)
    """

    def __init__(
        self,
        api_key: str | None = None,
        days_from: int = 3,
    ):
        self._api_key = api_key or os.environ.get("THE_ODDS_API_KEY", "")
        self._days_from = days_from
        self._base_url = "https://api.the-odds-api.com/v4"
        # Cache: sport_key → list of event dicts from the API
        self._cache: dict[str, list[dict]] = {}
        # Configured sport keys from agents.yaml (exact, no wildcards)
        self._configured_keys = _load_configured_sport_keys()

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _sport_keys_for(self, sport: Sport) -> list[str]:
        """Get API sport keys to query for a given Sport enum value."""
        # First try explicit keys from agents.yaml
        configured = self._configured_keys.get(sport.value, [])
        if configured:
            return configured
        # Fall back to prefix-based defaults
        prefixes = _SPORT_TO_API_KEYS.get(sport.value, [])
        if not prefixes:
            return []
        # We don't know all available keys without calling /sports,
        # so return the prefixes as-is (won't match exact keys).
        # For common sports the agents.yaml config should have them.
        return []

    def _fetch_scores(self, sport_key: str) -> list[dict]:
        """Fetch scores for a sport key, with caching."""
        if sport_key in self._cache:
            return self._cache[sport_key]

        url = f"{self._base_url}/sports/{sport_key}/scores"
        try:
            resp = requests.get(
                url,
                params={
                    "apiKey": self._api_key,
                    "daysFrom": self._days_from,
                    "dateFormat": "iso",
                },
                timeout=15,
            )
            resp.raise_for_status()
            events = resp.json()
            self._cache[sport_key] = events
            logger.info(
                "Fetched %d events from %s/scores (daysFrom=%d)",
                len(events), sport_key, self._days_from,
            )
            return events
        except requests.RequestException as exc:
            logger.warning("Failed to fetch scores for %s: %s", sport_key, exc)
            self._cache[sport_key] = []
            return []

    def _match_event(
        self, match: Match, events: list[dict],
    ) -> dict | None:
        """Find the API event that corresponds to a DB match.

        Tries exact team name match first, then falls back to
        case-insensitive substring matching.
        """
        db_home = match.home_team.strip().lower()
        db_away = match.away_team.strip().lower()

        # Pass 1: exact case-insensitive match
        for ev in events:
            api_home = (ev.get("home_team") or "").strip().lower()
            api_away = (ev.get("away_team") or "").strip().lower()
            if api_home == db_home and api_away == db_away:
                return ev

        # Pass 2: substring match (handles "Bayern Munich" vs "FC Bayern München")
        for ev in events:
            api_home = (ev.get("home_team") or "").strip().lower()
            api_away = (ev.get("away_team") or "").strip().lower()
            home_ok = (
                db_home in api_home or api_home in db_home
                or _fuzzy_team_match(db_home, api_home)
            )
            away_ok = (
                db_away in api_away or api_away in db_away
                or _fuzzy_team_match(db_away, api_away)
            )
            if home_ok and away_ok:
                logger.debug(
                    "Fuzzy matched: DB(%s vs %s) → API(%s vs %s)",
                    match.home_team, match.away_team,
                    ev.get("home_team"), ev.get("away_team"),
                )
                return ev

        return None

    def fetch_result(self, match: Match) -> MatchResult | None:
        """Fetch the result for a single match from The Odds API."""
        if not self._api_key:
            return None

        sport_keys = self._sport_keys_for(match.sport)
        if not sport_keys:
            logger.debug("No API sport keys for %s", match.sport.value)
            return None

        for sport_key in sport_keys:
            events = self._fetch_scores(sport_key)
            ev = self._match_event(match, events)
            if ev is None:
                continue

            completed = ev.get("completed", False)
            scores = ev.get("scores")
            if not scores:
                if completed:
                    # Completed but no scores (cancelled?)
                    return MatchResult(
                        home_team=match.home_team,
                        away_team=match.away_team,
                        home_score=0, away_score=0,
                        is_finished=False,
                    )
                return None

            # Parse scores list: [{"name": "Team A", "score": "2"}, ...]
            home_score = 0
            away_score = 0
            api_home = (ev.get("home_team") or "").strip().lower()
            for s in scores:
                sname = (s.get("name") or "").strip().lower()
                sval = int(s.get("score", 0))
                if sname == api_home:
                    home_score = sval
                else:
                    away_score = sval

            return MatchResult(
                home_team=match.home_team,
                away_team=match.away_team,
                home_score=home_score,
                away_score=away_score,
                is_finished=completed,
            )

        return None


def _fuzzy_team_match(name_a: str, name_b: str) -> bool:
    """Check if two team names likely refer to the same team.

    Compares significant tokens (length >= 4) and requires at least one
    overlap.  Handles cases like:
      - "borussia dortmund" vs "bvb dortmund"
      - "charlotte hornets" vs "cha hornets"
    """
    tokens_a = {t for t in name_a.split() if len(t) >= 4}
    tokens_b = {t for t in name_b.split() if len(t) >= 4}
    if not tokens_a or not tokens_b:
        return False
    return bool(tokens_a & tokens_b)


def _get_default_backend() -> ResultsBackend:
    """Return TheOddsAPIResultsBackend if API key is set, else ManualResultsBackend."""
    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    if api_key:
        backend = TheOddsAPIResultsBackend(api_key=api_key)
        logger.info("Using TheOddsAPIResultsBackend for results ingestion")
        return backend
    logger.warning(
        "THE_ODDS_API_KEY not set — using ManualResultsBackend (no-op). "
        "Set THE_ODDS_API_KEY to enable automatic results ingestion."
    )
    return ManualResultsBackend()


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
        backend: Pluggable results source. Defaults to TheOddsAPIResultsBackend
                 when THE_ODDS_API_KEY is set, else ManualResultsBackend (no-op).
        before_date: Only process matches before this date.

    Returns:
        FetchResult summary of the run.
    """
    if backend is None:
        backend = _get_default_backend()

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
