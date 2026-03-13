"""Flashscore / Sofascore Tennis Results Backend.

Third-priority fallback for tennis match results, activated after
TheOddsAPI (3-day window) and API-Sports (no tennis support) fail.

Strategy:
  1. Sofascore JSON API — clean REST endpoints, cached per day
  2. Flashscore via Cloudflare Browser Rendering — async batch fallback

Only handles **tennis** matches.  Other sports have reliable
backends (TheOddsAPI + API-Sports) and should never reach this layer.

Environment variables:
  SOFASCORE_ENABLED=1      Enable Sofascore backend (default: enabled)
  FLASHSCORE_CF_ENABLED=1  Enable Cloudflare-based Flashscore (default: disabled)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests

from bet_agent.db.models import Match, Sport
from bet_agent.tools.results_fetcher import (
    MatchResult,
    _extract_last_name,
    _normalize_name,
    _tennis_name_match,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

_SOFASCORE_BASE = "https://api.sofascore.com/api/v1"

_SOFASCORE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "Cache-Control": "no-cache",
}

# Sofascore status codes that indicate a finished match
_FINISHED_CODES = {100}  # 100 = "Ended"
_FINISHED_DESCRIPTIONS = {"ended", "finished", "retired", "walkover", "defaulted"}

_REQUEST_TIMEOUT = 15


# ── Data structures ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SofascoreEvent:
    """Parsed Sofascore event for internal matching."""

    event_id: int
    home_name: str
    away_name: str
    home_score: int
    away_score: int
    is_finished: bool
    tournament: str


# ── Sofascore API fetching ───────────────────────────────────────────────


def _fetch_sofascore_tennis_events(
    target_date: date,
) -> list[_SofascoreEvent]:
    """Fetch all tennis events for a date from Sofascore API.

    Endpoint: /sport/tennis/scheduled-events/{YYYY-MM-DD}
    Returns parsed events with scores and status.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"{_SOFASCORE_BASE}/sport/tennis/scheduled-events/{date_str}"

    try:
        resp = requests.get(url, headers=_SOFASCORE_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("Sofascore API request failed: %s", exc)
        return []

    events_raw = data.get("events", [])
    parsed: list[_SofascoreEvent] = []

    for ev in events_raw:
        try:
            home_team = ev.get("homeTeam", {})
            away_team = ev.get("awayTeam", {})
            home_score_data = ev.get("homeScore", {})
            away_score_data = ev.get("awayScore", {})
            status = ev.get("status", {})

            # Extract player names (prefer shortName, fall back to name)
            home_name = home_team.get("shortName") or home_team.get("name", "")
            away_name = away_team.get("shortName") or away_team.get("name", "")

            if not home_name or not away_name:
                continue

            # Tennis scores: "current" is total sets won
            home_score = home_score_data.get("current", 0) or 0
            away_score = away_score_data.get("current", 0) or 0

            # Check if finished
            status_code = status.get("code", 0)
            status_desc = (status.get("description") or "").lower()
            is_finished = (
                status_code in _FINISHED_CODES
                or status_desc in _FINISHED_DESCRIPTIONS
            )

            tournament_name = ev.get("tournament", {}).get("name", "")

            parsed.append(_SofascoreEvent(
                event_id=ev.get("id", 0),
                home_name=home_name,
                away_name=away_name,
                home_score=int(home_score),
                away_score=int(away_score),
                is_finished=is_finished,
                tournament=tournament_name,
            ))

        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed Sofascore event: %s", exc)
            continue

    logger.info(
        "Sofascore: fetched %d tennis events for %s (%d finished)",
        len(parsed), date_str,
        sum(1 for e in parsed if e.is_finished),
    )
    return parsed


# ── Match reconciliation ─────────────────────────────────────────────────


def _match_sofascore_event(
    match: Match,
    events: list[_SofascoreEvent],
) -> _SofascoreEvent | None:
    """Find the Sofascore event matching a DB match.

    Multi-pass strategy (same as TheOddsAPI matcher):
      1. Exact normalized name match
      2. Tennis abbreviation matching (J. Sinner <-> Jannik Sinner)
      3. Last-name-only matching (1v1 sport)
    """
    db_home = _normalize_name(match.home_team)
    db_away = _normalize_name(match.away_team)

    # Pass 1: exact normalized match
    for ev in events:
        if (
            _normalize_name(ev.home_name) == db_home
            and _normalize_name(ev.away_name) == db_away
        ):
            return ev

    # Pass 2: tennis abbreviation matching
    for ev in events:
        home_ok = (
            _normalize_name(ev.home_name) == db_home
            or _tennis_name_match(match.home_team, ev.home_name)
        )
        away_ok = (
            _normalize_name(ev.away_name) == db_away
            or _tennis_name_match(match.away_team, ev.away_name)
        )
        if home_ok and away_ok:
            logger.debug(
                "Sofascore tennis abbreviation match: "
                "DB(%s vs %s) -> API(%s vs %s)",
                match.home_team, match.away_team,
                ev.home_name, ev.away_name,
            )
            return ev

    # Pass 3: last-name-only fallback
    db_home_last = _extract_last_name(match.home_team)
    db_away_last = _extract_last_name(match.away_team)
    if db_home_last and db_away_last:
        for ev in events:
            api_home_last = _extract_last_name(ev.home_name)
            api_away_last = _extract_last_name(ev.away_name)
            if (
                api_home_last and api_away_last
                and db_home_last == api_home_last
                and db_away_last == api_away_last
            ):
                logger.debug(
                    "Sofascore last-name match: "
                    "DB(%s vs %s) -> API(%s vs %s)",
                    match.home_team, match.away_team,
                    ev.home_name, ev.away_name,
                )
                return ev

    return None


# ── Backend class ────────────────────────────────────────────────────────


class FlashscoreResultsBackend:
    """Tennis results backend using Sofascore API as primary source.

    Only processes tennis matches — returns None immediately for other sports.
    Caches API responses per date to minimize requests (one call covers
    all tennis events for a given day).

    Implements the ResultsBackend protocol:
        fetch_result(match: Match) -> MatchResult | None
    """

    def __init__(self) -> None:
        self._enabled = os.environ.get("SOFASCORE_ENABLED", "1") == "1"
        # Cache: date_str -> list of events (one API call per day)
        self._cache: dict[str, list[_SofascoreEvent]] = {}

    @property
    def is_available(self) -> bool:
        return self._enabled

    def _get_events(self, target_date: date) -> list[_SofascoreEvent]:
        """Get events for a date, using cache if available."""
        date_key = target_date.strftime("%Y-%m-%d")
        if date_key not in self._cache:
            self._cache[date_key] = _fetch_sofascore_tennis_events(target_date)
        return self._cache[date_key]

    def fetch_result(self, match: Match) -> MatchResult | None:
        """Fetch tennis result from Sofascore.

        Returns None immediately for non-tennis matches or if disabled.
        """
        if not self._enabled:
            return None

        # Only handle tennis
        if match.sport != Sport.TENNIS:
            return None

        match_date = match.scheduled_at.date()
        events = self._get_events(match_date)

        if not events:
            return None

        event = _match_sofascore_event(match, events)
        if event is None:
            logger.debug(
                "Sofascore: no match for %s vs %s on %s",
                match.home_team, match.away_team, match_date,
            )
            return None

        if not event.is_finished:
            logger.debug(
                "Sofascore: found %s vs %s but not finished yet",
                match.home_team, match.away_team,
            )
            return None

        # Store provenance
        stats = dict(match.live_stats) if match.live_stats else {}
        stats["sofascore_event_id"] = event.event_id
        data_sources = stats.get("data_sources", [])
        if "sofascore" not in data_sources:
            data_sources.append("sofascore")
        stats["data_sources"] = data_sources
        match.live_stats = stats

        logger.info(
            "Sofascore result: %s vs %s -> %d-%d (tournament: %s)",
            match.home_team, match.away_team,
            event.home_score, event.away_score,
            event.tournament,
        )

        return MatchResult(
            home_team=match.home_team,
            away_team=match.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            is_finished=True,
        )
