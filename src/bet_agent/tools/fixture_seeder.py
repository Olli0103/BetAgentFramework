"""Fixture Seeder — populate Match rows for an operational window.

The backfill engine only updates *existing* Match rows (gap-filling).
This module creates the rows in the first place by fetching upcoming
fixtures from external APIs and upserting them into the Match table.

Sources (priority order):
  1. TheOddsAPI /v4/sports/{key}/events — all sports including tennis
  2. API-Sports fetch_fixtures_by_date — football, basketball, hockey, NFL

Idempotent via the ``uq_match_identity`` unique constraint:
  (sport, league, home_team, away_team, scheduled_at)

Environment:
  THE_ODDS_API_KEY — enables TheOddsAPI seeding (primary)
  API_SPORTS_KEY   — enables API-Sports seeding (fallback)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import requests
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from bet_agent.db.models import Match, MatchState, Sport

logger = logging.getLogger(__name__)

# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class SeedResult:
    """Summary of a fixture seeding run."""

    window_start: str
    window_end: str
    fixtures_fetched: int = 0
    fixtures_inserted: int = 0
    fixtures_skipped: int = 0
    api_calls: int = 0
    sources_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sport_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "fixtures_fetched": self.fixtures_fetched,
            "fixtures_inserted": self.fixtures_inserted,
            "fixtures_skipped": self.fixtures_skipped,
            "api_calls": self.api_calls,
            "sources_used": self.sources_used,
            "errors": self.errors[:20],
            "sport_counts": self.sport_counts,
        }


# ── Sport enum mapping ───────────────────────────────────────────────────

# TheOddsAPI sport key prefix → internal Sport enum
_PREFIX_TO_SPORT: dict[str, Sport] = {
    "soccer_": Sport.FOOTBALL,
    "basketball_": Sport.BASKETBALL,
    "americanfootball_": Sport.AMERICAN_FOOTBALL,
    "icehockey_": Sport.ICE_HOCKEY,
    "tennis_": Sport.TENNIS,
    "darts_": Sport.DARTS,
}

# API-Sports sport value → internal Sport enum
_API_SPORTS_TO_SPORT: dict[str, Sport] = {
    "football": Sport.FOOTBALL,
    "basketball": Sport.BASKETBALL,
    "ice_hockey": Sport.ICE_HOCKEY,
    "american_football": Sport.AMERICAN_FOOTBALL,
}


def _sport_from_odds_key(sport_key: str) -> Sport | None:
    """Map an OddsAPI sport key (e.g. 'soccer_germany_bundesliga') to Sport enum."""
    for prefix, sport in _PREFIX_TO_SPORT.items():
        if sport_key.startswith(prefix):
            return sport
    return None


def _league_from_odds_key(sport_key: str) -> str:
    """Extract a league name from an OddsAPI sport key.

    e.g. 'soccer_germany_bundesliga' → 'Germany Bundesliga'
         'tennis_atp_french_open'    → 'ATP French Open'
    """
    for prefix in _PREFIX_TO_SPORT:
        if sport_key.startswith(prefix):
            remainder = sport_key[len(prefix):]
            return remainder.replace("_", " ").title()
    return sport_key.replace("_", " ").title()


# ── TheOddsAPI fixture fetching ──────────────────────────────────────────


def _fetch_odds_api_sport_keys(api_key: str) -> list[str]:
    """Discover all active sport keys from /v4/sports."""
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        keys = [s["key"] for s in resp.json() if not s.get("has_outrights")]
        logger.info("OddsAPI: discovered %d active sport keys", len(keys))
        return keys
    except requests.RequestException as exc:
        logger.warning("OddsAPI sport discovery failed: %s", exc)
        return []


def _fetch_odds_api_events(
    api_key: str,
    sport_key: str,
) -> list[dict]:
    """Fetch upcoming events for a sport key from /v4/sports/{key}/events.

    Returns raw event dicts with home_team, away_team, commence_time, etc.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    try:
        resp = requests.get(
            url,
            params={"apiKey": api_key, "dateFormat": "iso"},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
        return events
    except requests.RequestException as exc:
        logger.warning("OddsAPI events fetch failed for %s: %s", sport_key, exc)
        return []


def _parse_odds_api_event(event: dict, sport_key: str) -> dict | None:
    """Parse a single OddsAPI event into a Match-compatible dict.

    Returns None if the event cannot be parsed.
    """
    sport = _sport_from_odds_key(sport_key)
    if sport is None:
        return None

    home = event.get("home_team", "")
    away = event.get("away_team", "")
    commence = event.get("commence_time", "")

    if not home or not away or not commence:
        return None

    # Parse ISO timestamp
    try:
        scheduled_at = datetime.fromisoformat(commence.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    league = _league_from_odds_key(sport_key)

    return {
        "sport": sport,
        "league": league,
        "home_team": home,
        "away_team": away,
        "scheduled_at": scheduled_at,
        "source": "the_odds_api",
        "odds_event_id": event.get("id"),
        "odds_sport_key": sport_key,
    }


# ── API-Sports fixture fetching ──────────────────────────────────────────


def _fetch_api_sports_fixtures(
    date_str: str,
    sports: list[str] | None = None,
) -> list[dict]:
    """Fetch fixtures from API-Sports for a date, returning Match-compatible dicts."""
    from bet_agent.tools.api_sports_client import APISportsClient, _SPORT_TO_API

    client = APISportsClient()
    if not client.is_available:
        return []

    target_sports = sports or list(_SPORT_TO_API.keys())
    results: list[dict] = []

    for sport_val in target_sports:
        if sport_val not in _SPORT_TO_API:
            continue
        if not client.remaining_budget(_SPORT_TO_API[sport_val]):
            logger.info("API-Sports budget exhausted for %s", sport_val)
            continue

        fixtures = client.fetch_fixtures_by_date(sport_val, date_str)
        sport_enum = _API_SPORTS_TO_SPORT.get(sport_val)
        if not sport_enum:
            continue

        for fix in fixtures:
            parsed = _parse_api_sports_fixture(fix, sport_enum)
            if parsed:
                results.append(parsed)

    return results


def _parse_api_sports_fixture(fixture: dict, sport: Sport) -> dict | None:
    """Parse an API-Sports fixture into a Match-compatible dict."""
    teams = fixture.get("teams", {})
    home_name = teams.get("home", {}).get("name", "")
    away_name = teams.get("away", {}).get("name", "")

    if not home_name or not away_name:
        return None

    # Extract scheduled time
    fix_data = fixture.get("fixture", fixture)
    date_str = fix_data.get("date", "")
    if not date_str:
        return None

    try:
        scheduled_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    # Extract league
    league_data = fixture.get("league", {})
    league = league_data.get("name", "Unknown")
    country = league_data.get("country", "")
    if country:
        league = f"{country} {league}"

    return {
        "sport": sport,
        "league": league,
        "home_team": home_name,
        "away_team": away_name,
        "scheduled_at": scheduled_at,
        "source": "api_sports",
        "api_sports_fixture_id": fix_data.get("id"),
    }


# ── Core seeding function ────────────────────────────────────────────────


def seed_fixtures_for_window(
    session: Session,
    window_start: datetime,
    window_end: datetime,
    sports: list[str] | None = None,
) -> SeedResult:
    """Seed Match rows for all fixtures in an operational window.

    Fetches upcoming fixtures from TheOddsAPI (primary) and API-Sports
    (fallback), then upserts into the Match table. Idempotent via the
    ``uq_match_identity`` unique constraint — safe for repeated cron runs.

    Args:
        session: SQLAlchemy session (caller manages transaction).
        window_start: Window start (inclusive), typically today 07:00 UTC.
        window_end: Window end (exclusive), typically tomorrow 07:00 UTC.
        sports: Optional list of Sport enum values to seed. Defaults to all.

    Returns:
        SeedResult summary.
    """
    result = SeedResult(
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
    )

    sport_filter = set(sports) if sports else None
    candidates: list[dict] = []

    # ── Source 1: TheOddsAPI (all sports including tennis) ────────────
    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    if api_key:
        result.sources_used.append("the_odds_api")
        sport_keys = _fetch_odds_api_sport_keys(api_key)
        result.api_calls += 1

        for sk in sport_keys:
            sport_enum = _sport_from_odds_key(sk)
            if sport_enum is None:
                continue
            if sport_filter and sport_enum.value not in sport_filter:
                continue

            events = _fetch_odds_api_events(api_key, sk)
            result.api_calls += 1

            for ev in events:
                parsed = _parse_odds_api_event(ev, sk)
                if parsed:
                    candidates.append(parsed)

        logger.info(
            "OddsAPI: %d candidate fixtures from %d sport keys",
            len(candidates), len(sport_keys),
        )

    # ── Source 2: API-Sports (football/basketball/hockey/NFL) ─────────
    # Collect dates in window for API-Sports queries
    window_dates: list[str] = []
    current = window_start.date()
    while current <= window_end.date():
        window_dates.append(current.isoformat())
        current += timedelta(days=1)

    api_sports_candidates: list[dict] = []
    target_sports = (
        [s for s in _API_SPORTS_TO_SPORT if not sport_filter or s in sport_filter]
    )
    if target_sports:
        for date_str in window_dates:
            fixtures = _fetch_api_sports_fixtures(date_str, target_sports)
            if fixtures:
                result.api_calls += 1
                if "api_sports" not in result.sources_used:
                    result.sources_used.append("api_sports")
                api_sports_candidates.extend(fixtures)

        if api_sports_candidates:
            logger.info(
                "API-Sports: %d candidate fixtures for %d dates",
                len(api_sports_candidates), len(window_dates),
            )

    # ── Filter to window ─────────────────────────────────────────────
    all_candidates = candidates + api_sports_candidates
    in_window: list[dict] = []
    for c in all_candidates:
        sched = c["scheduled_at"]
        if window_start <= sched < window_end:
            in_window.append(c)

    result.fixtures_fetched = len(in_window)
    logger.info(
        "Seeder: %d fixtures in window %s → %s",
        len(in_window), window_start.isoformat(), window_end.isoformat(),
    )

    if not in_window:
        return result

    # ── Deduplicate by (sport, home_team, away_team, scheduled_at) ───
    # Prefer TheOddsAPI over API-Sports when both have the same fixture
    seen: set[tuple] = set()
    unique: list[dict] = []
    for c in in_window:
        key = (
            c["sport"].value,
            c["home_team"].strip().lower(),
            c["away_team"].strip().lower(),
            c["scheduled_at"].isoformat(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # ── Upsert into Match table ──────────────────────────────────────
    for c in unique:
        existing = session.execute(
            select(Match).where(
                Match.sport == c["sport"],
                Match.league == c["league"],
                Match.home_team == c["home_team"],
                Match.away_team == c["away_team"],
                Match.scheduled_at == c["scheduled_at"],
            )
        ).scalar_one_or_none()

        if existing:
            result.fixtures_skipped += 1
            # Update provenance if missing
            if not existing.source:
                existing.source = c["source"]
            # Store OddsAPI event binding for future direct lookups
            if c.get("odds_event_id") and existing.live_stats:
                stats = dict(existing.live_stats)
                if "odds_event_id" not in stats:
                    stats["odds_event_id"] = c["odds_event_id"]
                    stats["odds_sport_key"] = c.get("odds_sport_key", "")
                    existing.live_stats = stats
            elif c.get("odds_event_id") and not existing.live_stats:
                existing.live_stats = {
                    "odds_event_id": c["odds_event_id"],
                    "odds_sport_key": c.get("odds_sport_key", ""),
                    "data_sources": [c["source"]],
                }
            continue

        # Insert new Match row
        live_stats: dict = {"data_sources": [c["source"]]}
        if c.get("odds_event_id"):
            live_stats["odds_event_id"] = c["odds_event_id"]
            live_stats["odds_sport_key"] = c.get("odds_sport_key", "")
        if c.get("api_sports_fixture_id"):
            live_stats["api_sports_fixture_id"] = c["api_sports_fixture_id"]

        match = Match(
            sport=c["sport"],
            league=c["league"],
            home_team=c["home_team"],
            away_team=c["away_team"],
            scheduled_at=c["scheduled_at"],
            match_state=MatchState.NOT_STARTED,
            is_live=False,
            source=c["source"],
            live_stats=live_stats,
        )
        session.add(match)
        result.fixtures_inserted += 1

        sport_val = c["sport"].value
        result.sport_counts[sport_val] = result.sport_counts.get(sport_val, 0) + 1

    session.flush()

    logger.info(
        "Seeder complete: %d fetched, %d inserted, %d skipped (sports: %s)",
        result.fixtures_fetched,
        result.fixtures_inserted,
        result.fixtures_skipped,
        result.sport_counts,
    )

    return result


# ── Convenience function for operational window ──────────────────────────


def seed_today_window(
    session: Session,
    sports: list[str] | None = None,
) -> SeedResult:
    """Seed fixtures for today's operational window (07:00 UTC → next day 06:59 UTC).

    Convenience wrapper around seed_fixtures_for_window().
    """
    now = datetime.now(timezone.utc)
    today_7am = datetime.combine(now.date(), datetime.min.time().replace(hour=7), tzinfo=timezone.utc)

    if now < today_7am:
        # Before 07:00 — window is yesterday 07:00 → today 07:00
        window_start = today_7am - timedelta(days=1)
        window_end = today_7am
    else:
        # After 07:00 — window is today 07:00 → tomorrow 07:00
        window_start = today_7am
        window_end = today_7am + timedelta(days=1)

    return seed_fixtures_for_window(session, window_start, window_end, sports)
