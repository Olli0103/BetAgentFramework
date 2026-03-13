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
import unicodedata
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
    voided_stale_paper: int = 0


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


# ── Unicode-safe team name normalization ─────────────────────────────


def _normalize_name(raw: str) -> str:
    """Normalize a team name for comparison.

    - Strip whitespace
    - Lowercase
    - Unicode NFKD decomposition (ü → u, é → e, etc.)
    - Remove common prefixes/suffixes that vary between sources
    """
    s = raw.strip().lower()
    # Decompose Unicode: "München" → "Munchen", "Nîmes" → "Nimes"
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


def _name_tokens(name: str, min_len: int = 3) -> set[str]:
    """Extract significant tokens from a normalized name.

    Args:
        name: Raw name string.
        min_len: Minimum token length.  Default 3 (was 4) to catch
                 short first names common in tennis (e.g. "Ana", "Ben").
    """
    return {t for t in _normalize_name(name).split() if len(t) >= min_len}


# ── Tennis-specific name matching helpers ─────────────────────────────


def _tennis_name_match(name_a: str, name_b: str) -> bool:
    """Check if two tennis player names refer to the same person.

    Handles abbreviated formats common in different data sources:
      - "J. Sinner" ↔ "Jannik Sinner"
      - "Sinner J." ↔ "Jannik Sinner"
      - "Sinner, Jannik" ↔ "Jannik Sinner"

    Tries both directions (either could be abbreviated).
    """
    from bet_agent.ingest.alias_resolver import (
        is_abbreviated_tennis_name,
        try_match_abbreviated,
    )

    a_stripped = name_a.strip()
    b_stripped = name_b.strip()

    # If either is abbreviated, try matching against the other
    if is_abbreviated_tennis_name(a_stripped):
        if try_match_abbreviated(a_stripped, b_stripped):
            return True
    if is_abbreviated_tennis_name(b_stripped):
        if try_match_abbreviated(b_stripped, a_stripped):
            return True

    return False


def _extract_last_name(name: str) -> str:
    """Extract the last name from a player name for fallback matching.

    Handles common formats:
      - "Jannik Sinner" → "sinner"
      - "J. Sinner" → "sinner"
      - "Sinner J." → "sinner"  (last name is the long part)
      - "Sinner, Jannik" → "sinner"

    Returns lowercase normalized last name.
    """
    s = _normalize_name(name)
    if not s:
        return ""

    # "Lastname, Firstname" → lastname
    if "," in s:
        return s.split(",")[0].strip()

    parts = s.split()
    if len(parts) < 2:
        return s  # single name

    # "Sinner J." pattern — last name is first, initial is short
    # "J. Sinner" pattern — initial is first, last name is last
    # Heuristic: the longest token is the last name
    longest = max(parts, key=len)
    return longest


# ── Sport → Odds API sport keys mapping ──────────────────────────────

_AGENTS_YAML = Path(__file__).resolve().parents[3] / "config" / "agents.yaml"

# Internal Sport enum → Odds API key prefix for dynamic discovery
_SPORT_PREFIX: dict[str, str] = {
    "football": "soccer_",
    "basketball": "basketball_",
    "american_football": "americanfootball_",
    "ice_hockey": "icehockey_",
    "tennis": "tennis_",
    "darts": "darts_",
}

# League hint → preferred API sport key (for faster matching)
_LEAGUE_HINTS: dict[str, str] = {
    # Football / Soccer
    "bundesliga": "soccer_germany_bundesliga",
    "2. bundesliga": "soccer_germany_bundesliga2",
    "premier league": "soccer_epl",
    "epl": "soccer_epl",
    "la liga": "soccer_spain_la_liga",
    "serie a": "soccer_italy_serie_a",
    "champions league": "soccer_uefa_champs_league",
    # Basketball
    "nba": "basketball_nba",
    # American Football
    "nfl": "americanfootball_nfl",
    # Ice Hockey
    "nhl": "icehockey_nhl",
    # Tennis — ATP
    "atp": "tennis_atp_aus_open",
    "atp australian open": "tennis_atp_aus_open",
    "atp french open": "tennis_atp_french_open",
    "atp us open": "tennis_atp_us_open",
    "atp wimbledon": "tennis_atp_wimbledon",
    # Tennis — WTA
    "wta": "tennis_wta_aus_open",
    "wta australian open": "tennis_wta_aus_open",
    "wta french open": "tennis_wta_french_open",
    "wta us open": "tennis_wta_us_open",
    "wta wimbledon": "tennis_wta_wimbledon",
}

# Additional league substrings → sport keys for partial matching.
# Used when exact league name doesn't match any hint above.
_LEAGUE_SUBSTRING_HINTS: dict[str, list[str]] = {
    "atp": ["tennis_atp_"],
    "wta": ["tennis_wta_"],
    "itf": ["tennis_itf_"],
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

    Matching strategy (in priority order):
      0. Direct lookup by ``odds_event_id`` stored in ``Match.live_stats``
      1. Exact home_team + away_team match (Unicode-normalized)
      2. Fuzzy token scoring (shared significant tokens ≥ 4 chars)

    On successful match, binds the event by storing ``odds_event_id``
    and ``odds_sport_key`` in ``Match.live_stats`` for future direct lookups.

    Sport key selection:
      - League hint mapping (e.g. "Bundesliga" → soccer_germany_bundesliga)
      - Explicit keys from agents.yaml
      - Dynamic discovery via ``/v4/sports`` endpoint (prefix-filtered)
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
        # Discovered sport keys from /v4/sports (lazy-loaded)
        self._discovered_keys: dict[str, list[str]] | None = None

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _discover_sport_keys(self) -> dict[str, list[str]]:
        """Fetch available sport keys from /v4/sports and group by prefix."""
        if self._discovered_keys is not None:
            return self._discovered_keys

        self._discovered_keys = {}
        try:
            resp = requests.get(
                f"{self._base_url}/sports",
                params={"apiKey": self._api_key},
                timeout=15,
            )
            resp.raise_for_status()
            sports = resp.json()
            for s in sports:
                key = s.get("key", "")
                for sport_name, prefix in _SPORT_PREFIX.items():
                    if key.startswith(prefix):
                        self._discovered_keys.setdefault(sport_name, []).append(key)
            logger.info(
                "Discovered %d sport keys from /v4/sports",
                sum(len(v) for v in self._discovered_keys.values()),
            )
        except requests.RequestException as exc:
            logger.warning("Failed to discover sport keys: %s", exc)
            self._discovered_keys = {}

        return self._discovered_keys

    def _sport_keys_for(self, sport: Sport, league: str | None = None) -> list[str]:
        """Get API sport keys to query, ordered by relevance.

        Priority:
          1. League hint — exact match (most specific)
          2. League substring hints (e.g. "ATP Roland Garros" matches "atp")
          3. Explicit keys from agents.yaml
          4. Dynamic discovery from /v4/sports (filtered by prefix)
        """
        keys: list[str] = []
        seen: set[str] = set()

        if league:
            league_lower = league.strip().lower()

            # 1. Exact league hint
            hint = _LEAGUE_HINTS.get(league_lower)
            if hint and hint not in seen:
                keys.append(hint)
                seen.add(hint)

            # 2. Substring hint — e.g. league "ATP Indian Wells" contains "atp"
            if not keys:
                for substr, prefixes in _LEAGUE_SUBSTRING_HINTS.items():
                    if substr in league_lower:
                        # Need discovered keys filtered by prefix
                        discovered = self._discover_sport_keys()
                        for k in discovered.get(sport.value, []):
                            for pfx in prefixes:
                                if k.startswith(pfx) and k not in seen:
                                    keys.append(k)
                                    seen.add(k)
                        break

        # 3. Explicit keys from agents.yaml
        for k in self._configured_keys.get(sport.value, []):
            if k not in seen:
                keys.append(k)
                seen.add(k)

        # 4. Dynamic discovery (only if we still have nothing)
        if not keys:
            discovered = self._discover_sport_keys()
            for k in discovered.get(sport.value, []):
                if k not in seen:
                    keys.append(k)
                    seen.add(k)

        return keys

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

        Pass 0: Direct event ID lookup (from previous binding)
        Pass 1: Exact normalized name match
        Pass 2: Tennis abbreviation matching (J. Sinner ↔ Jannik Sinner)
        Pass 3: Fuzzy token scoring + substring containment
        Pass 4: Tennis last-name-only matching (1v1 sport — last names
                 are usually unique within a day's events)
        """
        # Pass 0: direct event ID from live_stats
        bound_id = (match.live_stats or {}).get("odds_event_id")
        if bound_id:
            for ev in events:
                if ev.get("id") == bound_id:
                    return ev

        db_home = _normalize_name(match.home_team)
        db_away = _normalize_name(match.away_team)
        is_tennis = match.sport == Sport.TENNIS

        # Pass 1: exact normalized match
        for ev in events:
            api_home = _normalize_name(ev.get("home_team") or "")
            api_away = _normalize_name(ev.get("away_team") or "")
            if api_home == db_home and api_away == db_away:
                return ev

        # Pass 2: tennis abbreviation matching
        # Handles "J. Sinner" ↔ "Jannik Sinner", "Sinner J." ↔ "Jannik Sinner"
        if is_tennis:
            for ev in events:
                api_home_raw = ev.get("home_team") or ""
                api_away_raw = ev.get("away_team") or ""
                home_ok = (
                    _normalize_name(api_home_raw) == db_home
                    or _tennis_name_match(match.home_team, api_home_raw)
                )
                away_ok = (
                    _normalize_name(api_away_raw) == db_away
                    or _tennis_name_match(match.away_team, api_away_raw)
                )
                if home_ok and away_ok:
                    logger.debug(
                        "Tennis abbreviation matched: DB(%s vs %s) → API(%s vs %s)",
                        match.home_team, match.away_team, api_home_raw, api_away_raw,
                    )
                    return ev

        # Pass 3: fuzzy token scoring
        db_home_tokens = _name_tokens(match.home_team)
        db_away_tokens = _name_tokens(match.away_team)

        best_ev = None
        best_score = 0

        for ev in events:
            api_home_tokens = _name_tokens(ev.get("home_team") or "")
            api_away_tokens = _name_tokens(ev.get("away_team") or "")

            home_overlap = len(db_home_tokens & api_home_tokens)
            away_overlap = len(db_away_tokens & api_away_tokens)

            # Also check substring containment (normalized)
            api_home_norm = _normalize_name(ev.get("home_team") or "")
            api_away_norm = _normalize_name(ev.get("away_team") or "")
            if db_home in api_home_norm or api_home_norm in db_home:
                home_overlap = max(home_overlap, 1)
            if db_away in api_away_norm or api_away_norm in db_away:
                away_overlap = max(away_overlap, 1)

            if home_overlap > 0 and away_overlap > 0:
                score = home_overlap + away_overlap
                if score > best_score:
                    best_score = score
                    best_ev = ev

        if best_ev:
            logger.debug(
                "Fuzzy matched (score=%d): DB(%s vs %s) → API(%s vs %s)",
                best_score, match.home_team, match.away_team,
                best_ev.get("home_team"), best_ev.get("away_team"),
            )
            return best_ev

        # Pass 4: tennis last-name-only match (1v1 — last names unique per day)
        if is_tennis:
            db_home_last = _extract_last_name(match.home_team)
            db_away_last = _extract_last_name(match.away_team)
            if db_home_last and db_away_last:
                for ev in events:
                    api_home_last = _extract_last_name(ev.get("home_team") or "")
                    api_away_last = _extract_last_name(ev.get("away_team") or "")
                    if (
                        api_home_last and api_away_last
                        and db_home_last == api_home_last
                        and db_away_last == api_away_last
                    ):
                        logger.debug(
                            "Tennis last-name matched: DB(%s vs %s) → API(%s vs %s)",
                            match.home_team, match.away_team,
                            ev.get("home_team"), ev.get("away_team"),
                        )
                        return ev

        return None

    def _bind_event(self, match: Match, event: dict, sport_key: str) -> None:
        """Store event binding in Match.live_stats for future direct lookup."""
        event_id = event.get("id")
        if not event_id:
            return

        stats = dict(match.live_stats) if match.live_stats else {}
        if stats.get("odds_event_id") == event_id:
            return  # already bound

        stats["odds_event_id"] = event_id
        stats["odds_sport_key"] = sport_key
        match.live_stats = stats
        logger.debug(
            "Bound match %s vs %s → event %s (%s)",
            match.home_team, match.away_team, event_id, sport_key,
        )

    def fetch_result(self, match: Match) -> MatchResult | None:
        """Fetch the result for a single match from The Odds API."""
        if not self._api_key:
            return None

        # If we have a bound sport key, try that first
        bound_key = (match.live_stats or {}).get("odds_sport_key")
        sport_keys = self._sport_keys_for(match.sport, match.league)
        if bound_key and bound_key not in sport_keys:
            sport_keys.insert(0, bound_key)

        if not sport_keys:
            logger.debug("No API sport keys for %s", match.sport.value)
            return None

        for sport_key in sport_keys:
            events = self._fetch_scores(sport_key)
            ev = self._match_event(match, events)
            if ev is None:
                continue

            # Bind event for future direct lookups
            self._bind_event(match, ev, sport_key)

            completed = ev.get("completed", False)
            scores = ev.get("scores")
            if not scores:
                if completed:
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
            api_home = _normalize_name(ev.get("home_team") or "")
            for s in scores:
                sname = _normalize_name(s.get("name") or "")
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
    """Check if two team/player names likely refer to the same entity.

    Compares significant tokens (length >= 3) after Unicode normalization.
    Also checks tennis abbreviation patterns as a fallback.
    """
    tokens_a = _name_tokens(name_a)
    tokens_b = _name_tokens(name_b)
    if tokens_a and tokens_b and (tokens_a & tokens_b):
        return True
    # Fallback: tennis abbreviation check
    return _tennis_name_match(name_a, name_b)


class APISportsResultsBackend:
    """Fetch results via API-Sports as a fallback backend.

    Uses the APISportsClient to fetch fixture results by date,
    matching against DB matches by team name comparison.
    """

    def __init__(self):
        from bet_agent.tools.api_sports_client import APISportsClient
        self._client = APISportsClient()

    @property
    def is_available(self) -> bool:
        return self._client.is_available

    def fetch_result(self, match: Match) -> MatchResult | None:
        if not self._client.is_available:
            return None

        date_str = match.scheduled_at.strftime("%Y-%m-%d")
        fixtures = self._client.fetch_fixtures_by_date(match.sport.value, date_str)
        if not fixtures:
            return None

        from bet_agent.tools.backfill_engine import _match_fixture_to_db
        api_match = _match_fixture_to_db(match, fixtures, match.sport.value)
        if not api_match:
            return None

        # Extract scores
        goals = api_match.get("goals", {})
        if isinstance(goals, dict) and goals.get("home") is not None:
            home_score = goals["home"]
            away_score = goals.get("away", 0)
        else:
            scores = api_match.get("scores", {})
            if isinstance(scores, dict):
                hs = scores.get("home")
                home_score = hs.get("total", 0) if isinstance(hs, dict) else (hs or 0)
                aws = scores.get("away")
                away_score = aws.get("total", 0) if isinstance(aws, dict) else (aws or 0)
            else:
                return None

        # Check if finished
        status = api_match.get("fixture", {}).get("status", {})
        short_status = status.get("short", "") if isinstance(status, dict) else ""
        is_finished = short_status in ("FT", "AET", "PEN", "AOT")

        # Store fixture ID for provenance
        fix_data = api_match.get("fixture", api_match)
        if isinstance(fix_data, dict) and fix_data.get("id"):
            stats = dict(match.live_stats) if match.live_stats else {}
            stats["api_sports_fixture_id"] = fix_data["id"]
            data_sources = stats.get("data_sources", [])
            if "api_sports" not in data_sources:
                data_sources.append("api_sports")
            stats["data_sources"] = data_sources
            match.live_stats = stats

        return MatchResult(
            home_team=match.home_team,
            away_team=match.away_team,
            home_score=home_score,
            away_score=away_score,
            is_finished=is_finished,
        )


class ChainedResultsBackend:
    """Try multiple backends in order, return first successful result."""

    def __init__(self, backends: list):
        self._backends = backends

    def fetch_result(self, match: Match) -> MatchResult | None:
        for backend in self._backends:
            try:
                result = backend.fetch_result(match)
                if result is not None:
                    return result
            except Exception as exc:
                logger.warning(
                    "Backend %s failed for %s vs %s: %s",
                    type(backend).__name__, match.home_team, match.away_team, exc,
                )
        return None


def _get_default_backend() -> ResultsBackend:
    """Return a chained backend: TheOddsAPI → API-Sports → Manual.

    Priority:
      1. TheOddsAPIResultsBackend (if THE_ODDS_API_KEY set)
      2. APISportsResultsBackend (if API_SPORTS_KEY set)
      3. ManualResultsBackend (no-op fallback)
    """
    backends = []

    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    if api_key:
        backends.append(TheOddsAPIResultsBackend(api_key=api_key))
        logger.info("Results backend: TheOddsAPI (primary)")

    api_sports_backend = APISportsResultsBackend()
    if api_sports_backend.is_available:
        backends.append(api_sports_backend)
        logger.info("Results backend: API-Sports (fallback)")

    if not backends:
        logger.warning(
            "No API keys set — using ManualResultsBackend (no-op). "
            "Set THE_ODDS_API_KEY or API_SPORTS_KEY for automatic results."
        )
        return ManualResultsBackend()

    return ChainedResultsBackend(backends)


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

    # Void stale unmatched PAPER bets that are beyond the API's score window
    # Void stale unmatched PAPER bets that are beyond the API's score window
    voided = _void_stale_unmatched_paper_bets(session)

    session.flush()
    return FetchResult(
        matches_checked=len(matches),
        matches_updated=updated,
        matches_not_found=not_found,
        voided_stale_paper=voided,
    )


# ── Stale paper bet terminal fallback ───────────────────────────────

# TheOddsAPI only returns scores for completed matches up to 3 days old.
# PAPER bets on matches older than this that still have no result are
# irrecoverable — void them so they don't block metrics forever.
_STALE_PAPER_DAYS = int(os.environ.get("BETAGENT_STALE_PAPER_DAYS", "4"))


def _void_stale_unmatched_paper_bets(session: Session) -> int:
    """Void PAPER bets on unfinished matches older than the score window.

    These are bets where the API score feed no longer returns the match
    (e.g. tennis bracket advanced, event removed). Since they're PAPER
    (simulated) and unresolvable, voiding is the cleanest outcome.

    REAL bets are never auto-voided here — they need manual review.
    """
    from bet_agent.db.models import LedgerType

    cutoff = datetime.now(timezone.utc) - timedelta(days=_STALE_PAPER_DAYS)

    stale_bets = list(
        session.execute(
            select(PlacedBet)
            .join(Match, Match.id == PlacedBet.match_id)
            .where(
                PlacedBet.ledger_type == LedgerType.PAPER,
                PlacedBet.status == BetStatus.PLACED,
                Match.match_state != MatchState.FINISHED,
                Match.scheduled_at < cutoff,
            )
        ).scalars().all()
    )

    for bet in stale_bets:
        bet.status = BetStatus.VOID
        bet.pnl_eur = bet.stake_eur  # refund stake (deduct-at-placement model)
        bet.resolved_at = datetime.now(timezone.utc)

        # Refund stake to PAPER ledger
        from bet_agent.db.models import BankrollLedger
        paper_ledger = session.execute(
            select(BankrollLedger).where(
                BankrollLedger.ledger_type == LedgerType.PAPER,
            )
        ).scalar_one_or_none()
        if paper_ledger:
            paper_ledger.balance += bet.stake_eur

        logger.info(
            "Voided stale PAPER bet: %s @%.2f (%.2f EUR) — match >%dd old, no result",
            bet.selection, float(bet.odds_at_placement),
            float(bet.stake_eur), _STALE_PAPER_DAYS,
        )

    if stale_bets:
        logger.info("Voided %d stale unmatched PAPER bet(s)", len(stale_bets))

    return len(stale_bets)
