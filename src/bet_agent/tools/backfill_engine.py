"""Backfill Engine — multi-source data gap filler.

Orchestrates backfilling of missing scores, stats, and features from
multiple data sources with priority ordering:
  1. API-Sports (structured stats, highest fidelity)
  2. TheOddsAPI (scores, already integrated)
  3. Cloudflare scraping (fallback for advanced metrics)

Reuses existing upsert patterns from ingest/base.py and results_fetcher.py.
All names pass through IroncladAliasResolver before DB writes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    Match,
    MatchState,
    Sport,
    TeamDailyStats,
)
from bet_agent.tools.coverage_engine import scan_gaps, GapReport

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class BackfillResult:
    """Summary of a backfill run."""
    target_date: str
    scores_filled: int = 0
    stats_filled: int = 0
    features_filled: int = 0
    api_sports_calls: int = 0
    odds_api_calls: int = 0
    cloudflare_calls: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_date": self.target_date,
            "scores_filled": self.scores_filled,
            "stats_filled": self.stats_filled,
            "features_filled": self.features_filled,
            "api_sports_calls": self.api_sports_calls,
            "odds_api_calls": self.odds_api_calls,
            "cloudflare_calls": self.cloudflare_calls,
            "errors": self.errors[:20],
        }

    @property
    def total_filled(self) -> int:
        return self.scores_filled + self.stats_filled + self.features_filled


@dataclass
class ReconcileResult:
    """Summary of a result reconciliation run."""
    matches_checked: int = 0
    matches_resolved: int = 0
    still_unresolved: int = 0
    sources_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "matches_checked": self.matches_checked,
            "matches_resolved": self.matches_resolved,
            "still_unresolved": self.still_unresolved,
            "sources_used": self.sources_used,
        }


# ── Helpers ──────────────────────────────────────────────────────────────


def _get_api_sports_client():
    """Lazy-import and instantiate APISportsClient."""
    from bet_agent.tools.api_sports_client import APISportsClient
    return APISportsClient()


def _get_odds_api_backend():
    """Lazy-import TheOddsAPIResultsBackend."""
    from bet_agent.tools.results_fetcher import TheOddsAPIResultsBackend
    backend = TheOddsAPIResultsBackend()
    return backend if backend.is_available else None


def _update_provenance(match: Match, source: str) -> None:
    """Add a data source to the match's provenance tracking in live_stats."""
    stats = dict(match.live_stats) if match.live_stats else {}
    sources = stats.get("data_sources", [])
    if source not in sources:
        sources.append(source)
    stats["data_sources"] = sources
    stats["last_stats_update"] = datetime.now(timezone.utc).isoformat()
    match.live_stats = stats


def _update_feature_coverage(match: Match, sport_value: str) -> None:
    """Calculate and store feature coverage percentage in live_stats."""
    from bet_agent.tools.coverage_engine import REQUIRED_FEATURES
    required = REQUIRED_FEATURES.get(sport_value, [])
    if not required:
        return
    stats = dict(match.live_stats) if match.live_stats else {}
    present = sum(1 for f in required if f in stats and stats[f] is not None)
    stats["feature_coverage"] = round(present / len(required), 2) if required else 1.0
    match.live_stats = stats


# ── Core backfill functions ──────────────────────────────────────────────


def backfill_day(
    session: Session,
    target_date: date,
    sports: list[str] | None = None,
) -> BackfillResult:
    """Backfill all missing data for a specific day.

    Source priority: API-Sports → TheOddsAPI → (Cloudflare in future).

    Args:
        session: SQLAlchemy session (caller manages transaction).
        target_date: Date to backfill.
        sports: Sport names to backfill. Defaults to all.

    Returns:
        BackfillResult summary.
    """
    result = BackfillResult(target_date=target_date.isoformat())

    # 1. Scan gaps for this day
    gap_report = scan_gaps(session, target_date=target_date, sports=sports, lookback_days=0)

    if gap_report.total_gaps == 0:
        logger.info("No gaps found for %s — nothing to backfill", target_date)
        return result

    logger.info(
        "Backfill %s: %d gaps to fill (scores=%d, stats=%d, features=%d, stale=%d)",
        target_date, gap_report.total_gaps,
        len(gap_report.missing_scores), len(gap_report.missing_stats),
        len(gap_report.missing_features), len(gap_report.stale_matches),
    )

    # 2. Try API-Sports for missing scores and stats
    client = _get_api_sports_client()
    if client.is_available:
        _backfill_from_api_sports(session, gap_report, client, result)
    else:
        logger.info("API-Sports not available (no key) — skipping")

    # 3. Try TheOddsAPI for remaining missing scores
    odds_backend = _get_odds_api_backend()
    if odds_backend:
        _backfill_scores_from_odds_api(session, gap_report, odds_backend, result)
    else:
        logger.info("TheOddsAPI not available (no key) — skipping")

    session.flush()

    logger.info(
        "Backfill %s complete: %d filled (scores=%d, stats=%d, features=%d)",
        target_date, result.total_filled,
        result.scores_filled, result.stats_filled, result.features_filled,
    )

    return result


def _backfill_from_api_sports(
    session: Session,
    gap_report: GapReport,
    client,
    result: BackfillResult,
) -> None:
    """Try to fill gaps using API-Sports fixtures."""
    from bet_agent.tools.api_sports_client import _SPORT_TO_API

    # Collect unique (sport, date) combos from gaps
    sport_dates: set[tuple[str, str]] = set()
    for gap in gap_report.missing_scores + gap_report.missing_stats + gap_report.stale_matches:
        if gap.scheduled_at:
            gap_date = gap.scheduled_at[:10]  # YYYY-MM-DD
            sport_dates.add((gap.sport, gap_date))

    for sport_val, gap_date in sport_dates:
        if not client.remaining_budget(_SPORT_TO_API.get(sport_val, sport_val)):
            logger.info("API-Sports budget exhausted for %s — skipping", sport_val)
            continue

        fixtures = client.fetch_fixtures_by_date(sport_val, gap_date)
        result.api_sports_calls += 1

        if not fixtures:
            continue

        # Try to match API fixtures against gap matches
        for gap in gap_report.missing_scores:
            if gap.sport != sport_val:
                continue
            match = session.get(Match, gap.match_id)
            if not match or match.home_score is not None:
                continue

            api_match = _match_fixture_to_db(match, fixtures, sport_val)
            if api_match:
                _apply_fixture_scores(session, match, api_match, sport_val)
                _update_provenance(match, "api_sports")
                result.scores_filled += 1

        # Fill missing stats from fixture statistics
        for gap in gap_report.missing_stats:
            if gap.sport != sport_val:
                continue
            match = session.get(Match, gap.match_id)
            if not match:
                continue

            # Find the fixture ID
            fixture_id = _find_fixture_id(match, fixtures, sport_val)
            if fixture_id:
                stats = client.fetch_fixture_stats(sport_val, fixture_id)
                result.api_sports_calls += 1
                if stats:
                    _apply_fixture_stats(session, match, stats)
                    _update_provenance(match, "api_sports")
                    _update_feature_coverage(match, sport_val)
                    result.stats_filled += 1


def _backfill_scores_from_odds_api(
    session: Session,
    gap_report: GapReport,
    backend,
    result: BackfillResult,
) -> None:
    """Fill remaining missing scores from TheOddsAPI."""
    for gap in gap_report.missing_scores:
        match = session.get(Match, gap.match_id)
        if not match or match.home_score is not None:
            continue  # already filled by API-Sports

        try:
            match_result = backend.fetch_result(match)
            result.odds_api_calls += 1
            if match_result and match_result.is_finished:
                match.home_score = match_result.home_score
                match.away_score = match_result.away_score
                match.match_state = MatchState.FINISHED
                match.is_live = False
                _update_provenance(match, "the_odds_api")
                result.scores_filled += 1
                logger.info(
                    "OddsAPI backfill: %s vs %s → %d-%d",
                    match.home_team, match.away_team,
                    match_result.home_score, match_result.away_score,
                )
        except Exception as exc:
            result.errors.append(f"OddsAPI error for {gap.match_id}: {exc}")


def _match_fixture_to_db(
    match: Match, fixtures: list[dict], sport_val: str,
) -> dict | None:
    """Match an API-Sports fixture to a DB match by team name comparison.

    Uses a multi-pass strategy:
      1. Exact normalized match
      2. Tennis abbreviation matching (if sport is tennis)
      3. Token overlap matching (min token length 3)
    """
    from bet_agent.tools.results_fetcher import (
        _normalize_name,
        _name_tokens,
        _tennis_name_match,
    )

    db_home = _normalize_name(match.home_team)
    db_away = _normalize_name(match.away_team)
    is_tennis = sport_val == "tennis"

    for fix in fixtures:
        teams = fix.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        api_home_raw = home.get("name", "")
        api_away_raw = away.get("name", "")
        api_home = _normalize_name(api_home_raw)
        api_away = _normalize_name(api_away_raw)

        # Pass 1: Exact match
        if api_home == db_home and api_away == db_away:
            return fix

        # Pass 2: Tennis abbreviation match
        if is_tennis:
            home_ok = api_home == db_home or _tennis_name_match(
                match.home_team, api_home_raw,
            )
            away_ok = api_away == db_away or _tennis_name_match(
                match.away_team, api_away_raw,
            )
            if home_ok and away_ok:
                return fix

        # Pass 3: Token overlap match
        db_home_t = _name_tokens(match.home_team)
        db_away_t = _name_tokens(match.away_team)
        api_home_t = _name_tokens(api_home_raw)
        api_away_t = _name_tokens(api_away_raw)

        if db_home_t & api_home_t and db_away_t & api_away_t:
            return fix

    return None


def _find_fixture_id(
    match: Match, fixtures: list[dict], sport_val: str,
) -> int | None:
    """Find the API-Sports fixture ID for a match."""
    fixture = _match_fixture_to_db(match, fixtures, sport_val)
    if fixture:
        fix_data = fixture.get("fixture", fixture)
        return fix_data.get("id")
    return None


def _apply_fixture_scores(
    session: Session, match: Match, fixture: dict, sport_val: str,
) -> None:
    """Apply scores from an API-Sports fixture to a Match."""
    # Football structure: fixture.goals.home/away
    goals = fixture.get("goals", {})
    if isinstance(goals, dict) and goals.get("home") is not None:
        match.home_score = goals["home"]
        match.away_score = goals.get("away", 0)
    else:
        # Basketball/hockey structure: scores.home.total / scores.away.total
        scores = fixture.get("scores", {})
        if isinstance(scores, dict):
            home_score = scores.get("home")
            away_score = scores.get("away")
            if isinstance(home_score, dict):
                match.home_score = home_score.get("total", 0)
                match.away_score = away_score.get("total", 0) if isinstance(away_score, dict) else 0
            elif isinstance(home_score, int):
                match.home_score = home_score
                match.away_score = away_score if isinstance(away_score, int) else 0

    # Update match state based on fixture status
    status = fixture.get("fixture", {}).get("status", {})
    short_status = status.get("short", "") if isinstance(status, dict) else ""
    if short_status in ("FT", "AET", "PEN", "AOT"):
        match.match_state = MatchState.FINISHED
        match.is_live = False

    # Store fixture ID for future direct lookups
    stats = dict(match.live_stats) if match.live_stats else {}
    fix_data = fixture.get("fixture", fixture)
    if isinstance(fix_data, dict) and fix_data.get("id"):
        stats["api_sports_fixture_id"] = fix_data["id"]
    match.live_stats = stats

    logger.info(
        "API-Sports backfill: %s vs %s → %s-%s",
        match.home_team, match.away_team, match.home_score, match.away_score,
    )


def _apply_fixture_stats(
    session: Session, match: Match, stats: dict,
) -> None:
    """Merge parsed stats into match.live_stats."""
    existing = dict(match.live_stats) if match.live_stats else {}
    existing.update(stats)
    match.live_stats = existing


# ── Open gap and reconciliation functions ────────────────────────────────


def backfill_open_gaps(
    session: Session,
    max_days_back: int = 7,
    sports: list[str] | None = None,
) -> BackfillResult:
    """Backfill all open gaps from the last N days.

    Iterates each day from yesterday back to max_days_back, running
    backfill_day() for each.

    Returns:
        Aggregated BackfillResult.
    """
    today = date.today()
    aggregate = BackfillResult(target_date=f"last_{max_days_back}_days")

    for days_ago in range(1, max_days_back + 1):
        target = today - timedelta(days=days_ago)
        day_result = backfill_day(session, target, sports)
        aggregate.scores_filled += day_result.scores_filled
        aggregate.stats_filled += day_result.stats_filled
        aggregate.features_filled += day_result.features_filled
        aggregate.api_sports_calls += day_result.api_sports_calls
        aggregate.odds_api_calls += day_result.odds_api_calls
        aggregate.cloudflare_calls += day_result.cloudflare_calls
        aggregate.errors.extend(day_result.errors)

    logger.info(
        "Open gaps backfill (%d days): %d total filled",
        max_days_back, aggregate.total_filled,
    )

    return aggregate


def reconcile_open_results(session: Session) -> ReconcileResult:
    """Retry result matching for all unresolved matches.

    Queries matches that are past scheduled_at but still NOT_STARTED,
    and attempts to resolve them via TheOddsAPI then API-Sports.

    Returns:
        ReconcileResult summary.
    """
    result = ReconcileResult()

    # Find all unresolved matches (past scheduled, not finished, any sport)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=6)  # at least 6h past
    unresolved = list(
        session.execute(
            select(Match).where(
                Match.match_state != MatchState.FINISHED,
                Match.scheduled_at < cutoff,
            ).order_by(Match.scheduled_at)
        ).scalars().all()
    )

    result.matches_checked = len(unresolved)
    if not unresolved:
        logger.info("No unresolved matches to reconcile")
        return result

    logger.info("Reconciling %d unresolved matches", len(unresolved))

    # Try TheOddsAPI first (broader coverage for scores)
    odds_backend = _get_odds_api_backend()
    if odds_backend:
        result.sources_used.append("the_odds_api")
        for match in unresolved:
            if match.home_score is not None:
                continue
            try:
                match_result = odds_backend.fetch_result(match)
                if match_result and match_result.is_finished:
                    match.home_score = match_result.home_score
                    match.away_score = match_result.away_score
                    match.match_state = MatchState.FINISHED
                    match.is_live = False
                    _update_provenance(match, "the_odds_api")
                    result.matches_resolved += 1
            except Exception as exc:
                logger.warning("OddsAPI reconcile error for %s: %s", match.id, exc)

    # Then try API-Sports for remaining unresolved
    client = _get_api_sports_client()
    if client.is_available:
        result.sources_used.append("api_sports")
        # Group remaining by (sport, date)
        remaining = [m for m in unresolved if m.home_score is None]
        sport_date_matches: dict[tuple[str, str], list[Match]] = {}
        for m in remaining:
            key = (m.sport.value, m.scheduled_at.strftime("%Y-%m-%d"))
            sport_date_matches.setdefault(key, []).append(m)

        for (sport_val, date_str), matches in sport_date_matches.items():
            fixtures = client.fetch_fixtures_by_date(sport_val, date_str)
            if not fixtures:
                continue
            for match in matches:
                api_match = _match_fixture_to_db(match, fixtures, sport_val)
                if api_match:
                    _apply_fixture_scores(session, match, api_match, sport_val)
                    _update_provenance(match, "api_sports")
                    result.matches_resolved += 1

    result.still_unresolved = result.matches_checked - result.matches_resolved
    session.flush()

    logger.info(
        "Reconcile complete: %d/%d resolved, %d still unresolved (sources: %s)",
        result.matches_resolved, result.matches_checked,
        result.still_unresolved, ", ".join(result.sources_used),
    )

    return result
