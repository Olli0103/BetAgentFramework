"""Historical Data Importer — populates matches + team_daily_stats from historical_matches.

Reads historical_matches (already ingested via the ingest/ package), resolves
team/player names through the IroncladAliasResolver, and produces:
  1. Resolved Match records in the `matches` table (with correct canonical names)
  2. Historical TeamDailyStats snapshots (point-in-time rolling stats)

Golden Rule #1: NO LLM MATH.  All aggregation is deterministic Python.
Golden Rule #2: STATEFUL MEMORY.  Everything goes to PostgreSQL.
Golden Rule #3: NO RAW STRINGS.  Every name passes through IroncladAliasResolver.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Sequence

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    HistoricalMatch,
    Match,
    MatchState,
    Sport,
    TeamAlias,
    TeamDailyStats,
)
from bet_agent.ingest.alias_resolver import IroncladAliasResolver

logger = logging.getLogger(__name__)


# ── Alias Resolution (backward-compatible wrapper) ──────────────────


class AliasResolver:
    """Legacy wrapper around IroncladAliasResolver.

    Maintains the same public API (resolve, add_alias) but delegates
    to sport-specific IroncladAliasResolvers under the hood.
    Also supports sport-agnostic resolution for backward compatibility.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._resolvers: dict[Sport, IroncladAliasResolver] = {}
        # Legacy cache for sport-agnostic aliases
        self._global_cache: dict[str, str] = {}
        self._load_global_cache()

    def _load_global_cache(self) -> None:
        """Load all aliases (any sport) into the global cache."""
        from bet_agent.ingest.alias_resolver import make_key

        rows = self._session.execute(select(TeamAlias)).scalars().all()
        for row in rows:
            key = make_key(row.alias)
            self._global_cache[key] = row.canonical_name
            self._global_cache[make_key(row.canonical_name)] = row.canonical_name

    def _get_resolver(self, sport: Sport) -> IroncladAliasResolver:
        if sport not in self._resolvers:
            self._resolvers[sport] = IroncladAliasResolver(self._session, sport)
        return self._resolvers[sport]

    def resolve(self, name: str, sport: Sport | None = None) -> str:
        """Return canonical name for a team/player.

        If sport is None, does a global lookup across all aliases (legacy behavior).
        """
        if sport is not None:
            return self._get_resolver(sport).resolve(name)

        from bet_agent.ingest.alias_resolver import make_key

        key = make_key(name)

        # Legacy: check global cache
        if key in self._global_cache:
            return self._global_cache[key]

        # Passthrough if no alias matched
        return name.strip()

    def add_alias(self, canonical: str, alias: str, source: str = "historical_import",
                  sport: Sport | None = None) -> None:
        """Register a new alias mapping (persists to DB)."""
        existing = self._session.execute(
            select(TeamAlias).where(TeamAlias.alias == alias)
        ).scalar_one_or_none()
        if existing:
            return
        self._session.add(TeamAlias(
            canonical_name=canonical,
            alias=alias,
            source=source,
            sport=sport,
        ))
        # Update caches
        from bet_agent.ingest.alias_resolver import make_key

        self._global_cache[make_key(alias)] = canonical
        self._global_cache[make_key(canonical)] = canonical
        if sport and sport in self._resolvers:
            self._resolvers[sport]._cache[make_key(alias)] = canonical
            self._resolvers[sport]._canonicals[make_key(canonical)] = canonical


# ── Match Population ─────────────────────────────────────────────────


def import_historical_to_matches(
    session: Session,
    sport: Sport | None = None,
    season: str | None = None,
    resolver: AliasResolver | None = None,
) -> int:
    """Import historical_matches into the matches table with alias resolution.

    Args:
        session: SQLAlchemy session (caller manages transaction).
        sport: Optional filter — only import this sport.
        season: Optional filter — only import this season.
        resolver: AliasResolver instance. Created automatically if None.

    Returns:
        Number of match records created/updated.
    """
    if resolver is None:
        resolver = AliasResolver(session)

    query = select(HistoricalMatch)
    if sport is not None:
        query = query.where(HistoricalMatch.sport == sport)
    if season is not None:
        query = query.where(HistoricalMatch.season == season)
    query = query.order_by(HistoricalMatch.match_date)

    hist_matches: Sequence[HistoricalMatch] = session.execute(query).scalars().all()
    logger.info("Found %d historical matches to import", len(hist_matches))

    count = 0
    for hm in hist_matches:
        try:
            home = resolver.resolve(hm.home_team, sport=hm.sport)
            away = resolver.resolve(hm.away_team, sport=hm.sport)
            league = hm.division or "unknown"

            scheduled_at = datetime.combine(hm.match_date, time(15, 0), tzinfo=timezone.utc)

            # Check if already exists
            existing = session.execute(
                select(Match).where(
                    Match.sport == hm.sport,
                    Match.league == league,
                    Match.home_team == home,
                    Match.away_team == away,
                    Match.scheduled_at == scheduled_at,
                )
            ).scalar_one_or_none()

            if existing:
                existing.home_score = hm.home_score
                existing.away_score = hm.away_score
                existing.match_state = MatchState.FINISHED
                existing.live_stats = _build_live_stats(hm)
            else:
                match = Match(
                    sport=hm.sport,
                    league=league,
                    home_team=home,
                    away_team=away,
                    scheduled_at=scheduled_at,
                    home_score=hm.home_score,
                    away_score=hm.away_score,
                    match_state=MatchState.FINISHED,
                    is_live=False,
                    live_stats=_build_live_stats(hm),
                )
                session.add(match)
            count += 1
        except Exception as exc:
            logger.warning(
                "Skipping historical match %s vs %s (%s): %s",
                hm.home_team, hm.away_team, hm.match_date, exc,
            )

    logger.info("Imported %d matches into matches table", count)
    return count


def _build_live_stats(hm: HistoricalMatch) -> dict:
    """Merge match_stats + advanced_stats into live_stats for the Match record."""
    stats: dict = {}
    stats.update(hm.match_stats or {})
    stats["result"] = hm.result
    stats["source"] = hm.source
    if hm.odds:
        stats["closing_odds"] = hm.odds
    return stats


# ── TeamDailyStats Population (Historical Snapshots) ─────────────────


# Sport-specific stat keys to track as rolling features
_SPORT_STAT_KEYS: dict[Sport, list[str]] = {
    Sport.FOOTBALL: [
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
    ],
    Sport.BASKETBALL: [
        "q1_home", "q2_home", "q3_home", "q4_home",
        "q1_away", "q2_away", "q3_away", "q4_away",
    ],
    Sport.ICE_HOCKEY: [
        "home_shots", "away_shots", "home_power_play_goals",
        "away_power_play_goals", "home_hits", "away_hits",
        "home_blocked_shots", "away_blocked_shots",
    ],
    Sport.AMERICAN_FOOTBALL: [],
    Sport.TENNIS: [],
    Sport.DARTS: [],
}

# Rolling window sizes for historical snapshots
_ROLLING_WINDOWS = [5, 10, 20]


def build_historical_daily_stats(
    session: Session,
    sport: Sport,
    season: str | None = None,
    resolver: AliasResolver | None = None,
) -> int:
    """Build TeamDailyStats snapshots from historical_matches.

    For each team, at each match date, computes rolling averages over
    their previous N matches (point-in-time: only data known BEFORE that date).

    Args:
        session: SQLAlchemy session.
        sport: Which sport to process.
        season: Optional season filter.
        resolver: AliasResolver instance.

    Returns:
        Number of daily stat rows created.
    """
    if resolver is None:
        resolver = AliasResolver(session)

    query = (
        select(HistoricalMatch)
        .where(HistoricalMatch.sport == sport)
        .order_by(HistoricalMatch.match_date)
    )
    if season is not None:
        query = query.where(HistoricalMatch.season == season)

    matches: Sequence[HistoricalMatch] = session.execute(query).scalars().all()
    logger.info("Building daily stats for %s from %d matches", sport.value, len(matches))

    # Build per-team match history (ordered by date)
    team_history: dict[str, list[dict]] = {}

    for hm in matches:
        home = resolver.resolve(hm.home_team, sport=sport)
        away = resolver.resolve(hm.away_team, sport=sport)

        home_record = _extract_team_record(hm, home, is_home=True)
        away_record = _extract_team_record(hm, away, is_home=False)

        team_history.setdefault(home, []).append(home_record)
        team_history.setdefault(away, []).append(away_record)

    # Now generate point-in-time rolling stats for each team at each date
    count = 0
    for team, records in team_history.items():
        for i, rec in enumerate(records):
            stats = _compute_rolling_stats(records[:i], sport)
            stats["games_played"] = i
            stats["is_home"] = rec["is_home"]

            # Include season record
            wins = sum(1 for r in records[:i] if r["won"])
            losses = sum(1 for r in records[:i] if not r["won"] and not r.get("draw"))
            draws = sum(1 for r in records[:i] if r.get("draw"))
            stats["season_wins"] = wins
            stats["season_losses"] = losses
            stats["season_draws"] = draws
            if i > 0:
                stats["win_pct"] = round(wins / i, 4)

            _upsert_daily_stat(
                session, sport, team, rec["date"], stats, rec.get("league", "unknown"),
            )
            count += 1

    logger.info("Created %d daily stat snapshots for %s", count, sport.value)
    return count


def _extract_team_record(hm: HistoricalMatch, team: str, is_home: bool) -> dict:
    """Extract a single team's record from a historical match."""
    goals_for = hm.home_score if is_home else hm.away_score
    goals_against = hm.away_score if is_home else hm.home_score

    record: dict = {
        "date": hm.match_date,
        "is_home": is_home,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "total_goals": hm.home_score + hm.away_score,
        "won": goals_for > goals_against,
        "draw": goals_for == goals_against,
        "league": hm.division,
    }

    # Sport-specific stats from match_stats JSONB
    prefix = "home" if is_home else "away"
    for key in _SPORT_STAT_KEYS.get(hm.sport, []):
        if key.startswith(prefix + "_"):
            stat_name = key[len(prefix) + 1:]
            val = (hm.match_stats or {}).get(key)
            if val is not None:
                record[stat_name] = val

    # Advanced stats (NHL rolling, tennis serve stats, etc.)
    for key, val in (hm.advanced_stats or {}).items():
        if key.startswith(prefix + "_"):
            record[key[len(prefix) + 1:]] = val

    return record


def _compute_rolling_stats(history: list[dict], sport: Sport) -> dict:
    """Compute rolling averages over the last N games for a team."""
    stats: dict = {}

    if not history:
        return stats

    for window in _ROLLING_WINDOWS:
        recent = history[-window:]
        n = len(recent)
        if n == 0:
            continue

        prefix = f"roll_{window}"

        # Universal stats
        stats[f"{prefix}_goals_for"] = round(sum(r["goals_for"] for r in recent) / n, 3)
        stats[f"{prefix}_goals_against"] = round(sum(r["goals_against"] for r in recent) / n, 3)
        stats[f"{prefix}_total_goals"] = round(sum(r["total_goals"] for r in recent) / n, 3)
        stats[f"{prefix}_win_pct"] = round(sum(1 for r in recent if r["won"]) / n, 3)

        # Sport-specific rolling stats
        for key in _SPORT_STAT_KEYS.get(sport, []):
            # Strip home_/away_ prefix to get base stat name
            base = key.split("_", 1)[1] if "_" in key else key
            vals = [r.get(base) for r in recent if r.get(base) is not None]
            if vals:
                stats[f"{prefix}_{base}"] = round(sum(vals) / len(vals), 3)

    return stats


def _upsert_daily_stat(
    session: Session,
    sport: Sport,
    team: str,
    stat_date: date,
    stats: dict,
    league: str,
) -> None:
    """Insert or merge a TeamDailyStats row."""
    existing = session.execute(
        select(TeamDailyStats).where(
            TeamDailyStats.sport == sport,
            TeamDailyStats.team_name == team,
            TeamDailyStats.stat_date == stat_date,
        )
    ).scalar_one_or_none()

    if existing:
        existing.stats = {**existing.stats, **stats}
    else:
        session.add(TeamDailyStats(
            sport=sport,
            team_name=team,
            league=league,
            stat_date=stat_date,
            stats=stats,
            source_url="historical_import",
        ))


# ── Full Import Pipeline ─────────────────────────────────────────────


def run_full_historical_import(
    session: Session,
    sport: Sport | None = None,
    season: str | None = None,
) -> dict[str, int]:
    """Run the complete historical import pipeline.

    1. Resolve aliases (via IroncladAliasResolver)
    2. Populate matches table
    3. Build team_daily_stats snapshots

    Returns:
        Dict with counts: {"matches": N, "daily_stats": N}
    """
    resolver = AliasResolver(session)

    match_count = import_historical_to_matches(session, sport, season, resolver)

    daily_count = 0
    if sport is not None:
        daily_count = build_historical_daily_stats(session, sport, season, resolver)
    else:
        for s in Sport:
            daily_count += build_historical_daily_stats(session, s, season, resolver)

    session.flush()
    logger.info(
        "Full import complete: %d matches, %d daily stats",
        match_count, daily_count,
    )
    return {"matches": match_count, "daily_stats": daily_count}
