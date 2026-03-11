"""Point-in-Time Feature Factory for ML training and inference.

Generates feature vectors for any match using ONLY data available BEFORE
that match started (no future leakage). Features come from team_daily_stats
and historical_matches tables.

Golden Rule #1: NO LLM MATH.  All feature engineering is deterministic Python.
Golden Rule #2: STATEFUL MEMORY.  All data sourced from PostgreSQL.

Performance: build_training_dataset uses bulk-loading to avoid N+1 queries.
For 10k matches this means ~3 queries instead of ~60k.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    HistoricalMatch,
    Sport,
    TeamDailyStats,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """A complete feature vector for a single match."""

    match_date: date
    sport: Sport
    home_team: str
    away_team: str
    features: dict[str, float] = field(default_factory=dict)
    # Target variables (only set for training data)
    target_result: str | None = None       # H/D/A
    target_total_goals: int | None = None  # total score


# ── Feature Definitions per Sport ────────────────────────────────────

# Each sport has a list of (feature_name, extractor_key) pairs.
# The extractor_key refers to a key in the TeamDailyStats.stats JSONB.

_UNIVERSAL_FEATURES = [
    "roll_5_goals_for", "roll_5_goals_against", "roll_5_total_goals", "roll_5_win_pct",
    "roll_10_goals_for", "roll_10_goals_against", "roll_10_total_goals", "roll_10_win_pct",
    "roll_20_goals_for", "roll_20_goals_against", "roll_20_total_goals", "roll_20_win_pct",
    "season_wins", "season_losses", "season_draws", "win_pct", "games_played",
]

_FOOTBALL_FEATURES = [
    "roll_5_shots", "roll_5_shots_target", "roll_5_corners", "roll_5_fouls",
    "roll_10_shots", "roll_10_shots_target", "roll_10_corners", "roll_10_fouls",
    "roll_20_shots", "roll_20_shots_target",
]

_TENNIS_FEATURES = [
    "ace_pct", "first_serve_pct", "first_serve_won_pct", "second_serve_won_pct",
    "bp_saved_pct", "return_points_won_pct", "elo_rating",
]

_ICE_HOCKEY_FEATURES = [
    "roll_5_shots", "roll_5_power_play_goals", "roll_5_hits", "roll_5_blocked_shots",
    "roll_10_shots", "roll_10_power_play_goals", "roll_10_hits", "roll_10_blocked_shots",
    "corsi_for_pct", "fenwick_for_pct", "pp_pct", "pk_pct", "sv_pct",
]

_BASKETBALL_FEATURES = [
    "pace", "off_rtg", "def_rtg", "net_rtg",
]

_AMERICAN_FOOTBALL_FEATURES = [
    "off_epa", "def_epa", "yards_per_play",
]

SPORT_FEATURES: dict[Sport, list[str]] = {
    Sport.FOOTBALL: _UNIVERSAL_FEATURES + _FOOTBALL_FEATURES,
    Sport.TENNIS: _UNIVERSAL_FEATURES + _TENNIS_FEATURES,
    Sport.ICE_HOCKEY: _UNIVERSAL_FEATURES + _ICE_HOCKEY_FEATURES,
    Sport.BASKETBALL: _UNIVERSAL_FEATURES + _BASKETBALL_FEATURES,
    Sport.AMERICAN_FOOTBALL: _UNIVERSAL_FEATURES + _AMERICAN_FOOTBALL_FEATURES,
    Sport.DARTS: _UNIVERSAL_FEATURES,
}


# ── Point-in-Time Feature Extraction ────────────────────────────────


def get_team_features_at_date(
    session: Session,
    sport: Sport,
    team: str,
    at_date: date,
) -> dict[str, float]:
    """Get the most recent TeamDailyStats snapshot for a team BEFORE at_date.

    This is the core Point-in-Time lookup: we only use data that was
    available before the match started.
    """
    row = session.execute(
        select(TeamDailyStats)
        .where(
            TeamDailyStats.sport == sport,
            TeamDailyStats.team_name == team,
            TeamDailyStats.stat_date < at_date,
        )
        .order_by(TeamDailyStats.stat_date.desc())
        .limit(1)
    ).scalar_one_or_none()

    if row is None:
        return {}

    return row.stats or {}


def build_feature_vector(
    session: Session,
    sport: Sport,
    home_team: str,
    away_team: str,
    match_date: date,
    include_target: bool = False,
) -> FeatureVector:
    """Build a complete feature vector for a match.

    Uses Point-in-Time data only (stats before match_date).
    Prefixes home features with 'h_' and away features with 'a_'.
    Also computes differential features (home - away).
    """
    home_stats = get_team_features_at_date(session, sport, home_team, match_date)
    away_stats = get_team_features_at_date(session, sport, away_team, match_date)

    feature_keys = SPORT_FEATURES.get(sport, _UNIVERSAL_FEATURES)
    features: dict[str, float] = {}

    # Home features
    for key in feature_keys:
        val = home_stats.get(key)
        if val is not None:
            features[f"h_{key}"] = float(val)
        else:
            features[f"h_{key}"] = 0.0

    # Away features
    for key in feature_keys:
        val = away_stats.get(key)
        if val is not None:
            features[f"a_{key}"] = float(val)
        else:
            features[f"a_{key}"] = 0.0

    # Differential features (home - away) for key rolling stats
    for window in [5, 10, 20]:
        for stat in ["goals_for", "goals_against", "win_pct"]:
            h_key = f"roll_{window}_{stat}"
            h_val = home_stats.get(h_key, 0.0)
            a_val = away_stats.get(h_key, 0.0)
            if h_val is not None and a_val is not None:
                features[f"diff_{h_key}"] = float(h_val) - float(a_val)

    # Head-to-head record (last 5 meetings)
    h2h = _head_to_head_stats(session, sport, home_team, away_team, match_date, n=5)
    features.update(h2h)

    # Rest days
    home_rest = _days_since_last_match(session, sport, home_team, match_date)
    away_rest = _days_since_last_match(session, sport, away_team, match_date)
    features["h_rest_days"] = float(home_rest) if home_rest is not None else 7.0
    features["a_rest_days"] = float(away_rest) if away_rest is not None else 7.0
    features["rest_diff"] = features["h_rest_days"] - features["a_rest_days"]

    fv = FeatureVector(
        match_date=match_date,
        sport=sport,
        home_team=home_team,
        away_team=away_team,
        features=features,
    )

    if include_target:
        target = _get_match_target(session, sport, home_team, away_team, match_date)
        if target:
            fv.target_result = target["result"]
            fv.target_total_goals = target["total"]

    return fv


def build_training_dataset(
    session: Session,
    sport: Sport,
    season: str | None = None,
    min_games_played: int = 5,
) -> list[FeatureVector]:
    """Build a complete training dataset from historical matches.

    Uses bulk-loading to avoid the N+1 query problem: instead of running
    2 queries per match (home stats + away stats) plus H2H + rest-day lookups,
    we load ALL TeamDailyStats and HistoricalMatches for the sport in ~3 queries
    and resolve features in-memory.

    Only includes matches where both teams have at least `min_games_played`
    games of history (to avoid cold-start noise).

    Returns:
        List of FeatureVector with targets populated.
    """
    query = (
        select(HistoricalMatch)
        .where(HistoricalMatch.sport == sport)
        .order_by(HistoricalMatch.match_date)
    )
    if season is not None:
        query = query.where(HistoricalMatch.season == season)

    matches: Sequence[HistoricalMatch] = session.execute(query).scalars().all()
    logger.info("Building training set for %s from %d matches", sport.value, len(matches))

    if not matches:
        return []

    # ── Bulk-load all TeamDailyStats for this sport ──
    stats_cache = _bulk_load_team_stats(session, sport)

    # ── Bulk-load all historical matches for H2H and rest-day lookups ──
    all_hist = _bulk_load_historical(session, sport)

    feature_keys = SPORT_FEATURES.get(sport, _UNIVERSAL_FEATURES)

    dataset: list[FeatureVector] = []
    for hm in matches:
        home_stats = _lookup_stats_at_date(stats_cache, hm.home_team, hm.match_date)
        away_stats = _lookup_stats_at_date(stats_cache, hm.away_team, hm.match_date)

        features: dict[str, float] = {}

        # Home + Away features
        for key in feature_keys:
            features[f"h_{key}"] = float(home_stats.get(key, 0.0) or 0.0)
            features[f"a_{key}"] = float(away_stats.get(key, 0.0) or 0.0)

        # Differential features
        for window in [5, 10, 20]:
            for stat in ["goals_for", "goals_against", "win_pct"]:
                h_key = f"roll_{window}_{stat}"
                h_val = float(home_stats.get(h_key, 0.0) or 0.0)
                a_val = float(away_stats.get(h_key, 0.0) or 0.0)
                features[f"diff_{h_key}"] = h_val - a_val

        # H2H (in-memory)
        h2h = _h2h_from_cache(all_hist, hm.home_team, hm.away_team, hm.match_date, n=5)
        features.update(h2h)

        # Rest days (in-memory)
        home_rest = _rest_days_from_cache(all_hist, hm.home_team, hm.match_date)
        away_rest = _rest_days_from_cache(all_hist, hm.away_team, hm.match_date)
        features["h_rest_days"] = float(home_rest) if home_rest is not None else 7.0
        features["a_rest_days"] = float(away_rest) if away_rest is not None else 7.0
        features["rest_diff"] = features["h_rest_days"] - features["a_rest_days"]

        # Skip if insufficient history
        if features.get("h_games_played", 0) < min_games_played:
            continue
        if features.get("a_games_played", 0) < min_games_played:
            continue

        fv = FeatureVector(
            match_date=hm.match_date,
            sport=sport,
            home_team=hm.home_team,
            away_team=hm.away_team,
            features=features,
            target_result=hm.result,
            target_total_goals=hm.home_score + hm.away_score,
        )
        dataset.append(fv)

    logger.info("Built %d training samples (filtered from %d)", len(dataset), len(matches))
    return dataset


# ── Bulk-Loading Helpers (eliminate N+1) ─────────────────────────────


def _bulk_load_team_stats(
    session: Session, sport: Sport,
) -> dict[str, list[tuple[date, dict]]]:
    """Load ALL TeamDailyStats for a sport into a dict keyed by team_name.

    Returns:
        {team_name: [(stat_date, stats_dict), ...]} sorted by date ascending.
    """
    rows = session.execute(
        select(TeamDailyStats)
        .where(TeamDailyStats.sport == sport)
        .order_by(TeamDailyStats.team_name, TeamDailyStats.stat_date)
    ).scalars().all()

    cache: dict[str, list[tuple[date, dict]]] = defaultdict(list)
    for row in rows:
        cache[row.team_name].append((row.stat_date, row.stats or {}))

    return dict(cache)


def _lookup_stats_at_date(
    cache: dict[str, list[tuple[date, dict]]],
    team: str,
    at_date: date,
) -> dict:
    """Binary-search the bulk cache for the latest stats BEFORE at_date."""
    entries = cache.get(team)
    if not entries:
        return {}

    # Entries are sorted by date ascending. Find rightmost entry < at_date.
    lo, hi = 0, len(entries) - 1
    result_idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] < at_date:
            result_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if result_idx < 0:
        return {}
    return entries[result_idx][1]


def _bulk_load_historical(
    session: Session, sport: Sport,
) -> list[HistoricalMatch]:
    """Load ALL historical matches for a sport, sorted by date ascending."""
    return list(
        session.execute(
            select(HistoricalMatch)
            .where(HistoricalMatch.sport == sport)
            .order_by(HistoricalMatch.match_date)
        ).scalars().all()
    )


def _h2h_from_cache(
    all_matches: list[HistoricalMatch],
    home: str,
    away: str,
    before_date: date,
    n: int = 5,
) -> dict[str, float]:
    """Compute H2H stats from in-memory match list."""
    meetings = []
    for m in reversed(all_matches):
        if m.match_date >= before_date:
            continue
        if (m.home_team == home and m.away_team == away) or \
           (m.home_team == away and m.away_team == home):
            meetings.append(m)
            if len(meetings) >= n:
                break

    if not meetings:
        return {"h2h_matches": 0.0, "h2h_home_wins": 0.0, "h2h_away_wins": 0.0, "h2h_draws": 0.0, "h2h_home_win_pct": 0.0}

    home_wins = away_wins = draws = 0
    for m in meetings:
        if m.home_team == home:
            if m.result == "H":
                home_wins += 1
            elif m.result == "A":
                away_wins += 1
            else:
                draws += 1
        else:
            if m.result == "A":
                home_wins += 1
            elif m.result == "H":
                away_wins += 1
            else:
                draws += 1

    total = len(meetings)
    return {
        "h2h_matches": float(total),
        "h2h_home_wins": float(home_wins),
        "h2h_away_wins": float(away_wins),
        "h2h_draws": float(draws),
        "h2h_home_win_pct": round(home_wins / total, 3) if total > 0 else 0.0,
    }


def _rest_days_from_cache(
    all_matches: list[HistoricalMatch],
    team: str,
    before_date: date,
) -> int | None:
    """Find days since team's last match from in-memory list."""
    for m in reversed(all_matches):
        if m.match_date >= before_date:
            continue
        if m.home_team == team or m.away_team == team:
            return (before_date - m.match_date).days
    return None


# ── Helper: Head-to-Head Stats ───────────────────────────────────────


def _head_to_head_stats(
    session: Session,
    sport: Sport,
    home: str,
    away: str,
    before_date: date,
    n: int = 5,
) -> dict[str, float]:
    """Last N meetings between two teams before a date."""
    meetings = session.execute(
        select(HistoricalMatch)
        .where(
            HistoricalMatch.sport == sport,
            HistoricalMatch.match_date < before_date,
            (
                and_(HistoricalMatch.home_team == home, HistoricalMatch.away_team == away)
                | and_(HistoricalMatch.home_team == away, HistoricalMatch.away_team == home)
            ),
        )
        .order_by(HistoricalMatch.match_date.desc())
        .limit(n)
    ).scalars().all()

    if not meetings:
        return {"h2h_matches": 0.0, "h2h_home_wins": 0.0, "h2h_away_wins": 0.0, "h2h_draws": 0.0}

    home_wins = 0
    away_wins = 0
    draws = 0

    for m in meetings:
        if m.home_team == home:
            if m.result == "H":
                home_wins += 1
            elif m.result == "A":
                away_wins += 1
            else:
                draws += 1
        else:
            # Teams are swapped in this meeting
            if m.result == "A":
                home_wins += 1
            elif m.result == "H":
                away_wins += 1
            else:
                draws += 1

    total = len(meetings)
    return {
        "h2h_matches": float(total),
        "h2h_home_wins": float(home_wins),
        "h2h_away_wins": float(away_wins),
        "h2h_draws": float(draws),
        "h2h_home_win_pct": round(home_wins / total, 3) if total > 0 else 0.0,
    }


# ── Helper: Rest Days ────────────────────────────────────────────────


def _days_since_last_match(
    session: Session,
    sport: Sport,
    team: str,
    before_date: date,
) -> int | None:
    """Days since team's most recent match before the given date."""
    last = session.execute(
        select(HistoricalMatch.match_date)
        .where(
            HistoricalMatch.sport == sport,
            HistoricalMatch.match_date < before_date,
            (HistoricalMatch.home_team == team) | (HistoricalMatch.away_team == team),
        )
        .order_by(HistoricalMatch.match_date.desc())
        .limit(1)
    ).scalar_one_or_none()

    if last is None:
        return None
    return (before_date - last).days


# ── Helper: Match Target ─────────────────────────────────────────────


def _get_match_target(
    session: Session,
    sport: Sport,
    home: str,
    away: str,
    match_date: date,
) -> dict | None:
    """Get result and total score for a specific match."""
    hm = session.execute(
        select(HistoricalMatch)
        .where(
            HistoricalMatch.sport == sport,
            HistoricalMatch.home_team == home,
            HistoricalMatch.away_team == away,
            HistoricalMatch.match_date == match_date,
        )
        .limit(1)
    ).scalar_one_or_none()

    if hm is None:
        return None

    return {
        "result": hm.result,
        "total": hm.home_score + hm.away_score,
    }


# ── Feature Name Registry ────────────────────────────────────────────


def get_feature_names(sport: Sport) -> list[str]:
    """Return ordered list of feature names for a sport.

    Useful for ensuring consistent column ordering in training/inference.
    """
    feature_keys = SPORT_FEATURES.get(sport, _UNIVERSAL_FEATURES)
    names: list[str] = []

    # Home + Away features
    for prefix in ["h", "a"]:
        for key in feature_keys:
            names.append(f"{prefix}_{key}")

    # Differential features
    for window in [5, 10, 20]:
        for stat in ["goals_for", "goals_against", "win_pct"]:
            names.append(f"diff_roll_{window}_{stat}")

    # H2H features
    names.extend(["h2h_matches", "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_home_win_pct"])

    # Rest days
    names.extend(["h_rest_days", "a_rest_days", "rest_diff"])

    return names
