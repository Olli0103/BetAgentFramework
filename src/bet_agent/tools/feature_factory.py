"""Point-in-Time Feature Factory for ML training and inference.

Generates feature vectors for any match using ONLY data available BEFORE
that match started (no future leakage). Features come from team_daily_stats
and historical_matches tables.

PHASE 3 ENHANCEMENT:
  Dynamic feature extraction — reads ALL keys from TeamDailyStats JSONB
  profiles instead of relying on hardcoded sport-specific lists. This means
  new features added by ingesters (roll_3_shots, opp_season_win_pct, etc.)
  are automatically consumed without code changes here.

  Metadata features for tennis: age_diff, height_diff, hand encoding,
  hand_matchup interaction feature.

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


# ── Hand Encoding for Tennis ────────────────────────────────────────

def _encode_hand(hand: str | None) -> float:
    """Encode playing hand: R=1.0, L=-1.0, U/unknown=0.0."""
    if not hand:
        return 0.0
    h = str(hand).strip().upper()
    if h == "R":
        return 1.0
    elif h == "L":
        return -1.0
    return 0.0


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


# ── Static Feature Definitions (fallback when no JSONB profiles exist) ──

_UNIVERSAL_FEATURES = [
    "roll_5_goals_for", "roll_5_goals_against", "roll_5_total_goals", "roll_5_win_pct",
    "roll_10_goals_for", "roll_10_goals_against", "roll_10_total_goals", "roll_10_win_pct",
    "roll_20_goals_for", "roll_20_goals_against", "roll_20_total_goals", "roll_20_win_pct",
    "season_wins", "season_losses", "season_draws", "win_pct", "games_played",
]

# Keys that are metadata, NOT numeric features — never feed to XGBoost
_METADATA_KEYS = {"hand", "height_cm", "age", "country", "surface", "name"}

# Keys from JSONB profiles to skip during dynamic feature extraction
# (they are used for metadata features, not directly as model features)
_SKIP_PROFILE_KEYS = _METADATA_KEYS | {"elo_rating"}


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
    Dynamically extracts ALL numeric keys from JSONB profiles,
    plus metadata features (tennis: age_diff, height_diff, hand).
    """
    home_stats = get_team_features_at_date(session, sport, home_team, match_date)
    away_stats = get_team_features_at_date(session, sport, away_team, match_date)

    features = _extract_dynamic_features(home_stats, away_stats, sport)

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

    PHASE 3: Dynamic feature extraction from JSONB profiles. All roll_*,
    season_*, opp_* keys are automatically consumed. Tennis metadata
    (age_diff, height_diff, hand) is extracted as interaction features.

    Only includes matches where both teams have at least `min_games_played`
    games of history (to avoid cold-start noise).

    Returns:
        List of FeatureVector with targets populated, sorted chronologically.
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

    dataset: list[FeatureVector] = []
    for hm in matches:
        home_stats = _lookup_stats_at_date(stats_cache, hm.home_team, hm.match_date)
        away_stats = _lookup_stats_at_date(stats_cache, hm.away_team, hm.match_date)

        # Dynamic feature extraction from JSONB profiles
        features = _extract_dynamic_features(home_stats, away_stats, sport)

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


# ── Dynamic Feature Extraction ───────────────────────────────────────


def _extract_dynamic_features(
    home_stats: dict,
    away_stats: dict,
    sport: Sport,
) -> dict[str, float]:
    """Extract features dynamically from JSONB profiles.

    Instead of hardcoded feature lists, reads ALL numeric keys from
    the profiles. This means new features added by ingesters (roll_3_shots,
    opp_season_win_pct, etc.) flow through automatically.

    Also computes:
      - Differential features for all shared roll_* keys
      - Tennis metadata: age_diff, height_diff, hand encoding, hand_matchup
    """
    features: dict[str, float] = {}

    # Collect all numeric keys from both profiles (union)
    all_keys: set[str] = set()
    for stats in (home_stats, away_stats):
        for k, v in stats.items():
            if k in _SKIP_PROFILE_KEYS:
                continue
            # Only include numeric values
            if isinstance(v, (int, float)):
                all_keys.add(k)
            elif isinstance(v, str):
                try:
                    float(v)
                    all_keys.add(k)
                except (ValueError, TypeError):
                    pass

    # Sorted for deterministic ordering
    sorted_keys = sorted(all_keys)

    # Home + Away features
    for key in sorted_keys:
        h_val = home_stats.get(key)
        a_val = away_stats.get(key)
        features[f"h_{key}"] = float(h_val) if h_val is not None else 0.0
        features[f"a_{key}"] = float(a_val) if a_val is not None else 0.0

    # Differential features for all roll_* keys (automatic)
    for key in sorted_keys:
        if key.startswith("roll_") or key in ("win_pct", "season_win_pct"):
            h_val = float(home_stats.get(key, 0.0) or 0.0)
            a_val = float(away_stats.get(key, 0.0) or 0.0)
            features[f"diff_{key}"] = h_val - a_val

    # ── Tennis Metadata Features ────────────────────────────────────
    if sport == Sport.TENNIS:
        # Age difference (home - away)
        h_age = home_stats.get("age")
        a_age = away_stats.get("age")
        if h_age is not None and a_age is not None:
            features["age_diff"] = float(h_age) - float(a_age)

        # Height difference (home - away)
        h_height = home_stats.get("height_cm")
        a_height = away_stats.get("height_cm")
        if h_height is not None and a_height is not None:
            features["height_diff"] = float(h_height) - float(a_height)

        # Hand encoding
        h_hand = _encode_hand(home_stats.get("hand"))
        a_hand = _encode_hand(away_stats.get("hand"))
        features["h_hand"] = h_hand
        features["a_hand"] = a_hand
        # Interaction: same-hand matchup (1.0) vs cross-hand (-1.0)
        features["hand_matchup"] = h_hand * a_hand

        # ELO rating (special — kept as direct feature, not skipped)
        h_elo = home_stats.get("elo_rating")
        a_elo = away_stats.get("elo_rating")
        if h_elo is not None:
            features["h_elo_rating"] = float(h_elo)
        if a_elo is not None:
            features["a_elo_rating"] = float(a_elo)
        if h_elo is not None and a_elo is not None:
            features["elo_diff"] = float(h_elo) - float(a_elo)

    return features


# ── Feature Name Registry ────────────────────────────────────────────


def get_feature_names(sport: Sport, dataset: list[FeatureVector] | None = None) -> list[str]:
    """Return ordered list of feature names for a sport.

    PHASE 3: If a dataset is provided, dynamically extracts feature names
    from the union of all FeatureVector.features keys. This handles
    variable-width JSONB profiles correctly.

    If no dataset is given, falls back to _UNIVERSAL_FEATURES for backward
    compatibility (inference without a training set).
    """
    if dataset:
        # Dynamic: union of all feature keys across the dataset
        all_keys: set[str] = set()
        for fv in dataset:
            all_keys.update(fv.features.keys())
        return sorted(all_keys)

    # Fallback: static universal features (for inference with no dataset)
    names: list[str] = []
    for prefix in ["h", "a"]:
        for key in _UNIVERSAL_FEATURES:
            names.append(f"{prefix}_{key}")

    for window in [5, 10, 20]:
        for stat in ["goals_for", "goals_against", "win_pct"]:
            names.append(f"diff_roll_{window}_{stat}")

    names.extend(["h2h_matches", "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_home_win_pct"])
    names.extend(["h_rest_days", "a_rest_days", "rest_diff"])

    return names
