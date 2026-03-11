"""Analytical Parameter Estimator — bridges team_daily_stats to prob models.

When no trained XGBoost model is available, the Quant still needs to feed
smart parameters into the analytical probability models (Poisson, Normal, etc.).

This module queries team_daily_stats and estimates the required input parameters
for each sport's analytical model:
  - Football: home_xg, away_xg from rolling goals/xG
  - Tennis: p_serve_home, p_serve_away from serve stats
  - Ice Hockey: home_xg, away_xg from Corsi-adjusted rates
  - Basketball: off_rtg, def_rtg, pace from team stats
  - American Football: home_power_rtg, away_power_rtg from EPA

Golden Rule #1: NO LLM MATH.  All estimation is deterministic Python.
Golden Rule #2: STATEFUL MEMORY.  All data sourced from PostgreSQL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import Sport, TeamDailyStats

logger = logging.getLogger(__name__)


@dataclass
class EstimatedParams:
    """Estimated analytical model parameters for a match."""

    sport: Sport
    home_team: str
    away_team: str
    params: dict[str, float] = field(default_factory=dict)
    confidence: str = "low"  # "low", "medium", "high"
    data_points_home: int = 0
    data_points_away: int = 0


# ── Sport-Specific Default Parameters ────────────────────────────────

_DEFAULTS: dict[Sport, dict[str, float]] = {
    Sport.FOOTBALL: {"home_xg": 1.35, "away_xg": 1.10},
    Sport.TENNIS: {"p_serve_home": 0.64, "p_serve_away": 0.62, "best_of": 3},
    Sport.ICE_HOCKEY: {"home_xg": 2.85, "away_xg": 2.65},
    Sport.BASKETBALL: {
        "home_off_rtg": 112.0, "home_def_rtg": 112.0,
        "away_off_rtg": 112.0, "away_def_rtg": 112.0,
        "pace": 100.0,
    },
    Sport.AMERICAN_FOOTBALL: {"home_power_rtg": 0.0, "away_power_rtg": 0.0},
    Sport.DARTS: {"p_leg_home": 0.50},
}


# ── Main Estimator ───────────────────────────────────────────────────


def estimate_params(
    session: Session,
    sport: Sport,
    home_team: str,
    away_team: str,
    at_date: date,
) -> EstimatedParams:
    """Estimate analytical model parameters from team_daily_stats.

    Queries the most recent stats snapshot for each team (point-in-time)
    and maps sport-specific metrics to the analytical model's expected inputs.

    Falls back to league-average defaults when no data is available.
    """
    home_stats = _get_latest_stats(session, sport, home_team, at_date)
    away_stats = _get_latest_stats(session, sport, away_team, at_date)

    estimator = _SPORT_ESTIMATORS.get(sport, _estimate_default)
    result = estimator(sport, home_team, away_team, home_stats, away_stats)

    # Determine confidence level
    result.data_points_home = home_stats.get("games_played", 0) if home_stats else 0
    result.data_points_away = away_stats.get("games_played", 0) if away_stats else 0

    min_games = min(result.data_points_home, result.data_points_away)
    if min_games >= 20:
        result.confidence = "high"
    elif min_games >= 5:
        result.confidence = "medium"
    else:
        result.confidence = "low"

    logger.info(
        "Estimated %s params for %s vs %s: %s (confidence=%s)",
        sport.value, home_team, away_team, result.params, result.confidence,
    )
    return result


# ── Sport-Specific Estimators ────────────────────────────────────────


def _estimate_football(
    sport: Sport, home_team: str, away_team: str,
    home_stats: dict | None, away_stats: dict | None,
) -> EstimatedParams:
    """Football: estimate xG from rolling goals or xG if available."""
    defaults = _DEFAULTS[Sport.FOOTBALL]

    home_xg = defaults["home_xg"]
    away_xg = defaults["away_xg"]

    if home_stats:
        # Prefer actual xG if crawled, otherwise use rolling goals_for
        home_xg = (
            home_stats.get("xg")
            or home_stats.get("roll_10_goals_for")
            or home_stats.get("roll_5_goals_for")
            or defaults["home_xg"]
        )

    if away_stats:
        away_xg = (
            away_stats.get("xg")
            or away_stats.get("roll_10_goals_for")
            or away_stats.get("roll_5_goals_for")
            or defaults["away_xg"]
        )

    return EstimatedParams(
        sport=sport, home_team=home_team, away_team=away_team,
        params={"home_xg": float(home_xg), "away_xg": float(away_xg)},
    )


def _estimate_tennis(
    sport: Sport, home_team: str, away_team: str,
    home_stats: dict | None, away_stats: dict | None,
) -> EstimatedParams:
    """Tennis: estimate serve win probability from first-serve-won %."""
    defaults = _DEFAULTS[Sport.TENNIS]

    p_serve_home = defaults["p_serve_home"]
    p_serve_away = defaults["p_serve_away"]

    if home_stats:
        # first_serve_won_pct is the most direct indicator
        fsw = home_stats.get("first_serve_won_pct")
        if fsw is not None:
            # Convert: overall serve hold probability ≈ weighted average
            # Typical: 65-70% 1st serve in, 72-80% 1st serve won, 48-55% 2nd serve won
            p_serve_home = float(fsw)
        else:
            # Fallback: estimate from ace% and bp_saved%
            ace_pct = home_stats.get("ace_pct", 0.0)
            bp_saved = home_stats.get("bp_saved_pct", 0.6)
            if ace_pct and bp_saved:
                p_serve_home = min(0.80, 0.55 + float(ace_pct) * 0.5 + float(bp_saved) * 0.1)

    if away_stats:
        fsw = away_stats.get("first_serve_won_pct")
        if fsw is not None:
            p_serve_away = float(fsw)
        else:
            ace_pct = away_stats.get("ace_pct", 0.0)
            bp_saved = away_stats.get("bp_saved_pct", 0.6)
            if ace_pct and bp_saved:
                p_serve_away = min(0.80, 0.55 + float(ace_pct) * 0.5 + float(bp_saved) * 0.1)

    return EstimatedParams(
        sport=sport, home_team=home_team, away_team=away_team,
        params={
            "p_serve_home": round(p_serve_home, 4),
            "p_serve_away": round(p_serve_away, 4),
            "best_of": int(defaults["best_of"]),
        },
    )


def _estimate_ice_hockey(
    sport: Sport, home_team: str, away_team: str,
    home_stats: dict | None, away_stats: dict | None,
) -> EstimatedParams:
    """Ice Hockey: estimate xG from Corsi-adjusted shot rates or rolling goals."""
    defaults = _DEFAULTS[Sport.ICE_HOCKEY]

    home_xg = defaults["home_xg"]
    away_xg = defaults["away_xg"]

    if home_stats:
        # Prefer Corsi-based estimate if available
        corsi_for = home_stats.get("corsi_for_pct")
        if corsi_for is not None:
            # Corsi for % ≈ shot attempt share → scale to xG
            # Average ~30 shots/game, ~9% shooting → ~2.7 goals
            home_xg = float(corsi_for) / 50.0 * 2.8
        else:
            home_xg = (
                home_stats.get("roll_10_goals_for")
                or home_stats.get("roll_5_goals_for")
                or defaults["home_xg"]
            )

    if away_stats:
        corsi_for = away_stats.get("corsi_for_pct")
        if corsi_for is not None:
            away_xg = float(corsi_for) / 50.0 * 2.8
        else:
            away_xg = (
                away_stats.get("roll_10_goals_for")
                or away_stats.get("roll_5_goals_for")
                or defaults["away_xg"]
            )

    return EstimatedParams(
        sport=sport, home_team=home_team, away_team=away_team,
        params={"home_xg": round(float(home_xg), 2), "away_xg": round(float(away_xg), 2)},
    )


def _estimate_basketball(
    sport: Sport, home_team: str, away_team: str,
    home_stats: dict | None, away_stats: dict | None,
) -> EstimatedParams:
    """Basketball: estimate off/def ratings and pace from team stats."""
    defaults = _DEFAULTS[Sport.BASKETBALL]

    params = dict(defaults)

    if home_stats:
        params["home_off_rtg"] = float(home_stats.get("off_rtg", defaults["home_off_rtg"]))
        params["home_def_rtg"] = float(home_stats.get("def_rtg", defaults["home_def_rtg"]))
        params["pace"] = float(home_stats.get("pace", defaults["pace"]))

    if away_stats:
        params["away_off_rtg"] = float(away_stats.get("off_rtg", defaults["away_off_rtg"]))
        params["away_def_rtg"] = float(away_stats.get("def_rtg", defaults["away_def_rtg"]))
        # Average both paces
        if home_stats and home_stats.get("pace"):
            away_pace = float(away_stats.get("pace", defaults["pace"]))
            params["pace"] = (params["pace"] + away_pace) / 2.0

    return EstimatedParams(
        sport=sport, home_team=home_team, away_team=away_team,
        params={k: round(v, 2) for k, v in params.items()},
    )


def _estimate_american_football(
    sport: Sport, home_team: str, away_team: str,
    home_stats: dict | None, away_stats: dict | None,
) -> EstimatedParams:
    """American Football: estimate power ratings from EPA data."""
    defaults = _DEFAULTS[Sport.AMERICAN_FOOTBALL]

    home_power = defaults["home_power_rtg"]
    away_power = defaults["away_power_rtg"]

    if home_stats:
        off_epa = home_stats.get("off_epa", 0.0)
        def_epa = home_stats.get("def_epa", 0.0)
        if off_epa is not None and def_epa is not None:
            # Power rating ≈ offensive EPA - defensive EPA (higher = better)
            home_power = float(off_epa) - float(def_epa)
        else:
            # Fallback: estimate from win_pct
            win_pct = home_stats.get("win_pct", 0.5)
            home_power = (float(win_pct) - 0.5) * 10.0  # Scale to ~-5 to +5

    if away_stats:
        off_epa = away_stats.get("off_epa", 0.0)
        def_epa = away_stats.get("def_epa", 0.0)
        if off_epa is not None and def_epa is not None:
            away_power = float(off_epa) - float(def_epa)
        else:
            win_pct = away_stats.get("win_pct", 0.5)
            away_power = (float(win_pct) - 0.5) * 10.0

    return EstimatedParams(
        sport=sport, home_team=home_team, away_team=away_team,
        params={
            "home_power_rtg": round(float(home_power), 2),
            "away_power_rtg": round(float(away_power), 2),
        },
    )


def _estimate_default(
    sport: Sport, home_team: str, away_team: str,
    home_stats: dict | None, away_stats: dict | None,
) -> EstimatedParams:
    """Default fallback: return league-average defaults."""
    return EstimatedParams(
        sport=sport, home_team=home_team, away_team=away_team,
        params=dict(_DEFAULTS.get(sport, {})),
    )


_SPORT_ESTIMATORS = {
    Sport.FOOTBALL: _estimate_football,
    Sport.TENNIS: _estimate_tennis,
    Sport.ICE_HOCKEY: _estimate_ice_hockey,
    Sport.BASKETBALL: _estimate_basketball,
    Sport.AMERICAN_FOOTBALL: _estimate_american_football,
}


# ── Helper: Stats Lookup ─────────────────────────────────────────────


def _get_latest_stats(
    session: Session,
    sport: Sport,
    team: str,
    before_date: date,
) -> dict | None:
    """Get the most recent TeamDailyStats for a team before a given date."""
    row = session.execute(
        select(TeamDailyStats)
        .where(
            TeamDailyStats.sport == sport,
            TeamDailyStats.team_name == team,
            TeamDailyStats.stat_date < before_date,
        )
        .order_by(TeamDailyStats.stat_date.desc())
        .limit(1)
    ).scalar_one_or_none()

    if row is None:
        return None
    return row.stats
