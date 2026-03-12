"""Deterministic Expected Value calculator — sport-aware.

Golden Rule #1: NO LLM MATH. This is pure Python + scipy.
Delegates live probability updates to sport-specific models via registry.
Supports ML-based inference when trained XGBoost models are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from bet_agent.tools.prob_models.registry import get_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EVResult:
    """Result of an EV calculation."""

    updated_prob: float
    implied_prob: float
    prob_edge: float  # updated_prob - implied_prob (probability difference)
    ev: float
    is_positive_ev: bool
    model_source: str = "analytical"  # "analytical" or "xgboost"


def calculate_live_ev(
    sport: str,
    pre_match_prob: float,
    live_score: tuple[int, int],
    live_time: float,
    live_odds: float,
    live_stats: dict | None = None,
) -> EVResult:
    """Calculate expected value for a live betting opportunity.

    Args:
        sport: Sport identifier (e.g., "football", "tennis").
        pre_match_prob: Pre-match model probability of the outcome.
        live_score: Current score as (home, away) tuple.
        live_time: Sport-specific time/progress metric.
        live_odds: Current live decimal odds being offered.
        live_stats: Optional sport-specific live statistics.

    Returns:
        EVResult with updated probability, implied probability, edge, and EV.

    Raises:
        KeyError: If sport has no registered probability model.
        ValueError: If inputs are out of valid ranges.
    """
    # Input validation
    if not 0.0 <= pre_match_prob <= 1.0:
        raise ValueError(f"pre_match_prob must be in [0, 1], got {pre_match_prob}")
    if live_odds <= 1.0:
        raise ValueError(f"live_odds must be > 1.0, got {live_odds}")
    if live_time < 0.0:
        raise ValueError(f"live_time must be >= 0, got {live_time}")

    # Get sport-specific model and compute updated probability
    model = get_model(sport)
    updated_prob = model.live_update(pre_match_prob, live_score, live_time, live_stats)

    # Clamp to valid probability range (defense-in-depth)
    import math
    if math.isnan(updated_prob) or math.isinf(updated_prob):
        logger.warning("live_update returned %s for %s — falling back to pre_match_prob", updated_prob, sport)
        updated_prob = pre_match_prob
    updated_prob = max(0.0, min(1.0, updated_prob))

    # EV calculation (deterministic math)
    implied_prob = 1.0 / live_odds
    prob_edge = updated_prob - implied_prob
    ev = (updated_prob * (live_odds - 1.0)) - (1.0 - updated_prob)

    return EVResult(
        updated_prob=round(updated_prob, 6),
        implied_prob=round(implied_prob, 6),
        prob_edge=round(prob_edge, 6),
        ev=round(ev, 6),
        is_positive_ev=ev > 0.0,
        model_source="analytical",
    )


def calculate_pre_match_ev(
    model_prob: float,
    odds: float,
) -> EVResult:
    """Calculate EV for a pre-match bet (no live update needed).

    Args:
        model_prob: Model's estimated probability of the outcome.
        odds: Decimal odds being offered.

    Returns:
        EVResult with probability, implied probability, edge, and EV.
    """
    if not 0.0 <= model_prob <= 1.0:
        raise ValueError(f"model_prob must be in [0, 1], got {model_prob}")
    if odds <= 1.0:
        raise ValueError(f"odds must be > 1.0, got {odds}")

    implied_prob = 1.0 / odds
    prob_edge = model_prob - implied_prob
    ev = (model_prob * (odds - 1.0)) - (1.0 - model_prob)

    return EVResult(
        updated_prob=round(model_prob, 6),
        implied_prob=round(implied_prob, 6),
        prob_edge=round(prob_edge, 6),
        ev=round(ev, 6),
        is_positive_ev=ev > 0.0,
    )


# ── ML-Powered Pre-Match EV ─────────────────────────────────────────


def calculate_ml_pre_match_ev(
    session,
    sport_str: str,
    home_team: str,
    away_team: str,
    match_date: date,
    odds_home: float,
    odds_draw: float | None = None,
    odds_away: float | None = None,
    model_dir: Path | None = None,
) -> dict[str, EVResult]:
    """Calculate EV using trained XGBoost model probabilities.

    Loads the latest trained model for the sport, builds a Point-in-Time
    feature vector from team_daily_stats, and computes ML-based probabilities.
    Falls back to analytical models if no trained model is available.

    Args:
        session: SQLAlchemy session for feature lookups.
        sport_str: Sport string (e.g., "football").
        home_team: Home team canonical name.
        away_team: Away team canonical name.
        match_date: Date of the match.
        odds_home: Decimal odds for home win.
        odds_draw: Decimal odds for draw (optional, not used for tennis).
        odds_away: Decimal odds for away win (optional).
        model_dir: Directory containing trained models.

    Returns:
        Dict of market→EVResult, e.g. {"home": EVResult, "draw": EVResult, "away": EVResult}.
    """
    from bet_agent.db.models import Sport
    from bet_agent.ml.trainer import find_latest_model, predict_match_winner
    from bet_agent.tools.feature_factory import build_feature_vector, get_feature_names

    sport = Sport(sport_str)

    # Try to load trained model
    artifact = find_latest_model(sport, "match_winner", model_dir)

    if artifact is None:
        logger.info("No trained model for %s, falling back to analytical", sport_str)
        # Fallback: use analytical model's default probabilities
        return _analytical_fallback(sport_str, odds_home, odds_draw, odds_away)

    # Build feature vector
    fv = build_feature_vector(session, sport, home_team, away_team, match_date)
    # Use feature names from the trained model artifact (not static fallback)
    # to ensure the feature vector matches what the model was trained on.
    feature_names = artifact.feature_names or get_feature_names(sport)

    X = np.zeros((1, len(feature_names)))
    for j, fname in enumerate(feature_names):
        X[0, j] = fv.features.get(fname, 0.0)

    # ML inference
    probs = predict_match_winner(artifact.file_path, X)
    logger.info(
        "ML prediction for %s vs %s: H=%.3f D=%.3f A=%.3f",
        home_team, away_team, probs["home"], probs["draw"], probs["away"],
    )

    results: dict[str, EVResult] = {}

    # Home win EV
    if odds_home > 1.0:
        results["home"] = _make_ev_result(probs["home"], odds_home, "xgboost")

    # Draw EV
    if odds_draw is not None and odds_draw > 1.0:
        results["draw"] = _make_ev_result(probs["draw"], odds_draw, "xgboost")

    # Away win EV
    if odds_away is not None and odds_away > 1.0:
        results["away"] = _make_ev_result(probs["away"], odds_away, "xgboost")

    return results


def _make_ev_result(model_prob: float, odds: float, source: str) -> EVResult:
    """Create an EVResult from probability and odds."""
    implied_prob = 1.0 / odds
    prob_edge = model_prob - implied_prob
    ev = (model_prob * (odds - 1.0)) - (1.0 - model_prob)
    return EVResult(
        updated_prob=round(model_prob, 6),
        implied_prob=round(implied_prob, 6),
        prob_edge=round(prob_edge, 6),
        ev=round(ev, 6),
        is_positive_ev=ev > 0.0,
        model_source=source,
    )


def _analytical_fallback(
    sport: str,
    odds_home: float,
    odds_draw: float | None,
    odds_away: float | None,
) -> dict[str, EVResult]:
    """Fallback using implied probabilities with no edge (conservative)."""
    results: dict[str, EVResult] = {}
    if odds_home > 1.0:
        results["home"] = calculate_pre_match_ev(1.0 / odds_home, odds_home)
    if odds_draw is not None and odds_draw > 1.0:
        results["draw"] = calculate_pre_match_ev(1.0 / odds_draw, odds_draw)
    if odds_away is not None and odds_away > 1.0:
        results["away"] = calculate_pre_match_ev(1.0 / odds_away, odds_away)
    return results
