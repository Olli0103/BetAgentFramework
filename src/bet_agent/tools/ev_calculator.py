"""Deterministic Expected Value calculator — sport-aware.

Golden Rule #1: NO LLM MATH. This is pure Python + scipy.
Delegates live probability updates to sport-specific models via registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from bet_agent.tools.prob_models.registry import get_model


@dataclass(frozen=True)
class EVResult:
    """Result of an EV calculation."""

    updated_prob: float
    implied_prob: float
    edge: float
    ev: float
    is_positive_ev: bool


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
    updated_prob = max(0.0, min(1.0, updated_prob))

    # EV calculation (deterministic math)
    implied_prob = 1.0 / live_odds
    edge = updated_prob - implied_prob
    ev = (updated_prob * (live_odds - 1.0)) - (1.0 - updated_prob)

    return EVResult(
        updated_prob=round(updated_prob, 6),
        implied_prob=round(implied_prob, 6),
        edge=round(edge, 6),
        ev=round(ev, 6),
        is_positive_ev=ev > 0.0,
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
    edge = model_prob - implied_prob
    ev = (model_prob * (odds - 1.0)) - (1.0 - model_prob)

    return EVResult(
        updated_prob=round(model_prob, 6),
        implied_prob=round(implied_prob, 6),
        edge=round(edge, 6),
        ev=round(ev, 6),
        is_positive_ev=ev > 0.0,
    )
