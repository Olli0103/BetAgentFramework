"""Deterministic Quarter-Kelly Criterion stake calculator.

Golden Rule #1: NO LLM MATH. Pure Python arithmetic.
Golden Rule #4: Moonshot parlays are hard-capped at 1.00 EUR by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from bet_agent.tools.odds import normalize_odds


@dataclass(frozen=True)
class KellyResult:
    """Result of a Kelly Criterion calculation."""

    stake_eur: float
    kelly_fraction: float
    full_kelly_fraction: float
    expected_profit: float  # (prob * odds) - 1, expected profit per unit staked
    reason: str | None


# Maximum fraction of bankroll per single bet (safety guardrail)
MAX_BANKROLL_FRACTION = 0.05  # 5%

# Minimum meaningful stake (below this, don't bother)
MIN_STAKE_EUR = 0.10


def calculate_quarter_kelly(
    prob: float,
    odds: float,
    bankroll: float,
) -> KellyResult:
    """Calculate the optimal stake using Quarter-Kelly Criterion.

    Args:
        prob: Model's estimated probability of winning (0-1).
        odds: Decimal odds offered by sportsbook (> 1.0).
        bankroll: Current bankroll balance in EUR.

    Returns:
        KellyResult with stake in EUR, Kelly fractions, and edge.

    Raises:
        ValueError: If inputs are out of valid ranges.
    """
    # Normalize odds format (auto-detect American → Decimal)
    odds = normalize_odds(odds)

    # Input validation
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prob must be in (0, 1), got {prob}")
    if bankroll < 0.0:
        raise ValueError(f"bankroll must be >= 0, got {bankroll}")

    # Expected profit per unit staked
    expected_profit = (prob * odds) - 1.0

    # Negative EV → don't bet
    if expected_profit <= 0.0:
        return KellyResult(
            stake_eur=0.0,
            kelly_fraction=0.0,
            full_kelly_fraction=0.0,
            expected_profit=round(expected_profit, 6),
            reason="negative_ev",
        )

    # Zero bankroll → can't bet
    if bankroll <= 0.0:
        return KellyResult(
            stake_eur=0.0,
            kelly_fraction=0.0,
            full_kelly_fraction=0.0,
            expected_profit=round(expected_profit, 6),
            reason="zero_bankroll",
        )

    # Full Kelly: f* = expected_profit / (odds - 1)
    full_kelly_fraction = expected_profit / (odds - 1.0)

    # Quarter Kelly (more conservative, reduces variance)
    quarter_kelly_fraction = full_kelly_fraction * 0.25

    # Calculate stake
    stake = bankroll * quarter_kelly_fraction

    # Apply guardrails
    stake = max(stake, 0.0)
    stake = min(stake, bankroll * MAX_BANKROLL_FRACTION)

    # Below minimum → not worth placing
    if stake < MIN_STAKE_EUR:
        return KellyResult(
            stake_eur=0.0,
            kelly_fraction=0.0,
            full_kelly_fraction=round(full_kelly_fraction, 6),
            expected_profit=round(expected_profit, 6),
            reason="below_minimum_stake",
        )

    # Round to 2 decimal places (cents)
    stake = round(stake, 2)

    return KellyResult(
        stake_eur=stake,
        kelly_fraction=round(quarter_kelly_fraction, 6),
        full_kelly_fraction=round(full_kelly_fraction, 6),
        expected_profit=round(expected_profit, 6),
        reason=None,
    )
