"""Odds format detection and conversion utilities.

Supports American (+270, -150), Decimal (3.70), and Fractional (27/10) odds.
All calculators in this project work with **European decimal odds** internally.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Decimal odds above this threshold are implausible — likely American format.
# The highest realistic decimal odds in practice are ~50.0 (5000/1 longshots).
# American odds start at ±100, so anything >= 100 is almost certainly American.
_AMERICAN_THRESHOLD = 99.0


def american_to_decimal(american: float) -> float:
    """Convert American odds to European decimal odds.

    +270 → 3.70  (profit $270 on $100 stake)
    -150 → 1.667 (must stake $150 to profit $100)
    """
    if american >= 100:
        return round(1.0 + american / 100.0, 4)
    elif american <= -100:
        return round(1.0 + 100.0 / abs(american), 4)
    else:
        raise ValueError(
            f"Invalid American odds: {american}. "
            f"Must be >= +100 or <= -100."
        )


def normalize_odds(odds: float) -> float:
    """Auto-detect odds format and return European decimal odds.

    Heuristic:
      - odds <= 0 or odds <= -100 → negative American (e.g., -150 → 1.667)
      - 1.0 < odds < 99.0        → already decimal (pass-through)
      - odds >= 99.0              → positive American (e.g., +270 → 3.70)
      - odds == 1.0               → invalid (no profit possible)
      - 0 < odds < 1.0            → invalid for all formats

    Raises:
        ValueError: If odds cannot be interpreted in any format.
    """
    if odds <= -100:
        result = american_to_decimal(odds)
        logger.info("Auto-converted American odds %+.0f → decimal %.4f", odds, result)
        return result

    if odds >= _AMERICAN_THRESHOLD:
        result = american_to_decimal(odds)
        logger.info("Auto-converted American odds %+.0f → decimal %.4f", odds, result)
        return result

    if odds > 1.0:
        return odds  # Already decimal

    raise ValueError(
        f"Invalid odds value: {odds}. "
        f"Expected decimal (> 1.0) or American (>= +100 / <= -100)."
    )
