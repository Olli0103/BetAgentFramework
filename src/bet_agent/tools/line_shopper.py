"""Line Shopper — finds the best available odds across sportsbooks.

Takes APPROVED predictions and scans odds_markets to find the highest
decimal odds for the same market and selection across all sportsbooks.

Compares apples to apples: same market_type AND canonicalized selection.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    MarketType,
    Match,
    OddsMarket,
    Prediction,
    PredictionStatus,
)

logger = logging.getLogger(__name__)


def _canonicalize_selection(selection: str) -> str:
    """Canonicalize a selection string for apples-to-apples comparison.

    Handles:
      - Case folding: "Home" → "home"
      - Whitespace/underscore normalization: "over 2.5" → "over_2.5"
      - Trailing-zero stripping on lines: "over_2.50" → "over_2.5"
      - Synonym mapping: "1" → "home", "x" → "draw", "2" → "away"
      - BTTS: "yes"/"no" pass through
      - Spread: "home_-1.5" keeps sign, strips trailing zeros
    """
    s = selection.strip().lower()

    # Normalize whitespace and common separators to underscore
    s = re.sub(r"[\s\-–]+(?=\d)", "_", s)  # "over 2.5" or "over-2.5" → "over_2.5"
    s = s.replace(" ", "_")

    # 1X2 synonym mapping (common sportsbook formats)
    _SYNONYMS = {
        "1": "home",
        "x": "draw",
        "2": "away",
        "h": "home",
        "d": "draw",
        "a": "away",
        "home_win": "home",
        "away_win": "away",
    }
    if s in _SYNONYMS:
        return _SYNONYMS[s]

    # Strip trailing zeros on numeric lines: "over_2.50" → "over_2.5"
    def _strip_trailing_zeros(m: re.Match) -> str:
        num = m.group(0)
        if "." in num:
            return num.rstrip("0").rstrip(".")
        return num

    s = re.sub(r"\d+\.\d+", _strip_trailing_zeros, s)

    return s


@dataclass(frozen=True)
class ShoppedLine:
    """Result of shopping a single prediction across sportsbooks."""

    prediction_id: uuid.UUID
    market_type: MarketType
    selection: str
    best_odds: Decimal
    best_sportsbook: str
    original_implied_prob: Decimal  # 1/original odds used in prediction
    new_implied_prob: Decimal  # 1/best_odds
    odds_improvement: Decimal  # best_odds - original odds
    all_odds: list[dict]  # [{"sportsbook": str, "odds": Decimal}, ...]


def shop_line(
    session: Session,
    prediction: Prediction,
) -> ShoppedLine | None:
    """Find the best available odds for a prediction's market+selection.

    Queries odds_markets for matching pre-match odds across all sportsbooks,
    comparing same market_type and selection for apples-to-apples comparison.

    Args:
        session: SQLAlchemy session.
        prediction: An APPROVED prediction to shop.

    Returns:
        ShoppedLine with best odds found, or None if no odds available.
    """
    # Canonicalize selection for robust matching across sportsbooks
    canonical_sel = _canonicalize_selection(prediction.selection)

    # Query all pre-match odds for this match, market type, and selection
    odds_query = (
        select(OddsMarket)
        .where(
            OddsMarket.match_id == prediction.match_id,
            OddsMarket.market_type == prediction.market_type,
            OddsMarket.is_live == False,
        )
        .order_by(OddsMarket.odds_decimal.desc())
    )

    all_odds_rows = list(session.execute(odds_query).scalars().all())

    # Filter to matching selection (canonicalized comparison)
    matching = [
        row for row in all_odds_rows
        if _canonicalize_selection(row.selection) == canonical_sel
    ]

    if not matching:
        logger.info(
            "No odds found for prediction %s (%s %s [canonical: %s])",
            prediction.id, prediction.market_type.value, prediction.selection,
            canonical_sel,
        )
        return None

    # Best odds is the first row (ordered desc)
    best = matching[0]

    # Original implied prob from the prediction
    original_odds = Decimal("1") / prediction.implied_prob if prediction.implied_prob > 0 else Decimal("0")

    all_odds_list = [
        {"sportsbook": row.sportsbook, "odds": row.odds_decimal}
        for row in matching
    ]

    new_implied = Decimal("1") / best.odds_decimal if best.odds_decimal > 0 else Decimal("0")

    return ShoppedLine(
        prediction_id=prediction.id,
        market_type=prediction.market_type,
        selection=prediction.selection,
        best_odds=best.odds_decimal,
        best_sportsbook=best.sportsbook,
        original_implied_prob=prediction.implied_prob,
        new_implied_prob=round(new_implied, 6),
        odds_improvement=best.odds_decimal - original_odds,
        all_odds=all_odds_list,
    )


def apply_shopped_line(
    session: Session,
    prediction: Prediction,
    shopped: ShoppedLine,
) -> None:
    """Store best odds and sportsbook on the prediction record."""
    prediction.best_odds = shopped.best_odds
    prediction.best_sportsbook = shopped.best_sportsbook

    # Recalculate EV with the better odds
    model_prob = float(prediction.model_prob)
    best_odds = float(shopped.best_odds)
    new_implied = 1.0 / best_odds
    new_edge = model_prob - new_implied
    new_ev = (model_prob * (best_odds - 1.0)) - (1.0 - model_prob)

    prediction.implied_prob = Decimal(str(round(new_implied, 6)))
    prediction.prob_edge = Decimal(str(round(new_edge, 6)))
    prediction.ev = Decimal(str(round(new_ev, 6)))


def shop_all_approved(
    session: Session,
    predictions: list[Prediction] | None = None,
) -> list[ShoppedLine]:
    """Shop lines for all APPROVED predictions.

    Args:
        session: SQLAlchemy session.
        predictions: Specific predictions to shop, or None to query all APPROVED.

    Returns:
        List of ShoppedLine results.
    """
    if predictions is None:
        predictions = list(
            session.execute(
                select(Prediction)
                .where(Prediction.status == PredictionStatus.APPROVED)
            ).scalars().all()
        )

    results: list[ShoppedLine] = []
    for pred in predictions:
        shopped = shop_line(session, pred)
        if shopped is not None:
            apply_shopped_line(session, pred, shopped)
            results.append(shopped)
        else:
            logger.warning(
                "No odds found for approved prediction %s, skipping",
                pred.id,
            )

    session.flush()
    logger.info("Line shopping complete: %d predictions shopped", len(results))
    return results
