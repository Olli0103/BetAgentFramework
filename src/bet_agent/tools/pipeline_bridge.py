"""Pipeline Bridge — converts approved+sized predictions into PlacedBet(PENDING).

This closes the gap between the prediction pipeline (Prediction → APPROVED)
and the settlement pipeline (PlacedBet → PLACED → WON/LOST).

Without this bridge, `/pending` shows nothing and settlement has no inputs.

Usage:
    ensure_pending_bets_from_approved(session)        # current day
    ensure_pending_bets_from_approved(session, date)   # specific day
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    BetStatus,
    LedgerType,
    MarketType,
    Match,
    PlacedBet,
    Prediction,
    PredictionStatus,
)
from bet_agent.tools.notifier import check_bet_readiness

logger = logging.getLogger(__name__)


def _determine_ledger_type(
    prediction: Prediction,
    match: Match,
    stake_eur: float,
) -> LedgerType:
    """Route bet to REAL or PAPER based on readiness gate.

    Bets that fail readiness checks are sandboxed to PAPER.
    """
    readiness = check_bet_readiness(prediction, match, stake_eur)
    if not readiness.is_ready:
        logger.info(
            "Routing %s to PAPER — %s",
            prediction.selection, readiness.reason,
        )
        return LedgerType.PAPER
    return LedgerType.REAL


def _find_existing_bet(
    session: Session,
    prediction: Prediction,
) -> PlacedBet | None:
    """Find an existing PlacedBet for this prediction's unique key."""
    return session.execute(
        select(PlacedBet).where(
            PlacedBet.match_id == prediction.match_id,
            PlacedBet.market_type == prediction.market_type,
            PlacedBet.selection == prediction.selection,
        ).limit(1)
    ).scalar_one_or_none()


def ensure_pending_bets_from_approved(
    session: Session,
    target_date: date | None = None,
) -> dict:
    """Create PlacedBet(PENDING) rows for all approved predictions missing one.

    This is idempotent: re-running never creates duplicates.

    Args:
        session: SQLAlchemy session.
        target_date: Only process matches on this date.  None = today.

    Returns:
        Dict with counts: created, skipped_existing, skipped_no_odds, total.
    """
    if target_date is None:
        target_date = date.today()

    # Window: full day in UTC
    day_start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    # Fetch approved predictions that have best_odds (were successfully shopped+sized)
    predictions = list(
        session.execute(
            select(Prediction)
            .join(Match, Match.id == Prediction.match_id)
            .where(
                Prediction.status == PredictionStatus.APPROVED,
                Match.scheduled_at >= day_start,
                Match.scheduled_at < day_end,
            )
        ).scalars().all()
    )

    created = 0
    updated_existing = 0
    skipped_existing = 0
    skipped_no_odds = 0

    for pred in predictions:
        # Skip if no best_odds (line shopping didn't run or found nothing)
        if not pred.best_odds or float(pred.best_odds) <= 0:
            skipped_no_odds += 1
            continue

        match = pred.match if pred.match else session.get(Match, pred.match_id)
        if match is None:
            logger.warning("No match found for prediction %s", pred.id)
            continue

        # Import here to avoid circular imports
        from bet_agent.tools.sizing_engine import size_bet
        sized = size_bet(session, pred)
        stake = sized.stake_eur

        if stake <= 0:
            skipped_no_odds += 1
            continue

        # Route to REAL or PAPER via readiness gate
        ledger_type = _determine_ledger_type(pred, match, stake)

        # Check for existing bet
        existing = _find_existing_bet(session, pred)
        if existing is not None:
            # Sync ledger_type on PENDING bets (re-route if readiness changed)
            if existing.status == BetStatus.PENDING and existing.ledger_type != ledger_type:
                logger.info(
                    "Re-routing existing bet %s: %s → %s",
                    existing.selection, existing.ledger_type.value, ledger_type.value,
                )
                existing.ledger_type = ledger_type
                updated_existing += 1
            else:
                skipped_existing += 1
            continue

        bet = PlacedBet(
            ledger_type=ledger_type,
            match_id=pred.match_id,
            market_type=pred.market_type,
            selection=pred.selection,
            odds_at_placement=pred.best_odds,
            stake_eur=Decimal(str(round(stake, 2))),
            model_prob=pred.model_prob,
            ev_at_placement=pred.ev,
            status=BetStatus.PENDING,
            is_live_bet=False,
        )
        session.add(bet)
        created += 1

        logger.info(
            "Created PENDING bet: %s %s @%.2f (%.2f EUR, %s)",
            pred.market_type.value, pred.selection,
            float(pred.best_odds), stake, ledger_type.value,
        )

    session.flush()

    result = {
        "total": len(predictions),
        "created": created,
        "updated_existing": updated_existing,
        "skipped_existing": skipped_existing,
        "skipped_no_odds": skipped_no_odds,
    }
    logger.info("Pipeline bridge: %s", result)
    return result
