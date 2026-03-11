"""Risk Manager Sizing Engine — strategic bet sizing with real DB data.

Fetches the current REAL ledger balance, calculates Quarter-Kelly stakes,
enforces daily/weekly loss limits, and routes unproven models to Paper.

Golden Rule #1: NO LLM MATH — delegates all arithmetic to kelly_calculator.
Golden Rule #3: Paper First — new models run on Paper until Auditor promotes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    BankrollLedger,
    BetStatus,
    LedgerType,
    PlacedBet,
    Prediction,
    PredictionStatus,
)
from bet_agent.tools.kelly_calculator import KellyResult, calculate_quarter_kelly

logger = logging.getLogger(__name__)

# Risk limits (must match agents.yaml)
MAX_SINGLE_BET_PCT = 5.0
MAX_DAILY_LOSS_PCT = 10.0
MAX_WEEKLY_LOSS_PCT = 20.0
MOONSHOT_HARD_CAP_EUR = 1.00

# Models considered "proven" (run on REAL ledger)
# Others go to PAPER until Auditor promotes them
_PROVEN_MODEL_PREFIXES = {"analytical_", "xgboost_"}


@dataclass(frozen=True)
class SizedBet:
    """A fully sized bet ready for alert or placement."""

    prediction_id: object  # UUID
    ledger_type: LedgerType
    stake_eur: float
    odds: float
    kelly: KellyResult
    reason: str | None  # None = good to go, else why rejected


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of daily/weekly loss limit checks."""

    daily_loss_eur: Decimal
    weekly_loss_eur: Decimal
    daily_limit_hit: bool
    weekly_limit_hit: bool
    bankroll: Decimal


# ── Bankroll queries ─────────────────────────────────────────────────


def get_bankroll(session: Session, ledger_type: LedgerType = LedgerType.REAL) -> Decimal:
    """Fetch the current balance from the bankroll_ledger table."""
    ledger = session.execute(
        select(BankrollLedger).where(BankrollLedger.ledger_type == ledger_type)
    ).scalar_one_or_none()

    if ledger is None:
        return Decimal("0.00")
    return ledger.balance


def get_daily_pnl(
    session: Session,
    check_date: date | None = None,
    ledger_type: LedgerType = LedgerType.REAL,
) -> Decimal:
    """Sum of PnL for resolved bets placed today."""
    if check_date is None:
        check_date = date.today()

    day_start = datetime.combine(check_date, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(check_date, time.max, tzinfo=timezone.utc)

    result = session.execute(
        select(func.coalesce(func.sum(PlacedBet.pnl_eur), 0))
        .where(
            PlacedBet.ledger_type == ledger_type,
            PlacedBet.placed_at >= day_start,
            PlacedBet.placed_at <= day_end,
            PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST]),
        )
    ).scalar()

    return Decimal(str(result))


def get_weekly_pnl(
    session: Session,
    check_date: date | None = None,
    ledger_type: LedgerType = LedgerType.REAL,
) -> Decimal:
    """Sum of PnL for resolved bets placed in the current week (Mon-Sun)."""
    if check_date is None:
        check_date = date.today()

    # Monday of current week
    monday = check_date - __import__("datetime").timedelta(days=check_date.weekday())
    week_start = datetime.combine(monday, time.min, tzinfo=timezone.utc)
    week_end = datetime.combine(check_date, time.max, tzinfo=timezone.utc)

    result = session.execute(
        select(func.coalesce(func.sum(PlacedBet.pnl_eur), 0))
        .where(
            PlacedBet.ledger_type == ledger_type,
            PlacedBet.placed_at >= week_start,
            PlacedBet.placed_at <= week_end,
            PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST]),
        )
    ).scalar()

    return Decimal(str(result))


def check_risk_limits(
    session: Session,
    ledger_type: LedgerType = LedgerType.REAL,
) -> RiskCheckResult:
    """Check if daily or weekly loss limits have been breached."""
    bankroll = get_bankroll(session, ledger_type)
    daily_pnl = get_daily_pnl(session, ledger_type=ledger_type)
    weekly_pnl = get_weekly_pnl(session, ledger_type=ledger_type)

    daily_limit = bankroll * Decimal(str(MAX_DAILY_LOSS_PCT / 100.0))
    weekly_limit = bankroll * Decimal(str(MAX_WEEKLY_LOSS_PCT / 100.0))

    return RiskCheckResult(
        daily_loss_eur=daily_pnl,
        weekly_loss_eur=weekly_pnl,
        daily_limit_hit=daily_pnl < -daily_limit if bankroll > 0 else False,
        weekly_limit_hit=weekly_pnl < -weekly_limit if bankroll > 0 else False,
        bankroll=bankroll,
    )


# ── Ledger routing ───────────────────────────────────────────────────


def assign_ledger_type(prediction: Prediction) -> LedgerType:
    """Decide REAL vs PAPER ledger based on model provenance.

    Proven models (analytical, xgboost) go to REAL.
    Unknown/experimental models go to PAPER for sandbox testing.
    """
    source = prediction.model_source or ""
    for prefix in _PROVEN_MODEL_PREFIXES:
        if source.startswith(prefix) or source == prefix.rstrip("_"):
            return LedgerType.REAL
    return LedgerType.PAPER


# ── Sizing ───────────────────────────────────────────────────────────


def size_bet(
    session: Session,
    prediction: Prediction,
    odds: float | None = None,
    is_parlay: bool = False,
) -> SizedBet:
    """Calculate optimal stake for a prediction using Quarter-Kelly.

    Args:
        session: SQLAlchemy session.
        prediction: APPROVED prediction with model_prob and best_odds.
        odds: Override odds (defaults to prediction.best_odds or implied).
        is_parlay: If True, hard-cap at MOONSHOT_HARD_CAP_EUR.

    Returns:
        SizedBet with stake, ledger type, and Kelly result.
    """
    # Determine ledger type
    ledger_type = assign_ledger_type(prediction)

    # Get bankroll
    bankroll = get_bankroll(session, ledger_type)

    # Determine odds to use
    if odds is not None:
        bet_odds = odds
    elif prediction.best_odds is not None:
        bet_odds = float(prediction.best_odds)
    elif prediction.implied_prob > 0:
        bet_odds = 1.0 / float(prediction.implied_prob)
    else:
        return SizedBet(
            prediction_id=prediction.id,
            ledger_type=ledger_type,
            stake_eur=0.0,
            odds=0.0,
            kelly=KellyResult(0.0, 0.0, 0.0, 0.0, "no_odds"),
            reason="no_odds_available",
        )

    model_prob = float(prediction.model_prob)

    # Check risk limits
    risk = check_risk_limits(session, ledger_type)
    if risk.daily_limit_hit:
        return SizedBet(
            prediction_id=prediction.id,
            ledger_type=ledger_type,
            stake_eur=0.0,
            odds=bet_odds,
            kelly=KellyResult(0.0, 0.0, 0.0, 0.0, "daily_limit_hit"),
            reason="daily_loss_limit_breached",
        )

    if risk.weekly_limit_hit:
        return SizedBet(
            prediction_id=prediction.id,
            ledger_type=ledger_type,
            stake_eur=0.0,
            odds=bet_odds,
            kelly=KellyResult(0.0, 0.0, 0.0, 0.0, "weekly_limit_hit"),
            reason="weekly_loss_limit_breached",
        )

    # Calculate Kelly
    kelly = calculate_quarter_kelly(
        prob=model_prob,
        odds=bet_odds,
        bankroll=float(bankroll),
    )

    stake = kelly.stake_eur

    # Apply parlay hard cap
    if is_parlay and stake > MOONSHOT_HARD_CAP_EUR:
        stake = MOONSHOT_HARD_CAP_EUR

    reason = kelly.reason

    return SizedBet(
        prediction_id=prediction.id,
        ledger_type=ledger_type,
        stake_eur=stake,
        odds=bet_odds,
        kelly=kelly,
        reason=reason,
    )


def deduct_stake_on_placement(
    session: Session,
    stake_eur: Decimal,
    ledger_type: LedgerType,
) -> Decimal:
    """Deduct the bet stake from the bankroll at placement time.

    This is the first half of the double-entry accounting model:
      1. Placement: balance -= stake  (this function)
      2. Settlement: balance += payout (in settlement_engine)

    Args:
        session: SQLAlchemy session.
        stake_eur: The stake amount to deduct.
        ledger_type: REAL or PAPER ledger.

    Returns:
        New bankroll balance after deduction.
    """
    ledger = session.execute(
        select(BankrollLedger).where(BankrollLedger.ledger_type == ledger_type)
    ).scalar_one_or_none()

    if ledger is None:
        logger.warning(
            "No %s ledger found — creating with negative balance", ledger_type.value,
        )
        ledger = BankrollLedger(ledger_type=ledger_type, balance=-stake_eur)
        session.add(ledger)
    else:
        ledger.balance -= stake_eur

    logger.info(
        "Deducted %.2f EUR from %s ledger (new balance: %.2f)",
        stake_eur, ledger_type.value, ledger.balance,
    )
    return ledger.balance


def size_all_approved(
    session: Session,
    predictions: list[Prediction] | None = None,
) -> list[SizedBet]:
    """Size bets for all APPROVED predictions with best_odds.

    Args:
        session: SQLAlchemy session.
        predictions: Specific predictions, or None to query all APPROVED.

    Returns:
        List of SizedBet results (includes $0 stakes for rejected bets).
    """
    if predictions is None:
        predictions = list(
            session.execute(
                select(Prediction)
                .where(Prediction.status == PredictionStatus.APPROVED)
            ).scalars().all()
        )

    results: list[SizedBet] = []
    for pred in predictions:
        sized = size_bet(session, pred)
        results.append(sized)

    placed = [s for s in results if s.stake_eur > 0]
    rejected = [s for s in results if s.stake_eur == 0]
    logger.info(
        "Sizing complete: %d bets sized, %d rejected out of %d",
        len(placed), len(rejected), len(results),
    )
    return results
