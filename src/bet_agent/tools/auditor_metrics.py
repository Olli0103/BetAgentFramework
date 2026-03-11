"""Auditor Metrics & MLOps — model health evaluation and self-learning triggers.

Evaluates the mathematical health of prediction models by:
  1. Fetching settled bets grouped by model_name
  2. Calculating Brier Score (MSE of predicted prob vs actual outcome 0/1)
  3. Calculating ROI % (total PnL / total staked)
  4. Writing daily stats to model_metrics table
  5. Checking degradation thresholds for self-learning triggers

Golden Rule #1: NO LLM MATH — pure Python arithmetic.
Golden Rule #3: STATEFUL MEMORY — all metrics go to PostgreSQL.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    BetStatus,
    LedgerType,
    Match,
    ModelMetrics,
    PlacedBet,
    Prediction,
    Sport,
)

logger = logging.getLogger(__name__)

# ── Degradation thresholds ───────────────────────────────────────────

BRIER_THRESHOLD = 0.22  # Above this = poorly calibrated
ROI_THRESHOLD = -5.0  # Below -5% ROI = losing money
ROLLING_WINDOW = 50  # Minimum bets for degradation check


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class ModelHealthReport:
    """Health report for a single model on a single ledger."""

    model_name: str
    ledger_type: LedgerType
    sport: str | None
    brier_score: float
    roi_pct: float
    total_bets: int
    record_win: int
    record_loss: int
    total_staked: Decimal
    total_pnl: Decimal
    is_degraded: bool
    degradation_reasons: list[str] = field(default_factory=list)


@dataclass
class AuditSummary:
    """Summary of a full audit run."""

    reports: list[ModelHealthReport]
    degraded_models: list[str]
    metrics_written: int


# ── Brier Score calculation ──────────────────────────────────────────


def calculate_brier_score(predictions: list[tuple[float, int]]) -> float:
    """Calculate the Brier Score (mean squared error of probability predictions).

    Args:
        predictions: List of (predicted_probability, actual_outcome) tuples.
                     actual_outcome is 1 for WON, 0 for LOST.

    Returns:
        Brier Score (lower is better, 0 = perfect, 0.25 = coin flip).
    """
    if not predictions:
        return 0.0

    total = sum((prob - outcome) ** 2 for prob, outcome in predictions)
    return round(total / len(predictions), 6)


def calculate_roi(total_staked: Decimal, total_pnl: Decimal) -> float:
    """Calculate Return on Investment as a percentage.

    ROI = (total_pnl / total_staked) * 100
    """
    if total_staked <= 0:
        return 0.0
    return round(float(total_pnl / total_staked) * 100.0, 2)


# ── Bet-to-model mapping ────────────────────────────────────────────


def _get_model_for_bet(
    session: Session,
    bet: PlacedBet,
) -> str | None:
    """Find the prediction model_name that generated this bet.

    Looks up predictions matching the bet's match_id, market_type, and selection.
    """
    pred = session.execute(
        select(Prediction).where(
            Prediction.match_id == bet.match_id,
            Prediction.market_type == bet.market_type,
            Prediction.selection == bet.selection,
        )
    ).scalar_one_or_none()

    if pred is not None:
        return pred.model_name
    # Fallback: use a generic name
    return "unknown"


# ── Core metrics evaluation ──────────────────────────────────────────


def evaluate_model_performance(
    session: Session,
    eval_date: date | None = None,
    ledger_type: LedgerType | None = None,
) -> list[ModelHealthReport]:
    """Evaluate all models based on their settled bets.

    Fetches settled bets (WON/LOST) and groups them by model_name,
    then calculates Brier Score, ROI, and degradation status.

    Args:
        session: SQLAlchemy session.
        eval_date: Only consider bets resolved on this date.
                   None = consider all resolved bets (rolling window).
        ledger_type: Filter by ledger type. None = evaluate both.

    Returns:
        List of ModelHealthReport objects.
    """
    # Build base query for settled bets
    query = (
        select(PlacedBet)
        .where(PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST]))
    )

    if ledger_type is not None:
        query = query.where(PlacedBet.ledger_type == ledger_type)

    if eval_date is not None:
        day_start = datetime.combine(eval_date, time.min, tzinfo=timezone.utc)
        day_end = datetime.combine(eval_date, time.max, tzinfo=timezone.utc)
        query = query.where(
            PlacedBet.resolved_at >= day_start,
            PlacedBet.resolved_at <= day_end,
        )

    settled_bets = list(session.execute(query).scalars().all())

    if not settled_bets:
        logger.info("No settled bets found for evaluation")
        return []

    # Group bets by (model_name, ledger_type)
    groups: dict[tuple[str, LedgerType], list[PlacedBet]] = defaultdict(list)

    for bet in settled_bets:
        model_name = _get_model_for_bet(session, bet)
        groups[(model_name, bet.ledger_type)].append(bet)

    # Evaluate each group
    reports: list[ModelHealthReport] = []

    for (model_name, lt), bets in groups.items():
        # Brier Score: compare predicted probability vs actual (1/0)
        brier_inputs: list[tuple[float, int]] = []
        for bet in bets:
            prob = float(bet.model_prob)
            outcome = 1 if bet.status == BetStatus.WON else 0
            brier_inputs.append((prob, outcome))

        brier = calculate_brier_score(brier_inputs)

        # ROI
        total_staked = sum(bet.stake_eur for bet in bets)
        total_pnl = sum(bet.pnl_eur or Decimal("0") for bet in bets)
        roi = calculate_roi(total_staked, total_pnl)

        wins = sum(1 for b in bets if b.status == BetStatus.WON)
        losses = sum(1 for b in bets if b.status == BetStatus.LOST)

        # Get sport from first bet's match
        sport = None
        match = bets[0].match if bets[0].match else session.get(Match, bets[0].match_id)
        if match:
            sport = match.sport.value

        # Degradation check (only if enough data)
        is_degraded = False
        reasons: list[str] = []

        if len(bets) >= ROLLING_WINDOW:
            if brier > BRIER_THRESHOLD:
                is_degraded = True
                reasons.append(
                    f"Brier Score {brier:.4f} > {BRIER_THRESHOLD} "
                    f"(over {len(bets)} bets)"
                )
            if roi < ROI_THRESHOLD:
                is_degraded = True
                reasons.append(
                    f"ROI {roi:.2f}% < {ROI_THRESHOLD}% "
                    f"(over {len(bets)} bets)"
                )

        report = ModelHealthReport(
            model_name=model_name,
            ledger_type=lt,
            sport=sport,
            brier_score=brier,
            roi_pct=roi,
            total_bets=len(bets),
            record_win=wins,
            record_loss=losses,
            total_staked=total_staked,
            total_pnl=total_pnl,
            is_degraded=is_degraded,
            degradation_reasons=reasons,
        )
        reports.append(report)

    return reports


# ── Write metrics to DB ──────────────────────────────────────────────


def write_daily_metrics(
    session: Session,
    reports: list[ModelHealthReport],
    metrics_date: date | None = None,
) -> int:
    """Write model health reports to the model_metrics table.

    Upserts on (model_name, date, ledger_type) unique constraint.

    Returns:
        Number of metrics rows written/updated.
    """
    if metrics_date is None:
        metrics_date = date.today()

    count = 0
    for report in reports:
        # Check for existing row
        existing = session.execute(
            select(ModelMetrics).where(
                ModelMetrics.model_name == report.model_name,
                ModelMetrics.date == metrics_date,
                ModelMetrics.ledger_type == report.ledger_type,
            )
        ).scalar_one_or_none()

        if existing:
            existing.brier_score = Decimal(str(report.brier_score))
            existing.roi_pct = Decimal(str(report.roi_pct))
            existing.total_bets = report.total_bets
            existing.record_win = report.record_win
            existing.record_loss = report.record_loss
        else:
            session.add(ModelMetrics(
                model_name=report.model_name,
                date=metrics_date,
                brier_score=Decimal(str(report.brier_score)),
                roi_pct=Decimal(str(report.roi_pct)),
                total_bets=report.total_bets,
                record_win=report.record_win,
                record_loss=report.record_loss,
                ledger_type=report.ledger_type,
            ))
        count += 1

    session.flush()
    return count


# ── Full audit pipeline ──────────────────────────────────────────────


def run_daily_audit(
    session: Session,
    eval_date: date | None = None,
) -> AuditSummary:
    """Run the complete daily audit pipeline.

    1. Evaluate all models with settled bets
    2. Write metrics to DB
    3. Identify degraded models
    4. Return summary for Master Agent alerting

    Args:
        session: SQLAlchemy session.
        eval_date: Date to evaluate. None = all settled bets.

    Returns:
        AuditSummary with reports and degradation alerts.
    """
    reports = evaluate_model_performance(session, eval_date)
    metrics_written = write_daily_metrics(session, reports, eval_date or date.today())

    degraded = [r.model_name for r in reports if r.is_degraded]

    if degraded:
        logger.warning(
            "DEGRADED MODELS DETECTED: %s — halting betting until human review",
            ", ".join(degraded),
        )

    for report in reports:
        logger.info(
            "Model %s [%s]: Brier=%.4f ROI=%.2f%% W:%d L:%d (%d bets) %s",
            report.model_name,
            report.ledger_type.value,
            report.brier_score,
            report.roi_pct,
            report.record_win,
            report.record_loss,
            report.total_bets,
            "DEGRADED" if report.is_degraded else "OK",
        )

    return AuditSummary(
        reports=reports,
        degraded_models=degraded,
        metrics_written=metrics_written,
    )


# ── Rolling window check ────────────────────────────────────────────


def check_rolling_degradation(
    session: Session,
    model_name: str,
    ledger_type: LedgerType = LedgerType.REAL,
    window: int = ROLLING_WINDOW,
) -> ModelHealthReport | None:
    """Check a specific model's health over its last N bets.

    This is used for on-demand checks outside the daily audit.

    Returns:
        ModelHealthReport or None if insufficient data.
    """
    # Get the last N settled bets for this model via predictions
    pred_subq = (
        select(Prediction.match_id, Prediction.market_type, Prediction.selection)
        .where(Prediction.model_name == model_name)
        .subquery()
    )

    bets = list(
        session.execute(
            select(PlacedBet)
            .where(
                PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST]),
                PlacedBet.ledger_type == ledger_type,
                PlacedBet.match_id.in_(select(pred_subq.c.match_id)),
            )
            .order_by(PlacedBet.resolved_at.desc())
            .limit(window)
        ).scalars().all()
    )

    if len(bets) < window:
        return None

    brier_inputs = [(float(b.model_prob), 1 if b.status == BetStatus.WON else 0) for b in bets]
    brier = calculate_brier_score(brier_inputs)

    total_staked = sum(b.stake_eur for b in bets)
    total_pnl = sum(b.pnl_eur or Decimal("0") for b in bets)
    roi = calculate_roi(total_staked, total_pnl)

    is_degraded = brier > BRIER_THRESHOLD or roi < ROI_THRESHOLD
    reasons = []
    if brier > BRIER_THRESHOLD:
        reasons.append(f"Brier {brier:.4f} > {BRIER_THRESHOLD}")
    if roi < ROI_THRESHOLD:
        reasons.append(f"ROI {roi:.2f}% < {ROI_THRESHOLD}%")

    return ModelHealthReport(
        model_name=model_name,
        ledger_type=ledger_type,
        sport=None,
        brier_score=brier,
        roi_pct=roi,
        total_bets=len(bets),
        record_win=sum(1 for b in bets if b.status == BetStatus.WON),
        record_loss=sum(1 for b in bets if b.status == BetStatus.LOST),
        total_staked=total_staked,
        total_pnl=total_pnl,
        is_degraded=is_degraded,
        degradation_reasons=reasons,
    )
