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
    MarketType,
    Match,
    ModelMetrics,
    PlacedBet,
    Prediction,
    Sport,
)

logger = logging.getLogger(__name__)

# ── Degradation thresholds ───────────────────────────────────────────

BRIER_THRESHOLD = 0.22  # Default (fallback)
ROI_THRESHOLD = -5.0  # Below -5% ROI = losing money
ROLLING_WINDOW = 50  # Minimum bets for degradation check

# Per-market Brier thresholds: 3-way markets are harder to calibrate
_BRIER_THRESHOLDS: dict[str, float] = {
    "match_winner": 0.18,  # 3-way (Home/Draw/Away) — bookmaker ~0.16-0.18
    "over_under": 0.22,    # 2-way — bookmaker ~0.20-0.22
    "btts": 0.22,          # 2-way
    "spread": 0.22,        # 2-way
}


def _brier_threshold_for_model(model_name: str) -> float:
    """Return the appropriate Brier threshold based on model name.

    Match winner models (3-way) get a tighter threshold since they're
    evaluated with multi-class Brier. 2-way models use the default.
    """
    name = model_name.lower()
    if "match_winner" in name:
        return _BRIER_THRESHOLDS["match_winner"]
    if "over_under" in name:
        return _BRIER_THRESHOLDS["over_under"]
    return BRIER_THRESHOLD


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
    """Calculate the binary Brier Score (mean squared error of probability predictions).

    Suitable for 2-way markets (Over/Under, BTTS). For 3-way markets
    (Match Winner), use calculate_multiclass_brier_score() instead.

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


def calculate_multiclass_brier_score(
    prob_vectors: list[tuple[list[float], list[int]]],
) -> float:
    """Calculate the multi-class Brier Score for N-way markets.

    For 3-way markets (Home/Draw/Away), the binary formula is incorrect:
    it only evaluates the probability of the selected outcome, missing
    how probability mass was distributed across all 3 outcomes.

    Multi-class Brier: (1/N) * sum_i( sum_j( (p_ij - o_ij)^2 ) )
    where j iterates over all outcomes (home, draw, away).

    Args:
        prob_vectors: List of (predicted_probs, outcome_vector) tuples.
                      predicted_probs: [p_home, p_draw, p_away]
                      outcome_vector:  [1, 0, 0] for home win, etc.

    Returns:
        Multi-class Brier Score (lower is better).
    """
    if not prob_vectors:
        return 0.0

    total = 0.0
    for probs, outcomes in prob_vectors:
        total += sum((p - o) ** 2 for p, o in zip(probs, outcomes))

    return round(total / len(prob_vectors), 6)


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


def _compute_brier_for_bets(
    session: Session,
    bets: list[PlacedBet],
    model_name: str,
) -> float:
    """Compute Brier Score using the correct formula per market type.

    For MATCH_WINNER (3-way): uses multi-class Brier by looking up sibling
    predictions from the same model to reconstruct the full probability
    vector [p_home, p_draw, p_away] vs outcome vector [1, 0, 0].

    For OVER_UNDER and other 2-way markets: uses standard binary Brier.

    This prevents the common error of evaluating a 3-way model as if
    it were binary, which would mask poor probability distribution.
    """
    binary_inputs: list[tuple[float, int]] = []
    multiclass_inputs: list[tuple[list[float], list[int]]] = []

    # Cache: match_id → {selection: model_prob} for 3-way lookups
    _mw_cache: dict[object, dict[str, float]] = {}

    for bet in bets:
        if bet.market_type == MarketType.MATCH_WINNER:
            # Multi-class: reconstruct full probability vector
            match_id = bet.match_id

            if match_id not in _mw_cache:
                # Fetch all predictions for this match/model
                sibling_preds = session.execute(
                    select(Prediction).where(
                        Prediction.match_id == match_id,
                        Prediction.model_name == model_name,
                        Prediction.market_type == MarketType.MATCH_WINNER,
                    )
                ).scalars().all()
                _mw_cache[match_id] = {
                    p.selection.lower(): float(p.model_prob) for p in sibling_preds
                }

            probs_dict = _mw_cache[match_id]

            # Get the actual match result
            match = bet.match if bet.match else session.get(Match, bet.match_id)
            if match is None or match.home_score is None or match.away_score is None:
                continue

            if match.home_score > match.away_score:
                outcome_vec = [1, 0, 0]
            elif match.home_score == match.away_score:
                outcome_vec = [0, 1, 0]
            else:
                outcome_vec = [0, 0, 1]

            prob_vec = [
                probs_dict.get("home", 0.33),
                probs_dict.get("draw", 0.33),
                probs_dict.get("away", 0.33),
            ]

            multiclass_inputs.append((prob_vec, outcome_vec))
        else:
            # Binary: Over/Under, BTTS, Spread — (p, outcome) is correct
            prob = float(bet.model_prob)
            outcome = 1 if bet.status == BetStatus.WON else 0
            binary_inputs.append((prob, outcome))

    # Combine: weighted average of both Brier scores
    scores: list[float] = []
    weights: list[int] = []

    if multiclass_inputs:
        mc_brier = calculate_multiclass_brier_score(multiclass_inputs)
        scores.append(mc_brier)
        weights.append(len(multiclass_inputs))

    if binary_inputs:
        bin_brier = calculate_brier_score(binary_inputs)
        scores.append(bin_brier)
        weights.append(len(binary_inputs))

    if not scores:
        return 0.0

    # Weighted average across market types
    total_weight = sum(weights)
    combined = sum(s * w for s, w in zip(scores, weights)) / total_weight
    return round(combined, 6)


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
        # Brier Score: use multi-class for 3-way markets, binary for 2-way
        brier = _compute_brier_for_bets(session, bets, model_name)

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
            brier_thresh = _brier_threshold_for_model(model_name)
            if brier > brier_thresh:
                is_degraded = True
                reasons.append(
                    f"Brier Score {brier:.4f} > {brier_thresh} "
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
    # Get the last N settled bets for this model via full (match_id, market_type, selection) join
    pred_subq = (
        select(Prediction.match_id, Prediction.market_type, Prediction.selection)
        .where(Prediction.model_name == model_name)
        .subquery()
    )

    bets = list(
        session.execute(
            select(PlacedBet)
            .join(
                pred_subq,
                and_(
                    PlacedBet.match_id == pred_subq.c.match_id,
                    PlacedBet.market_type == pred_subq.c.market_type,
                    PlacedBet.selection == pred_subq.c.selection,
                ),
            )
            .where(
                PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST]),
                PlacedBet.ledger_type == ledger_type,
            )
            .order_by(PlacedBet.resolved_at.desc())
            .limit(window)
        ).scalars().all()
    )

    if len(bets) < window:
        return None

    # Use the correct Brier implementation (multiclass for 3-way markets)
    brier = _compute_brier_for_bets(session, bets, model_name)

    total_staked = sum(b.stake_eur for b in bets)
    total_pnl = sum(b.pnl_eur or Decimal("0") for b in bets)
    roi = calculate_roi(total_staked, total_pnl)

    brier_thresh = _brier_threshold_for_model(model_name)
    is_degraded = brier > brier_thresh or roi < ROI_THRESHOLD
    reasons = []
    if brier > brier_thresh:
        reasons.append(f"Brier {brier:.4f} > {brier_thresh}")
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
