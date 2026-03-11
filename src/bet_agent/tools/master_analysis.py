"""Master Analysis — read-only intelligence tools for the Master Agent.

Provides the Master Agent with database query tools to answer
human questions from the Telegram Concierge. All tools are strictly
READ-ONLY and contain zero betting logic.

Used for: portfolio summaries, model health reports, veto explanations,
sport exposure breakdowns, and recent activity feeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    BankrollLedger,
    BetStatus,
    LedgerType,
    MarketType,
    Match,
    MatchState,
    ModelMetrics,
    PlacedBet,
    Prediction,
    PredictionStatus,
    Sport,
)

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PortfolioSummary:
    """Snapshot of the fund's current state."""

    real_balance: Decimal
    paper_balance: Decimal
    pending_bets_count: int
    today_bets_count: int
    today_pnl_real: Decimal
    today_pnl_paper: Decimal
    total_exposure_real: Decimal
    total_exposure_paper: Decimal
    sport_exposure: dict[str, Decimal]
    win_rate_7d: float
    total_bets_7d: int


@dataclass(frozen=True)
class ModelHealthSummary:
    """Health status for a single model."""

    model_name: str
    sport: str
    latest_brier: float
    latest_roi: float
    total_bets: int
    record_win: int
    record_loss: int
    is_degraded: bool
    trend: str  # "improving", "stable", "declining"


@dataclass(frozen=True)
class VetoExplanation:
    """Full explanation of why a prediction was vetoed."""

    match_description: str
    sport: str
    league: str
    selection: str
    model_prob: float
    ev: float
    veto_reason: str | None
    prediction_status: str


@dataclass(frozen=True)
class RecentActivity:
    """Summary of recent pipeline activity."""

    predictions_today: int
    approved_today: int
    vetoed_today: int
    placed_today: int
    settled_today: int
    last_settlement_pnl: Decimal


@dataclass(frozen=True)
class SportExposure:
    """Exposure breakdown for a single sport."""

    sport: str
    pending_count: int
    total_stake: Decimal
    avg_odds: float
    avg_ev: float


# ── Core query tools ─────────────────────────────────────────────────


def fetch_portfolio_summary(
    session: Session,
    as_of: date | None = None,
) -> PortfolioSummary:
    """Build a complete portfolio snapshot for the concierge.

    Args:
        session: SQLAlchemy session.
        as_of: Date for "today" calculations. Defaults to today.

    Returns:
        PortfolioSummary with balances, exposure, and performance.
    """
    if as_of is None:
        as_of = date.today()

    day_start = datetime.combine(as_of, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(as_of, time.max, tzinfo=timezone.utc)

    # Bankroll balances
    real_balance = Decimal("0.00")
    paper_balance = Decimal("0.00")
    for ledger in session.execute(select(BankrollLedger)).scalars().all():
        if ledger.ledger_type == LedgerType.REAL:
            real_balance = ledger.balance
        elif ledger.ledger_type == LedgerType.PAPER:
            paper_balance = ledger.balance

    # Pending bets (open exposure)
    pending_bets = list(
        session.execute(
            select(PlacedBet).where(PlacedBet.status == BetStatus.PENDING)
        ).scalars().all()
    )

    # Exposure by sport
    sport_exposure: dict[str, Decimal] = {}
    total_exp_real = Decimal("0.00")
    total_exp_paper = Decimal("0.00")

    for bet in pending_bets:
        match = bet.match if bet.match else session.get(Match, bet.match_id)
        if match:
            sport_name = match.sport.value
            sport_exposure[sport_name] = sport_exposure.get(sport_name, Decimal("0.00")) + bet.stake_eur
        if bet.ledger_type == LedgerType.REAL:
            total_exp_real += bet.stake_eur
        else:
            total_exp_paper += bet.stake_eur

    # Today's bets and PnL
    today_bets = list(
        session.execute(
            select(PlacedBet).where(
                PlacedBet.placed_at >= day_start,
                PlacedBet.placed_at <= day_end,
            )
        ).scalars().all()
    )

    today_pnl_real = sum(
        (b.pnl_eur or Decimal("0")) for b in today_bets
        if b.ledger_type == LedgerType.REAL and b.pnl_eur is not None
    )
    today_pnl_paper = sum(
        (b.pnl_eur or Decimal("0")) for b in today_bets
        if b.ledger_type == LedgerType.PAPER and b.pnl_eur is not None
    )

    # 7-day win rate
    week_ago = datetime.combine(as_of - timedelta(days=7), time.min, tzinfo=timezone.utc)
    settled_7d = list(
        session.execute(
            select(PlacedBet).where(
                PlacedBet.resolved_at >= week_ago,
                PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST]),
            )
        ).scalars().all()
    )
    wins_7d = sum(1 for b in settled_7d if b.status == BetStatus.WON)
    win_rate = round(wins_7d / len(settled_7d) * 100, 1) if settled_7d else 0.0

    return PortfolioSummary(
        real_balance=real_balance,
        paper_balance=paper_balance,
        pending_bets_count=len(pending_bets),
        today_bets_count=len(today_bets),
        today_pnl_real=today_pnl_real,
        today_pnl_paper=today_pnl_paper,
        total_exposure_real=total_exp_real,
        total_exposure_paper=total_exp_paper,
        sport_exposure=sport_exposure,
        win_rate_7d=win_rate,
        total_bets_7d=len(settled_7d),
    )


def fetch_model_health(
    session: Session,
    sport: str | None = None,
    days: int = 30,
) -> list[ModelHealthSummary]:
    """Fetch model health metrics, optionally filtered by sport.

    Args:
        session: SQLAlchemy session.
        sport: Filter by sport name (e.g., 'football'). None = all.
        days: Lookback window for trend analysis.

    Returns:
        List of ModelHealthSummary objects.
    """
    cutoff = date.today() - timedelta(days=days)

    query = select(ModelMetrics).where(ModelMetrics.date >= cutoff)

    # Filter by sport via model_name heuristic (model names contain sport)
    if sport:
        query = query.where(ModelMetrics.model_name.ilike(f"%{sport}%"))

    metrics = list(
        session.execute(query.order_by(ModelMetrics.date.desc())).scalars().all()
    )

    if not metrics:
        return []

    # Group by model_name and take the latest entry + compute trend
    from collections import defaultdict
    by_model: dict[str, list[ModelMetrics]] = defaultdict(list)
    for m in metrics:
        by_model[m.model_name].append(m)

    results: list[ModelHealthSummary] = []
    for model_name, entries in by_model.items():
        entries.sort(key=lambda x: x.date, reverse=True)
        latest = entries[0]

        # Trend: compare latest Brier to average of older entries
        trend = "stable"
        if len(entries) >= 3:
            recent_brier = float(entries[0].brier_score)
            older_brier = sum(float(e.brier_score) for e in entries[1:4]) / min(3, len(entries) - 1)
            if recent_brier < older_brier - 0.02:
                trend = "improving"
            elif recent_brier > older_brier + 0.02:
                trend = "declining"

        # Detect sport from model_name
        detected_sport = "unknown"
        for s in Sport:
            if s.value in model_name.lower():
                detected_sport = s.value
                break

        is_degraded = (
            float(latest.brier_score) > 0.22 and latest.total_bets >= 50
        ) or (
            float(latest.roi_pct) < -5.0 and latest.total_bets >= 50
        )

        results.append(ModelHealthSummary(
            model_name=model_name,
            sport=detected_sport,
            latest_brier=round(float(latest.brier_score), 4),
            latest_roi=round(float(latest.roi_pct), 2),
            total_bets=latest.total_bets,
            record_win=latest.record_win,
            record_loss=latest.record_loss,
            is_degraded=is_degraded,
            trend=trend,
        ))

    return results


def explain_veto_reason(
    session: Session,
    match_id: object,
) -> list[VetoExplanation]:
    """Explain why predictions on a match were vetoed (or approved).

    Args:
        session: SQLAlchemy session.
        match_id: UUID of the match.

    Returns:
        List of VetoExplanation for each prediction on this match.
    """
    match = session.get(Match, match_id)
    if match is None:
        return []

    predictions = list(
        session.execute(
            select(Prediction).where(Prediction.match_id == match_id)
        ).scalars().all()
    )

    return [
        VetoExplanation(
            match_description=f"{match.home_team} vs {match.away_team}",
            sport=match.sport.value,
            league=match.league,
            selection=p.selection,
            model_prob=round(float(p.model_prob), 4),
            ev=round(float(p.ev), 4),
            veto_reason=p.veto_reason,
            prediction_status=p.status.value,
        )
        for p in predictions
    ]


def fetch_recent_activity(
    session: Session,
    as_of: date | None = None,
) -> RecentActivity:
    """Fetch today's pipeline activity summary.

    Args:
        session: SQLAlchemy session.
        as_of: Date for "today". Defaults to today.

    Returns:
        RecentActivity summary.
    """
    if as_of is None:
        as_of = date.today()

    day_start = datetime.combine(as_of, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(as_of, time.max, tzinfo=timezone.utc)

    # Predictions created today
    preds = list(
        session.execute(
            select(Prediction).where(
                Prediction.created_at >= day_start,
                Prediction.created_at <= day_end,
            )
        ).scalars().all()
    )

    approved = sum(1 for p in preds if p.status == PredictionStatus.APPROVED)
    vetoed = sum(1 for p in preds if p.status == PredictionStatus.VETOED)
    placed = sum(1 for p in preds if p.status == PredictionStatus.PLACED)

    # Settled today
    settled = list(
        session.execute(
            select(PlacedBet).where(
                PlacedBet.resolved_at >= day_start,
                PlacedBet.resolved_at <= day_end,
            )
        ).scalars().all()
    )

    last_pnl = sum(b.pnl_eur or Decimal("0") for b in settled)

    return RecentActivity(
        predictions_today=len(preds),
        approved_today=approved,
        vetoed_today=vetoed,
        placed_today=placed,
        settled_today=len(settled),
        last_settlement_pnl=last_pnl,
    )


def fetch_sport_exposure(
    session: Session,
) -> list[SportExposure]:
    """Get pending bet exposure broken down by sport.

    Returns:
        List of SportExposure objects.
    """
    pending = list(
        session.execute(
            select(PlacedBet).where(PlacedBet.status == BetStatus.PENDING)
        ).scalars().all()
    )

    if not pending:
        return []

    from collections import defaultdict
    by_sport: dict[str, list[PlacedBet]] = defaultdict(list)

    for bet in pending:
        match = bet.match if bet.match else session.get(Match, bet.match_id)
        if match:
            by_sport[match.sport.value].append(bet)

    results = []
    for sport_name, bets in by_sport.items():
        total_stake = sum(b.stake_eur for b in bets)
        avg_odds = sum(float(b.odds_at_placement) for b in bets) / len(bets)
        avg_ev = sum(float(b.ev_at_placement) for b in bets) / len(bets)

        results.append(SportExposure(
            sport=sport_name,
            pending_count=len(bets),
            total_stake=total_stake,
            avg_odds=round(avg_odds, 3),
            avg_ev=round(avg_ev, 4),
        ))

    return sorted(results, key=lambda x: x.total_stake, reverse=True)


def fetch_pending_for_human(
    session: Session,
) -> list[dict]:
    """Fetch bets waiting for human execution (PUSHED_TO_HUMAN status).

    Returns list of dicts with match details, selection, stake, odds.
    """
    bets = list(
        session.execute(
            select(PlacedBet).where(
                PlacedBet.status.in_([BetStatus.PENDING, BetStatus.PUSHED_TO_HUMAN])
            )
        ).scalars().all()
    )

    results = []
    for bet in bets:
        match = bet.match if bet.match else session.get(Match, bet.match_id)
        if match is None:
            continue
        results.append({
            "bet_id": str(bet.id),
            "match": f"{match.home_team} vs {match.away_team}",
            "sport": match.sport.value,
            "league": match.league,
            "selection": bet.selection,
            "market": bet.market_type.value,
            "stake_eur": float(bet.stake_eur),
            "odds": float(bet.odds_at_placement),
            "ledger": bet.ledger_type.value,
            "scheduled_at": match.scheduled_at.isoformat() if match.scheduled_at else "",
        })

    return results


def fetch_pnl_timeseries(
    session: Session,
    days: int = 30,
    ledger_type: LedgerType | None = None,
) -> list[dict]:
    """Fetch daily PnL timeseries for chart rendering.

    Returns list of {date, pnl, cumulative_pnl, bets_count} dicts.
    """
    cutoff = datetime.combine(
        date.today() - timedelta(days=days), time.min, tzinfo=timezone.utc
    )

    # Compute pre-lookback baseline: sum of all PnL resolved before the window
    baseline_query = (
        select(func.coalesce(func.sum(PlacedBet.pnl_eur), 0))
        .where(
            PlacedBet.resolved_at < cutoff,
            PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST]),
        )
    )
    if ledger_type:
        baseline_query = baseline_query.where(PlacedBet.ledger_type == ledger_type)

    baseline = Decimal(str(session.execute(baseline_query).scalar()))

    # Fetch bets within the lookback window
    query = (
        select(PlacedBet)
        .where(
            PlacedBet.resolved_at >= cutoff,
            PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST]),
        )
        .order_by(PlacedBet.resolved_at)
    )

    if ledger_type:
        query = query.where(PlacedBet.ledger_type == ledger_type)

    bets = list(session.execute(query).scalars().all())

    if not bets:
        return []

    # Group by date
    from collections import defaultdict
    daily: dict[date, list[PlacedBet]] = defaultdict(list)
    for b in bets:
        if b.resolved_at:
            daily[b.resolved_at.date()].append(b)

    cumulative = baseline  # Start from pre-lookback history
    timeseries = []
    for d in sorted(daily.keys()):
        day_pnl = sum(b.pnl_eur or Decimal("0") for b in daily[d])
        cumulative += day_pnl
        timeseries.append({
            "date": d.isoformat(),
            "pnl": float(day_pnl),
            "cumulative_pnl": float(cumulative),
            "bets_count": len(daily[d]),
        })

    return timeseries


def mark_bet_placed_by_user(
    session: Session,
    bet_id: str,
    placed_by: str,
    actual_odds: float | None = None,
    actual_stake: float | None = None,
) -> dict:
    """Mark a pending bet as placed by a syndicate member.

    Captures the ACTUAL odds and stake from the sportsbook (which may differ
    from the model's original values due to line movement or manual sizing).
    Recalculates EV on the actual odds and warns if the bet is now -EV.
    Deducts the actual stake from the bankroll ledger.

    Args:
        session: SQLAlchemy session.
        bet_id: UUID string of the bet.
        placed_by: Display name of the person who placed.
        actual_odds: The real odds obtained at the sportsbook.
        actual_stake: The real stake placed (EUR).

    Returns:
        Dict with bet details, EV assessment, and any warnings.
    """
    import uuid as _uuid
    try:
        uid = _uuid.UUID(bet_id)
    except ValueError:
        return {"error": f"Invalid bet_id: {bet_id}"}

    bet = session.get(PlacedBet, uid)
    if bet is None:
        return {"error": f"Bet {bet_id} not found"}

    if bet.status not in (BetStatus.PENDING, BetStatus.PUSHED_TO_HUMAN):
        return {"error": f"Bet {bet_id} is already {bet.status.value}"}

    # Capture original values for comparison
    original_odds = float(bet.odds_at_placement)
    original_stake = float(bet.stake_eur)

    # Override with actual values from the sportsbook
    if actual_odds is not None:
        bet.odds_at_placement = Decimal(str(actual_odds))
    if actual_stake is not None:
        bet.stake_eur = Decimal(str(actual_stake))

    # Recalculate EV on the actual odds
    model_prob = float(bet.model_prob)
    used_odds = float(bet.odds_at_placement)
    new_ev = (model_prob * (used_odds - 1.0)) - (1.0 - model_prob)
    bet.ev_at_placement = Decimal(str(round(new_ev, 6)))

    # Deduct actual stake from bankroll
    from bet_agent.tools.sizing_engine import deduct_stake_on_placement
    deduct_stake_on_placement(session, bet.stake_eur, bet.ledger_type)

    # Mark as placed
    bet.status = BetStatus.PUSHED_TO_HUMAN
    session.flush()

    match = bet.match if bet.match else session.get(Match, bet.match_id)
    match_desc = f"{match.home_team} vs {match.away_team}" if match else "Unknown"

    # Build warnings
    warnings = []
    if new_ev < 0:
        warnings.append(
            f"NEGATIVE EV ({new_ev:+.4f})! "
            f"Quote dropped from {original_odds:.2f} to {used_odds:.2f}. "
            f"This bet has negative expected value at the actual odds."
        )
    if actual_odds is not None and actual_odds < original_odds * 0.95:
        warnings.append(
            f"Significant slippage: model assumed {original_odds:.2f}, "
            f"you got {actual_odds:.2f} ({(actual_odds/original_odds - 1)*100:+.1f}%)"
        )

    return {
        "success": True,
        "bet_id": bet_id,
        "match": match_desc,
        "selection": bet.selection,
        "stake_eur": float(bet.stake_eur),
        "odds": float(bet.odds_at_placement),
        "original_odds": original_odds,
        "original_stake": original_stake,
        "ev": round(new_ev, 4),
        "placed_by": placed_by,
        "warnings": warnings,
    }


# ── Formatting helpers (for Telegram responses) ─────────────────────


def format_portfolio_text(summary: PortfolioSummary) -> str:
    """Format portfolio summary as a Telegram-friendly text block."""
    lines = [
        "PORTFOLIO STATUS",
        "",
        f"REAL Balance:  {summary.real_balance:>10.2f} EUR",
        f"PAPER Balance: {summary.paper_balance:>10.2f} EUR",
        "",
        f"Open Bets:  {summary.pending_bets_count}",
        f"Today Bets: {summary.today_bets_count}",
        "",
        f"Today PnL (REAL):  {summary.today_pnl_real:>+.2f} EUR",
        f"Today PnL (PAPER): {summary.today_pnl_paper:>+.2f} EUR",
        "",
        f"Exposure REAL:  {summary.total_exposure_real:.2f} EUR",
        f"Exposure PAPER: {summary.total_exposure_paper:.2f} EUR",
        "",
        f"7d Win Rate: {summary.win_rate_7d:.1f}% ({summary.total_bets_7d} bets)",
    ]

    if summary.sport_exposure:
        lines.append("")
        lines.append("Exposure by Sport:")
        for sport, amount in sorted(
            summary.sport_exposure.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {sport:<20s} {amount:.2f} EUR")

    return "\n".join(lines)


def format_pending_text(pending: list[dict]) -> str:
    """Format pending bets list for Telegram."""
    if not pending:
        return "No pending bets waiting for execution."

    lines = [f"PENDING BETS ({len(pending)})", ""]

    for i, b in enumerate(pending, 1):
        lines.append(
            f"{i}. {b['match']} ({b['sport']})\n"
            f"   {b['selection']} @ {b['odds']:.2f} | "
            f"{b['stake_eur']:.2f}EUR [{b['ledger'].upper()}]\n"
            f"   ID: {b['bet_id'][:8]}..."
        )

    lines.append("")
    lines.append("Use /placed <bet_id> <actual_odds> <actual_stake> to confirm.")
    return "\n".join(lines)


def format_pnl_text(
    session: Session,
) -> str:
    """Format a quick PnL summary for Telegram."""
    summary = fetch_portfolio_summary(session)
    lines = [
        "PnL REPORT",
        "",
        f"REAL:  {summary.real_balance:>10.2f} EUR  (today: {summary.today_pnl_real:>+.2f})",
        f"PAPER: {summary.paper_balance:>10.2f} EUR  (today: {summary.today_pnl_paper:>+.2f})",
        "",
        f"7d Win Rate: {summary.win_rate_7d:.1f}% over {summary.total_bets_7d} bets",
    ]
    return "\n".join(lines)
