"""Settlement Engine — resolves placed bets against final match scores.

Deterministic tool: compares bet selections against actual results,
calculates PnL, updates bet status, and adjusts bankroll ledgers.

Golden Rule #1: NO LLM MATH — pure Python arithmetic for all PnL.
Golden Rule #2: STATEFUL MEMORY — all settlements go to PostgreSQL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    BankrollLedger,
    BetStatus,
    LedgerType,
    MarketType,
    Match,
    MatchState,
    PlacedBet,
)

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class SettlementResult:
    """Result of settling a single bet."""

    bet_id: object  # UUID
    old_status: BetStatus
    new_status: BetStatus
    pnl_eur: Decimal
    reason: str


@dataclass(frozen=True)
class SettlementSummary:
    """Summary of a settlement run."""

    total_settled: int
    won: int
    lost: int
    voided: int
    pushed: int  # bets pushed back to human (e.g., postponed match)
    total_pnl_real: Decimal
    total_pnl_paper: Decimal


# ── Outcome determination ────────────────────────────────────────────


def determine_outcome(
    bet: PlacedBet,
    match: Match,
) -> BetStatus:
    """Determine whether a bet WON, LOST, or is VOID based on final score.

    Args:
        bet: The placed bet to evaluate.
        match: The match with final scores.

    Returns:
        BetStatus.WON, BetStatus.LOST, or BetStatus.VOID.
    """
    if match.home_score is None or match.away_score is None:
        return BetStatus.VOID

    home = match.home_score
    away = match.away_score
    selection = bet.selection.lower()

    if bet.market_type == MarketType.MATCH_WINNER:
        return _settle_match_winner(selection, home, away)

    elif bet.market_type == MarketType.OVER_UNDER:
        return _settle_over_under(selection, home, away)

    elif bet.market_type == MarketType.BTTS:
        return _settle_btts(selection, home, away)

    elif bet.market_type == MarketType.SPREAD:
        return _settle_spread(selection, home, away)

    # Unknown market type → push to human
    return BetStatus.PUSHED_TO_HUMAN


def _settle_match_winner(selection: str, home: int, away: int) -> BetStatus:
    """Settle a match winner bet."""
    if selection == "home":
        return BetStatus.WON if home > away else BetStatus.LOST
    elif selection == "away":
        return BetStatus.WON if away > home else BetStatus.LOST
    elif selection == "draw":
        return BetStatus.WON if home == away else BetStatus.LOST
    return BetStatus.PUSHED_TO_HUMAN


def _settle_over_under(selection: str, home: int, away: int) -> BetStatus:
    """Settle an over/under bet.

    Selection format: 'over_2.5' or 'under_2.5'.
    """
    total = home + away

    parts = selection.split("_", 1)
    if len(parts) != 2:
        return BetStatus.PUSHED_TO_HUMAN

    direction = parts[0]
    try:
        line = float(parts[1])
    except ValueError:
        return BetStatus.PUSHED_TO_HUMAN

    if direction == "over":
        if total > line:
            return BetStatus.WON
        elif total == line:
            return BetStatus.VOID  # Push on exact line
        else:
            return BetStatus.LOST
    elif direction == "under":
        if total < line:
            return BetStatus.WON
        elif total == line:
            return BetStatus.VOID
        else:
            return BetStatus.LOST

    return BetStatus.PUSHED_TO_HUMAN


def _settle_btts(selection: str, home: int, away: int) -> BetStatus:
    """Settle a Both Teams To Score bet."""
    both_scored = home > 0 and away > 0

    if selection == "yes":
        return BetStatus.WON if both_scored else BetStatus.LOST
    elif selection == "no":
        return BetStatus.WON if not both_scored else BetStatus.LOST

    return BetStatus.PUSHED_TO_HUMAN


def _settle_spread(selection: str, home: int, away: int) -> BetStatus:
    """Settle a spread/handicap bet.

    Selection format: 'home_-1.5' or 'away_+1.5'.
    """
    parts = selection.split("_", 1)
    if len(parts) != 2:
        return BetStatus.PUSHED_TO_HUMAN

    side = parts[0]
    try:
        spread = float(parts[1])
    except ValueError:
        return BetStatus.PUSHED_TO_HUMAN

    if side == "home":
        adjusted = home + spread
        if adjusted > away:
            return BetStatus.WON
        elif adjusted == away:
            return BetStatus.VOID
        else:
            return BetStatus.LOST
    elif side == "away":
        adjusted = away + spread
        if adjusted > home:
            return BetStatus.WON
        elif adjusted == home:
            return BetStatus.VOID
        else:
            return BetStatus.LOST

    return BetStatus.PUSHED_TO_HUMAN


# ── PnL calculation ──────────────────────────────────────────────────


def calculate_pnl(bet: PlacedBet, outcome: BetStatus) -> Decimal:
    """Calculate the bankroll adjustment at settlement time.

    Accounting model: stake is DEDUCTED from bankroll at placement time
    (via ``deduct_stake_on_placement``).  At settlement we return money:

    WON:  +stake * odds          (full payout: original stake + profit)
    LOST:  0.00                  (stake already deducted — nothing to return)
    VOID: +stake                 (refund the deducted stake)

    The ``pnl_eur`` stored on the bet record represents the NET profit/loss
    for reporting purposes (not the bankroll delta, which is the return value):
      WON  net = stake * (odds - 1)   (profit)
      LOST net = -stake               (loss)
      VOID net = 0                    (break-even)
    """
    if outcome == BetStatus.WON:
        # Bankroll gets full payout (stake was already deducted)
        return round(bet.stake_eur * bet.odds_at_placement, 2)
    elif outcome == BetStatus.LOST:
        # Stake already gone at placement — nothing to return
        return Decimal("0.00")
    elif outcome == BetStatus.VOID:
        # Refund the stake that was deducted at placement
        return bet.stake_eur
    else:
        # PUSHED_TO_HUMAN → no change
        return Decimal("0.00")


# ── Bankroll update ──────────────────────────────────────────────────


def update_bankroll(
    session: Session,
    ledger_type: LedgerType,
    pnl: Decimal,
) -> Decimal:
    """Apply settlement return to the bankroll ledger.

    Accounting model (deduct-at-placement):
      - Stake was deducted when the bet was placed
      - WON:  balance += stake * odds   (full payout returned)
      - LOST: balance += 0              (stake already gone)
      - VOID: balance += stake          (refund)
    """
    ledger = session.execute(
        select(BankrollLedger)
        .where(BankrollLedger.ledger_type == ledger_type)
        .with_for_update()
    ).scalar_one_or_none()

    if ledger is None:
        logger.warning("No %s ledger found, creating with PnL as balance", ledger_type.value)
        ledger = BankrollLedger(ledger_type=ledger_type, balance=pnl)
        session.add(ledger)
    else:
        ledger.balance += pnl

    return ledger.balance


# ── Core settlement ──────────────────────────────────────────────────


def _net_pnl(bet: PlacedBet, outcome: BetStatus) -> Decimal:
    """Compute net profit/loss for reporting (stored on ``bet.pnl_eur``).

    This is the human-readable number:
      WON:  +stake * (odds - 1)   (profit)
      LOST: -stake                (loss)
      VOID:  0                    (break-even)
    """
    if outcome == BetStatus.WON:
        return round(bet.stake_eur * (bet.odds_at_placement - Decimal("1")), 2)
    elif outcome == BetStatus.LOST:
        return -bet.stake_eur
    else:
        return Decimal("0.00")


def settle_bet(
    session: Session,
    bet: PlacedBet,
    match: Match,
) -> SettlementResult:
    """Settle a single bet against a finished match.

    Updates bet status, PnL, resolved_at, and adjusts the bankroll.

    Accounting: stake was deducted at placement.  At settlement we
    return money to the bankroll (WON → full payout, VOID → refund).
    The ``bet.pnl_eur`` stores the human-readable net profit/loss.

    Idempotent: if the bet is already settled (WON/LOST/VOID), returns
    the existing result without modifying the bankroll again.
    """
    # Guard: skip already-settled bets to prevent double bankroll adjustment
    if bet.status in (BetStatus.WON, BetStatus.LOST, BetStatus.VOID):
        logger.debug("Bet %s already settled (%s), skipping", bet.id, bet.status.value)
        return SettlementResult(
            bet_id=bet.id,
            old_status=bet.status,
            new_status=bet.status,
            pnl_eur=bet.pnl_eur or Decimal("0.00"),
            reason="already_settled",
        )

    old_status = bet.status
    outcome = determine_outcome(bet, match)
    bankroll_delta = calculate_pnl(bet, outcome)
    net = _net_pnl(bet, outcome)

    # Update the bet record (net PnL for reporting)
    bet.status = outcome
    bet.pnl_eur = net
    bet.resolved_at = datetime.now(timezone.utc)

    # Update the bankroll (return money: payout or refund)
    update_bankroll(session, bet.ledger_type, bankroll_delta)

    return SettlementResult(
        bet_id=bet.id,
        old_status=old_status,
        new_status=outcome,
        pnl_eur=net,
        reason=f"{bet.selection} vs score {match.home_score}-{match.away_score}",
    )


def expire_stale_bets(
    session: Session,
    max_age_minutes: int = 60,
) -> list[dict]:
    """Expire PENDING bets that are older than max_age_minutes.

    When the HitL doesn't place a bet within the window, the line has
    likely moved and the edge is gone.  We VOID these bets.

    ACCOUNTING: PENDING bets have NO stake deducted (deduct-at-placement
    model deducts only when human confirms via /placed → PLACED status).
    So no bankroll refund is needed — just mark VOID for bookkeeping.

    Args:
        session: SQLAlchemy session.
        max_age_minutes: How long a PENDING bet can sit before expiry.

    Returns:
        List of dicts describing each expired bet (for Telegram alerts).
    """
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

    stale = list(
        session.execute(
            select(PlacedBet).where(
                PlacedBet.status == BetStatus.PENDING,
                PlacedBet.placed_at <= cutoff,
            )
        ).scalars().all()
    )

    if not stale:
        return []

    expired: list[dict] = []
    for bet in stale:
        # No bankroll refund needed — PENDING bets never had stake deducted
        bet.status = BetStatus.VOID
        bet.pnl_eur = Decimal("0.00")
        bet.resolved_at = datetime.now(timezone.utc)

        match = bet.match if bet.match else session.get(Match, bet.match_id)
        match_desc = f"{match.home_team} vs {match.away_team}" if match else "Unknown"

        expired.append({
            "bet_id": str(bet.id),
            "match": match_desc,
            "selection": bet.selection,
            "stake_eur": float(bet.stake_eur),
            "age_minutes": int((datetime.now(timezone.utc) - bet.placed_at).total_seconds() / 60),
        })

        logger.info(
            "Expired stale bet %s (%s) — edge likely gone after %d minutes",
            bet.id, match_desc, expired[-1]["age_minutes"],
        )

    session.flush()
    return expired


def void_unplaced_finished(session: Session) -> int:
    """Auto-VOID PENDING bets on FINISHED matches (never placed at sportsbook).

    These bets were generated by the model but the human never confirmed them.
    Since stake was NOT deducted for PENDING bets (deduct-at-placement model),
    no bankroll adjustment is needed — just mark them VOID for bookkeeping.

    Returns:
        Number of bets voided.
    """
    unplaced = list(
        session.execute(
            select(PlacedBet)
            .join(Match)
            .where(
                PlacedBet.status == BetStatus.PENDING,
                Match.match_state == MatchState.FINISHED,
            )
        ).scalars().all()
    )

    for bet in unplaced:
        bet.status = BetStatus.VOID
        bet.pnl_eur = Decimal("0.00")
        bet.resolved_at = datetime.now(timezone.utc)
        logger.info(
            "Auto-voided unplaced bet %s (match finished, never placed at sportsbook)",
            bet.id,
        )

    if unplaced:
        session.flush()
        logger.info("Voided %d unplaced bets on finished matches", len(unplaced))

    return len(unplaced)


def settle_finished_matches(
    session: Session,
) -> SettlementSummary:
    """Settle all PLACED bets on FINISHED matches.

    CRITICAL: Only settles bets the human actually placed at the sportsbook
    (status=PLACED). PENDING bets are auto-voided separately via
    void_unplaced_finished() — they were never placed and must NOT affect
    the bankroll with phantom wins/losses.

    This is the main entry point for the settlement pipeline.

    Returns:
        SettlementSummary with counts and total PnL.
    """
    # First: auto-void any PENDING bets on finished matches (no bankroll change)
    voided_unplaced = void_unplaced_finished(session)

    # Then: settle only bets the human actually placed at the sportsbook
    placed_bets = list(
        session.execute(
            select(PlacedBet)
            .join(Match)
            .where(
                PlacedBet.status == BetStatus.PLACED,
                Match.match_state == MatchState.FINISHED,
            )
        ).scalars().all()
    )

    logger.info("Found %d placed bets on finished matches", len(placed_bets))

    results: list[SettlementResult] = []

    for bet in placed_bets:
        match = bet.match
        if match is None:
            match = session.get(Match, bet.match_id)
        if match is None:
            logger.warning("Match not found for bet %s", bet.id)
            continue

        result = settle_bet(session, bet, match)
        results.append(result)
        logger.info(
            "Settled bet %s: %s → %s (PnL: %s EUR)",
            bet.id, result.old_status.value, result.new_status.value, result.pnl_eur,
        )

    session.flush()

    won = sum(1 for r in results if r.new_status == BetStatus.WON)
    lost = sum(1 for r in results if r.new_status == BetStatus.LOST)
    voided = sum(1 for r in results if r.new_status == BetStatus.VOID)
    pushed = sum(1 for r in results if r.new_status == BetStatus.PUSHED_TO_HUMAN)

    # Sum PnL by ledger type
    pnl_real = Decimal("0.00")
    pnl_paper = Decimal("0.00")
    for r, bet in zip(results, placed_bets):
        if bet.ledger_type == LedgerType.REAL:
            pnl_real += r.pnl_eur
        else:
            pnl_paper += r.pnl_eur

    summary = SettlementSummary(
        total_settled=len(results),
        won=won,
        lost=lost,
        voided=voided,
        pushed=pushed,
        total_pnl_real=pnl_real,
        total_pnl_paper=pnl_paper,
    )

    logger.info(
        "Settlement complete: %d settled (W:%d L:%d V:%d P:%d) "
        "PnL Real: %s EUR, Paper: %s EUR | %d unplaced bets auto-voided",
        summary.total_settled, won, lost, voided, pushed,
        summary.total_pnl_real, summary.total_pnl_paper, voided_unplaced,
    )
    return summary
