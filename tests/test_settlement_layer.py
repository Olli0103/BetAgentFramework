"""Tests for Phase 4: Settlement & Self-Learning Layer.

Covers results_fetcher, settlement_engine, and auditor_metrics.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import (
    BankrollLedger,
    Base,
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


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _yesterday():
    return date.today() - timedelta(days=1)


def _add_match(
    session,
    sport=Sport.FOOTBALL,
    home="FC Bayern",
    away="BVB Dortmund",
    state=MatchState.NOT_STARTED,
    scheduled=None,
    home_score=None,
    away_score=None,
):
    if scheduled is None:
        scheduled = datetime.combine(
            _yesterday(), datetime.min.time(), tzinfo=timezone.utc
        )
    m = Match(
        sport=sport,
        league="Bundesliga",
        home_team=home,
        away_team=away,
        scheduled_at=scheduled,
        match_state=state,
        home_score=home_score,
        away_score=away_score,
    )
    session.add(m)
    session.flush()
    return m


def _add_bet(
    session,
    match,
    selection="home",
    market_type=MarketType.MATCH_WINNER,
    odds=Decimal("2.10"),
    stake=Decimal("10.00"),
    model_prob=Decimal("0.55"),
    ledger=LedgerType.REAL,
    status=BetStatus.PENDING,
):
    b = PlacedBet(
        match_id=match.id,
        ledger_type=ledger,
        market_type=market_type,
        selection=selection,
        odds_at_placement=odds,
        stake_eur=stake,
        model_prob=model_prob,
        ev_at_placement=Decimal("0.08"),
        status=status,
    )
    session.add(b)
    session.flush()
    return b


def _add_prediction(session, match, selection="home", model_name="analytical_football"):
    p = Prediction(
        match_id=match.id,
        model_name=model_name,
        market_type=MarketType.MATCH_WINNER,
        selection=selection,
        model_prob=Decimal("0.55"),
        implied_prob=Decimal("0.4762"),
        prob_edge=Decimal("0.0738"),
        ev=Decimal("0.08"),
        model_source="analytical",
        status=PredictionStatus.PLACED,
    )
    session.add(p)
    session.flush()
    return p


def _add_bankroll(session, ledger=LedgerType.REAL, balance=Decimal("1000.00")):
    b = BankrollLedger(ledger_type=ledger, balance=balance)
    session.add(b)
    session.flush()
    return b


# ── Results Fetcher Tests ────────────────────────────────────────────


class TestResultsFetcher:
    """Tests for the results fetcher."""

    def test_get_unsettled_matches(self, db_session):
        """Finds past matches with pending bets that aren't finished."""
        from bet_agent.tools.results_fetcher import get_unsettled_matches

        match = _add_match(db_session, state=MatchState.NOT_STARTED)
        _add_bet(db_session, match)

        unsettled = get_unsettled_matches(db_session)
        assert len(unsettled) == 1
        assert unsettled[0].id == match.id

    def test_ignores_finished_matches(self, db_session):
        """Already finished matches should not appear."""
        from bet_agent.tools.results_fetcher import get_unsettled_matches

        match = _add_match(db_session, state=MatchState.FINISHED, home_score=2, away_score=1)
        _add_bet(db_session, match)

        unsettled = get_unsettled_matches(db_session)
        assert len(unsettled) == 0

    def test_ignores_matches_without_pending_bets(self, db_session):
        """Matches with no pending bets should not appear."""
        from bet_agent.tools.results_fetcher import get_unsettled_matches

        _add_match(db_session, state=MatchState.NOT_STARTED)
        # No bet added

        unsettled = get_unsettled_matches(db_session)
        assert len(unsettled) == 0

    def test_update_match_result(self, db_session):
        """update_match_result sets scores and state."""
        from bet_agent.tools.results_fetcher import MatchResult, update_match_result

        match = _add_match(db_session)
        result = MatchResult("FC Bayern", "BVB Dortmund", 3, 1, is_finished=True)

        update_match_result(db_session, match, result)

        assert match.home_score == 3
        assert match.away_score == 1
        assert match.match_state == MatchState.FINISHED
        assert match.is_live is False

    def test_fetch_and_update_with_manual_backend(self, db_session):
        """Full flow: fetch results using ManualResultsBackend."""
        from bet_agent.tools.results_fetcher import (
            ManualResultsBackend,
            MatchResult,
            fetch_and_update_results,
        )

        match = _add_match(db_session)
        _add_bet(db_session, match)

        backend = ManualResultsBackend()
        backend.add_result(
            "FC Bayern vs BVB Dortmund",
            MatchResult("FC Bayern", "BVB Dortmund", 2, 0, is_finished=True),
        )

        result = fetch_and_update_results(db_session, backend)

        assert result.matches_checked == 1
        assert result.matches_updated == 1
        assert result.matches_not_found == 0
        assert match.match_state == MatchState.FINISHED
        assert match.home_score == 2

    def test_missing_result_counted(self, db_session):
        """Matches with no result from backend are counted as not_found."""
        from bet_agent.tools.results_fetcher import (
            ManualResultsBackend,
            fetch_and_update_results,
        )

        match = _add_match(db_session)
        _add_bet(db_session, match)

        result = fetch_and_update_results(db_session, ManualResultsBackend())

        assert result.matches_not_found == 1
        assert match.match_state == MatchState.NOT_STARTED


# ── Settlement Engine Tests ──────────────────────────────────────────


class TestDetermineOutcome:
    """Tests for bet outcome determination."""

    def test_home_win(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=2, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home")

        assert determine_outcome(bet, match) == BetStatus.WON

    def test_home_loss(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=0, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home")

        assert determine_outcome(bet, match) == BetStatus.LOST

    def test_away_win(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=0, away_score=2, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="away")

        assert determine_outcome(bet, match) == BetStatus.WON

    def test_draw_win(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=1, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="draw")

        assert determine_outcome(bet, match) == BetStatus.WON

    def test_draw_loss(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=2, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="draw")

        assert determine_outcome(bet, match) == BetStatus.LOST

    def test_over_win(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=2, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="over_2.5", market_type=MarketType.OVER_UNDER)

        assert determine_outcome(bet, match) == BetStatus.WON

    def test_over_loss(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=1, away_score=0, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="over_2.5", market_type=MarketType.OVER_UNDER)

        assert determine_outcome(bet, match) == BetStatus.LOST

    def test_under_win(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=1, away_score=0, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="under_2.5", market_type=MarketType.OVER_UNDER)

        assert determine_outcome(bet, match) == BetStatus.WON

    def test_over_under_push_on_exact_line(self, db_session):
        """Total exactly on the line → VOID (push, stake returned)."""
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=1, away_score=2, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="over_3.0", market_type=MarketType.OVER_UNDER)

        assert determine_outcome(bet, match) == BetStatus.VOID

    def test_btts_yes_win(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=2, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="yes", market_type=MarketType.BTTS)

        assert determine_outcome(bet, match) == BetStatus.WON

    def test_btts_yes_loss(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="yes", market_type=MarketType.BTTS)

        assert determine_outcome(bet, match) == BetStatus.LOST

    def test_btts_no_win(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=1, away_score=0, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="no", market_type=MarketType.BTTS)

        assert determine_outcome(bet, match) == BetStatus.WON

    def test_spread_home_win(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=3, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home_-1.5", market_type=MarketType.SPREAD)

        assert determine_outcome(bet, match) == BetStatus.WON

    def test_spread_home_loss(self, db_session):
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=2, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home_-1.5", market_type=MarketType.SPREAD)

        assert determine_outcome(bet, match) == BetStatus.LOST

    def test_spread_push(self, db_session):
        """Spread exactly covers → VOID."""
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, home_score=2, away_score=1, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home_-1.0", market_type=MarketType.SPREAD)

        assert determine_outcome(bet, match) == BetStatus.VOID

    def test_no_scores_void(self, db_session):
        """Missing scores → VOID."""
        from bet_agent.tools.settlement_engine import determine_outcome

        match = _add_match(db_session, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home")

        assert determine_outcome(bet, match) == BetStatus.VOID


class TestPnLCalculation:
    """Tests for PnL arithmetic."""

    def test_won_pnl(self, db_session):
        from bet_agent.tools.settlement_engine import calculate_pnl

        match = _add_match(db_session)
        bet = _add_bet(db_session, match, odds=Decimal("2.50"), stake=Decimal("10.00"))

        pnl = calculate_pnl(bet, BetStatus.WON)
        assert pnl == Decimal("15.00")  # 10 * (2.50 - 1)

    def test_lost_pnl(self, db_session):
        from bet_agent.tools.settlement_engine import calculate_pnl

        match = _add_match(db_session)
        bet = _add_bet(db_session, match, stake=Decimal("25.00"))

        pnl = calculate_pnl(bet, BetStatus.LOST)
        assert pnl == Decimal("-25.00")

    def test_void_pnl(self, db_session):
        from bet_agent.tools.settlement_engine import calculate_pnl

        match = _add_match(db_session)
        bet = _add_bet(db_session, match)

        pnl = calculate_pnl(bet, BetStatus.VOID)
        assert pnl == Decimal("0.00")


class TestSettlementEngine:
    """Tests for the full settlement pipeline."""

    def test_settle_won_bet(self, db_session):
        """Settling a winning bet updates status, PnL, and bankroll."""
        from bet_agent.tools.settlement_engine import settle_bet

        match = _add_match(db_session, home_score=3, away_score=0, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home", odds=Decimal("2.00"), stake=Decimal("20.00"))
        _add_bankroll(db_session, LedgerType.REAL, Decimal("980.00"))

        result = settle_bet(db_session, bet, match)

        assert result.new_status == BetStatus.WON
        assert result.pnl_eur == Decimal("20.00")
        assert bet.status == BetStatus.WON
        assert bet.pnl_eur == Decimal("20.00")
        assert bet.resolved_at is not None

    def test_settle_lost_bet(self, db_session):
        """Settling a losing bet updates bankroll negatively."""
        from bet_agent.tools.settlement_engine import settle_bet

        match = _add_match(db_session, home_score=0, away_score=2, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home", stake=Decimal("15.00"))
        _add_bankroll(db_session, LedgerType.REAL, Decimal("985.00"))

        result = settle_bet(db_session, bet, match)

        assert result.new_status == BetStatus.LOST
        assert result.pnl_eur == Decimal("-15.00")

    def test_settle_void_bet_no_pnl(self, db_session):
        """VOID bet has zero PnL."""
        from bet_agent.tools.settlement_engine import settle_bet

        match = _add_match(db_session, state=MatchState.FINISHED)
        # No scores → VOID
        bet = _add_bet(db_session, match)
        _add_bankroll(db_session)

        result = settle_bet(db_session, bet, match)
        assert result.new_status == BetStatus.VOID
        assert result.pnl_eur == Decimal("0.00")

    def test_settle_finished_matches_batch(self, db_session):
        """settle_finished_matches processes all pending bets on finished matches."""
        from bet_agent.tools.settlement_engine import settle_finished_matches

        m1 = _add_match(db_session, home="A", away="B", home_score=2, away_score=0, state=MatchState.FINISHED)
        m2 = _add_match(db_session, home="C", away="D", home_score=1, away_score=3, state=MatchState.FINISHED)

        _add_bet(db_session, m1, selection="home", stake=Decimal("10.00"), odds=Decimal("2.00"))
        _add_bet(db_session, m2, selection="home", stake=Decimal("10.00"), odds=Decimal("2.00"))

        _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        summary = settle_finished_matches(db_session)

        assert summary.total_settled == 2
        assert summary.won == 1
        assert summary.lost == 1

    def test_skips_already_settled_bets(self, db_session):
        """Already WON/LOST bets should not be re-settled."""
        from bet_agent.tools.settlement_engine import settle_finished_matches

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        _add_bet(db_session, match, status=BetStatus.WON)
        _add_bankroll(db_session)

        summary = settle_finished_matches(db_session)
        assert summary.total_settled == 0

    def test_bankroll_updated_correctly(self, db_session):
        """Bankroll should reflect PnL after settlement."""
        from bet_agent.tools.settlement_engine import settle_bet

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        bet = _add_bet(db_session, match, selection="home", odds=Decimal("2.50"), stake=Decimal("10.00"))
        ledger = _add_bankroll(db_session, LedgerType.REAL, Decimal("990.00"))

        settle_bet(db_session, bet, match)
        db_session.flush()

        assert ledger.balance == Decimal("1005.00")  # 990 + 15 (10 * 1.5)

    def test_paper_and_real_ledgers_separate(self, db_session):
        """REAL and PAPER bets go to their respective ledgers."""
        from bet_agent.tools.settlement_engine import settle_finished_matches

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        _add_bet(db_session, match, selection="home", ledger=LedgerType.REAL,
                 odds=Decimal("2.00"), stake=Decimal("10.00"))
        _add_bet(db_session, match, selection="away", ledger=LedgerType.PAPER,
                 odds=Decimal("3.00"), stake=Decimal("5.00"))

        real_ledger = _add_bankroll(db_session, LedgerType.REAL, Decimal("990.00"))
        paper_ledger = _add_bankroll(db_session, LedgerType.PAPER, Decimal("995.00"))

        summary = settle_finished_matches(db_session)

        assert summary.total_pnl_real == Decimal("10.00")  # won: 10 * (2-1)
        assert summary.total_pnl_paper == Decimal("-5.00")  # lost: -5
        assert real_ledger.balance == Decimal("1000.00")
        assert paper_ledger.balance == Decimal("990.00")


# ── Auditor Metrics Tests ────────────────────────────────────────────


class TestBrierScore:
    """Tests for Brier Score calculation."""

    def test_perfect_prediction(self):
        from bet_agent.tools.auditor_metrics import calculate_brier_score

        # Perfect: predicted 1.0 and won, predicted 0.0 and lost
        preds = [(1.0, 1), (0.0, 0)]
        assert calculate_brier_score(preds) == 0.0

    def test_worst_prediction(self):
        from bet_agent.tools.auditor_metrics import calculate_brier_score

        # Worst: predicted 1.0 and lost, predicted 0.0 and won
        preds = [(1.0, 0), (0.0, 1)]
        assert calculate_brier_score(preds) == 1.0

    def test_coin_flip_prediction(self):
        from bet_agent.tools.auditor_metrics import calculate_brier_score

        # Coin flip: always predict 0.5
        preds = [(0.5, 1), (0.5, 0)]
        assert calculate_brier_score(preds) == 0.25

    def test_empty_predictions(self):
        from bet_agent.tools.auditor_metrics import calculate_brier_score

        assert calculate_brier_score([]) == 0.0

    def test_realistic_calibrated_model(self):
        """A well-calibrated model should have Brier < 0.25."""
        from bet_agent.tools.auditor_metrics import calculate_brier_score

        # Model says 0.7 and wins 7 out of 10
        preds = [(0.7, 1)] * 7 + [(0.7, 0)] * 3
        brier = calculate_brier_score(preds)
        assert brier < 0.25


class TestROI:
    """Tests for ROI calculation."""

    def test_positive_roi(self):
        from bet_agent.tools.auditor_metrics import calculate_roi

        roi = calculate_roi(Decimal("100"), Decimal("15"))
        assert roi == 15.0

    def test_negative_roi(self):
        from bet_agent.tools.auditor_metrics import calculate_roi

        roi = calculate_roi(Decimal("200"), Decimal("-30"))
        assert roi == -15.0

    def test_zero_staked(self):
        from bet_agent.tools.auditor_metrics import calculate_roi

        assert calculate_roi(Decimal("0"), Decimal("0")) == 0.0

    def test_breakeven(self):
        from bet_agent.tools.auditor_metrics import calculate_roi

        assert calculate_roi(Decimal("500"), Decimal("0")) == 0.0


class TestModelEvaluation:
    """Tests for the full model evaluation pipeline."""

    def test_evaluate_with_settled_bets(self, db_session):
        """Evaluate produces a report for settled bets."""
        from bet_agent.tools.auditor_metrics import evaluate_model_performance

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        _add_prediction(db_session, match, selection="home")

        # Simulate settled bets
        b1 = _add_bet(db_session, match, selection="home", odds=Decimal("2.00"),
                       stake=Decimal("10.00"), model_prob=Decimal("0.60"),
                       status=BetStatus.WON)
        b1.pnl_eur = Decimal("10.00")
        b1.resolved_at = datetime.now(timezone.utc)
        db_session.flush()

        reports = evaluate_model_performance(db_session)

        assert len(reports) == 1
        assert reports[0].model_name == "analytical_football"
        assert reports[0].total_bets == 1
        assert reports[0].record_win == 1
        assert reports[0].roi_pct > 0

    def test_evaluate_no_settled_bets(self, db_session):
        """No settled bets → empty reports."""
        from bet_agent.tools.auditor_metrics import evaluate_model_performance

        reports = evaluate_model_performance(db_session)
        assert len(reports) == 0

    def test_degradation_detection_brier(self, db_session):
        """Model with high Brier Score over 50+ bets should be flagged."""
        from bet_agent.tools.auditor_metrics import evaluate_model_performance

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        _add_prediction(db_session, match, selection="home")

        # Create 50+ badly calibrated bets (predicted 0.9 but only won 20%)
        for i in range(60):
            b = _add_bet(
                db_session, match, selection="home",
                odds=Decimal("2.00"), stake=Decimal("10.00"),
                model_prob=Decimal("0.90"),
                status=BetStatus.WON if i < 12 else BetStatus.LOST,
            )
            b.pnl_eur = Decimal("10.00") if i < 12 else Decimal("-10.00")
            b.resolved_at = datetime.now(timezone.utc)
        db_session.flush()

        reports = evaluate_model_performance(db_session)

        assert len(reports) >= 1
        degraded = [r for r in reports if r.is_degraded]
        assert len(degraded) >= 1
        # Check that Brier is the reason
        assert any("Brier" in reason for r in degraded for reason in r.degradation_reasons)

    def test_degradation_detection_roi(self, db_session):
        """Model with negative ROI over 50+ bets should be flagged."""
        from bet_agent.tools.auditor_metrics import evaluate_model_performance

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        _add_prediction(db_session, match, selection="home")

        # 50 bets, all losses → bad ROI
        for i in range(55):
            b = _add_bet(
                db_session, match, selection="home",
                odds=Decimal("2.00"), stake=Decimal("10.00"),
                model_prob=Decimal("0.55"),
                status=BetStatus.LOST,
            )
            b.pnl_eur = Decimal("-10.00")
            b.resolved_at = datetime.now(timezone.utc)
        db_session.flush()

        reports = evaluate_model_performance(db_session)

        degraded = [r for r in reports if r.is_degraded]
        assert len(degraded) >= 1
        assert any("ROI" in reason for r in degraded for reason in r.degradation_reasons)

    def test_no_degradation_under_threshold(self, db_session):
        """Model with too few bets should not be flagged even if metrics are bad."""
        from bet_agent.tools.auditor_metrics import evaluate_model_performance

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        _add_prediction(db_session, match, selection="home")

        # Only 10 bets, all losses → bad but below window
        for i in range(10):
            b = _add_bet(
                db_session, match, selection="home",
                odds=Decimal("2.00"), stake=Decimal("10.00"),
                model_prob=Decimal("0.90"),
                status=BetStatus.LOST,
            )
            b.pnl_eur = Decimal("-10.00")
            b.resolved_at = datetime.now(timezone.utc)
        db_session.flush()

        reports = evaluate_model_performance(db_session)

        assert all(not r.is_degraded for r in reports)


class TestWriteMetrics:
    """Tests for writing metrics to DB."""

    def test_writes_new_metrics(self, db_session):
        from bet_agent.tools.auditor_metrics import ModelHealthReport, write_daily_metrics

        report = ModelHealthReport(
            model_name="analytical_football",
            ledger_type=LedgerType.REAL,
            sport="football",
            brier_score=0.20,
            roi_pct=5.5,
            total_bets=30,
            record_win=18,
            record_loss=12,
            total_staked=Decimal("300"),
            total_pnl=Decimal("16.50"),
            is_degraded=False,
        )

        count = write_daily_metrics(db_session, [report])
        assert count == 1

        # Verify in DB
        from sqlalchemy import select
        metrics = db_session.execute(select(ModelMetrics)).scalars().all()
        assert len(metrics) == 1
        assert metrics[0].model_name == "analytical_football"
        assert metrics[0].brier_score == Decimal("0.20")
        assert metrics[0].roi_pct == Decimal("5.50")

    def test_upsert_existing_metrics(self, db_session):
        """Writing metrics for same model+date+ledger updates instead of duplicating."""
        from bet_agent.tools.auditor_metrics import ModelHealthReport, write_daily_metrics

        report1 = ModelHealthReport(
            model_name="test_model", ledger_type=LedgerType.REAL,
            sport="football", brier_score=0.25, roi_pct=-2.0,
            total_bets=10, record_win=4, record_loss=6,
            total_staked=Decimal("100"), total_pnl=Decimal("-2"),
            is_degraded=False,
        )
        write_daily_metrics(db_session, [report1])

        report2 = ModelHealthReport(
            model_name="test_model", ledger_type=LedgerType.REAL,
            sport="football", brier_score=0.22, roi_pct=1.0,
            total_bets=15, record_win=8, record_loss=7,
            total_staked=Decimal("150"), total_pnl=Decimal("1.50"),
            is_degraded=False,
        )
        write_daily_metrics(db_session, [report2])

        from sqlalchemy import select
        metrics = db_session.execute(select(ModelMetrics)).scalars().all()
        assert len(metrics) == 1
        assert metrics[0].total_bets == 15


class TestDailyAudit:
    """Tests for the full audit pipeline."""

    def test_run_daily_audit(self, db_session):
        """Full pipeline: evaluate → write → detect degradation."""
        from bet_agent.tools.auditor_metrics import run_daily_audit

        match = _add_match(db_session, home_score=2, away_score=0, state=MatchState.FINISHED)
        _add_prediction(db_session, match, selection="home")

        b = _add_bet(
            db_session, match, selection="home",
            odds=Decimal("2.00"), stake=Decimal("10.00"),
            model_prob=Decimal("0.60"), status=BetStatus.WON,
        )
        b.pnl_eur = Decimal("10.00")
        b.resolved_at = datetime.now(timezone.utc)
        db_session.flush()

        audit = run_daily_audit(db_session)

        assert len(audit.reports) >= 1
        assert audit.metrics_written >= 1
        assert len(audit.degraded_models) == 0  # Only 1 bet, below threshold


# ── Integration: Full Settlement + Audit Pipeline ────────────────────


class TestFullPipeline:
    """Integration test: Results → Settle → Audit."""

    def test_full_settlement_to_audit(self, db_session):
        """End-to-end: fetch results, settle, audit."""
        from bet_agent.tools.auditor_metrics import run_daily_audit
        from bet_agent.tools.results_fetcher import (
            ManualResultsBackend,
            MatchResult,
            fetch_and_update_results,
        )
        from bet_agent.tools.settlement_engine import settle_finished_matches

        # Setup: match from yesterday with pending bet
        match = _add_match(db_session)
        bet = _add_bet(db_session, match, selection="home",
                       odds=Decimal("2.00"), stake=Decimal("10.00"),
                       model_prob=Decimal("0.60"))
        _add_prediction(db_session, match, selection="home")
        _add_bankroll(db_session, LedgerType.REAL, Decimal("990.00"))

        # Step 1: Fetch results
        backend = ManualResultsBackend()
        backend.add_result(
            "FC Bayern vs BVB Dortmund",
            MatchResult("FC Bayern", "BVB Dortmund", 2, 1, is_finished=True),
        )
        fetch_result = fetch_and_update_results(db_session, backend)
        assert fetch_result.matches_updated == 1
        assert match.match_state == MatchState.FINISHED

        # Step 2: Settle
        settlement = settle_finished_matches(db_session)
        assert settlement.total_settled == 1
        assert settlement.won == 1
        assert bet.status == BetStatus.WON
        assert bet.pnl_eur == Decimal("10.00")

        # Step 3: Audit
        audit = run_daily_audit(db_session)
        assert len(audit.reports) >= 1
        assert audit.metrics_written >= 1
