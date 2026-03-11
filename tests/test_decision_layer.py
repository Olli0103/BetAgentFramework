"""Tests for Phase 3: Decision & Execution Layer.

Covers veto_engine, line_shopper, sizing_engine, and notifier.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
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
    OddsMarket,
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


def _add_match(session, sport=Sport.FOOTBALL, home="FC Bayern", away="BVB Dortmund"):
    m = Match(
        sport=sport,
        league="Bundesliga",
        home_team=home,
        away_team=away,
        scheduled_at=datetime.combine(date.today(), datetime.min.time(), tzinfo=timezone.utc),
        match_state=MatchState.NOT_STARTED,
    )
    session.add(m)
    session.flush()
    return m


def _add_prediction(session, match, selection="home", ev="0.08", status=PredictionStatus.PENDING):
    p = Prediction(
        match_id=match.id,
        model_name="analytical_football",
        market_type=MarketType.MATCH_WINNER,
        selection=selection,
        model_prob=Decimal("0.55"),
        implied_prob=Decimal("0.4762"),
        prob_edge=Decimal("0.0738"),
        ev=Decimal(ev),
        model_source="analytical",
        status=status,
    )
    session.add(p)
    session.flush()
    return p


def _add_odds(session, match, sportsbook="bet365", selection="home", odds=Decimal("2.10")):
    o = OddsMarket(
        match_id=match.id,
        sportsbook=sportsbook,
        market_type=MarketType.MATCH_WINNER,
        selection=selection,
        odds_decimal=odds,
        is_live=False,
    )
    session.add(o)
    session.flush()
    return o


def _add_bankroll(session, ledger_type=LedgerType.REAL, balance=Decimal("1000.00")):
    b = BankrollLedger(ledger_type=ledger_type, balance=balance)
    session.add(b)
    session.flush()
    return b


# ── Veto Engine Tests ────────────────────────────────────────────────


class TestVetoEngine:
    """Tests for the Devil's Advocate veto engine."""

    def test_approve_when_no_news(self, db_session):
        """No risk factors found → APPROVE."""
        from bet_agent.tools.veto_engine import DefaultNewsSearch, veto_check

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)

        result = veto_check(db_session, pred, search_backend=DefaultNewsSearch())

        assert result.decision == "APPROVE"
        assert "No risk factors" in result.reason

    def test_veto_when_multiple_risks(self, db_session):
        """Multiple risk keywords in news → VETO."""
        from bet_agent.tools.veto_engine import veto_check

        class MockSearch:
            def search(self, query, max_results=5):
                return [
                    {"title": "Star player injured in training",
                     "snippet": "Key striker suffered a hamstring injury and is doubtful for tomorrow"},
                    {"title": "Manager sacked after poor run",
                     "snippet": "Club confirms manager sacking, interim coach appointed"},
                ]

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)

        result = veto_check(db_session, pred, search_backend=MockSearch())

        assert result.decision == "VETO"
        assert len(result.risk_factors) >= 2

    def test_approve_with_minor_risk(self, db_session):
        """Single risk factor below threshold → APPROVE with note."""
        from bet_agent.tools.veto_engine import veto_check

        class MockSearch:
            def search(self, query, max_results=5):
                return [
                    {"title": "Slight injury concern",
                     "snippet": "Backup goalkeeper sustained minor injury in training"},
                ]

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)

        result = veto_check(db_session, pred, search_backend=MockSearch(), risk_threshold=2)

        assert result.decision == "APPROVE"
        assert len(result.risk_factors) >= 1
        assert "Minor risk" in result.reason or "below threshold" in result.reason

    def test_apply_veto_updates_db(self, db_session):
        """apply_veto_result should update prediction status and reason."""
        from bet_agent.tools.veto_engine import VetoResult, apply_veto_result

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)

        result = VetoResult(
            prediction_id=pred.id,
            decision="VETO",
            reason="Key player injured",
            risk_factors=["injured", "doubtful"],
        )
        apply_veto_result(db_session, pred, result)

        assert pred.status == PredictionStatus.VETOED
        assert pred.veto_reason == "Key player injured"

    def test_apply_approve_updates_db(self, db_session):
        """apply_veto_result for APPROVE should set status and clear reason."""
        from bet_agent.tools.veto_engine import VetoResult, apply_veto_result

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)

        result = VetoResult(
            prediction_id=pred.id,
            decision="APPROVE",
            reason="No risks found",
        )
        apply_veto_result(db_session, pred, result)

        assert pred.status == PredictionStatus.APPROVED
        assert pred.veto_reason is None

    def test_batch_veto_checks(self, db_session):
        """run_veto_checks processes multiple predictions."""
        from bet_agent.tools.veto_engine import DefaultNewsSearch, run_veto_checks

        match = _add_match(db_session)
        p1 = _add_prediction(db_session, match, selection="home")
        p2 = _add_prediction(db_session, match, selection="away")
        # Fix unique constraint — different selection for p2
        p2.model_name = "analytical_football_v2"
        db_session.flush()

        results = run_veto_checks(
            db_session,
            predictions=[p1, p2],
            search_backend=DefaultNewsSearch(),
        )

        assert len(results) == 2
        assert all(r.decision == "APPROVE" for r in results)
        assert p1.status == PredictionStatus.APPROVED
        assert p2.status == PredictionStatus.APPROVED

    def test_search_backend_error_handled(self, db_session):
        """Search errors should not crash the veto check."""
        from bet_agent.tools.veto_engine import veto_check

        class FailingSearch:
            def search(self, query, max_results=5):
                raise ConnectionError("Network error")

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)

        result = veto_check(db_session, pred, search_backend=FailingSearch())

        # Should still approve (no evidence of risk)
        assert result.decision == "APPROVE"

    def test_risk_keyword_extraction(self):
        """Test _extract_risk_factors finds multiple patterns."""
        from bet_agent.tools.veto_engine import _extract_risk_factors

        snippets = [
            "Star player suffered a hamstring injury in training",
            "He is listed as doubtful for the match",
            "Weather warning issued for stadium area",
        ]

        factors = _extract_risk_factors(snippets)
        assert len(factors) >= 3
        # Should find injury, hamstring, doubtful, weather warning
        factor_text = " ".join(f.lower() for f in factors)
        assert "hamstring" in factor_text or "injur" in factor_text


# ── Line Shopper Tests ───────────────────────────────────────────────


class TestLineShopper:
    """Tests for the odds optimizer."""

    def test_finds_best_odds(self, db_session):
        """shop_line should return the highest odds across sportsbooks."""
        from bet_agent.tools.line_shopper import shop_line

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)

        # Add odds from multiple sportsbooks
        _add_odds(db_session, match, "tipico", "home", Decimal("2.00"))
        _add_odds(db_session, match, "bet365", "home", Decimal("2.15"))
        _add_odds(db_session, match, "bwin", "home", Decimal("2.05"))

        result = shop_line(db_session, pred)

        assert result is not None
        assert result.best_odds == Decimal("2.15")
        assert result.best_sportsbook == "bet365"
        assert len(result.all_odds) == 3

    def test_no_odds_returns_none(self, db_session):
        """shop_line returns None when no odds are available."""
        from bet_agent.tools.line_shopper import shop_line

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)

        result = shop_line(db_session, pred)
        assert result is None

    def test_apply_shopped_line_updates_prediction(self, db_session):
        """apply_shopped_line should update prediction with best odds and recalculate EV."""
        from bet_agent.tools.line_shopper import ShoppedLine, apply_shopped_line

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)
        original_ev = pred.ev

        shopped = ShoppedLine(
            prediction_id=pred.id,
            market_type=MarketType.MATCH_WINNER,
            selection="home",
            best_odds=Decimal("2.20"),
            best_sportsbook="bet365",
            original_implied_prob=Decimal("0.4762"),
            new_implied_prob=Decimal("0.4545"),
            odds_improvement=Decimal("0.10"),
            all_odds=[],
        )
        apply_shopped_line(db_session, pred, shopped)

        assert pred.best_odds == Decimal("2.20")
        assert pred.best_sportsbook == "bet365"
        # EV should have improved with better odds
        assert pred.ev > original_ev

    def test_selection_matching_case_insensitive(self, db_session):
        """Odds matching should be case-insensitive."""
        from bet_agent.tools.line_shopper import shop_line

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, selection="home", status=PredictionStatus.APPROVED)

        # Add odds with different casing
        _add_odds(db_session, match, "bet365", "Home", Decimal("2.10"))

        result = shop_line(db_session, pred)
        assert result is not None
        assert result.best_odds == Decimal("2.10")

    def test_shop_all_approved(self, db_session):
        """shop_all_approved processes multiple predictions."""
        from bet_agent.tools.line_shopper import shop_all_approved

        match = _add_match(db_session)
        p1 = _add_prediction(db_session, match, selection="home", status=PredictionStatus.APPROVED)
        p2 = _add_prediction(db_session, match, selection="away", status=PredictionStatus.APPROVED)
        p2.model_name = "analytical_football_v2"
        db_session.flush()

        _add_odds(db_session, match, "bet365", "home", Decimal("2.10"))
        _add_odds(db_session, match, "bet365", "away", Decimal("3.50"))

        results = shop_all_approved(db_session, predictions=[p1, p2])

        assert len(results) == 2
        assert p1.best_odds == Decimal("2.10")
        assert p2.best_odds == Decimal("3.50")

    def test_ignores_live_odds(self, db_session):
        """Live odds should not be considered by the line shopper."""
        from bet_agent.tools.line_shopper import shop_line

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)

        # Only add live odds
        o = OddsMarket(
            match_id=match.id,
            sportsbook="bet365",
            market_type=MarketType.MATCH_WINNER,
            selection="home",
            odds_decimal=Decimal("2.50"),
            is_live=True,
        )
        db_session.add(o)
        db_session.flush()

        result = shop_line(db_session, pred)
        assert result is None


# ── Sizing Engine Tests ──────────────────────────────────────────────


class TestSizingEngine:
    """Tests for the Quarter-Kelly sizing engine."""

    def test_basic_sizing(self, db_session):
        """size_bet calculates a positive stake for a +EV bet."""
        from bet_agent.tools.sizing_engine import size_bet

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)
        pred.best_odds = Decimal("2.10")
        pred.model_prob = Decimal("0.55")
        db_session.flush()

        _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        sized = size_bet(db_session, pred)

        assert sized.stake_eur > 0
        assert sized.stake_eur <= 50.0  # Max 5% of 1000
        assert sized.ledger_type == LedgerType.REAL
        assert sized.reason is None

    def test_zero_bankroll_returns_no_stake(self, db_session):
        """No bankroll → no stake."""
        from bet_agent.tools.sizing_engine import size_bet

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)
        pred.best_odds = Decimal("2.10")
        db_session.flush()

        # No bankroll added

        sized = size_bet(db_session, pred)
        assert sized.stake_eur == 0.0

    def test_negative_ev_returns_zero(self, db_session):
        """Negative EV bet should return 0 stake."""
        from bet_agent.tools.sizing_engine import size_bet

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, ev="-0.05", status=PredictionStatus.APPROVED)
        pred.best_odds = Decimal("1.80")
        pred.model_prob = Decimal("0.50")  # 50% prob at 1.80 odds = negative EV
        db_session.flush()

        _add_bankroll(db_session)

        sized = size_bet(db_session, pred)
        assert sized.stake_eur == 0.0
        assert sized.reason == "negative_ev"

    def test_parlay_hard_cap(self, db_session):
        """Parlay stakes should be capped at 1.00 EUR."""
        from bet_agent.tools.sizing_engine import size_bet

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)
        pred.best_odds = Decimal("5.00")
        pred.model_prob = Decimal("0.30")
        db_session.flush()

        _add_bankroll(db_session, balance=Decimal("10000.00"))

        sized = size_bet(db_session, pred, is_parlay=True)
        assert sized.stake_eur <= 1.00

    def test_max_5_pct_cap(self, db_session):
        """Single bet should never exceed 5% of bankroll."""
        from bet_agent.tools.sizing_engine import size_bet

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)
        pred.best_odds = Decimal("10.00")
        pred.model_prob = Decimal("0.80")  # Very high edge to force large Kelly
        db_session.flush()

        _add_bankroll(db_session, balance=Decimal("1000.00"))

        sized = size_bet(db_session, pred)
        assert sized.stake_eur <= 50.0  # 5% of 1000

    def test_get_bankroll(self, db_session):
        """get_bankroll fetches current balance."""
        from bet_agent.tools.sizing_engine import get_bankroll

        _add_bankroll(db_session, LedgerType.REAL, Decimal("500.00"))
        _add_bankroll(db_session, LedgerType.PAPER, Decimal("10000.00"))

        assert get_bankroll(db_session, LedgerType.REAL) == Decimal("500.00")
        assert get_bankroll(db_session, LedgerType.PAPER) == Decimal("10000.00")

    def test_get_bankroll_missing_returns_zero(self, db_session):
        """Missing ledger returns 0."""
        from bet_agent.tools.sizing_engine import get_bankroll

        assert get_bankroll(db_session, LedgerType.REAL) == Decimal("0.00")

    def test_assign_ledger_proven_model(self, db_session):
        """Analytical models go to REAL ledger."""
        from bet_agent.tools.sizing_engine import assign_ledger_type

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)
        pred.model_source = "analytical"
        assert assign_ledger_type(pred) == LedgerType.REAL

    def test_assign_ledger_unknown_model(self, db_session):
        """Unknown model sources go to PAPER ledger."""
        from bet_agent.tools.sizing_engine import assign_ledger_type

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)
        pred.model_source = "experimental_v3"
        assert assign_ledger_type(pred) == LedgerType.PAPER

    def test_check_risk_limits(self, db_session):
        """check_risk_limits reports correct daily/weekly status."""
        from bet_agent.tools.sizing_engine import check_risk_limits

        _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        risk = check_risk_limits(db_session, LedgerType.REAL)

        assert risk.bankroll == Decimal("1000.00")
        assert risk.daily_loss_eur == Decimal("0")
        assert risk.daily_limit_hit is False
        assert risk.weekly_limit_hit is False

    def test_no_odds_returns_reason(self, db_session):
        """No odds on prediction → reason 'no_odds_available'."""
        from bet_agent.tools.sizing_engine import size_bet

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)
        pred.best_odds = None
        pred.implied_prob = Decimal("0")  # Force no odds
        db_session.flush()

        _add_bankroll(db_session)

        sized = size_bet(db_session, pred)
        assert sized.reason == "no_odds_available"

    def test_size_all_approved(self, db_session):
        """size_all_approved processes a batch of predictions."""
        from bet_agent.tools.sizing_engine import size_all_approved

        match = _add_match(db_session)
        p1 = _add_prediction(db_session, match, selection="home", status=PredictionStatus.APPROVED)
        p1.best_odds = Decimal("2.10")
        p2 = _add_prediction(db_session, match, selection="away", status=PredictionStatus.APPROVED)
        p2.model_name = "analytical_football_v2"
        p2.best_odds = Decimal("3.50")
        db_session.flush()

        _add_bankroll(db_session)

        results = size_all_approved(db_session, predictions=[p1, p2])
        assert len(results) == 2


# ── Notifier Tests ───────────────────────────────────────────────────


class TestNotifier:
    """Tests for the notification system."""

    def test_build_ticket(self, db_session):
        """build_ticket creates a proper BetTicket."""
        from bet_agent.tools.notifier import build_ticket

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, status=PredictionStatus.APPROVED)
        pred.best_odds = Decimal("2.15")
        pred.best_sportsbook = "bet365"
        pred.prob_edge = Decimal("0.084")
        db_session.flush()

        ticket = build_ticket(pred, match, stake_eur=42.50, ledger_type=LedgerType.REAL)

        assert ticket.match_description == "FC Bayern vs BVB Dortmund"
        assert ticket.best_odds == 2.15
        assert ticket.best_sportsbook == "bet365"
        assert ticket.stake_eur == 42.50
        assert ticket.ledger_type == "REAL"
        assert ticket.model_edge_pct == 8.4

    def test_format_ticket_message(self, db_session):
        """format_ticket_message includes all required fields."""
        from bet_agent.tools.notifier import BetTicket, format_ticket_message

        ticket = BetTicket(
            prediction_id=uuid.uuid4(),
            sport="football",
            match_description="Zverev vs Alcaraz",
            league="ATP Finals",
            market="Match Winner - Alcaraz",
            selection="away",
            stake_eur=42.50,
            best_odds=1.95,
            best_sportsbook="Bet365",
            model_edge_pct=8.4,
            ev=0.12,
            veto_status="PASSED",
            ledger_type="REAL",
            model_source="xgboost",
        )

        msg = format_ticket_message(ticket)

        assert "Zverev vs Alcaraz" in msg
        assert "42.50EUR" in msg
        assert "1.95" in msg
        assert "Bet365" in msg
        assert "+8.4%" in msg
        assert "PASSED" in msg

    def test_format_daily_summary_empty(self):
        """Empty tickets list produces a 'no bets' message."""
        from bet_agent.tools.notifier import format_daily_summary

        msg = format_daily_summary([])
        assert "No +EV bets found" in msg

    def test_format_daily_summary_with_tickets(self):
        """Summary includes count and total stake."""
        from bet_agent.tools.notifier import BetTicket, format_daily_summary

        tickets = [
            BetTicket(
                prediction_id=uuid.uuid4(),
                sport="football", match_description="A vs B",
                league="Test", market="MW", selection="home",
                stake_eur=25.0, best_odds=2.0, best_sportsbook="bet365",
                model_edge_pct=5.0, ev=0.08, veto_status="PASSED",
                ledger_type="REAL", model_source="analytical",
            ),
            BetTicket(
                prediction_id=uuid.uuid4(),
                sport="tennis", match_description="C vs D",
                league="Test2", market="MW", selection="away",
                stake_eur=15.0, best_odds=1.80, best_sportsbook="tipico",
                model_edge_pct=3.0, ev=0.05, veto_status="PASSED",
                ledger_type="PAPER", model_source="analytical",
            ),
        ]

        msg = format_daily_summary(tickets)
        assert "Bets: 2" in msg
        assert "1 real" in msg
        assert "1 paper" in msg
        assert "40.00EUR" in msg

    def test_console_notifier(self):
        """ConsoleNotifier always succeeds."""
        from bet_agent.tools.notifier import ConsoleNotifier

        notifier = ConsoleNotifier()
        assert notifier.send("test message") is True

    def test_telegram_notifier_no_config(self):
        """TelegramNotifier without config returns False."""
        from bet_agent.tools.notifier import TelegramNotifier

        notifier = TelegramNotifier(bot_token="", chat_id="")
        assert notifier.send("test") is False

    def test_push_alert_uses_all_notifiers(self, db_session):
        """push_alert sends through all provided notifiers."""
        from bet_agent.tools.notifier import BetTicket, ConsoleNotifier, push_alert

        ticket = BetTicket(
            prediction_id=uuid.uuid4(),
            sport="football", match_description="A vs B",
            league="Test", market="MW", selection="home",
            stake_eur=25.0, best_odds=2.0, best_sportsbook="bet365",
            model_edge_pct=5.0, ev=0.08, veto_status="PASSED",
            ledger_type="REAL", model_source="analytical",
        )

        result = push_alert(ticket, notifiers=[ConsoleNotifier()])
        assert result is True

    def test_push_daily_summary(self):
        """push_daily_summary works with empty list."""
        from bet_agent.tools.notifier import ConsoleNotifier, push_daily_summary

        result = push_daily_summary([], notifiers=[ConsoleNotifier()])
        assert result is True


# ── Integration: Full Pipeline Test ──────────────────────────────────


class TestFullPipeline:
    """Integration test: PENDING → Veto → Line Shop → Size → Alert."""

    def test_full_pipeline(self, db_session):
        """End-to-end: prediction goes through veto, shopping, sizing, alert."""
        from bet_agent.tools.line_shopper import shop_all_approved
        from bet_agent.tools.notifier import ConsoleNotifier, build_ticket, push_alert
        from bet_agent.tools.sizing_engine import size_bet
        from bet_agent.tools.veto_engine import DefaultNewsSearch, run_veto_checks

        # Setup
        match = _add_match(db_session)
        pred = _add_prediction(db_session, match, ev="0.08")
        _add_odds(db_session, match, "bet365", "home", Decimal("2.15"))
        _add_odds(db_session, match, "tipico", "home", Decimal("2.05"))
        _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        # Step 1: Veto check
        veto_results = run_veto_checks(
            db_session, [pred], search_backend=DefaultNewsSearch()
        )
        assert veto_results[0].decision == "APPROVE"
        assert pred.status == PredictionStatus.APPROVED

        # Step 2: Line shopping
        shopped = shop_all_approved(db_session, [pred])
        assert len(shopped) == 1
        assert pred.best_odds == Decimal("2.15")
        assert pred.best_sportsbook == "bet365"

        # Step 3: Sizing
        sized = size_bet(db_session, pred)
        assert sized.stake_eur > 0
        assert sized.ledger_type == LedgerType.REAL

        # Step 4: Alert
        ticket = build_ticket(pred, match, sized.stake_eur, sized.ledger_type)
        assert ticket.best_odds == 2.15
        assert push_alert(ticket, [ConsoleNotifier()]) is True

    def test_vetoed_prediction_stops_pipeline(self, db_session):
        """A VETOED prediction should not proceed to sizing."""
        from bet_agent.tools.veto_engine import VetoResult, apply_veto_result

        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)

        veto = VetoResult(
            prediction_id=pred.id,
            decision="VETO",
            reason="Key player injured",
            risk_factors=["injured", "hamstring"],
        )
        apply_veto_result(db_session, pred, veto)

        assert pred.status == PredictionStatus.VETOED
        assert pred.veto_reason == "Key player injured"
        # Pipeline stops — sizing should not be called for vetoed predictions


# ── Prediction Model Columns Test ────────────────────────────────────


class TestPredictionNewColumns:
    """Test the new veto_reason, best_odds, best_sportsbook columns."""

    def test_veto_reason_stored(self, db_session):
        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)
        pred.veto_reason = "Goalkeeper injured"
        db_session.flush()

        loaded = db_session.get(Prediction, pred.id)
        assert loaded.veto_reason == "Goalkeeper injured"

    def test_best_odds_stored(self, db_session):
        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)
        pred.best_odds = Decimal("2.25")
        pred.best_sportsbook = "tipico"
        db_session.flush()

        loaded = db_session.get(Prediction, pred.id)
        assert loaded.best_odds == Decimal("2.25")
        assert loaded.best_sportsbook == "tipico"

    def test_null_by_default(self, db_session):
        match = _add_match(db_session)
        pred = _add_prediction(db_session, match)

        assert pred.veto_reason is None
        assert pred.best_odds is None
        assert pred.best_sportsbook is None
