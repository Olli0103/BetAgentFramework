"""Tests for pipeline_bridge + edge_plausible readiness check."""

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from bet_agent.db.models import (
    BankrollLedger,
    Base,
    BetStatus,
    LedgerType,
    MarketType,
    Match,
    MatchState,
    PlacedBet,
    Prediction,
    PredictionStatus,
    Sport,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _add_match(session, home="FC Bayern", away="BVB Dortmund"):
    m = Match(
        sport=Sport.FOOTBALL,
        league="Bundesliga",
        home_team=home,
        away_team=away,
        scheduled_at=datetime.combine(date.today(), datetime.min.time(), tzinfo=timezone.utc),
        match_state=MatchState.NOT_STARTED,
    )
    session.add(m)
    session.flush()
    return m


def _add_prediction(
    session, match, selection="home", ev="0.08",
    model_prob="0.55", implied_prob="0.4762",
    status=PredictionStatus.APPROVED,
    best_odds="2.10", best_sportsbook="bet365",
):
    p = Prediction(
        match_id=match.id,
        model_name="analytical_football",
        market_type=MarketType.MATCH_WINNER,
        selection=selection,
        model_prob=Decimal(model_prob),
        implied_prob=Decimal(implied_prob),
        prob_edge=Decimal(str(float(model_prob) - float(implied_prob))),
        ev=Decimal(ev),
        model_source="analytical",
        status=status,
        best_odds=Decimal(best_odds) if best_odds else None,
        best_sportsbook=best_sportsbook,
    )
    session.add(p)
    session.flush()
    return p


def _add_bankroll(session, ledger=LedgerType.REAL, balance=Decimal("1000.00")):
    b = BankrollLedger(ledger_type=ledger, balance=balance)
    session.add(b)
    session.flush()
    return b


# ── Pipeline Bridge Tests ───────────────────────────────────────────


class TestPipelineBridge:
    def test_creates_pending_bet(self, db_session):
        """Approved prediction with best_odds produces a PENDING PlacedBet."""
        from bet_agent.tools.pipeline_bridge import ensure_pending_bets_from_approved

        _add_bankroll(db_session)
        match = _add_match(db_session)
        _add_prediction(db_session, match)

        result = ensure_pending_bets_from_approved(db_session)

        assert result["created"] == 1
        bets = list(db_session.execute(select(PlacedBet)).scalars().all())
        assert len(bets) == 1
        assert bets[0].status == BetStatus.PENDING
        assert bets[0].selection == "home"

    def test_idempotent(self, db_session):
        """Re-running bridge doesn't create duplicates."""
        from bet_agent.tools.pipeline_bridge import ensure_pending_bets_from_approved

        _add_bankroll(db_session)
        match = _add_match(db_session)
        _add_prediction(db_session, match)

        r1 = ensure_pending_bets_from_approved(db_session)
        r2 = ensure_pending_bets_from_approved(db_session)

        assert r1["created"] == 1
        assert r2["created"] == 0
        assert r2["skipped_existing"] == 1

        bets = list(db_session.execute(select(PlacedBet)).scalars().all())
        assert len(bets) == 1

    def test_skips_no_odds(self, db_session):
        """Prediction without best_odds is skipped."""
        from bet_agent.tools.pipeline_bridge import ensure_pending_bets_from_approved

        _add_bankroll(db_session)
        match = _add_match(db_session)
        _add_prediction(db_session, match, best_odds=None, best_sportsbook=None)

        result = ensure_pending_bets_from_approved(db_session)
        assert result["created"] == 0
        assert result["skipped_no_odds"] == 1

    def test_skips_pending_predictions(self, db_session):
        """Only APPROVED predictions are processed."""
        from bet_agent.tools.pipeline_bridge import ensure_pending_bets_from_approved

        _add_bankroll(db_session)
        match = _add_match(db_session)
        _add_prediction(db_session, match, status=PredictionStatus.PENDING)

        result = ensure_pending_bets_from_approved(db_session)
        assert result["total"] == 0
        assert result["created"] == 0

    def test_reroutes_existing_pending_on_ledger_change(self, db_session):
        """Re-running bridge updates ledger_type on existing PENDING bets."""
        from bet_agent.tools.pipeline_bridge import ensure_pending_bets_from_approved

        _add_bankroll(db_session, ledger=LedgerType.REAL)
        _add_bankroll(db_session, ledger=LedgerType.PAPER, balance=Decimal("10000.00"))
        match = _add_match(db_session)

        # First: create with normal edge → REAL
        pred = _add_prediction(db_session, match)
        r1 = ensure_pending_bets_from_approved(db_session)
        assert r1["created"] == 1

        bet = db_session.execute(select(PlacedBet)).scalar_one()
        assert bet.ledger_type == LedgerType.REAL

        # Now make edge implausible (simulate model update)
        pred.model_prob = Decimal("0.40")
        pred.implied_prob = Decimal("0.10")
        pred.prob_edge = Decimal("0.30")
        pred.ev = Decimal("3.00")
        pred.best_odds = Decimal("10.00")
        db_session.flush()

        r2 = ensure_pending_bets_from_approved(db_session)
        assert r2["updated_existing"] == 1
        assert r2["created"] == 0

        db_session.refresh(bet)
        assert bet.ledger_type == LedgerType.PAPER

    def test_no_reroute_on_placed_bet(self, db_session):
        """Already PLACED bets are not re-routed (only PENDING can change)."""
        from bet_agent.tools.pipeline_bridge import ensure_pending_bets_from_approved

        _add_bankroll(db_session, ledger=LedgerType.REAL)
        _add_bankroll(db_session, ledger=LedgerType.PAPER, balance=Decimal("10000.00"))
        match = _add_match(db_session)

        pred = _add_prediction(db_session, match)
        r1 = ensure_pending_bets_from_approved(db_session)
        assert r1["created"] == 1

        # Simulate human confirmation
        bet = db_session.execute(select(PlacedBet)).scalar_one()
        bet.status = BetStatus.PLACED
        db_session.flush()

        # Make edge implausible
        pred.model_prob = Decimal("0.40")
        pred.implied_prob = Decimal("0.10")
        pred.prob_edge = Decimal("0.30")
        pred.ev = Decimal("3.00")
        pred.best_odds = Decimal("10.00")
        db_session.flush()

        r2 = ensure_pending_bets_from_approved(db_session)
        assert r2["skipped_existing"] == 1
        assert r2["updated_existing"] == 0

        db_session.refresh(bet)
        assert bet.ledger_type == LedgerType.REAL  # unchanged

    def test_routes_implausible_edge_to_paper(self, db_session):
        """Extreme edge gets routed to PAPER via readiness gate."""
        from bet_agent.tools.pipeline_bridge import ensure_pending_bets_from_approved

        _add_bankroll(db_session, ledger=LedgerType.REAL)
        _add_bankroll(db_session, ledger=LedgerType.PAPER, balance=Decimal("10000.00"))
        match = _add_match(db_session)
        # model_prob=0.40, implied_prob=0.10 → edge=30pp > 15pp threshold
        _add_prediction(
            db_session, match,
            model_prob="0.40", implied_prob="0.10",
            ev="3.00", best_odds="10.00",
        )

        result = ensure_pending_bets_from_approved(db_session)
        assert result["created"] == 1

        bet = db_session.execute(select(PlacedBet)).scalar_one()
        assert bet.ledger_type == LedgerType.PAPER


# ── Edge Plausibility Readiness Check Tests ─────────────────────────


class TestEdgePlausible:
    def test_normal_edge_passes(self, db_session):
        """Normal edge (~7pp) passes edge_plausible check."""
        from bet_agent.tools.notifier import check_bet_readiness

        match = _add_match(db_session)
        pred = _add_prediction(
            db_session, match,
            model_prob="0.55", implied_prob="0.4762",
        )

        result = check_bet_readiness(pred, match, stake_eur=10.0)
        assert result.checks["edge_plausible"] is True
        assert result.is_ready is True

    def test_extreme_edge_fails(self, db_session):
        """Extreme edge (~30pp) fails edge_plausible check."""
        from bet_agent.tools.notifier import check_bet_readiness

        match = _add_match(db_session)
        pred = _add_prediction(
            db_session, match,
            model_prob="0.40", implied_prob="0.10",
            ev="3.00", best_odds="10.00",
        )

        result = check_bet_readiness(pred, match, stake_eur=10.0)
        assert result.checks["edge_plausible"] is False
        assert result.is_ready is False
        assert "edge_plausible" in result.reason

    def test_boundary_at_threshold(self, db_session):
        """Edge exactly at threshold (15pp) should pass."""
        from bet_agent.tools.notifier import check_bet_readiness

        match = _add_match(db_session)
        # model_prob=0.55, implied_prob=0.40 → edge=15pp exactly
        pred = _add_prediction(
            db_session, match,
            model_prob="0.55", implied_prob="0.40",
            ev="0.10", best_odds="2.50",
        )

        result = check_bet_readiness(pred, match, stake_eur=10.0)
        assert result.checks["edge_plausible"] is True

    def test_just_over_threshold_fails(self, db_session):
        """Edge just over threshold (15.1pp) should fail."""
        from bet_agent.tools.notifier import check_bet_readiness

        match = _add_match(db_session)
        # model_prob=0.551, implied_prob=0.40 → edge=15.1pp
        pred = _add_prediction(
            db_session, match,
            model_prob="0.551", implied_prob="0.40",
            ev="0.10", best_odds="2.50",
        )

        result = check_bet_readiness(pred, match, stake_eur=10.0)
        assert result.checks["edge_plausible"] is False

    def test_ticket_downgraded_on_implausible_edge(self, db_session):
        """build_ticket downgrades REAL to PAPER when edge is implausible."""
        from bet_agent.tools.notifier import build_ticket

        match = _add_match(db_session)
        pred = _add_prediction(
            db_session, match,
            model_prob="0.40", implied_prob="0.10",
            ev="3.00", best_odds="10.00",
        )

        ticket = build_ticket(pred, match, stake_eur=10.0, ledger_type=LedgerType.REAL)
        assert ticket.ledger_type == "PAPER"
        assert "READINESS FAIL" in ticket.veto_status
        assert "edge_plausible" in ticket.veto_status
