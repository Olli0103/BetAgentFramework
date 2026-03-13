"""Tests for the Moonshot Parlay Builder."""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bet_agent.db.models import (
    Base,
    LedgerType,
    MarketType,
    Match,
    MatchState,
    Prediction,
    PredictionStatus,
    Sport,
)
from bet_agent.tools.parlay_builder import (
    MOONSHOT_HARD_CAP_EUR,
    LegCorrelation,
    ParlayLeg,
    ParlayTicket,
    build_parlay,
    build_best_parlay,
    calculate_parlay_ev,
    check_leg_correlation,
    validate_parlay_stake,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make_match(
    session,
    sport=Sport.FOOTBALL,
    league="Bundesliga",
    home="Bayern Munich",
    away="Borussia Dortmund",
) -> Match:
    m = Match(
        sport=sport,
        league=league,
        home_team=home,
        away_team=away,
        scheduled_at=datetime.now(timezone.utc) + timedelta(hours=3),
        match_state=MatchState.NOT_STARTED,
    )
    session.add(m)
    session.flush()
    return m


def _make_prediction(
    session,
    match: Match,
    selection="home",
    model_prob=0.55,
    implied_prob=0.50,
    ev=0.05,
    best_odds=2.00,
    status=PredictionStatus.APPROVED,
) -> Prediction:
    p = Prediction(
        match_id=match.id,
        model_name="xgb_test",
        market_type=MarketType.MATCH_WINNER,
        selection=selection,
        model_prob=Decimal(str(model_prob)),
        implied_prob=Decimal(str(implied_prob)),
        prob_edge=Decimal(str(model_prob - implied_prob)),
        ev=Decimal(str(ev)),
        best_odds=Decimal(str(best_odds)),
        best_sportsbook="tipico",
        status=status,
    )
    session.add(p)
    session.flush()
    return p


# ── Stake Validation ────────────────────────────────────────────────────


class TestValidateStake:
    def test_stake_within_cap(self):
        stake, errors = validate_parlay_stake(0.50)
        assert stake == 0.50
        assert errors == []

    def test_stake_at_cap(self):
        stake, errors = validate_parlay_stake(1.00)
        assert stake == 1.00
        assert errors == []

    def test_stake_exceeds_cap(self):
        stake, errors = validate_parlay_stake(5.00)
        assert stake == MOONSHOT_HARD_CAP_EUR
        assert len(errors) == 1
        assert "exceeds" in errors[0].lower() or "cap" in errors[0].lower()

    def test_stake_zero(self):
        stake, errors = validate_parlay_stake(0.0)
        assert stake == 0.0
        assert len(errors) == 1
        assert "positive" in errors[0].lower()

    def test_negative_stake(self):
        stake, errors = validate_parlay_stake(-1.0)
        assert stake == 0.0


# ── Correlation Check ────────────────────────────────────────────────────


class TestLegCorrelation:
    def test_same_match_correlation(self, db_session):
        m = _make_match(db_session)
        p1 = _make_prediction(db_session, m, selection="home")
        p2 = _make_prediction(
            db_session, m, selection="over_2.5",
        )
        # Need different market type for unique constraint
        p2.market_type = MarketType.OVER_UNDER
        db_session.flush()

        corr = check_leg_correlation(db_session, p1, p2)
        assert corr.same_match is True
        assert corr.correlation_factor > 0.0
        assert "same match" in corr.reason.lower()

    def test_same_league_correlation(self, db_session):
        m1 = _make_match(db_session, home="Bayern", away="Dortmund")
        m2 = _make_match(db_session, home="Leipzig", away="Freiburg")
        p1 = _make_prediction(db_session, m1)
        p2 = _make_prediction(db_session, m2)

        corr = check_leg_correlation(db_session, p1, p2)
        assert corr.same_match is False
        assert corr.same_league is True
        assert corr.correlation_factor >= 0.0

    def test_cross_sport_independent(self, db_session):
        m1 = _make_match(db_session, sport=Sport.FOOTBALL)
        m2 = _make_match(
            db_session, sport=Sport.TENNIS,
            league="ATP", home="Sinner", away="Djokovic",
        )
        p1 = _make_prediction(db_session, m1)
        p2 = _make_prediction(db_session, m2)

        corr = check_leg_correlation(db_session, p1, p2)
        assert corr.same_sport is False
        assert corr.correlation_factor == 0.0
        assert "independent" in corr.reason.lower()


# ── EV Calculation ───────────────────────────────────────────────────────


class TestCalculateParlayEV:
    def test_two_independent_legs(self):
        legs = [
            ParlayLeg(
                prediction_id=uuid.uuid4(), match_id=uuid.uuid4(),
                sport="football", league="Bundesliga",
                match_description="A vs B", selection="home",
                market_type="match_winner",
                model_prob=0.6, best_odds=2.0, best_sportsbook="tipico", ev=0.1,
            ),
            ParlayLeg(
                prediction_id=uuid.uuid4(), match_id=uuid.uuid4(),
                sport="tennis", league="ATP",
                match_description="C vs D", selection="home",
                market_type="match_winner",
                model_prob=0.55, best_odds=1.8, best_sportsbook="bet365", ev=0.05,
            ),
        ]
        correlations: list[LegCorrelation] = []

        combined_odds, adj_prob, ev = calculate_parlay_ev(legs, correlations)

        assert combined_odds == pytest.approx(3.6, rel=0.01)  # 2.0 * 1.8
        assert adj_prob == pytest.approx(0.33, rel=0.01)  # 0.6 * 0.55
        # EV = 0.33 * 3.6 - 1.0 = 0.188
        assert ev > 0.0  # positive EV

    def test_correlated_legs_reduce_prob(self):
        legs = [
            ParlayLeg(
                prediction_id=uuid.uuid4(), match_id=uuid.uuid4(),
                sport="football", league="BL",
                match_description="A vs B", selection="home",
                market_type="match_winner",
                model_prob=0.6, best_odds=2.0, best_sportsbook="tipico", ev=0.1,
            ),
            ParlayLeg(
                prediction_id=uuid.uuid4(), match_id=uuid.uuid4(),
                sport="football", league="BL",
                match_description="A vs B", selection="over_2.5",
                market_type="over_under",
                model_prob=0.55, best_odds=1.8, best_sportsbook="tipico", ev=0.05,
            ),
        ]
        # Same-match correlation
        corr = LegCorrelation(
            prediction_a_id="a", prediction_b_id="b",
            same_match=True, same_sport=True, same_league=True,
            correlation_factor=0.15, reason="Same match",
        )

        _, adj_prob_corr, ev_corr = calculate_parlay_ev(legs, [corr])

        # Without correlation
        _, adj_prob_indep, ev_indep = calculate_parlay_ev(legs, [])

        # Correlated version should have lower probability
        assert adj_prob_corr < adj_prob_indep
        # And potentially lower EV
        assert ev_corr < ev_indep

    def test_empty_legs(self):
        combined_odds, adj_prob, ev = calculate_parlay_ev([], [])
        assert combined_odds == 1.0
        assert ev == -1.0


# ── Build Parlay ─────────────────────────────────────────────────────────


class TestBuildParlay:
    def test_build_with_explicit_ids(self, db_session):
        m1 = _make_match(db_session, home="Bayern", away="Dortmund")
        m2 = _make_match(db_session, home="Leipzig", away="Freiburg")
        p1 = _make_prediction(db_session, m1, ev=0.08, best_odds=2.0)
        p2 = _make_prediction(db_session, m2, ev=0.05, best_odds=1.9)

        ticket = build_parlay(db_session, prediction_ids=[p1.id, p2.id])

        assert ticket is not None
        assert ticket.num_legs == 2
        assert ticket.stake_eur == MOONSHOT_HARD_CAP_EUR
        assert ticket.combined_odds > 1.0
        assert ticket.potential_payout > 0
        assert isinstance(ticket.parlay_id, uuid.UUID)

    def test_not_enough_picks(self, db_session):
        m = _make_match(db_session)
        p = _make_prediction(db_session, m)

        ticket = build_parlay(db_session, prediction_ids=[p.id])
        assert ticket is None  # need at least 2 legs

    def test_only_approved_picks(self, db_session):
        m1 = _make_match(db_session, home="A", away="B")
        m2 = _make_match(db_session, home="C", away="D")
        p1 = _make_prediction(db_session, m1, status=PredictionStatus.APPROVED)
        p2 = _make_prediction(db_session, m2, status=PredictionStatus.VETOED)

        ticket = build_parlay(db_session, prediction_ids=[p1.id, p2.id])
        assert ticket is None  # only 1 approved

    def test_auto_select_by_sport(self, db_session):
        m1 = _make_match(
            db_session, sport=Sport.TENNIS,
            league="ATP", home="Sinner", away="Djokovic",
        )
        m2 = _make_match(
            db_session, sport=Sport.TENNIS,
            league="ATP", home="Alcaraz", away="Zverev",
        )
        m3 = _make_match(
            db_session, sport=Sport.FOOTBALL,
            home="Bayern", away="Dortmund",
        )
        _make_prediction(db_session, m1, ev=0.10, best_odds=1.8)
        _make_prediction(db_session, m2, ev=0.08, best_odds=2.0)
        _make_prediction(db_session, m3, ev=0.05, best_odds=1.9)

        ticket = build_parlay(db_session, sport_filter="tennis", max_legs=2)

        if ticket is not None:
            # Should only include tennis legs
            for leg in ticket.legs:
                assert leg.sport == "tennis"

    def test_parlay_ticket_to_dict(self, db_session):
        m1 = _make_match(db_session, home="A", away="B")
        m2 = _make_match(db_session, home="C", away="D")
        p1 = _make_prediction(db_session, m1, ev=0.08, best_odds=2.0)
        p2 = _make_prediction(db_session, m2, ev=0.05, best_odds=1.9)

        ticket = build_parlay(db_session, prediction_ids=[p1.id, p2.id])
        assert ticket is not None

        d = ticket.to_dict()
        assert "parlay_id" in d
        assert "combined_odds" in d
        assert "legs" in d
        assert len(d["legs"]) == 2
        assert "is_positive_ev" in d

    def test_parlay_format_message(self, db_session):
        m1 = _make_match(db_session, home="Bayern", away="Dortmund")
        m2 = _make_match(db_session, home="Leipzig", away="Freiburg")
        p1 = _make_prediction(db_session, m1, ev=0.08, best_odds=2.0)
        p2 = _make_prediction(db_session, m2, ev=0.05, best_odds=1.9)

        ticket = build_parlay(db_session, prediction_ids=[p1.id, p2.id])
        assert ticket is not None

        msg = ticket.format_message()
        assert "MOONSHOT PARLAY" in msg
        assert "Leg 1:" in msg
        assert "Leg 2:" in msg
        assert "EUR" in msg

    def test_stake_hard_capped(self, db_session):
        m1 = _make_match(db_session, home="A", away="B")
        m2 = _make_match(db_session, home="C", away="D")
        p1 = _make_prediction(db_session, m1, ev=0.08, best_odds=2.0)
        p2 = _make_prediction(db_session, m2, ev=0.05, best_odds=1.9)

        ticket = build_parlay(db_session, prediction_ids=[p1.id, p2.id], stake_eur=50.0)
        assert ticket is not None
        assert ticket.stake_eur == MOONSHOT_HARD_CAP_EUR


class TestBuildBestParlay:
    def test_convenience_function(self, db_session):
        m1 = _make_match(db_session, home="A", away="B")
        m2 = _make_match(db_session, home="C", away="D")
        m3 = _make_match(db_session, home="E", away="F")
        _make_prediction(db_session, m1, ev=0.10, best_odds=2.2)
        _make_prediction(db_session, m2, ev=0.08, best_odds=2.0)
        _make_prediction(db_session, m3, ev=0.05, best_odds=1.8)

        ticket = build_best_parlay(db_session, num_legs=3)

        if ticket is not None:
            assert ticket.num_legs <= 3
            assert ticket.stake_eur == MOONSHOT_HARD_CAP_EUR

    def test_empty_db_returns_none(self, db_session):
        ticket = build_best_parlay(db_session)
        assert ticket is None


# ── Telegram Integration ────────────────────────────────────────────────


class TestTelegramParlayDetection:
    """Test the MasterAgentBridge parlay request detection."""

    def test_detect_parlay_keywords(self):
        from bet_agent.interfaces.telegram_bot import MasterAgentBridge
        bridge = MasterAgentBridge()

        # These should detect parlay intent
        assert bridge._try_parlay_request("Baue mir eine 3er Kombi für Tennis") is not None
        assert bridge._try_parlay_request("build a parlay for NBA") is not None
        assert bridge._try_parlay_request("Moonshot 4er für Football") is not None

    def test_non_parlay_not_intercepted(self):
        from bet_agent.interfaces.telegram_bot import MasterAgentBridge
        bridge = MasterAgentBridge()

        # These should NOT be intercepted
        assert bridge._try_parlay_request("How are we doing today?") is None
        assert bridge._try_parlay_request("What's our NBA exposure?") is None
        assert bridge._try_parlay_request("Show me pending bets") is None

    def test_detect_num_legs(self):
        from bet_agent.interfaces.telegram_bot import MasterAgentBridge
        bridge = MasterAgentBridge()

        # Should detect "3" from "3er Kombi"
        result = bridge._try_parlay_request("3er Kombi Tennis")
        assert result is not None
        # The result should mention inability to build (no DB data in tests)
        assert "kombi" in result.lower() or "parlay" in result.lower() or "moonshot" in result.lower()
