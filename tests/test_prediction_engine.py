"""Tests for Phase 3: Prediction Engine & Live Betting Integration.

Covers:
  - Prediction model (DB table)
  - Parameter Estimator (sport-specific analytical parameter estimation)
  - Prediction Runner (daily pipeline, idempotency, ML/analytical fallback)
"""

from datetime import date, datetime, time, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import (
    Base,
    MarketType,
    Match,
    MatchState,
    OddsMarket,
    Prediction,
    PredictionStatus,
    Sport,
    TeamDailyStats,
)


@pytest.fixture
def db_session():
    """In-memory SQLite for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _add_match(session, sport=Sport.FOOTBALL, **overrides) -> Match:
    """Helper to create a NOT_STARTED match for today."""
    defaults = dict(
        sport=sport,
        league="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        scheduled_at=datetime.combine(date.today(), time(15, 0), tzinfo=timezone.utc),
        match_state=MatchState.NOT_STARTED,
        is_live=False,
    )
    defaults.update(overrides)
    m = Match(**defaults)
    session.add(m)
    session.flush()
    return m


def _add_odds(session, match, home=2.10, draw=3.50, away=3.80):
    """Add match winner odds for a match."""
    for sel, odds in [("home", home), ("draw", draw), ("away", away)]:
        if odds is not None:
            session.add(OddsMarket(
                match_id=match.id,
                sportsbook="test_book",
                market_type=MarketType.MATCH_WINNER,
                selection=sel,
                odds_decimal=Decimal(str(odds)),
                is_live=False,
            ))
    session.flush()


def _add_daily_stats(session, sport, team, stat_date, stats):
    """Add team daily stats."""
    session.add(TeamDailyStats(
        sport=sport,
        team_name=team,
        league="Test",
        stat_date=stat_date,
        stats=stats,
    ))
    session.flush()


# ── Prediction Model ─────────────────────────────────────────────────


class TestPredictionModel:
    def test_create_prediction(self, db_session):
        match = _add_match(db_session)

        pred = Prediction(
            match_id=match.id,
            model_name="xgb_football_match_winner",
            market_type=MarketType.MATCH_WINNER,
            selection="home",
            model_prob=Decimal("0.55"),
            implied_prob=Decimal("0.476190"),
            prob_edge=Decimal("0.073810"),
            ev=Decimal("0.155"),
            model_source="xgboost",
            status=PredictionStatus.PENDING,
        )
        db_session.add(pred)
        db_session.flush()

        result = db_session.query(Prediction).one()
        assert result.model_source == "xgboost"
        assert result.status == PredictionStatus.PENDING
        assert float(result.model_prob) == 0.55

    def test_prediction_status_transitions(self, db_session):
        match = _add_match(db_session)
        pred = Prediction(
            match_id=match.id,
            model_name="test",
            market_type=MarketType.MATCH_WINNER,
            selection="home",
            model_prob=Decimal("0.55"),
            implied_prob=Decimal("0.50"),
            prob_edge=Decimal("0.05"),
            ev=Decimal("0.10"),
        )
        db_session.add(pred)
        db_session.flush()

        assert pred.status == PredictionStatus.PENDING

        pred.status = PredictionStatus.APPROVED
        db_session.flush()
        assert db_session.query(Prediction).one().status == PredictionStatus.APPROVED

        pred.status = PredictionStatus.PLACED
        db_session.flush()
        assert db_session.query(Prediction).one().status == PredictionStatus.PLACED

    def test_prediction_unique_constraint(self, db_session):
        match = _add_match(db_session)

        pred1 = Prediction(
            match_id=match.id, model_name="model_a",
            market_type=MarketType.MATCH_WINNER, selection="home",
            model_prob=Decimal("0.55"), implied_prob=Decimal("0.50"),
            prob_edge=Decimal("0.05"), ev=Decimal("0.10"),
        )
        db_session.add(pred1)
        db_session.flush()

        # Same match, same model, same market, same selection → should conflict
        pred2 = Prediction(
            match_id=match.id, model_name="model_a",
            market_type=MarketType.MATCH_WINNER, selection="home",
            model_prob=Decimal("0.60"), implied_prob=Decimal("0.50"),
            prob_edge=Decimal("0.10"), ev=Decimal("0.20"),
        )
        db_session.add(pred2)
        with pytest.raises(Exception):  # IntegrityError
            db_session.flush()

    def test_repr(self, db_session):
        match = _add_match(db_session)
        pred = Prediction(
            match_id=match.id, model_name="test",
            market_type=MarketType.MATCH_WINNER, selection="home",
            model_prob=Decimal("0.55"), implied_prob=Decimal("0.50"),
            prob_edge=Decimal("0.05"), ev=Decimal("0.10"),
        )
        assert "home" in repr(pred)


# ── Parameter Estimator ──────────────────────────────────────────────


class TestParamEstimator:
    def test_football_with_xg(self, db_session):
        from bet_agent.tools.param_estimator import estimate_params

        _add_daily_stats(db_session, Sport.FOOTBALL, "Arsenal", date(2025, 1, 14), {
            "xg": 1.8, "games_played": 20,
        })
        _add_daily_stats(db_session, Sport.FOOTBALL, "Chelsea", date(2025, 1, 14), {
            "xg": 1.2, "games_played": 20,
        })

        result = estimate_params(db_session, Sport.FOOTBALL, "Arsenal", "Chelsea", date(2025, 1, 15))
        assert result.params["home_xg"] == 1.8
        assert result.params["away_xg"] == 1.2
        assert result.confidence == "high"

    def test_football_falls_back_to_rolling_goals(self, db_session):
        from bet_agent.tools.param_estimator import estimate_params

        _add_daily_stats(db_session, Sport.FOOTBALL, "Arsenal", date(2025, 1, 14), {
            "roll_10_goals_for": 1.6, "games_played": 12,
        })

        result = estimate_params(db_session, Sport.FOOTBALL, "Arsenal", "Chelsea", date(2025, 1, 15))
        assert result.params["home_xg"] == 1.6
        # Away has no data → default
        assert result.params["away_xg"] == 1.10

    def test_football_default_when_no_data(self, db_session):
        from bet_agent.tools.param_estimator import estimate_params

        result = estimate_params(db_session, Sport.FOOTBALL, "NewTeam", "OtherTeam", date(2025, 1, 15))
        assert result.params["home_xg"] == 1.35
        assert result.params["away_xg"] == 1.10
        assert result.confidence == "low"

    def test_tennis_serve_estimation(self, db_session):
        from bet_agent.tools.param_estimator import estimate_params

        _add_daily_stats(db_session, Sport.TENNIS, "Sinner", date(2025, 1, 14), {
            "first_serve_won_pct": 0.72, "games_played": 25,
        })
        _add_daily_stats(db_session, Sport.TENNIS, "Djokovic", date(2025, 1, 14), {
            "first_serve_won_pct": 0.68, "games_played": 25,
        })

        result = estimate_params(db_session, Sport.TENNIS, "Sinner", "Djokovic", date(2025, 1, 15))
        assert result.params["p_serve_home"] == 0.72
        assert result.params["p_serve_away"] == 0.68
        assert result.params["best_of"] == 3

    def test_ice_hockey_corsi_estimation(self, db_session):
        from bet_agent.tools.param_estimator import estimate_params

        _add_daily_stats(db_session, Sport.ICE_HOCKEY, "Avalanche", date(2025, 1, 14), {
            "corsi_for_pct": 55.0, "games_played": 30,
        })
        _add_daily_stats(db_session, Sport.ICE_HOCKEY, "Stars", date(2025, 1, 14), {
            "corsi_for_pct": 48.0, "games_played": 30,
        })

        result = estimate_params(db_session, Sport.ICE_HOCKEY, "Avalanche", "Stars", date(2025, 1, 15))
        # Corsi 55% → 55/50 * 2.8 = 3.08
        assert result.params["home_xg"] == 3.08
        # Corsi 48% → 48/50 * 2.8 = 2.69
        assert result.params["away_xg"] == 2.69
        assert result.confidence == "high"

    def test_basketball_rating_estimation(self, db_session):
        from bet_agent.tools.param_estimator import estimate_params

        _add_daily_stats(db_session, Sport.BASKETBALL, "Celtics", date(2025, 1, 14), {
            "off_rtg": 118.5, "def_rtg": 108.2, "pace": 99.0, "games_played": 40,
        })
        _add_daily_stats(db_session, Sport.BASKETBALL, "Lakers", date(2025, 1, 14), {
            "off_rtg": 115.0, "def_rtg": 112.0, "pace": 101.0, "games_played": 40,
        })

        result = estimate_params(db_session, Sport.BASKETBALL, "Celtics", "Lakers", date(2025, 1, 15))
        assert result.params["home_off_rtg"] == 118.5
        assert result.params["away_off_rtg"] == 115.0
        assert result.params["pace"] == 100.0  # average of 99 and 101

    def test_american_football_epa_estimation(self, db_session):
        from bet_agent.tools.param_estimator import estimate_params

        _add_daily_stats(db_session, Sport.AMERICAN_FOOTBALL, "Chiefs", date(2025, 1, 14), {
            "off_epa": 3.5, "def_epa": -1.2, "games_played": 10,
        })

        result = estimate_params(db_session, Sport.AMERICAN_FOOTBALL, "Chiefs", "Ravens", date(2025, 1, 15))
        # Power = off_epa - def_epa = 3.5 - (-1.2) = 4.7
        assert result.params["home_power_rtg"] == 4.7

    def test_confidence_levels(self, db_session):
        from bet_agent.tools.param_estimator import estimate_params

        _add_daily_stats(db_session, Sport.FOOTBALL, "TeamA", date(2025, 1, 14), {"games_played": 3})
        _add_daily_stats(db_session, Sport.FOOTBALL, "TeamB", date(2025, 1, 14), {"games_played": 3})

        result = estimate_params(db_session, Sport.FOOTBALL, "TeamA", "TeamB", date(2025, 1, 15))
        assert result.confidence == "low"

        _add_daily_stats(db_session, Sport.FOOTBALL, "TeamC", date(2025, 1, 14), {"games_played": 10})
        _add_daily_stats(db_session, Sport.FOOTBALL, "TeamD", date(2025, 1, 14), {"games_played": 10})
        result = estimate_params(db_session, Sport.FOOTBALL, "TeamC", "TeamD", date(2025, 1, 15))
        assert result.confidence == "medium"


# ── Prediction Runner ────────────────────────────────────────────────


class TestPredictionRunner:
    def test_generates_predictions_for_today(self, db_session):
        from bet_agent.tools.prediction_runner import run_daily_predictions

        match = _add_match(db_session)
        _add_odds(db_session, match)

        # Add some team stats so analytical model has data
        _add_daily_stats(db_session, Sport.FOOTBALL, "Arsenal", date.today(), {
            "roll_10_goals_for": 1.8, "games_played": 15,
        })
        _add_daily_stats(db_session, Sport.FOOTBALL, "Chelsea", date.today(), {
            "roll_10_goals_for": 1.2, "games_played": 15,
        })

        preds = run_daily_predictions(db_session, prediction_date=date.today())
        db_session.flush()

        assert len(preds) > 0
        assert all(isinstance(p, Prediction) for p in preds)
        assert all(p.status == PredictionStatus.PENDING for p in preds)

    def test_idempotent_predictions(self, db_session):
        from bet_agent.tools.prediction_runner import run_daily_predictions

        match = _add_match(db_session)
        _add_odds(db_session, match)

        # Run twice
        preds1 = run_daily_predictions(db_session, prediction_date=date.today())
        db_session.flush()
        count1 = db_session.query(Prediction).count()

        preds2 = run_daily_predictions(db_session, prediction_date=date.today())
        db_session.flush()
        count2 = db_session.query(Prediction).count()

        # Should not create duplicates
        assert count1 == count2

    def test_skips_finished_matches(self, db_session):
        from bet_agent.tools.prediction_runner import run_daily_predictions

        _add_match(db_session, match_state=MatchState.FINISHED)

        preds = run_daily_predictions(db_session, prediction_date=date.today())
        assert len(preds) == 0

    def test_no_odds_no_predictions(self, db_session):
        from bet_agent.tools.prediction_runner import run_daily_predictions

        _add_match(db_session)
        # No odds added

        preds = run_daily_predictions(db_session, prediction_date=date.today())
        assert len(preds) == 0

    def test_analytical_fallback_used(self, db_session):
        """Without trained ML model, should use analytical model."""
        from bet_agent.tools.prediction_runner import run_daily_predictions
        import tempfile
        from pathlib import Path

        match = _add_match(db_session)
        _add_odds(db_session, match)

        with tempfile.TemporaryDirectory() as tmp:
            preds = run_daily_predictions(
                db_session, prediction_date=date.today(),
                model_dir=Path(tmp),
            )
            db_session.flush()

            if preds:
                assert all(p.model_source == "analytical" for p in preds)

    def test_multiple_sports(self, db_session):
        from bet_agent.tools.prediction_runner import run_daily_predictions

        m1 = _add_match(db_session, sport=Sport.FOOTBALL, home_team="Arsenal", away_team="Chelsea")
        _add_odds(db_session, m1)

        m2 = _add_match(db_session, sport=Sport.ICE_HOCKEY, home_team="Avalanche", away_team="Stars", league="NHL")
        _add_odds(db_session, m2)

        preds = run_daily_predictions(db_session, prediction_date=date.today())
        db_session.flush()

        # Should have predictions for both matches
        match_ids = {p.match_id for p in preds}
        assert m1.id in match_ids or m2.id in match_ids

    def test_min_ev_filter(self, db_session):
        from bet_agent.tools.prediction_runner import run_daily_predictions

        match = _add_match(db_session)
        _add_odds(db_session, match)

        # Very high min_ev should filter out most predictions
        preds = run_daily_predictions(
            db_session, prediction_date=date.today(), min_ev=10.0,
        )
        assert len(preds) == 0


# ── Query Helpers ────────────────────────────────────────────────────


class TestQueryHelpers:
    def test_get_positive_ev_predictions(self, db_session):
        from bet_agent.tools.prediction_runner import get_positive_ev_predictions

        match = _add_match(db_session)

        # Add a positive EV prediction
        db_session.add(Prediction(
            match_id=match.id, model_name="test",
            market_type=MarketType.MATCH_WINNER, selection="home",
            model_prob=Decimal("0.60"), implied_prob=Decimal("0.50"),
            prob_edge=Decimal("0.10"), ev=Decimal("0.20"),
            status=PredictionStatus.PENDING,
        ))
        # Add a negative EV prediction
        db_session.add(Prediction(
            match_id=match.id, model_name="test",
            market_type=MarketType.MATCH_WINNER, selection="away",
            model_prob=Decimal("0.20"), implied_prob=Decimal("0.25"),
            prob_edge=Decimal("-0.05"), ev=Decimal("-0.10"),
            status=PredictionStatus.PENDING,
        ))
        db_session.flush()

        positives = get_positive_ev_predictions(db_session, min_ev=0.0)
        assert len(positives) == 1
        assert float(positives[0].ev) > 0

    def test_update_prediction_status(self, db_session):
        from bet_agent.tools.prediction_runner import update_prediction_status

        match = _add_match(db_session)
        pred = Prediction(
            match_id=match.id, model_name="test",
            market_type=MarketType.MATCH_WINNER, selection="home",
            model_prob=Decimal("0.55"), implied_prob=Decimal("0.50"),
            prob_edge=Decimal("0.05"), ev=Decimal("0.10"),
        )
        db_session.add(pred)
        db_session.flush()

        update_prediction_status(db_session, pred.id, PredictionStatus.APPROVED)
        db_session.flush()
        assert db_session.query(Prediction).one().status == PredictionStatus.APPROVED


# ── Odds Extraction ──────────────────────────────────────────────────


class TestOddsExtraction:
    def test_get_match_odds(self, db_session):
        from bet_agent.tools.prediction_runner import _get_match_odds

        match = _add_match(db_session)
        _add_odds(db_session, match, home=2.10, draw=3.50, away=3.80)

        # Also add O/U odds for two different lines
        db_session.add(OddsMarket(
            match_id=match.id, sportsbook="test",
            market_type=MarketType.OVER_UNDER, selection="over_2.5",
            odds_decimal=Decimal("1.90"), is_live=False,
        ))
        db_session.add(OddsMarket(
            match_id=match.id, sportsbook="test",
            market_type=MarketType.OVER_UNDER, selection="under_2.5",
            odds_decimal=Decimal("1.95"), is_live=False,
        ))
        db_session.add(OddsMarket(
            match_id=match.id, sportsbook="test",
            market_type=MarketType.OVER_UNDER, selection="over_3.5",
            odds_decimal=Decimal("2.40"), is_live=False,
        ))
        db_session.add(OddsMarket(
            match_id=match.id, sportsbook="test",
            market_type=MarketType.OVER_UNDER, selection="under_3.5",
            odds_decimal=Decimal("1.55"), is_live=False,
        ))
        db_session.flush()

        odds = _get_match_odds(db_session, match)

        # Match winner: nested under "match_winner" key
        assert odds["match_winner"]["home"] == 2.10
        assert odds["match_winner"]["draw"] == 3.50
        assert odds["match_winner"]["away"] == 3.80

        # Over/Under: nested by line, each with "over" and "under"
        assert odds["over_under"][2.5]["over"] == 1.90
        assert odds["over_under"][2.5]["under"] == 1.95
        assert odds["over_under"][3.5]["over"] == 2.40
        assert odds["over_under"][3.5]["under"] == 1.55

    def test_no_odds_returns_empty_structure(self, db_session):
        from bet_agent.tools.prediction_runner import _get_match_odds

        match = _add_match(db_session)
        odds = _get_match_odds(db_session, match)
        assert odds == {"match_winner": {}, "over_under": {}}
