"""Tests for the EV calculator."""

import pytest

# Import sport models to trigger registration
import bet_agent.tools.prob_models.football  # noqa: F401
import bet_agent.tools.prob_models.basketball  # noqa: F401
import bet_agent.tools.prob_models.tennis  # noqa: F401

from bet_agent.tools.ev_calculator import (
    EVResult,
    calculate_live_ev,
    calculate_pre_match_ev,
)


class TestPreMatchEV:
    def test_positive_ev(self):
        # Model says 55% but odds imply 50% → +EV
        result = calculate_pre_match_ev(model_prob=0.55, odds=2.0)
        assert result.is_positive_ev
        assert result.ev > 0
        assert result.edge > 0

    def test_negative_ev(self):
        # Model says 40% but odds imply 50% → -EV
        result = calculate_pre_match_ev(model_prob=0.40, odds=2.0)
        assert not result.is_positive_ev
        assert result.ev < 0
        assert result.edge < 0

    def test_fair_odds(self):
        # Model = implied → EV ≈ 0
        result = calculate_pre_match_ev(model_prob=0.50, odds=2.0)
        assert abs(result.ev) < 0.001

    def test_invalid_prob(self):
        with pytest.raises(ValueError, match="model_prob"):
            calculate_pre_match_ev(model_prob=1.5, odds=2.0)

    def test_invalid_odds(self):
        with pytest.raises(ValueError, match="odds"):
            calculate_pre_match_ev(model_prob=0.5, odds=0.5)


class TestLiveEV:
    def test_football_live_positive_ev(self):
        result = calculate_live_ev(
            sport="football",
            pre_match_prob=0.5,
            live_score=(2, 0),
            live_time=45.0,
            live_odds=1.40,
            live_stats={"home_xg": 1.8, "away_xg": 0.3},
        )
        assert isinstance(result, EVResult)
        assert result.updated_prob > 0.5  # Leading 2-0

    def test_basketball_live(self):
        result = calculate_live_ev(
            sport="basketball",
            pre_match_prob=0.5,
            live_score=(80, 65),
            live_time=36.0,
            live_odds=1.20,
        )
        assert result.updated_prob > 0.7

    def test_tennis_live(self):
        result = calculate_live_ev(
            sport="tennis",
            pre_match_prob=0.5,
            live_score=(1, 0),
            live_time=0,
            live_odds=1.60,
            live_stats={"p_serve_home": 0.65, "p_serve_away": 0.60, "best_of": 3},
        )
        assert result.updated_prob > 0.5

    def test_unknown_sport_raises(self):
        with pytest.raises(KeyError):
            calculate_live_ev(
                sport="cricket",
                pre_match_prob=0.5,
                live_score=(0, 0),
                live_time=0,
                live_odds=2.0,
            )

    def test_invalid_live_time(self):
        with pytest.raises(ValueError, match="live_time"):
            calculate_live_ev(
                sport="football",
                pre_match_prob=0.5,
                live_score=(0, 0),
                live_time=-5.0,
                live_odds=2.0,
            )

    def test_invalid_live_odds(self):
        with pytest.raises(ValueError, match="live_odds"):
            calculate_live_ev(
                sport="football",
                pre_match_prob=0.5,
                live_score=(0, 0),
                live_time=30.0,
                live_odds=0.8,
            )
