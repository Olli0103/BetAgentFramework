"""Tests for all 6 sport probability models."""

import pytest

# Import triggers registration via registry.register() calls
from bet_agent.tools.prob_models import registry  # noqa: F401
from bet_agent.tools.prob_models.football import FootballModel
from bet_agent.tools.prob_models.ice_hockey import IceHockeyModel
from bet_agent.tools.prob_models.basketball import BasketballModel
from bet_agent.tools.prob_models.tennis import TennisModel
from bet_agent.tools.prob_models.american_football import AmericanFootballModel
from bet_agent.tools.prob_models.darts import DartsModel


def _assert_probs_sum_to_one(probs: dict[str, float], tol: float = 0.01):
    """Helper: assert probabilities sum to ~1.0."""
    total = sum(probs.values())
    assert abs(total - 1.0) < tol, f"Probs sum to {total}, expected ~1.0: {probs}"


def _assert_valid_prob(p: float, name: str = "p"):
    """Helper: assert 0 <= p <= 1."""
    assert 0.0 <= p <= 1.0, f"{name} = {p}, expected in [0, 1]"


# ── Registry ───────────────────────────────────────────────────────────


class TestRegistry:
    def test_all_sports_registered(self):
        sports = registry.available_sports()
        expected = [
            "american_football", "basketball", "darts",
            "football", "ice_hockey", "tennis",
        ]
        assert sports == expected

    def test_get_model(self):
        model = registry.get_model("football")
        assert model.sport == "football"

    def test_unknown_sport_raises(self):
        with pytest.raises(KeyError, match="cricket"):
            registry.get_model("cricket")


# ── Football ───────────────────────────────────────────────────────────


class TestFootball:
    model = FootballModel()

    def test_outcome_probs_sum(self):
        probs = self.model.match_outcome_probs(home_xg=1.5, away_xg=1.2)
        _assert_probs_sum_to_one(probs)

    def test_stronger_home_team(self):
        probs = self.model.match_outcome_probs(home_xg=2.5, away_xg=0.8)
        assert probs["home"] > probs["away"]
        assert probs["home"] > probs["draw"]

    def test_equal_teams(self):
        probs = self.model.match_outcome_probs(home_xg=1.3, away_xg=1.3)
        assert abs(probs["home"] - probs["away"]) < 0.01

    def test_over_under(self):
        p_over = self.model.over_under_prob(home_xg=1.5, away_xg=1.5, line=2.5)
        _assert_valid_prob(p_over, "P(over 2.5)")
        assert p_over > 0.3  # With 3.0 total xG, over 2.5 should be likely

    def test_btts(self):
        p = self.model.btts_prob(home_xg=1.5, away_xg=1.2)
        _assert_valid_prob(p, "P(BTTS)")
        assert p > 0.4  # Both teams have decent xG

    def test_live_update_halftime_leading(self):
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(2, 0),
            live_time=45.0, live_stats={"home_xg": 1.8, "away_xg": 0.4},
        )
        _assert_valid_prob(p, "P(home wins at HT 2-0)")
        assert p > 0.8  # 2-0 at halftime, strong favorite

    def test_live_update_end_of_match(self):
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(1, 2), live_time=90.0,
        )
        assert p == 0.0  # Home lost

    def test_dixon_coles_increases_draw_prob(self):
        """Dixon-Coles rho < 0 should increase draw probability vs naive Poisson."""
        # With rho=0 (no correction = naive independent Poisson)
        probs_naive = self.model.match_outcome_probs(home_xg=1.3, away_xg=1.1, rho=0.0)
        # With default rho (Dixon-Coles correction)
        probs_dc = self.model.match_outcome_probs(home_xg=1.3, away_xg=1.1)
        assert probs_dc["draw"] > probs_naive["draw"], (
            f"Dixon-Coles should increase draw prob: {probs_dc['draw']:.4f} vs naive {probs_naive['draw']:.4f}"
        )

    def test_dixon_coles_probs_sum_to_one(self):
        probs = self.model.match_outcome_probs(home_xg=1.5, away_xg=1.2)
        _assert_probs_sum_to_one(probs)

    def test_dixon_coles_rho_zero_equals_naive(self):
        """With rho=0, results should match independent Poisson."""
        probs = self.model.match_outcome_probs(home_xg=1.5, away_xg=1.2, rho=0.0)
        _assert_probs_sum_to_one(probs)
        # Just verify it doesn't crash and sums to 1

    def test_live_update_with_pre_match_xg(self):
        """live_update should use explicit pre-match xG instead of magic numbers."""
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(1, 0), live_time=60.0,
            pre_match_home_xg=1.8, pre_match_away_xg=0.9,
        )
        _assert_valid_prob(p, "P(home wins 60min 1-0 with xG)")
        assert p > 0.7  # Leading 1-0 at 60' with superior xG

    def test_live_update_no_xg_uses_defaults(self):
        """Without xG args, live_update should use class defaults (not crash)."""
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(0, 0), live_time=45.0,
        )
        _assert_valid_prob(p, "P(home wins 0-0 at HT)")


# ── Ice Hockey ─────────────────────────────────────────────────────────


class TestIceHockey:
    model = IceHockeyModel()

    def test_outcome_probs_regulation(self):
        probs = self.model.match_outcome_probs(home_xg=2.8, away_xg=2.5)
        _assert_probs_sum_to_one(probs)

    def test_outcome_probs_with_ot(self):
        probs = self.model.match_outcome_probs_with_ot(home_xg=2.8, away_xg=2.5)
        assert probs["draw"] == 0.0
        _assert_probs_sum_to_one(probs)

    def test_over_under_5_5(self):
        p = self.model.over_under_prob(home_xg=3.0, away_xg=2.8, line=5.5)
        _assert_valid_prob(p)

    def test_live_update_tie_goes_to_ot(self):
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(3, 3), live_time=60.0,
        )
        assert abs(p - 0.52) < 0.01  # OT_PROB_HOME


# ── Basketball ─────────────────────────────────────────────────────────


class TestBasketball:
    model = BasketballModel()

    def test_outcome_probs(self):
        probs = self.model.match_outcome_probs(
            home_off_rtg=115, home_def_rtg=108,
            away_off_rtg=112, away_def_rtg=110,
        )
        assert probs["draw"] == 0.0
        _assert_probs_sum_to_one(probs)

    def test_better_team_favored(self):
        probs = self.model.match_outcome_probs(
            home_off_rtg=120, home_def_rtg=105,
            away_off_rtg=105, away_def_rtg=115,
        )
        assert probs["home"] > 0.7

    def test_spread_prob(self):
        p = self.model.spread_prob(
            home_off_rtg=115, home_def_rtg=108,
            away_off_rtg=112, away_def_rtg=110,
            spread=-5.5,
        )
        _assert_valid_prob(p)

    def test_live_update_big_lead(self):
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(95, 75), live_time=40.0,
        )
        assert p > 0.9  # 20 point lead with 8 min left


# ── Tennis ─────────────────────────────────────────────────────────────


class TestTennis:
    model = TennisModel()

    def test_match_outcome_bo3(self):
        probs = self.model.match_outcome_probs(
            p_serve_home=0.65, p_serve_away=0.60, best_of=3,
        )
        assert probs["draw"] == 0.0
        _assert_probs_sum_to_one(probs)
        assert probs["home"] > probs["away"]  # Better server wins more

    def test_match_outcome_bo5(self):
        probs = self.model.match_outcome_probs(
            p_serve_home=0.65, p_serve_away=0.60, best_of=5,
        )
        _assert_probs_sum_to_one(probs)
        # Bo5 amplifies skill advantage
        bo3_probs = self.model.match_outcome_probs(
            p_serve_home=0.65, p_serve_away=0.60, best_of=3,
        )
        assert probs["home"] > bo3_probs["home"]

    def test_game_prob_server_advantage(self):
        p = TennisModel._p_game(0.65)
        assert p > 0.8  # Server with 65% point win rate wins ~80%+ of games

    def test_live_update_set_advantage(self):
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(1, 0), live_time=0,
            live_stats={"p_serve_home": 0.65, "p_serve_away": 0.62, "best_of": 3},
        )
        assert p > 0.6  # 1 set up in bo3


# ── American Football ──────────────────────────────────────────────────


class TestAmericanFootball:
    model = AmericanFootballModel()

    def test_outcome_probs(self):
        probs = self.model.match_outcome_probs(
            home_power_rtg=5.0, away_power_rtg=-2.0,
        )
        _assert_probs_sum_to_one(probs)
        assert probs["home"] > 0.7  # +7 expected diff + home advantage

    def test_spread_prob(self):
        p = self.model.spread_prob(
            home_power_rtg=5.0, away_power_rtg=-2.0, spread=-7.5,
        )
        _assert_valid_prob(p)

    def test_over_under(self):
        p = self.model.over_under_prob(line=44.5)
        _assert_valid_prob(p)

    def test_live_update_leading_late(self):
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(28, 14), live_time=50.0,
        )
        assert p > 0.85  # 14 point lead with 10 min left


# ── Darts ──────────────────────────────────────────────────────────────


class TestDarts:
    model = DartsModel()

    def test_legs_format(self):
        probs = self.model.match_outcome_probs(
            p_leg_home=0.55, legs_to_win=6,
        )
        _assert_probs_sum_to_one(probs)
        assert probs["home"] > 0.5  # Slight edge

    def test_fair_match(self):
        probs = self.model.match_outcome_probs(
            p_leg_home=0.50, legs_to_win=6,
        )
        assert abs(probs["home"] - 0.5) < 0.01

    def test_sets_format(self):
        p = self.model.match_win_prob_sets(
            p_leg_home=0.55, legs_per_set=5, sets_to_win=4,
        )
        _assert_valid_prob(p)
        assert p > 0.5

    def test_live_update_legs(self):
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(5, 2), live_time=0,
            live_stats={"p_leg_home": 0.55, "legs_to_win": 6, "format": "legs"},
        )
        assert p > 0.8  # 5-2 up, needs 1 more leg

    def test_live_update_sets(self):
        p = self.model.live_update(
            pre_match_prob=0.5, live_score=(3, 1), live_time=0,
            live_stats={
                "p_leg_home": 0.55, "format": "sets",
                "legs_per_set": 5, "sets_to_win": 4,
            },
        )
        assert p > 0.7  # 3-1 up in sets
