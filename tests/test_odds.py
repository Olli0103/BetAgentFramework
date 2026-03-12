"""Tests for odds format detection and conversion."""

import pytest

from bet_agent.tools.odds import american_to_decimal, normalize_odds


class TestAmericanToDecimal:
    def test_positive_american(self):
        assert american_to_decimal(270) == pytest.approx(3.70)

    def test_positive_american_100(self):
        assert american_to_decimal(100) == pytest.approx(2.0)

    def test_positive_american_large(self):
        # +500 → 6.0
        assert american_to_decimal(500) == pytest.approx(6.0)

    def test_negative_american(self):
        # -150 → 1.6667
        assert american_to_decimal(-150) == pytest.approx(1.6667, abs=0.001)

    def test_negative_american_100(self):
        assert american_to_decimal(-100) == pytest.approx(2.0)

    def test_negative_american_heavy(self):
        # -300 → 1.3333
        assert american_to_decimal(-300) == pytest.approx(1.3333, abs=0.001)

    def test_invalid_american_between(self):
        with pytest.raises(ValueError, match="Invalid American odds"):
            american_to_decimal(50)


class TestNormalizeOdds:
    def test_decimal_passthrough(self):
        assert normalize_odds(2.50) == 2.50

    def test_decimal_low(self):
        assert normalize_odds(1.05) == 1.05

    def test_decimal_high(self):
        # 50.0 is a valid longshot decimal odd
        assert normalize_odds(50.0) == 50.0

    def test_american_positive(self):
        # +270 → 3.70
        assert normalize_odds(270) == pytest.approx(3.70)

    def test_american_positive_100(self):
        # +100 → 2.0
        assert normalize_odds(100) == pytest.approx(2.0)

    def test_american_negative(self):
        # -150 → 1.6667
        assert normalize_odds(-150) == pytest.approx(1.6667, abs=0.001)

    def test_american_negative_heavy(self):
        # -500 → 1.20
        assert normalize_odds(-500) == pytest.approx(1.20)

    def test_invalid_zero(self):
        with pytest.raises(ValueError):
            normalize_odds(0)

    def test_invalid_one(self):
        with pytest.raises(ValueError):
            normalize_odds(1.0)

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            normalize_odds(0.5)

    def test_invalid_small_negative(self):
        # -50 is not valid American
        with pytest.raises(ValueError):
            normalize_odds(-50)


class TestIntegrationEV:
    """Verify that EV calculator handles American odds without overflow."""

    def test_pre_match_ev_with_american_odds(self):
        from bet_agent.tools.ev_calculator import calculate_pre_match_ev

        # +270 should be auto-converted to 3.70 decimal
        result = calculate_pre_match_ev(model_prob=0.35, odds=270)
        assert result.ev == pytest.approx(
            (0.35 * (3.70 - 1.0)) - (1.0 - 0.35), abs=0.01
        )
        assert abs(result.ev) < 5.0  # Sanity: no overflow

    def test_pre_match_ev_with_negative_american(self):
        from bet_agent.tools.ev_calculator import calculate_pre_match_ev

        # -150 → 1.6667
        result = calculate_pre_match_ev(model_prob=0.65, odds=-150)
        assert abs(result.ev) < 5.0

    def test_live_ev_with_american_odds(self):
        import bet_agent.tools.prob_models.football  # noqa: F401
        from bet_agent.tools.ev_calculator import calculate_live_ev

        result = calculate_live_ev(
            sport="football",
            pre_match_prob=0.5,
            live_score=(1, 0),
            live_time=60.0,
            live_odds=150,  # American +150 → 2.50 decimal
        )
        assert abs(result.ev) < 5.0


class TestIntegrationKelly:
    """Verify that Kelly calculator handles American odds without overflow."""

    def test_kelly_with_american_odds(self):
        from bet_agent.tools.kelly_calculator import calculate_quarter_kelly

        # +200 → 3.0 decimal, prob=0.40, bankroll=1000
        result = calculate_quarter_kelly(prob=0.40, odds=200, bankroll=1000)
        # edge = (0.4 * 3.0) - 1 = 0.2; full_kelly = 0.2/2.0 = 0.1
        # quarter_kelly = 0.025; stake = 25.0
        assert result.stake_eur == pytest.approx(25.0, abs=1.0)
        assert result.stake_eur <= 50.0  # Max 5% of 1000
