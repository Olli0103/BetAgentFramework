"""Tests for the Quarter-Kelly Criterion calculator."""

import pytest

from bet_agent.tools.kelly_calculator import KellyResult, calculate_quarter_kelly


class TestQuarterKelly:
    def test_positive_ev_returns_stake(self):
        result = calculate_quarter_kelly(prob=0.55, odds=2.0, bankroll=1000.0)
        assert result.stake_eur > 0
        assert result.reason is None
        assert result.expected_profit > 0

    def test_negative_ev_returns_zero(self):
        result = calculate_quarter_kelly(prob=0.40, odds=2.0, bankroll=1000.0)
        assert result.stake_eur == 0.0
        assert result.reason == "negative_ev"

    def test_quarter_kelly_is_quarter_of_full(self):
        result = calculate_quarter_kelly(prob=0.60, odds=2.0, bankroll=1000.0)
        assert abs(result.kelly_fraction - result.full_kelly_fraction * 0.25) < 1e-6

    def test_max_5_percent_cap(self):
        # Very high edge should still be capped at 5% of bankroll
        result = calculate_quarter_kelly(prob=0.95, odds=5.0, bankroll=1000.0)
        assert result.stake_eur <= 50.0  # 5% of 1000

    def test_rounded_to_cents(self):
        result = calculate_quarter_kelly(prob=0.55, odds=2.0, bankroll=1000.0)
        # Check it's rounded to 2 decimal places
        assert result.stake_eur == round(result.stake_eur, 2)

    def test_zero_bankroll(self):
        result = calculate_quarter_kelly(prob=0.60, odds=2.0, bankroll=0.0)
        assert result.stake_eur == 0.0
        assert result.reason == "zero_bankroll"

    def test_small_bankroll_below_minimum(self):
        result = calculate_quarter_kelly(prob=0.55, odds=2.0, bankroll=1.0)
        # Quarter Kelly on 1 EUR with slight edge → below 0.10 minimum
        assert result.stake_eur == 0.0
        assert result.reason == "below_minimum_stake"

    def test_invalid_prob_zero(self):
        with pytest.raises(ValueError, match="prob"):
            calculate_quarter_kelly(prob=0.0, odds=2.0, bankroll=1000.0)

    def test_invalid_prob_one(self):
        with pytest.raises(ValueError, match="prob"):
            calculate_quarter_kelly(prob=1.0, odds=2.0, bankroll=1000.0)

    def test_invalid_odds(self):
        with pytest.raises(ValueError, match="odds"):
            calculate_quarter_kelly(prob=0.5, odds=1.0, bankroll=1000.0)

    def test_negative_bankroll(self):
        with pytest.raises(ValueError, match="bankroll"):
            calculate_quarter_kelly(prob=0.5, odds=2.0, bankroll=-100.0)

    def test_result_is_dataclass(self):
        result = calculate_quarter_kelly(prob=0.55, odds=2.0, bankroll=1000.0)
        assert isinstance(result, KellyResult)

    def test_concrete_calculation(self):
        # prob=0.6, odds=2.0, bankroll=1000
        # edge = 0.6*2.0 - 1 = 0.2
        # full_kelly = 0.2 / (2.0-1.0) = 0.2
        # quarter_kelly = 0.05
        # stake = 1000 * 0.05 = 50.0
        # But capped at 5% = 50.0 → exactly at cap
        result = calculate_quarter_kelly(prob=0.6, odds=2.0, bankroll=1000.0)
        assert result.stake_eur == 50.0
        assert abs(result.full_kelly_fraction - 0.2) < 1e-6
        assert abs(result.kelly_fraction - 0.05) < 1e-6
