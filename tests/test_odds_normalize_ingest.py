"""Tests for OddsMarket auto-normalization on attribute set."""

from decimal import Decimal

import pytest

from bet_agent.db.models import OddsMarket


class TestOddsMarketNormalization:
    def test_decimal_passthrough(self):
        om = OddsMarket()
        om.odds_decimal = Decimal("2.50")
        assert om.odds_decimal == Decimal("2.50")

    def test_american_positive_normalized(self):
        om = OddsMarket()
        om.odds_decimal = Decimal("270")  # American +270 → 3.70
        assert float(om.odds_decimal) == pytest.approx(3.70, abs=0.01)

    def test_american_negative_normalized(self):
        om = OddsMarket()
        om.odds_decimal = Decimal("-150")  # American -150 → 1.6667
        assert float(om.odds_decimal) == pytest.approx(1.6667, abs=0.01)

    def test_american_100_normalized(self):
        om = OddsMarket()
        om.odds_decimal = Decimal("100")  # American +100 → 2.0
        assert float(om.odds_decimal) == pytest.approx(2.0, abs=0.01)

    def test_low_decimal_passthrough(self):
        om = OddsMarket()
        om.odds_decimal = Decimal("1.05")
        assert om.odds_decimal == Decimal("1.05")

    def test_none_stays_none(self):
        om = OddsMarket()
        om.odds_decimal = None
        assert om.odds_decimal is None
