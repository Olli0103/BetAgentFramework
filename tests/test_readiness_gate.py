"""Tests for the bettable readiness gate in notifier."""

import uuid
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bet_agent.db.models import LedgerType, MarketType, Match, Prediction, Sport
from bet_agent.tools.notifier import build_ticket, check_bet_readiness


def _pred(**overrides):
    defaults = {
        "id": uuid.uuid4(),
        "best_odds": Decimal("2.10"),
        "best_sportsbook": "tipico",
        "ev": Decimal("0.0450"),
        "prob_edge": Decimal("0.045"),
        "selection": "home",
        "model_source": "analytical",
        "market_type": MarketType.MATCH_WINNER,
        "veto_reason": None,
    }
    defaults.update(overrides)
    p = MagicMock(spec=Prediction)
    for k, v in defaults.items():
        setattr(p, k, v)
    return p


def _match(**overrides):
    defaults = {
        "home_team": "Charlotte Hornets",
        "away_team": "Boston Celtics",
        "sport": MagicMock(value="basketball"),
        "league": "NBA",
    }
    defaults.update(overrides)
    m = MagicMock(spec=Match)
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


class TestCheckBetReadiness:
    def test_all_checks_pass(self):
        r = check_bet_readiness(_pred(), _match(), 5.0)
        assert r.is_ready is True
        assert all(r.checks.values())

    def test_no_odds(self):
        r = check_bet_readiness(_pred(best_odds=None), _match(), 5.0)
        assert r.is_ready is False
        assert "odds_available" in r.failed_checks

    def test_odds_too_high(self):
        r = check_bet_readiness(_pred(best_odds=Decimal("150.0")), _match(), 5.0)
        assert r.is_ready is False
        assert "odds_reasonable" in r.failed_checks

    def test_short_team_name(self):
        r = check_bet_readiness(_pred(), _match(home_team="CHA"), 5.0)
        assert r.is_ready is False
        assert "team_names_ok" in r.failed_checks

    def test_zero_stake(self):
        r = check_bet_readiness(_pred(), _match(), 0.0)
        assert r.is_ready is False
        assert "stake_positive" in r.failed_checks

    def test_negative_ev(self):
        r = check_bet_readiness(_pred(ev=Decimal("-0.02")), _match(), 5.0)
        assert r.is_ready is False
        assert "ev_positive" in r.failed_checks


class TestBuildTicketReadiness:
    def test_ready_ticket_stays_real(self):
        ticket = build_ticket(_pred(), _match(), 5.0, LedgerType.REAL)
        assert ticket.ledger_type == "REAL"
        assert "READINESS FAIL" not in ticket.veto_status

    def test_unready_ticket_downgraded_to_paper(self):
        ticket = build_ticket(
            _pred(best_odds=None), _match(), 5.0, LedgerType.REAL,
        )
        assert ticket.ledger_type == "PAPER"
        assert "READINESS FAIL" in ticket.veto_status

    def test_paper_stays_paper_even_if_unready(self):
        ticket = build_ticket(
            _pred(best_odds=None), _match(), 5.0, LedgerType.PAPER,
        )
        assert ticket.ledger_type == "PAPER"
