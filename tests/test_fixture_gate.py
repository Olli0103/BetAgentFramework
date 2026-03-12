"""Tests for the fixture validity gate in prediction_runner."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from bet_agent.db.models import Match, MatchState, Sport
from bet_agent.tools.prediction_runner import validate_fixture


def _make_match(**overrides):
    defaults = {
        "home_team": "Charlotte Hornets",
        "away_team": "Boston Celtics",
        "sport": Sport.BASKETBALL,
        "league": "NBA",
        "scheduled_at": datetime(2026, 3, 12, 19, 0, tzinfo=timezone.utc),
        "match_state": MatchState.NOT_STARTED,
    }
    defaults.update(overrides)
    m = MagicMock(spec=Match)
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


class TestValidateFixture:
    def test_valid_fixture(self):
        m = _make_match()
        odds = {"match_winner": {"home": 2.10, "draw": 3.50, "away": 3.80}}
        ok, reason = validate_fixture(m, odds)
        assert ok is True
        assert reason == ""

    def test_home_team_too_short(self):
        m = _make_match(home_team="CHA")
        odds = {"match_winner": {"home": 2.10}}
        ok, reason = validate_fixture(m, odds)
        assert ok is False
        assert "home_team" in reason and ("abbreviation" in reason or "too short" in reason)

    def test_away_team_too_short(self):
        m = _make_match(away_team="BOS")
        odds = {"match_winner": {"home": 2.10}}
        ok, reason = validate_fixture(m, odds)
        assert ok is False
        assert "away_team" in reason and ("abbreviation" in reason or "too short" in reason)

    def test_missing_scheduled_at(self):
        m = _make_match(scheduled_at=None)
        odds = {"match_winner": {"home": 2.10}}
        ok, reason = validate_fixture(m, odds)
        assert ok is False
        assert "missing scheduled_at" in reason

    def test_no_odds(self):
        m = _make_match()
        ok, reason = validate_fixture(m, {})
        assert ok is False
        assert "no match_winner odds" in reason

    def test_no_home_odds(self):
        m = _make_match()
        odds = {"match_winner": {"draw": 3.50, "away": 3.80}}
        ok, reason = validate_fixture(m, odds)
        assert ok is False
        assert "no match_winner odds" in reason

    def test_invalid_odds_value(self):
        m = _make_match()
        odds = {"match_winner": {"home": 0.5, "away": 3.80}}
        ok, reason = validate_fixture(m, odds)
        assert ok is False
        assert "invalid odds" in reason

    def test_tennis_short_names_ok(self):
        """Tennis players can have short last names — 4+ chars is fine."""
        m = _make_match(home_team="Sinner", away_team="Rune", sport=Sport.TENNIS)
        odds = {"match_winner": {"home": 1.50, "away": 2.80}}
        ok, reason = validate_fixture(m, odds)
        assert ok is True
