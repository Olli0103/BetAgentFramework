"""Tests for TheOddsAPIResultsBackend and fuzzy team matching."""

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import (
    Base,
    BetStatus,
    LedgerType,
    MarketType,
    Match,
    MatchState,
    PlacedBet,
    Sport,
)
from bet_agent.tools.results_fetcher import (
    MatchResult,
    TheOddsAPIResultsBackend,
    _fuzzy_team_match,
    _get_default_backend,
    ManualResultsBackend,
    fetch_and_update_results,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _add_match(
    session, sport=Sport.FOOTBALL, home="Bayern Munich", away="Borussia Dortmund",
    league="Bundesliga",
):
    m = Match(
        sport=sport,
        league=league,
        home_team=home,
        away_team=away,
        scheduled_at=datetime(2026, 3, 12, 18, 30, tzinfo=timezone.utc),
        match_state=MatchState.NOT_STARTED,
    )
    session.add(m)
    session.flush()
    return m


def _add_bet(session, match):
    b = PlacedBet(
        ledger_type=LedgerType.PAPER,
        match_id=match.id,
        market_type=MarketType.MATCH_WINNER,
        selection="home",
        odds_at_placement=Decimal("2.10"),
        stake_eur=Decimal("10.00"),
        model_prob=Decimal("0.55"),
        ev_at_placement=Decimal("0.08"),
        status=BetStatus.PLACED,
        is_live_bet=False,
    )
    session.add(b)
    session.flush()
    return b


# ── Fuzzy matching tests ────────────────────────────────────────────


class TestFuzzyTeamMatch:
    def test_exact_match(self):
        assert _fuzzy_team_match("bayern munich", "bayern munich") is True

    def test_shared_significant_token(self):
        assert _fuzzy_team_match("borussia dortmund", "bvb dortmund") is True

    def test_no_overlap(self):
        assert _fuzzy_team_match("bayern munich", "borussia dortmund") is False

    def test_short_tokens_ignored(self):
        # "fc" and "bvb" are < 4 chars, no overlap
        assert _fuzzy_team_match("fc bvb", "sc bvb") is False

    def test_hornets_match(self):
        assert _fuzzy_team_match("charlotte hornets", "cha hornets") is True


# ── TheOddsAPIResultsBackend tests ──────────────────────────────────


class TestTheOddsAPIResultsBackend:
    def test_not_available_without_key(self):
        backend = TheOddsAPIResultsBackend(api_key="")
        assert backend.is_available is False

    def test_available_with_key(self):
        backend = TheOddsAPIResultsBackend(api_key="test-key")
        assert backend.is_available is True

    def test_fetch_result_returns_none_without_key(self):
        backend = TheOddsAPIResultsBackend(api_key="")
        match = MagicMock()
        match.sport = Sport.FOOTBALL
        assert backend.fetch_result(match) is None

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_fetch_result_completed_match(self, mock_get):
        """Completed match returns MatchResult with scores."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "id": "abc123",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "commence_time": "2026-03-12T18:30:00Z",
                "completed": True,
                "scores": [
                    {"name": "Bayern Munich", "score": "3"},
                    {"name": "Borussia Dortmund", "score": "1"},
                ],
            }
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        # Inject configured keys so sport lookup works
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        match = MagicMock()
        match.sport = Sport.FOOTBALL
        match.home_team = "Bayern Munich"
        match.away_team = "Borussia Dortmund"

        result = backend.fetch_result(match)

        assert result is not None
        assert result.home_score == 3
        assert result.away_score == 1
        assert result.is_finished is True

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_fetch_result_not_completed(self, mock_get):
        """In-progress match returns None (no completed flag)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "abc123",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "completed": False,
                "scores": [
                    {"name": "Bayern Munich", "score": "1"},
                    {"name": "Borussia Dortmund", "score": "0"},
                ],
            }
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        match = MagicMock()
        match.sport = Sport.FOOTBALL
        match.home_team = "Bayern Munich"
        match.away_team = "Borussia Dortmund"

        result = backend.fetch_result(match)
        # Non-completed match with scores still returns result (for live tracking)
        assert result is not None
        assert result.is_finished is False

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_fetch_result_no_match_found(self, mock_get):
        """No matching event returns None."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "abc123",
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "completed": True,
                "scores": [
                    {"name": "Real Madrid", "score": "2"},
                    {"name": "Barcelona", "score": "1"},
                ],
            }
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        match = MagicMock()
        match.sport = Sport.FOOTBALL
        match.home_team = "Bayern Munich"
        match.away_team = "Borussia Dortmund"

        result = backend.fetch_result(match)
        assert result is None

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_caches_api_responses(self, mock_get):
        """Second call for same sport key uses cache."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        match = MagicMock()
        match.sport = Sport.FOOTBALL
        match.home_team = "Team A"
        match.away_team = "Team B"

        backend.fetch_result(match)
        backend.fetch_result(match)

        # Only one API call despite two fetch_result calls
        assert mock_get.call_count == 1

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_fuzzy_team_name_matching(self, mock_get):
        """Fuzzy matching resolves API vs DB name differences."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "abc123",
                "home_team": "FC Bayern Munich",
                "away_team": "BVB Dortmund",
                "completed": True,
                "scores": [
                    {"name": "FC Bayern Munich", "score": "2"},
                    {"name": "BVB Dortmund", "score": "0"},
                ],
            }
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        match = MagicMock()
        match.sport = Sport.FOOTBALL
        match.home_team = "Bayern Munich"  # DB has no "FC" prefix
        match.away_team = "Borussia Dortmund"  # DB has full name

        result = backend.fetch_result(match)
        assert result is not None
        assert result.home_score == 2
        assert result.away_score == 0


# ── Default backend selection tests ─────────────────────────────────


class TestDefaultBackend:
    @patch.dict("os.environ", {"THE_ODDS_API_KEY": "test-key-123"})
    def test_uses_odds_api_when_key_set(self):
        backend = _get_default_backend()
        assert isinstance(backend, TheOddsAPIResultsBackend)

    @patch.dict("os.environ", {"THE_ODDS_API_KEY": ""})
    def test_falls_back_to_manual_when_no_key(self):
        backend = _get_default_backend()
        assert isinstance(backend, ManualResultsBackend)

    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_to_manual_when_env_missing(self):
        backend = _get_default_backend()
        assert isinstance(backend, ManualResultsBackend)


# ── Integration: fetch_and_update_results with real DB ──────────────


class TestFetchAndUpdateIntegration:
    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_end_to_end_updates_match_state(self, mock_get, db_session):
        """Full pipeline: API returns scores → match becomes FINISHED."""
        match = _add_match(db_session)
        _add_bet(db_session, match)

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "ev123",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "completed": True,
                "scores": [
                    {"name": "Bayern Munich", "score": "2"},
                    {"name": "Borussia Dortmund", "score": "1"},
                ],
            }
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        # before_date = tomorrow so today's match is included
        result = fetch_and_update_results(
            db_session, backend, before_date=date(2026, 3, 14),
        )

        assert result.matches_checked == 1
        assert result.matches_updated == 1

        db_session.refresh(match)
        assert match.match_state == MatchState.FINISHED
        assert match.home_score == 2
        assert match.away_score == 1
