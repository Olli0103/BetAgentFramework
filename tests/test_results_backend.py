"""Tests for TheOddsAPIResultsBackend: Unicode normalization, event binding,
dynamic sport key discovery, league hints, fuzzy token scoring,
and stale paper bet terminal fallback.
"""

import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import (
    BankrollLedger,
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
    ManualResultsBackend,
    MatchResult,
    TheOddsAPIResultsBackend,
    _fuzzy_team_match,
    _get_default_backend,
    _normalize_name,
    _name_tokens,
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
    league="Bundesliga", scheduled_at=None,
):
    m = Match(
        sport=sport,
        league=league,
        home_team=home,
        away_team=away,
        scheduled_at=scheduled_at or datetime(2026, 3, 12, 18, 30, tzinfo=timezone.utc),
        match_state=MatchState.NOT_STARTED,
    )
    session.add(m)
    session.flush()
    return m


def _add_bet(session, match, ledger=LedgerType.PAPER, status=BetStatus.PLACED):
    b = PlacedBet(
        ledger_type=ledger,
        match_id=match.id,
        market_type=MarketType.MATCH_WINNER,
        selection="home",
        odds_at_placement=Decimal("2.10"),
        stake_eur=Decimal("10.00"),
        model_prob=Decimal("0.55"),
        ev_at_placement=Decimal("0.08"),
        status=status,
        is_live_bet=False,
    )
    session.add(b)
    session.flush()
    return b


def _add_bankroll(session, ledger=LedgerType.PAPER, balance=Decimal("10000.00")):
    b = BankrollLedger(ledger_type=ledger, balance=balance)
    session.add(b)
    session.flush()
    return b


# ── Unicode normalization tests ─────────────────────────────────────


class TestNormalization:
    def test_unicode_decomposition(self):
        assert _normalize_name("München") == "munchen"

    def test_accented_chars(self):
        assert _normalize_name("Nîmes Olympique") == "nimes olympique"

    def test_strip_and_lowercase(self):
        assert _normalize_name("  FC Bayern  ") == "fc bayern"

    def test_name_tokens_significant(self):
        tokens = _name_tokens("FC Bayern München")
        assert "bayern" in tokens
        assert "munchen" in tokens
        assert "fc" not in tokens  # < 4 chars


# ── Fuzzy matching tests ────────────────────────────────────────────


class TestFuzzyTeamMatch:
    def test_exact_match(self):
        assert _fuzzy_team_match("bayern munich", "bayern munich") is True

    def test_shared_significant_token(self):
        assert _fuzzy_team_match("borussia dortmund", "bvb dortmund") is True

    def test_no_overlap(self):
        assert _fuzzy_team_match("bayern munich", "borussia dortmund") is False

    def test_short_tokens_ignored(self):
        assert _fuzzy_team_match("fc bvb", "sc bvb") is False

    def test_hornets_match(self):
        assert _fuzzy_team_match("charlotte hornets", "cha hornets") is True

    def test_unicode_fuzzy(self):
        """Unicode normalization in fuzzy matching."""
        assert _fuzzy_team_match("FC Bayern München", "Bayern Munich") is True


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
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        match = MagicMock()
        match.sport = Sport.FOOTBALL
        match.league = "Bundesliga"
        match.home_team = "Bayern Munich"
        match.away_team = "Borussia Dortmund"
        match.live_stats = None

        result = backend.fetch_result(match)

        assert result is not None
        assert result.home_score == 3
        assert result.away_score == 1
        assert result.is_finished is True

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_fetch_result_not_completed(self, mock_get):
        """In-progress match with scores returns result with is_finished=False."""
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
        match.league = "Bundesliga"
        match.home_team = "Bayern Munich"
        match.away_team = "Borussia Dortmund"
        match.live_stats = None

        result = backend.fetch_result(match)
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
        match.league = "Bundesliga"
        match.home_team = "Bayern Munich"
        match.away_team = "Borussia Dortmund"
        match.live_stats = None

        result = backend.fetch_result(match)
        assert result is None

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_caches_api_responses(self, mock_get):
        """Second call for same sport key uses cache, no duplicate API call."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        match = MagicMock()
        match.sport = Sport.FOOTBALL
        match.league = "Bundesliga"
        match.home_team = "Team A"
        match.away_team = "Team B"
        match.live_stats = None

        backend.fetch_result(match)
        backend.fetch_result(match)

        # Only one API call (scores endpoint), not two
        scores_calls = [c for c in mock_get.call_args_list if "scores" in str(c)]
        assert len(scores_calls) == 1

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
        match.league = "Bundesliga"
        match.home_team = "Bayern Munich"
        match.away_team = "Borussia Dortmund"
        match.live_stats = None

        result = backend.fetch_result(match)
        assert result is not None
        assert result.home_score == 2
        assert result.away_score == 0


# ── Event binding tests ─────────────────────────────────────────────


class TestEventBinding:
    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_binds_event_id_on_match(self, mock_get, db_session):
        """Successful score match stores odds_event_id in live_stats."""
        match = _add_match(db_session)
        _add_bet(db_session, match)

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "evt_001",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "completed": True,
                "scores": [
                    {"name": "Bayern Munich", "score": "1"},
                    {"name": "Borussia Dortmund", "score": "0"},
                ],
            }
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        fetch_and_update_results(db_session, backend, before_date=date(2026, 3, 14))

        db_session.refresh(match)
        assert match.live_stats is not None
        assert match.live_stats["odds_event_id"] == "evt_001"
        assert match.live_stats["odds_sport_key"] == "soccer_germany_bundesliga"

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_direct_lookup_by_bound_event_id(self, mock_get):
        """Match with bound odds_event_id skips name matching."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "evt_bound",
                "home_team": "Completely Different Name",
                "away_team": "Also Different",
                "completed": True,
                "scores": [
                    {"name": "Completely Different Name", "score": "4"},
                    {"name": "Also Different", "score": "2"},
                ],
            }
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        match = MagicMock()
        match.sport = Sport.FOOTBALL
        match.league = "Bundesliga"
        match.home_team = "Bayern Munich"
        match.away_team = "Borussia Dortmund"
        # Previously bound event ID
        match.live_stats = {"odds_event_id": "evt_bound", "odds_sport_key": "soccer_germany_bundesliga"}

        result = backend.fetch_result(match)
        assert result is not None
        assert result.home_score == 4
        assert result.away_score == 2


# ── League hint tests ───────────────────────────────────────────────


class TestLeagueHints:
    def test_bundesliga_hint(self):
        backend = TheOddsAPIResultsBackend(api_key="test-key")
        keys = backend._sport_keys_for(Sport.FOOTBALL, league="Bundesliga")
        assert "soccer_germany_bundesliga" in keys

    def test_nba_hint(self):
        backend = TheOddsAPIResultsBackend(api_key="test-key")
        keys = backend._sport_keys_for(Sport.BASKETBALL, league="NBA")
        assert "basketball_nba" in keys

    def test_premier_league_hint(self):
        backend = TheOddsAPIResultsBackend(api_key="test-key")
        keys = backend._sport_keys_for(Sport.FOOTBALL, league="Premier League")
        assert "soccer_epl" in keys

    def test_hint_comes_first(self):
        """League hint key should be first in the list (most relevant)."""
        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_epl", "soccer_germany_bundesliga"]}
        keys = backend._sport_keys_for(Sport.FOOTBALL, league="Bundesliga")
        assert keys[0] == "soccer_germany_bundesliga"


# ── Dynamic sport key discovery tests ───────────────────────────────


class TestDynamicDiscovery:
    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_discovers_keys_from_api(self, mock_get):
        """Falls back to /v4/sports when no configured keys match."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"key": "soccer_germany_bundesliga", "title": "Bundesliga"},
            {"key": "soccer_epl", "title": "EPL"},
            {"key": "basketball_nba", "title": "NBA"},
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {}  # empty — force discovery

        keys = backend._sport_keys_for(Sport.FOOTBALL, league="Unknown League")
        assert "soccer_germany_bundesliga" in keys
        assert "soccer_epl" in keys
        assert "basketball_nba" not in keys

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_discovery_cached_across_calls(self, mock_get):
        """Sport key discovery only calls /v4/sports once."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"key": "soccer_germany_bundesliga"},
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {}

        backend._sport_keys_for(Sport.FOOTBALL, league=None)
        backend._sport_keys_for(Sport.FOOTBALL, league=None)

        # Only 1 call to /v4/sports
        sports_calls = [c for c in mock_get.call_args_list if "sports" in str(c)]
        assert len(sports_calls) == 1


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


# ── Stale paper bet terminal fallback tests ─────────────────────────


class TestStalePaperBetFallback:
    def test_voids_stale_paper_bets(self, db_session):
        """PAPER bets on matches older than score window get voided."""
        _add_bankroll(db_session, ledger=LedgerType.PAPER, balance=Decimal("9990.00"))

        # Match scheduled 5 days ago — beyond the 4-day stale threshold
        old_match = _add_match(
            db_session, home="Player A", away="Player B",
            sport=Sport.TENNIS, league="ATP",
            scheduled_at=datetime.now(timezone.utc) - timedelta(days=5),
        )
        bet = _add_bet(db_session, old_match, ledger=LedgerType.PAPER)

        # Use ManualResultsBackend (no results) to trigger the stale fallback
        result = fetch_and_update_results(
            db_session, ManualResultsBackend(),
            before_date=date.today() + timedelta(days=1),
        )

        assert result.voided_stale_paper == 1

        db_session.refresh(bet)
        assert bet.status == BetStatus.VOID
        assert bet.pnl_eur == Decimal("10.00")  # stake refunded
        assert bet.resolved_at is not None

    def test_does_not_void_recent_paper_bets(self, db_session):
        """PAPER bets within the score window are not voided."""
        _add_bankroll(db_session, ledger=LedgerType.PAPER)

        recent_match = _add_match(
            db_session, home="Player X", away="Player Y",
            sport=Sport.TENNIS, league="ATP",
            scheduled_at=datetime.now(timezone.utc) - timedelta(days=2),
        )
        bet = _add_bet(db_session, recent_match, ledger=LedgerType.PAPER)

        result = fetch_and_update_results(
            db_session, ManualResultsBackend(),
            before_date=date.today() + timedelta(days=1),
        )

        assert result.voided_stale_paper == 0
        db_session.refresh(bet)
        assert bet.status == BetStatus.PLACED

    def test_does_not_void_real_bets(self, db_session):
        """REAL bets are never auto-voided, even if stale."""
        _add_bankroll(db_session, ledger=LedgerType.REAL, balance=Decimal("990.00"))

        old_match = _add_match(
            db_session, home="Team A", away="Team B",
            scheduled_at=datetime.now(timezone.utc) - timedelta(days=5),
        )
        bet = _add_bet(db_session, old_match, ledger=LedgerType.REAL)

        result = fetch_and_update_results(
            db_session, ManualResultsBackend(),
            before_date=date.today() + timedelta(days=1),
        )

        assert result.voided_stale_paper == 0
        db_session.refresh(bet)
        assert bet.status == BetStatus.PLACED

    def test_refunds_stake_to_paper_ledger(self, db_session):
        """Voided stale PAPER bet refunds stake to PAPER ledger."""
        ledger = _add_bankroll(db_session, ledger=LedgerType.PAPER, balance=Decimal("9990.00"))

        old_match = _add_match(
            db_session, home="Player C", away="Player D",
            sport=Sport.TENNIS, league="WTA",
            scheduled_at=datetime.now(timezone.utc) - timedelta(days=5),
        )
        _add_bet(db_session, old_match, ledger=LedgerType.PAPER)

        fetch_and_update_results(
            db_session, ManualResultsBackend(),
            before_date=date.today() + timedelta(days=1),
        )

        db_session.refresh(ledger)
        assert ledger.balance == Decimal("10000.00")  # 9990 + 10 refund


# ── Integration: full pipeline ──────────────────────────────────────


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

        result = fetch_and_update_results(
            db_session, backend, before_date=date(2026, 3, 14),
        )

        assert result.matches_checked == 1
        assert result.matches_updated == 1

        db_session.refresh(match)
        assert match.match_state == MatchState.FINISHED
        assert match.home_score == 2
        assert match.away_score == 1

    @patch("bet_agent.tools.results_fetcher.requests.get")
    def test_unicode_names_matched(self, mock_get, db_session):
        """Unicode team names (umlauts, accents) are matched correctly."""
        match = _add_match(
            db_session, home="FC Bayern München", away="1. FC Nürnberg",
            league="Bundesliga",
        )
        _add_bet(db_session, match)

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "ev_unicode",
                "home_team": "FC Bayern Munchen",  # no umlaut in API
                "away_team": "1. FC Nurnberg",
                "completed": True,
                "scores": [
                    {"name": "FC Bayern Munchen", "score": "3"},
                    {"name": "1. FC Nurnberg", "score": "0"},
                ],
            }
        ]
        mock_get.return_value = mock_resp

        backend = TheOddsAPIResultsBackend(api_key="test-key")
        backend._configured_keys = {"football": ["soccer_germany_bundesliga"]}

        result = fetch_and_update_results(
            db_session, backend, before_date=date(2026, 3, 14),
        )

        assert result.matches_updated == 1
        db_session.refresh(match)
        assert match.home_score == 3
        assert match.match_state == MatchState.FINISHED
