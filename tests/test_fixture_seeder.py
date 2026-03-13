"""Tests for the Fixture Seeder — fixture creation for operational windows."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bet_agent.db.models import (
    Base,
    Match,
    MatchState,
    Sport,
)
from bet_agent.tools.fixture_seeder import (
    SeedResult,
    _league_from_odds_key,
    _parse_odds_api_event,
    _sport_from_odds_key,
    seed_fixtures_for_window,
    seed_today_window,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


# ── Unit tests for helpers ───────────────────────────────────────────────


class TestSportFromOddsKey:
    def test_football(self):
        assert _sport_from_odds_key("soccer_germany_bundesliga") == Sport.FOOTBALL

    def test_tennis(self):
        assert _sport_from_odds_key("tennis_atp_french_open") == Sport.TENNIS

    def test_basketball(self):
        assert _sport_from_odds_key("basketball_nba") == Sport.BASKETBALL

    def test_ice_hockey(self):
        assert _sport_from_odds_key("icehockey_nhl") == Sport.ICE_HOCKEY

    def test_unknown(self):
        assert _sport_from_odds_key("rugby_six_nations") is None


class TestLeagueFromOddsKey:
    def test_football_league(self):
        assert _league_from_odds_key("soccer_germany_bundesliga") == "Germany Bundesliga"

    def test_tennis_league(self):
        assert _league_from_odds_key("tennis_atp_french_open") == "Atp French Open"

    def test_unknown_key(self):
        result = _league_from_odds_key("rugby_six_nations")
        assert result == "Rugby Six Nations"


class TestParseOddsApiEvent:
    def test_valid_event(self):
        event = {
            "id": "abc123",
            "home_team": "Bayern Munich",
            "away_team": "Borussia Dortmund",
            "commence_time": "2026-03-13T15:30:00Z",
        }
        result = _parse_odds_api_event(event, "soccer_germany_bundesliga")
        assert result is not None
        assert result["sport"] == Sport.FOOTBALL
        assert result["home_team"] == "Bayern Munich"
        assert result["away_team"] == "Borussia Dortmund"
        assert result["source"] == "the_odds_api"
        assert result["odds_event_id"] == "abc123"

    def test_missing_home_team(self):
        event = {
            "id": "abc123",
            "home_team": "",
            "away_team": "Dortmund",
            "commence_time": "2026-03-13T15:30:00Z",
        }
        assert _parse_odds_api_event(event, "soccer_germany_bundesliga") is None

    def test_unsupported_sport_key(self):
        event = {
            "id": "abc123",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2026-03-13T15:30:00Z",
        }
        assert _parse_odds_api_event(event, "rugby_six_nations") is None


# ── SeedResult tests ─────────────────────────────────────────────────────


class TestSeedResult:
    def test_to_dict(self):
        r = SeedResult(
            window_start="2026-03-13T07:00:00+00:00",
            window_end="2026-03-14T07:00:00+00:00",
            fixtures_fetched=10,
            fixtures_inserted=7,
            fixtures_skipped=3,
        )
        d = r.to_dict()
        assert d["fixtures_inserted"] == 7
        assert d["fixtures_skipped"] == 3

    def test_errors_truncated(self):
        r = SeedResult(
            window_start="a", window_end="b",
            errors=[f"err{i}" for i in range(50)],
        )
        assert len(r.to_dict()["errors"]) == 20


# ── Integration tests ────────────────────────────────────────────────────


class TestSeedFixturesForWindow:
    def _make_window(self):
        now = datetime.now(timezone.utc)
        start = now.replace(hour=7, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end

    @patch("bet_agent.tools.fixture_seeder._fetch_api_sports_fixtures")
    @patch("bet_agent.tools.fixture_seeder._fetch_odds_api_events")
    @patch("bet_agent.tools.fixture_seeder._fetch_odds_api_sport_keys")
    def test_seeds_from_odds_api(
        self, mock_keys, mock_events, mock_api_sports, db_session,
    ):
        start, end = self._make_window()
        mid = start + timedelta(hours=6)

        mock_keys.return_value = ["soccer_germany_bundesliga"]
        mock_events.return_value = [
            {
                "id": "evt1",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "commence_time": mid.isoformat(),
            },
            {
                "id": "evt2",
                "home_team": "RB Leipzig",
                "away_team": "VfB Stuttgart",
                "commence_time": mid.isoformat(),
            },
        ]
        mock_api_sports.return_value = []

        with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
            result = seed_fixtures_for_window(db_session, start, end)
            db_session.commit()

        assert result.fixtures_inserted == 2
        assert result.fixtures_skipped == 0
        assert result.sport_counts.get("football", 0) == 2
        assert "the_odds_api" in result.sources_used

        # Verify DB rows
        matches = db_session.query(Match).all()
        assert len(matches) == 2
        assert all(m.sport == Sport.FOOTBALL for m in matches)
        assert all(m.match_state == MatchState.NOT_STARTED for m in matches)
        assert all(m.source == "the_odds_api" for m in matches)

    @patch("bet_agent.tools.fixture_seeder._fetch_api_sports_fixtures")
    @patch("bet_agent.tools.fixture_seeder._fetch_odds_api_sport_keys")
    def test_idempotent_no_duplicates(
        self, mock_keys, mock_api_sports, db_session,
    ):
        """Re-running seeding should not create duplicates."""
        start, end = self._make_window()
        mid = start + timedelta(hours=6)

        # Pre-existing match in DB
        existing = Match(
            sport=Sport.FOOTBALL,
            league="Germany Bundesliga",
            home_team="Bayern Munich",
            away_team="Borussia Dortmund",
            scheduled_at=mid,
            match_state=MatchState.NOT_STARTED,
            source="the_odds_api",
        )
        db_session.add(existing)
        db_session.commit()

        mock_keys.return_value = ["soccer_germany_bundesliga"]
        mock_api_sports.return_value = []

        with patch(
            "bet_agent.tools.fixture_seeder._fetch_odds_api_events"
        ) as mock_events:
            mock_events.return_value = [
                {
                    "id": "evt1",
                    "home_team": "Bayern Munich",
                    "away_team": "Borussia Dortmund",
                    "commence_time": mid.isoformat(),
                },
            ]
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                result = seed_fixtures_for_window(db_session, start, end)
                db_session.commit()

        assert result.fixtures_skipped == 1
        assert result.fixtures_inserted == 0

        # Still only 1 match in DB
        assert db_session.query(Match).count() == 1

    @patch("bet_agent.tools.fixture_seeder._fetch_api_sports_fixtures")
    @patch("bet_agent.tools.fixture_seeder._fetch_odds_api_sport_keys")
    def test_filters_out_of_window(
        self, mock_keys, mock_api_sports, db_session,
    ):
        """Events outside the window should not be seeded."""
        start, end = self._make_window()
        outside = end + timedelta(hours=2)  # past window end

        mock_keys.return_value = ["soccer_germany_bundesliga"]
        mock_api_sports.return_value = []

        with patch(
            "bet_agent.tools.fixture_seeder._fetch_odds_api_events"
        ) as mock_events:
            mock_events.return_value = [
                {
                    "id": "evt_outside",
                    "home_team": "Bayern Munich",
                    "away_team": "Dortmund",
                    "commence_time": outside.isoformat(),
                },
            ]
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                result = seed_fixtures_for_window(db_session, start, end)

        assert result.fixtures_fetched == 0
        assert result.fixtures_inserted == 0

    @patch("bet_agent.tools.fixture_seeder._fetch_api_sports_fixtures")
    @patch("bet_agent.tools.fixture_seeder._fetch_odds_api_sport_keys")
    def test_tennis_seeding(
        self, mock_keys, mock_api_sports, db_session,
    ):
        """Tennis fixtures should be seeded via OddsAPI."""
        start, end = self._make_window()
        mid = start + timedelta(hours=4)

        mock_keys.return_value = ["tennis_atp_french_open"]
        mock_api_sports.return_value = []

        with patch(
            "bet_agent.tools.fixture_seeder._fetch_odds_api_events"
        ) as mock_events:
            mock_events.return_value = [
                {
                    "id": "tennis1",
                    "home_team": "Jannik Sinner",
                    "away_team": "Carlos Alcaraz",
                    "commence_time": mid.isoformat(),
                },
            ]
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                result = seed_fixtures_for_window(db_session, start, end)
                db_session.commit()

        assert result.fixtures_inserted == 1
        assert result.sport_counts.get("tennis", 0) == 1

        match = db_session.query(Match).one()
        assert match.sport == Sport.TENNIS
        assert match.home_team == "Jannik Sinner"
        assert match.away_team == "Carlos Alcaraz"
        assert match.live_stats["odds_event_id"] == "tennis1"

    @patch("bet_agent.tools.fixture_seeder._fetch_api_sports_fixtures")
    def test_no_api_keys_returns_empty(self, mock_api_sports, db_session):
        """Without API keys, seeding should return empty result."""
        start, end = self._make_window()
        mock_api_sports.return_value = []

        with patch.dict("os.environ", {}, clear=True):
            result = seed_fixtures_for_window(db_session, start, end)

        assert result.fixtures_fetched == 0
        assert result.fixtures_inserted == 0

    @patch("bet_agent.tools.fixture_seeder._fetch_api_sports_fixtures")
    @patch("bet_agent.tools.fixture_seeder._fetch_odds_api_sport_keys")
    def test_sport_filter(
        self, mock_keys, mock_api_sports, db_session,
    ):
        """Sport filter should restrict which sports get seeded."""
        start, end = self._make_window()
        mid = start + timedelta(hours=6)

        mock_keys.return_value = [
            "soccer_germany_bundesliga",
            "tennis_atp_french_open",
        ]
        mock_api_sports.return_value = []

        def fake_events(api_key, sport_key):
            return [
                {
                    "id": f"evt_{sport_key}",
                    "home_team": "Team A",
                    "away_team": "Team B",
                    "commence_time": mid.isoformat(),
                },
            ]

        with patch(
            "bet_agent.tools.fixture_seeder._fetch_odds_api_events",
            side_effect=fake_events,
        ):
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                result = seed_fixtures_for_window(
                    db_session, start, end, sports=["tennis"],
                )
                db_session.commit()

        # Only tennis should be seeded
        assert result.fixtures_inserted == 1
        assert "tennis" in result.sport_counts
        assert "football" not in result.sport_counts
