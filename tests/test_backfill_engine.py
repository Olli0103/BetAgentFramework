"""Tests for the Backfill Engine — multi-source data gap filler."""

import uuid
from datetime import date, datetime, timedelta, timezone
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
from bet_agent.tools.backfill_engine import (
    BackfillResult,
    ReconcileResult,
    _odds_league_from_key,
    _odds_sport_from_key,
    _seed_matches_from_odds_api,
    _update_feature_coverage,
    _update_provenance,
    backfill_day,
    backfill_open_gaps,
    reconcile_open_results,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make_match(
    sport=Sport.FOOTBALL,
    league="Bundesliga",
    home="Bayern Munich",
    away="Borussia Dortmund",
    state=MatchState.NOT_STARTED,
    home_score=None,
    away_score=None,
    live_stats=None,
    scheduled_at=None,
) -> Match:
    if scheduled_at is None:
        scheduled_at = datetime.now(timezone.utc) - timedelta(hours=3)
    return Match(
        sport=sport,
        league=league,
        home_team=home,
        away_team=away,
        scheduled_at=scheduled_at,
        match_state=state,
        home_score=home_score,
        away_score=away_score,
        live_stats=live_stats,
    )


# ── BackfillResult Tests ─────────────────────────────────────────────────


class TestBackfillResult:
    def test_total_filled(self):
        r = BackfillResult(target_date="2026-03-13")
        assert r.total_filled == 0

        r.scores_filled = 5
        r.stats_filled = 3
        r.features_filled = 2
        assert r.total_filled == 10

    def test_to_dict(self):
        r = BackfillResult(
            target_date="2026-03-13",
            scores_filled=5,
            errors=["error1"],
        )
        d = r.to_dict()
        assert d["target_date"] == "2026-03-13"
        assert d["scores_filled"] == 5
        assert d["errors"] == ["error1"]


class TestReconcileResult:
    def test_to_dict(self):
        r = ReconcileResult(
            matches_checked=10,
            matches_resolved=7,
            still_unresolved=3,
            sources_used=["the_odds_api"],
        )
        d = r.to_dict()
        assert d["matches_checked"] == 10
        assert d["still_unresolved"] == 3


# ── Provenance Tests ─────────────────────────────────────────────────────


class TestProvenance:
    def test_update_provenance_new(self, db_session):
        m = _make_match(live_stats=None)
        db_session.add(m)
        db_session.flush()

        _update_provenance(m, "api_sports")
        assert "api_sports" in m.live_stats["data_sources"]
        assert "last_stats_update" in m.live_stats

    def test_update_provenance_append(self, db_session):
        m = _make_match(live_stats={"data_sources": ["the_odds_api"]})
        db_session.add(m)
        db_session.flush()

        _update_provenance(m, "api_sports")
        assert "api_sports" in m.live_stats["data_sources"]
        assert "the_odds_api" in m.live_stats["data_sources"]
        assert len(m.live_stats["data_sources"]) == 2

    def test_update_provenance_no_duplicate(self, db_session):
        m = _make_match(live_stats={"data_sources": ["api_sports"]})
        db_session.add(m)
        db_session.flush()

        _update_provenance(m, "api_sports")
        assert m.live_stats["data_sources"].count("api_sports") == 1

    def test_update_feature_coverage(self, db_session):
        m = _make_match(
            live_stats={
                "home_score": 2, "away_score": 1,
                "home_shots": 15, "away_shots": 8,
                "home_sot": 7, "away_sot": 3,
                "home_corners": 6, "away_corners": 4,
            }
        )
        db_session.add(m)
        db_session.flush()

        _update_feature_coverage(m, "football")
        assert "feature_coverage" in m.live_stats
        assert 0.0 <= m.live_stats["feature_coverage"] <= 1.0


# ── Backfill Day Tests ───────────────────────────────────────────────────


class TestBackfillDay:
    def test_no_gaps_no_backfill(self, db_session):
        """No matches = no backfill needed."""
        result = backfill_day(db_session, date.today())
        assert result.total_filled == 0

    def test_backfill_with_no_api_keys(self, db_session, monkeypatch):
        """Without API keys, backfill runs but fills nothing."""
        monkeypatch.delenv("API_SPORTS_KEY", raising=False)
        monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)

        m = _make_match()
        db_session.add(m)
        db_session.flush()

        result = backfill_day(db_session, date.today())
        # Should not crash, just report 0 fills
        assert isinstance(result, BackfillResult)

    def test_backfill_result_structure(self):
        r = BackfillResult(target_date="2026-03-13")
        d = r.to_dict()
        assert "api_sports_calls" in d
        assert "odds_api_calls" in d
        assert "cloudflare_calls" in d


# ── Open Gaps and Reconcile Tests ────────────────────────────────────────


class TestBackfillOpenGaps:
    def test_open_gaps_empty_db(self, db_session):
        result = backfill_open_gaps(db_session, max_days_back=3)
        assert result.total_filled == 0

    def test_open_gaps_aggregates_days(self, db_session, monkeypatch):
        monkeypatch.delenv("API_SPORTS_KEY", raising=False)
        monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)

        result = backfill_open_gaps(db_session, max_days_back=3)
        assert isinstance(result, BackfillResult)
        assert result.target_date == "last_3_days"


class TestReconcileResults:
    def test_reconcile_empty_db(self, db_session):
        result = reconcile_open_results(db_session)
        assert result.matches_checked == 0
        assert result.matches_resolved == 0

    def test_reconcile_with_unresolved_matches(self, db_session, monkeypatch):
        monkeypatch.delenv("API_SPORTS_KEY", raising=False)
        monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)

        # Add a match that's past scheduled but not finished
        m = _make_match(
            scheduled_at=datetime.now(timezone.utc) - timedelta(hours=12),
        )
        db_session.add(m)
        db_session.flush()

        result = reconcile_open_results(db_session)
        assert result.matches_checked == 1
        # Without API keys, nothing gets resolved
        assert result.still_unresolved == 1

    def test_reconcile_result_structure(self):
        r = ReconcileResult(
            matches_checked=5,
            matches_resolved=3,
            still_unresolved=2,
            sources_used=["the_odds_api", "api_sports"],
        )
        d = r.to_dict()
        assert d["matches_checked"] == 5
        assert len(d["sources_used"]) == 2


# ── Fixture Seeding Tests ────────────────────────────────────────────────


class TestOddsSportFromKey:
    def test_football(self):
        assert _odds_sport_from_key("soccer_germany_bundesliga") == Sport.FOOTBALL

    def test_tennis(self):
        assert _odds_sport_from_key("tennis_atp_french_open") == Sport.TENNIS

    def test_basketball(self):
        assert _odds_sport_from_key("basketball_nba") == Sport.BASKETBALL

    def test_unknown(self):
        assert _odds_sport_from_key("rugby_six_nations") is None


class TestOddsLeagueFromKey:
    def test_football_league(self):
        assert _odds_league_from_key("soccer_germany_bundesliga") == "Germany Bundesliga"

    def test_tennis_league(self):
        assert _odds_league_from_key("tennis_atp_french_open") == "Atp French Open"


def _mock_odds_api_response(events, sport_keys=None):
    """Create a side_effect for requests.get that returns sport keys and events."""
    if sport_keys is None:
        sport_keys = [{"key": "soccer_germany_bundesliga"}]

    def side_effect(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        if "/v4/sports" in url and "/events" not in url:
            mock_resp.json.return_value = sport_keys
        elif "/events" in url:
            mock_resp.json.return_value = events
        else:
            mock_resp.json.return_value = []
        return mock_resp

    return side_effect


class TestSeedMatchesFromOddsApi:
    def test_seeds_football_matches(self, db_session):
        """Seeds new football matches from OddsAPI events."""
        target = date(2026, 3, 13)
        commence = datetime(2026, 3, 13, 15, 30, tzinfo=timezone.utc).isoformat()

        events = [
            {
                "id": "evt1",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "commence_time": commence,
            },
            {
                "id": "evt2",
                "home_team": "RB Leipzig",
                "away_team": "VfB Stuttgart",
                "commence_time": commence,
            },
        ]

        result = BackfillResult(target_date=target.isoformat())
        with patch("bet_agent.tools.backfill_engine.requests.get",
                   side_effect=_mock_odds_api_response(events)):
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                _seed_matches_from_odds_api(db_session, target, None, result)
                db_session.commit()

        assert result.matches_seeded == 2
        matches = db_session.query(Match).all()
        assert len(matches) == 2
        assert all(m.sport == Sport.FOOTBALL for m in matches)
        assert all(m.match_state == MatchState.NOT_STARTED for m in matches)
        assert all(m.source == "the_odds_api" for m in matches)
        # Check provenance
        for m in matches:
            assert m.live_stats["odds_event_id"] in ("evt1", "evt2")
            assert m.live_stats["odds_sport_key"] == "soccer_germany_bundesliga"
            assert "the_odds_api" in m.live_stats["data_sources"]

    def test_idempotent_no_duplicates(self, db_session):
        """Re-running seeding does not create duplicates."""
        target = date(2026, 3, 13)
        commence = datetime(2026, 3, 13, 18, 0, tzinfo=timezone.utc).isoformat()

        # Pre-existing match
        existing = Match(
            sport=Sport.FOOTBALL,
            league="Germany Bundesliga",
            home_team="Bayern Munich",
            away_team="Borussia Dortmund",
            scheduled_at=datetime(2026, 3, 13, 18, 0, tzinfo=timezone.utc),
            match_state=MatchState.NOT_STARTED,
            source="the_odds_api",
        )
        db_session.add(existing)
        db_session.commit()

        events = [
            {
                "id": "evt1",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "commence_time": commence,
            },
        ]

        result = BackfillResult(target_date=target.isoformat())
        with patch("bet_agent.tools.backfill_engine.requests.get",
                   side_effect=_mock_odds_api_response(events)):
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                _seed_matches_from_odds_api(db_session, target, None, result)
                db_session.commit()

        assert result.matches_seeded == 0
        assert db_session.query(Match).count() == 1

    def test_seeds_tennis_matches(self, db_session):
        """Seeds tennis matches (the key missing sport)."""
        target = date(2026, 3, 13)
        commence = datetime(2026, 3, 13, 12, 0, tzinfo=timezone.utc).isoformat()

        events = [
            {
                "id": "tennis1",
                "home_team": "Jannik Sinner",
                "away_team": "Carlos Alcaraz",
                "commence_time": commence,
            },
        ]
        sport_keys = [{"key": "tennis_atp_indian_wells"}]

        result = BackfillResult(target_date=target.isoformat())
        with patch("bet_agent.tools.backfill_engine.requests.get",
                   side_effect=_mock_odds_api_response(events, sport_keys)):
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                _seed_matches_from_odds_api(db_session, target, None, result)
                db_session.commit()

        assert result.matches_seeded == 1
        match = db_session.query(Match).one()
        assert match.sport == Sport.TENNIS
        assert match.home_team == "Jannik Sinner"
        assert match.away_team == "Carlos Alcaraz"

    def test_filters_out_of_date_events(self, db_session):
        """Events not on target_date are skipped."""
        target = date(2026, 3, 13)
        wrong_day = datetime(2026, 3, 14, 15, 30, tzinfo=timezone.utc).isoformat()

        events = [
            {
                "id": "evt_wrong",
                "home_team": "Team A",
                "away_team": "Team B",
                "commence_time": wrong_day,
            },
        ]

        result = BackfillResult(target_date=target.isoformat())
        with patch("bet_agent.tools.backfill_engine.requests.get",
                   side_effect=_mock_odds_api_response(events)):
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                _seed_matches_from_odds_api(db_session, target, None, result)

        assert result.matches_seeded == 0
        assert db_session.query(Match).count() == 0

    def test_no_api_key_skips(self, db_session):
        """Without THE_ODDS_API_KEY, seeding is a no-op."""
        result = BackfillResult(target_date="2026-03-13")
        with patch.dict("os.environ", {}, clear=True):
            _seed_matches_from_odds_api(db_session, date(2026, 3, 13), None, result)

        assert result.matches_seeded == 0
        assert result.odds_api_calls == 0

    def test_sport_filter(self, db_session):
        """Sport filter restricts which sports get seeded."""
        target = date(2026, 3, 13)
        commence = datetime(2026, 3, 13, 15, 0, tzinfo=timezone.utc).isoformat()

        events = [
            {
                "id": "evt1",
                "home_team": "Team A",
                "away_team": "Team B",
                "commence_time": commence,
            },
        ]
        sport_keys = [
            {"key": "soccer_germany_bundesliga"},
            {"key": "tennis_atp_indian_wells"},
        ]

        result = BackfillResult(target_date=target.isoformat())
        with patch("bet_agent.tools.backfill_engine.requests.get",
                   side_effect=_mock_odds_api_response(events, sport_keys)):
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                _seed_matches_from_odds_api(db_session, target, ["tennis"], result)
                db_session.commit()

        # Only tennis should be seeded
        assert result.matches_seeded == 1
        match = db_session.query(Match).one()
        assert match.sport == Sport.TENNIS

    def test_max_keys_env_guard(self, db_session):
        """BETAGENT_ODDS_SEED_MAX_KEYS limits how many sport keys are queried."""
        target = date(2026, 3, 13)
        commence = datetime(2026, 3, 13, 15, 0, tzinfo=timezone.utc).isoformat()

        events = [
            {
                "id": "evt1",
                "home_team": "Team A",
                "away_team": "Team B",
                "commence_time": commence,
            },
        ]
        # 3 sport keys, but limit to 1
        sport_keys = [
            {"key": "soccer_germany_bundesliga"},
            {"key": "basketball_nba"},
            {"key": "tennis_atp_indian_wells"},
        ]

        result = BackfillResult(target_date=target.isoformat())
        with patch("bet_agent.tools.backfill_engine.requests.get",
                   side_effect=_mock_odds_api_response(events, sport_keys)):
            with patch.dict("os.environ", {
                "THE_ODDS_API_KEY": "test_key",
                "BETAGENT_ODDS_SEED_MAX_KEYS": "1",
            }):
                _seed_matches_from_odds_api(db_session, target, None, result)
                db_session.commit()

        # Only 1 sport key queried → 1 match
        assert result.matches_seeded == 1
        # 1 call for sport discovery + 1 for the single key
        assert result.odds_api_calls == 2

    def test_binds_odds_event_id_on_existing(self, db_session):
        """Existing matches get odds_event_id bound on re-seed."""
        target = date(2026, 3, 13)
        sched = datetime(2026, 3, 13, 18, 0, tzinfo=timezone.utc)

        existing = Match(
            sport=Sport.FOOTBALL,
            league="Germany Bundesliga",
            home_team="Bayern Munich",
            away_team="Borussia Dortmund",
            scheduled_at=sched,
            match_state=MatchState.NOT_STARTED,
            source="manual",
            live_stats={},
        )
        db_session.add(existing)
        db_session.commit()

        events = [
            {
                "id": "evt_new",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "commence_time": sched.isoformat(),
            },
        ]

        result = BackfillResult(target_date=target.isoformat())
        with patch("bet_agent.tools.backfill_engine.requests.get",
                   side_effect=_mock_odds_api_response(events)):
            with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                _seed_matches_from_odds_api(db_session, target, None, result)
                db_session.commit()

        assert result.matches_seeded == 0
        match = db_session.query(Match).one()
        assert match.live_stats["odds_event_id"] == "evt_new"
        assert "the_odds_api" in match.live_stats["data_sources"]


class TestBackfillDayWithSeeding:
    """Integration: backfill_day() now seeds before gap scan."""

    def test_backfill_result_includes_seeded(self):
        """BackfillResult.to_dict() includes matches_seeded field."""
        r = BackfillResult(target_date="2026-03-13", matches_seeded=5)
        d = r.to_dict()
        assert d["matches_seeded"] == 5

    def test_window_coverage_improvement(self, db_session):
        """Simulates the repro scenario: seeding increases match coverage."""
        target = date(2026, 3, 13)

        # Use a future date so scan_gaps doesn't trip on tz comparisons
        target = date.today() + timedelta(days=1)

        # Pre-seed only football/hockey (the "before" state)
        for i in range(5):
            db_session.add(Match(
                sport=Sport.FOOTBALL,
                league="Bundesliga",
                home_team=f"Home FC {i}",
                away_team=f"Away FC {i}",
                scheduled_at=datetime.combine(
                    target, datetime.min.time().replace(hour=15 + i),
                    tzinfo=timezone.utc,
                ),
                match_state=MatchState.NOT_STARTED,
            ))
        for i in range(2):
            db_session.add(Match(
                sport=Sport.ICE_HOCKEY,
                league="NHL",
                home_team=f"Home HC {i}",
                away_team=f"Away HC {i}",
                scheduled_at=datetime.combine(
                    target, datetime.min.time().replace(hour=19 + i),
                    tzinfo=timezone.utc,
                ),
                match_state=MatchState.NOT_STARTED,
            ))
        db_session.commit()

        before_count = db_session.query(Match).count()
        assert before_count == 7  # football=5, hockey=2

        # Mock OddsAPI to return basketball + tennis events
        sport_keys = [
            {"key": "basketball_nba"},
            {"key": "tennis_atp_indian_wells"},
        ]

        def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            if "/v4/sports" in url and "/events" not in url:
                mock_resp.json.return_value = sport_keys
            elif "basketball_nba" in url:
                mock_resp.json.return_value = [
                    {
                        "id": f"nba{i}",
                        "home_team": f"NBA Home {i}",
                        "away_team": f"NBA Away {i}",
                        "commence_time": datetime.combine(
                            target, datetime.min.time().replace(hour=10 + i),
                            tzinfo=timezone.utc,
                        ).isoformat(),
                    }
                    for i in range(8)
                ]
            elif "tennis_atp" in url:
                mock_resp.json.return_value = [
                    {
                        "id": "atp1",
                        "home_team": "Jannik Sinner",
                        "away_team": "Carlos Alcaraz",
                        "commence_time": datetime.combine(
                            target, datetime.min.time().replace(hour=11),
                            tzinfo=timezone.utc,
                        ).isoformat(),
                    },
                    {
                        "id": "atp2",
                        "home_team": "Alexander Zverev",
                        "away_team": "Daniil Medvedev",
                        "commence_time": datetime.combine(
                            target, datetime.min.time().replace(hour=14),
                            tzinfo=timezone.utc,
                        ).isoformat(),
                    },
                ]
            else:
                mock_resp.json.return_value = []
            return mock_resp

        # Patch scan_gaps since SQLite doesn't store tz info (pre-existing compat issue)
        from bet_agent.tools.coverage_engine import GapReport
        empty_gap = GapReport(scan_date=target.isoformat(), sports_scanned=[])

        with patch("bet_agent.tools.backfill_engine.requests.get", side_effect=mock_get):
            with patch("bet_agent.tools.backfill_engine.scan_gaps", return_value=empty_gap):
                with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                    result = backfill_day(db_session, target)
                    db_session.commit()

        # Verify coverage improvement
        after_count = db_session.query(Match).count()
        assert after_count == 17  # 7 existing + 8 basketball + 2 tennis

        # Verify sport distribution
        basketball = db_session.query(Match).filter(Match.sport == Sport.BASKETBALL).count()
        tennis = db_session.query(Match).filter(Match.sport == Sport.TENNIS).count()
        assert basketball == 8
        assert tennis == 2

        assert result.matches_seeded == 10  # 8 + 2

        # Verify re-run is idempotent
        with patch("bet_agent.tools.backfill_engine.requests.get", side_effect=mock_get):
            with patch("bet_agent.tools.backfill_engine.scan_gaps", return_value=empty_gap):
                with patch.dict("os.environ", {"THE_ODDS_API_KEY": "test_key"}):
                    result2 = backfill_day(db_session, target)
                    db_session.commit()

        assert result2.matches_seeded == 0
        assert db_session.query(Match).count() == 17  # no duplicates
