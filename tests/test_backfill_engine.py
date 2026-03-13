"""Tests for the Backfill Engine — multi-source data gap filler."""

import uuid
from datetime import date, datetime, timedelta, timezone

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
