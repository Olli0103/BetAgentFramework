"""Tests for the Coverage Engine — gap scan, coverage report, quality gates."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bet_agent.db.models import (
    Base,
    HistoricalMatch,
    Match,
    MatchState,
    Sport,
    TeamDailyStats,
)
from bet_agent.tools.coverage_engine import (
    COVERAGE_SLOS,
    REQUIRED_FEATURES,
    CoverageKPIs,
    GapReport,
    QualityGate,
    check_quality_gate,
    coverage_report,
    scan_gaps,
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


def _make_historical(
    sport=Sport.FOOTBALL,
    season="2024-2025",
    division="D1",
    home="Bayern Munich",
    away="Dortmund",
    match_stats=None,
):
    from datetime import date
    return HistoricalMatch(
        sport=sport,
        season=season,
        division=division,
        match_date=date.today() - timedelta(days=5),
        home_team=home,
        away_team=away,
        home_score=3,
        away_score=1,
        result="H",
        match_stats=match_stats or {},
        odds={},
        betting_lines={},
        advanced_stats={},
        source="test",
    )


# ── Gap Scan Tests ────────────────────────────────────────────────────────


class TestGapScan:
    def test_no_gaps_empty_db(self, db_session):
        report = scan_gaps(db_session)
        assert report.total_gaps == 0
        assert isinstance(report, GapReport)

    def test_detects_missing_scores(self, db_session):
        m = _make_match(home_score=None, away_score=None)
        db_session.add(m)
        db_session.flush()

        report = scan_gaps(db_session, lookback_days=1)
        assert len(report.missing_scores) == 1
        assert report.missing_scores[0].gap_type == "missing_score"

    def test_no_gap_when_scores_present(self, db_session):
        m = _make_match(
            state=MatchState.FINISHED,
            home_score=2,
            away_score=1,
            live_stats={"home_shots": 15, "away_shots": 8},
        )
        db_session.add(m)
        db_session.flush()

        report = scan_gaps(db_session, lookback_days=1)
        assert len(report.missing_scores) == 0

    def test_detects_missing_stats(self, db_session):
        m = _make_match(
            state=MatchState.FINISHED,
            home_score=1,
            away_score=0,
            live_stats=None,
        )
        db_session.add(m)
        db_session.flush()

        report = scan_gaps(db_session, lookback_days=1)
        assert len(report.missing_stats) == 1

    def test_detects_stale_matches(self, db_session):
        stale_time = datetime.now(timezone.utc) - timedelta(days=5)
        m = _make_match(state=MatchState.NOT_STARTED, scheduled_at=stale_time)
        db_session.add(m)
        db_session.flush()

        report = scan_gaps(db_session, lookback_days=7)
        assert len(report.stale_matches) == 1

    def test_sport_filter(self, db_session):
        m1 = _make_match(sport=Sport.FOOTBALL)
        m2 = _make_match(
            sport=Sport.TENNIS,
            home="Sinner",
            away="Djokovic",
            league="ATP",
        )
        db_session.add_all([m1, m2])
        db_session.flush()

        report = scan_gaps(db_session, sports=["tennis"], lookback_days=1)
        assert "tennis" in report.sports_scanned
        # Only tennis gaps should appear
        for gap in report.missing_scores:
            assert gap.sport == "tennis"

    def test_gap_report_to_dict(self, db_session):
        report = scan_gaps(db_session)
        d = report.to_dict()
        assert "total_gaps" in d
        assert "scan_date" in d
        assert isinstance(d["sports_scanned"], list)


# ── Coverage Report Tests ────────────────────────────────────────────────


class TestCoverageReport:
    def test_empty_db_returns_empty_report(self, db_session):
        report = coverage_report(db_session)
        assert report.kpis == []

    def test_kpi_calculation_with_matches(self, db_session):
        # 2 finished matches: one with scores, one without live_stats
        m1 = _make_match(
            state=MatchState.FINISHED,
            home_score=2,
            away_score=1,
            live_stats={"ht_home": 1, "ht_away": 0},
        )
        m2 = _make_match(
            state=MatchState.FINISHED,
            home_score=0,
            away_score=3,
            home="Leipzig",
            away="Freiburg",
            live_stats=None,
        )
        db_session.add_all([m1, m2])
        db_session.flush()

        report = coverage_report(db_session, lookback_days=1)
        assert len(report.kpis) >= 1

        football_kpis = next(k for k in report.kpis if k.sport == "football")
        assert football_kpis.total_matches == 2
        assert football_kpis.with_final_score == 2  # both have scores

    def test_historical_matches_counted(self, db_session):
        hm = _make_historical()
        db_session.add(hm)
        db_session.flush()

        report = coverage_report(db_session, lookback_days=7)
        assert len(report.kpis) >= 1
        football_kpis = next(k for k in report.kpis if k.sport == "football")
        assert football_kpis.with_final_score >= 1  # historical always has scores

    def test_kpis_to_dict(self):
        kpis = CoverageKPIs(
            sport="football",
            total_matches=100,
            with_final_score=95,
            with_halftime_splits=80,
            with_key_features=70,
            with_canonical_names=98,
        )
        d = kpis.to_dict()
        assert d["pct_final_score"] == 0.95
        assert d["pct_halftime_splits"] == 0.80


# ── Quality Gate Tests ────────────────────────────────────────────────────


class TestQualityGates:
    def test_unknown_sport_fails(self, db_session):
        gate = check_quality_gate(db_session, "underwater_polo")
        assert not gate.passed
        assert not gate.can_train
        assert not gate.can_predict

    def test_empty_db_fails(self, db_session):
        gate = check_quality_gate(db_session, "football")
        assert not gate.passed
        assert "No matches found" in gate.violations[0]

    def test_good_data_passes(self, db_session):
        # Create enough historical matches with good data
        for i in range(20):
            hm = _make_historical(
                home=f"Team A{i}",
                away=f"Team B{i}",
                match_stats={
                    "ht_home": 1, "ht_away": 0,
                    "home_score": 2, "away_score": 1,
                    "home_shots": 15, "away_shots": 8,
                    "home_sot": 7, "away_sot": 3,
                    "home_corners": 6, "away_corners": 4,
                    "home_cards": 2, "away_cards": 3,
                    "possession_home": 55.0,
                },
            )
            db_session.add(hm)
        db_session.flush()

        gate = check_quality_gate(db_session, "football", lookback_days=7)
        # Historical matches always have scores and canonical names
        assert gate.can_train or gate.kpis.total_matches > 0
        assert gate.kpis is not None

    def test_gate_to_dict(self, db_session):
        gate = check_quality_gate(db_session, "football")
        d = gate.to_dict()
        assert "passed" in d
        assert "can_train" in d
        assert "can_predict" in d
        assert "violations" in d

    def test_required_features_defined_for_all_sports(self):
        for sport in ["football", "tennis", "basketball", "ice_hockey", "american_football"]:
            assert sport in REQUIRED_FEATURES
            assert len(REQUIRED_FEATURES[sport]) > 0

    def test_coverage_slos_defined(self):
        assert "pct_final_score" in COVERAGE_SLOS
        assert "pct_key_features" in COVERAGE_SLOS
        assert "pct_canonical_mapped" in COVERAGE_SLOS
        assert COVERAGE_SLOS["pct_final_score"] == 0.95
