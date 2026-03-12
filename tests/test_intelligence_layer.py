"""Tests for Phase 2: Intelligence Layer.

Covers:
  - Historical Importer (alias resolution, match population, daily stats)
  - Feature Factory (PiT feature generation, training dataset)
  - ML Trainer (XGBoost training, inference, model artifacts)
  - Updated EV Calculator (ML-powered inference)
"""

import tempfile
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import (
    Base,
    HistoricalMatch,
    Match,
    MatchState,
    Sport,
    TeamAlias,
    TeamDailyStats,
)


@pytest.fixture
def db_session():
    """In-memory SQLite for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _add_historical_match(session, **overrides) -> HistoricalMatch:
    """Helper to add a HistoricalMatch with defaults."""
    defaults = dict(
        sport=Sport.FOOTBALL,
        season="2024-25",
        division="Premier League",
        match_date=date(2025, 1, 15),
        home_team="Arsenal",
        away_team="Chelsea",
        home_score=2,
        away_score=1,
        result="H",
        match_stats={"home_shots": 15, "away_shots": 8},
        odds={"b365_home": 1.90},
        betting_lines={},
        advanced_stats={},
        source="test",
    )
    defaults.update(overrides)
    hm = HistoricalMatch(**defaults)
    session.add(hm)
    session.flush()
    return hm


# ── Alias Resolver ───────────────────────────────────────────────────


class TestAliasResolver:
    def test_resolve_known_alias(self, db_session):
        from bet_agent.tools.historical_importer import AliasResolver

        db_session.add(TeamAlias(canonical_name="Arsenal", alias="Arsenal FC", source="test"))
        db_session.flush()

        resolver = AliasResolver(db_session)
        assert resolver.resolve("Arsenal FC") == "Arsenal"
        assert resolver.resolve("arsenal fc") == "Arsenal"  # case-insensitive

    def test_resolve_unknown_passthrough(self, db_session):
        from bet_agent.tools.historical_importer import AliasResolver

        resolver = AliasResolver(db_session)
        assert resolver.resolve("Unknown Team") == "Unknown Team"

    def test_add_alias(self, db_session):
        from bet_agent.tools.historical_importer import AliasResolver

        resolver = AliasResolver(db_session)
        resolver.add_alias("Bayern München", "FC Bayern", "test")
        assert resolver.resolve("FC Bayern") == "Bayern München"

    def test_add_alias_idempotent(self, db_session):
        from bet_agent.tools.historical_importer import AliasResolver

        db_session.add(TeamAlias(canonical_name="Arsenal", alias="Gunners", source="test"))
        db_session.flush()

        resolver = AliasResolver(db_session)
        resolver.add_alias("Arsenal", "Gunners", "test")  # should not raise
        assert resolver.resolve("Gunners") == "Arsenal"


# ── Historical Importer ─────────────────────────────────────────────


class TestHistoricalImporter:
    def test_import_creates_match_records(self, db_session):
        from bet_agent.tools.historical_importer import import_historical_to_matches

        _add_historical_match(db_session)
        _add_historical_match(
            db_session,
            match_date=date(2025, 1, 20),
            home_team="Liverpool",
            away_team="Man Utd",
            home_score=3,
            away_score=0,
        )

        count = import_historical_to_matches(db_session, sport=Sport.FOOTBALL)
        db_session.flush()
        assert count == 2
        assert db_session.query(Match).count() == 2

    def test_import_resolves_aliases(self, db_session):
        from bet_agent.tools.historical_importer import AliasResolver, import_historical_to_matches

        db_session.add(TeamAlias(canonical_name="Arsenal", alias="Arsenal FC", source="test"))
        db_session.flush()

        _add_historical_match(db_session, home_team="Arsenal FC")

        resolver = AliasResolver(db_session)
        count = import_historical_to_matches(db_session, resolver=resolver)
        db_session.flush()

        match = db_session.query(Match).one()
        assert match.home_team == "Arsenal"  # resolved

    def test_import_idempotent(self, db_session):
        from bet_agent.tools.historical_importer import import_historical_to_matches

        _add_historical_match(db_session)

        import_historical_to_matches(db_session)
        db_session.flush()
        assert db_session.query(Match).count() == 1

        # Second import should update, not duplicate
        import_historical_to_matches(db_session)
        db_session.flush()
        assert db_session.query(Match).count() == 1

    def test_match_state_is_finished(self, db_session):
        from bet_agent.tools.historical_importer import import_historical_to_matches

        _add_historical_match(db_session)
        import_historical_to_matches(db_session)
        db_session.flush()

        match = db_session.query(Match).one()
        assert match.match_state == MatchState.FINISHED
        assert match.home_score == 2
        assert match.away_score == 1


class TestDailyStatsBuilder:
    def test_builds_rolling_stats(self, db_session):
        from bet_agent.tools.historical_importer import build_historical_daily_stats

        # Create 6 matches for Arsenal (need > 5 for rolling averages)
        for i in range(6):
            _add_historical_match(
                db_session,
                match_date=date(2025, 1, 10 + i),
                home_team="Arsenal",
                away_team=f"Team{i}",
                home_score=2,
                away_score=i % 2,
            )

        count = build_historical_daily_stats(db_session, Sport.FOOTBALL)
        db_session.flush()
        assert count > 0
        assert db_session.query(TeamDailyStats).count() > 0

    def test_point_in_time_no_future_leakage(self, db_session):
        from bet_agent.tools.historical_importer import build_historical_daily_stats

        # Match on Jan 10
        _add_historical_match(db_session, match_date=date(2025, 1, 10))
        # Match on Jan 20
        _add_historical_match(
            db_session,
            match_date=date(2025, 1, 20),
            home_team="Arsenal",
            away_team="Liverpool",
            home_score=1,
            away_score=1,
            result="D",
        )

        build_historical_daily_stats(db_session, Sport.FOOTBALL)
        db_session.flush()

        # Stats for the Jan 10 match (first game) should have 0 games_played
        first_stat = db_session.query(TeamDailyStats).filter(
            TeamDailyStats.team_name == "Arsenal",
            TeamDailyStats.stat_date == date(2025, 1, 10),
        ).first()
        assert first_stat is not None
        assert first_stat.stats["games_played"] == 0

    def test_season_record_tracked(self, db_session):
        from bet_agent.tools.historical_importer import build_historical_daily_stats

        # 3 matches: W, W, L
        _add_historical_match(
            db_session, match_date=date(2025, 1, 10),
            home_team="Arsenal", away_team="TeamA", home_score=2, away_score=0, result="H",
        )
        _add_historical_match(
            db_session, match_date=date(2025, 1, 15),
            home_team="Arsenal", away_team="TeamB", home_score=3, away_score=1, result="H",
        )
        _add_historical_match(
            db_session, match_date=date(2025, 1, 20),
            home_team="Arsenal", away_team="TeamC", home_score=0, away_score=1, result="A",
        )

        build_historical_daily_stats(db_session, Sport.FOOTBALL)
        db_session.flush()

        # At the 3rd match, Arsenal should have 2 games played (PiT: before this match)
        stat = db_session.query(TeamDailyStats).filter(
            TeamDailyStats.team_name == "Arsenal",
            TeamDailyStats.stat_date == date(2025, 1, 20),
        ).first()
        assert stat is not None
        assert stat.stats["games_played"] == 2
        assert stat.stats["season_wins"] == 2


# ── Feature Factory ──────────────────────────────────────────────────


class TestFeatureFactory:
    def _setup_data(self, db_session):
        """Create test data with daily stats."""
        from bet_agent.tools.historical_importer import build_historical_daily_stats

        for i in range(8):
            _add_historical_match(
                db_session,
                match_date=date(2025, 1, 10 + i),
                home_team="Arsenal",
                away_team=f"Opp{i}",
                home_score=2,
                away_score=i % 3,
                result="H" if 2 > i % 3 else ("D" if 2 == i % 3 else "A"),
            )
        for i in range(8):
            _add_historical_match(
                db_session,
                match_date=date(2025, 1, 10 + i),
                home_team=f"Opp{i}",
                away_team="Chelsea",
                home_score=1,
                away_score=1 + i % 2,
                result="A" if (1 + i % 2) > 1 else "D",
            )

        build_historical_daily_stats(db_session, Sport.FOOTBALL)
        db_session.flush()

    def test_build_feature_vector(self, db_session):
        from bet_agent.tools.feature_factory import build_feature_vector

        self._setup_data(db_session)

        fv = build_feature_vector(
            db_session, Sport.FOOTBALL, "Arsenal", "Chelsea", date(2025, 1, 18),
        )
        assert fv.sport == Sport.FOOTBALL
        assert fv.home_team == "Arsenal"
        assert "h_roll_5_goals_for" in fv.features
        assert "a_roll_5_goals_for" in fv.features
        assert "diff_roll_5_goals_for" in fv.features
        assert "h_rest_days" in fv.features

    def test_feature_names_consistent(self):
        from bet_agent.tools.feature_factory import get_feature_names

        names = get_feature_names(Sport.FOOTBALL)
        assert len(names) > 0
        assert "h_roll_5_goals_for" in names
        assert "a_roll_5_goals_for" in names
        assert "diff_roll_5_goals_for" in names
        assert "h2h_matches" in names

    def test_build_training_dataset(self, db_session):
        from bet_agent.tools.feature_factory import build_training_dataset

        self._setup_data(db_session)

        dataset = build_training_dataset(db_session, Sport.FOOTBALL, min_games_played=3)
        # Should have some samples (those with >= 3 games history)
        assert isinstance(dataset, list)
        for fv in dataset:
            assert fv.target_result in ("H", "D", "A")
            assert fv.target_total_goals is not None

    def test_h2h_stats(self, db_session):
        from bet_agent.tools.feature_factory import _head_to_head_stats

        _add_historical_match(
            db_session, match_date=date(2025, 1, 10),
            home_team="Arsenal", away_team="Chelsea", result="H",
        )
        _add_historical_match(
            db_session, match_date=date(2025, 1, 20),
            home_team="Chelsea", away_team="Arsenal", result="H",  # Chelsea wins at home
        )

        h2h = _head_to_head_stats(db_session, Sport.FOOTBALL, "Arsenal", "Chelsea", date(2025, 2, 1))
        assert h2h["h2h_matches"] == 2.0
        assert h2h["h2h_home_wins"] == 1.0  # Arsenal won when they were home
        assert h2h["h2h_away_wins"] == 1.0  # Chelsea won when Arsenal was away

    def test_rest_days(self, db_session):
        from bet_agent.tools.feature_factory import _days_since_last_match

        _add_historical_match(db_session, match_date=date(2025, 1, 10))

        rest = _days_since_last_match(db_session, Sport.FOOTBALL, "Arsenal", date(2025, 1, 15))
        assert rest == 5

        rest_none = _days_since_last_match(db_session, Sport.FOOTBALL, "NewTeam", date(2025, 1, 15))
        assert rest_none is None


# ── ML Trainer ───────────────────────────────────────────────────────


class TestMLTrainer:
    def test_train_match_winner(self):
        from bet_agent.ml.trainer import ModelArtifact, load_model, train_match_winner

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            np.random.seed(42)

            X = np.random.rand(100, 10)
            y = np.random.randint(0, 3, 100)

            artifact = train_match_winner(
                X, y,
                feature_names=[f"f{i}" for i in range(10)],
                sport=Sport.FOOTBALL,
                model_dir=model_dir,
            )

            assert isinstance(artifact, ModelArtifact)
            assert artifact.model_type == "match_winner"
            assert artifact.train_samples == 100
            assert artifact.train_accuracy is not None
            assert artifact.train_accuracy > 0.0
            assert Path(artifact.file_path).exists()

            # Can load
            model = load_model(artifact.file_path)
            assert hasattr(model, "predict_proba")

    def test_train_over_under(self):
        from bet_agent.ml.trainer import train_over_under

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            np.random.seed(42)

            X = np.random.rand(100, 10)
            y = np.random.rand(100) * 5

            artifact = train_over_under(
                X, y,
                feature_names=[f"f{i}" for i in range(10)],
                sport=Sport.FOOTBALL,
                model_dir=model_dir,
            )

            assert artifact.model_type == "over_under"
            assert artifact.train_rmse is not None
            assert Path(artifact.file_path).exists()

    def test_predict_match_winner(self):
        from bet_agent.ml.trainer import predict_match_winner, train_match_winner

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            np.random.seed(42)

            X = np.random.rand(50, 5)
            y = np.random.randint(0, 3, 50)

            artifact = train_match_winner(
                X, y,
                feature_names=[f"f{i}" for i in range(5)],
                sport=Sport.FOOTBALL,
                model_dir=model_dir,
            )

            result = predict_match_winner(artifact.file_path, X[:1])
            assert "home" in result
            assert "draw" in result
            assert "away" in result
            assert abs(result["home"] + result["draw"] + result["away"] - 1.0) < 0.01

    def test_predict_match_winner_binary(self):
        """Binary sport (Tennis) returns draw=0.0 and probs sum to 1.0."""
        from bet_agent.ml.trainer import predict_match_winner, train_match_winner

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            np.random.seed(42)

            X = np.random.rand(50, 5)
            y = np.random.randint(0, 2, 50)  # Binary: 0=Home, 1=Away

            artifact = train_match_winner(
                X, y,
                feature_names=[f"f{i}" for i in range(5)],
                sport=Sport.TENNIS,
                model_dir=model_dir,
            )

            result = predict_match_winner(artifact.file_path, X[:1])
            assert "home" in result
            assert "draw" in result
            assert "away" in result
            assert result["draw"] == 0.0
            assert abs(result["home"] + result["away"] - 1.0) < 0.01

    def test_predict_total(self):
        from bet_agent.ml.trainer import predict_total, train_over_under

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            np.random.seed(42)

            X = np.random.rand(50, 5)
            y = np.random.rand(50) * 5

            artifact = train_over_under(
                X, y,
                feature_names=[f"f{i}" for i in range(5)],
                sport=Sport.FOOTBALL,
                model_dir=model_dir,
            )

            total = predict_total(artifact.file_path, X[:1])
            assert isinstance(total, float)
            assert total >= 0.0

    def test_find_latest_model(self):
        from bet_agent.ml.trainer import find_latest_model, train_match_winner

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            np.random.seed(42)

            X = np.random.rand(50, 5)
            y = np.random.randint(0, 3, 50)

            train_match_winner(
                X, y,
                feature_names=[f"f{i}" for i in range(5)],
                sport=Sport.FOOTBALL,
                model_dir=model_dir,
            )

            found = find_latest_model(Sport.FOOTBALL, "match_winner", model_dir)
            assert found is not None
            assert found.sport == Sport.FOOTBALL

            # No tennis model trained
            not_found = find_latest_model(Sport.TENNIS, "match_winner", model_dir)
            assert not_found is None

    def test_model_artifact_to_dict(self):
        from bet_agent.ml.trainer import ModelArtifact

        artifact = ModelArtifact(
            model_name="test",
            sport=Sport.FOOTBALL,
            model_type="match_winner",
            version="abc123",
            feature_names=["f1"],
            trained_at="2025-01-01",
            train_samples=100,
        )
        d = artifact.to_dict()
        assert d["sport"] == "football"
        assert d["model_name"] == "test"

    def test_load_model_lru_cache(self):
        """load_model should return the same object on repeated calls (LRU cached)."""
        from bet_agent.ml.trainer import load_model, train_match_winner

        # Clear cache from any prior tests
        load_model.cache_clear()

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            np.random.seed(42)
            X = np.random.rand(50, 5)
            y = np.random.randint(0, 3, 50)

            artifact = train_match_winner(
                X, y,
                feature_names=[f"f{i}" for i in range(5)],
                sport=Sport.FOOTBALL,
                model_dir=model_dir,
            )

            m1 = load_model(artifact.file_path)
            m2 = load_model(artifact.file_path)
            assert m1 is m2, "LRU cache should return the same object"

            info = load_model.cache_info()
            assert info.hits >= 1

        # Clean up
        load_model.cache_clear()


# ── Evaluate Model Performance ───────────────────────────────────────


class TestEvaluatePerformance:
    def test_no_bets_returns_no_data(self, db_session):
        from bet_agent.ml.trainer import evaluate_model_performance

        result = evaluate_model_performance(db_session, "test_model")
        assert result["total_bets"] == 0
        assert result["passes_threshold"] is False
        assert result["reason"] == "no_resolved_bets"


# ── Updated EV Calculator ────────────────────────────────────────────


class TestEVCalculatorML:
    def test_ev_result_has_model_source(self):
        from bet_agent.tools.ev_calculator import calculate_pre_match_ev

        result = calculate_pre_match_ev(0.6, 2.0)
        # Default source is "analytical" (no model_source kwarg in old API)
        assert hasattr(result, "model_source")

    def test_analytical_fallback(self):
        from bet_agent.tools.ev_calculator import _analytical_fallback

        results = _analytical_fallback("football", 2.0, 3.5, 4.0)
        assert "home" in results
        assert "draw" in results
        assert "away" in results
        # Implied prob = model prob → EV should be ≈ 0
        assert abs(results["home"].ev) < 0.01

    def test_make_ev_result(self):
        from bet_agent.tools.ev_calculator import _make_ev_result

        result = _make_ev_result(0.55, 2.0, "xgboost")
        assert result.model_source == "xgboost"
        assert result.updated_prob == 0.55
        assert result.implied_prob == 0.5
        assert result.is_positive_ev is True  # 0.55 > 0.50

    def test_calculate_ml_pre_match_ev_no_model(self, db_session):
        """Without a trained model, should fall back to analytical."""
        from bet_agent.tools.ev_calculator import calculate_ml_pre_match_ev

        with tempfile.TemporaryDirectory() as tmp:
            results = calculate_ml_pre_match_ev(
                db_session, "football", "Arsenal", "Chelsea",
                date(2025, 3, 15), 2.0, 3.5, 4.0,
                model_dir=Path(tmp),
            )
            assert "home" in results
            # Fallback means implied = model prob → near-zero EV
            assert abs(results["home"].ev) < 0.01


# ── Full Import Pipeline ─────────────────────────────────────────────


class TestFullPipeline:
    def test_run_full_historical_import(self, db_session):
        from bet_agent.tools.historical_importer import run_full_historical_import

        for i in range(5):
            _add_historical_match(
                db_session,
                match_date=date(2025, 1, 10 + i),
                home_team="Arsenal",
                away_team=f"Team{i}",
                home_score=2,
                away_score=i % 2,
            )

        result = run_full_historical_import(db_session, sport=Sport.FOOTBALL)
        assert result["matches"] == 5
        assert result["daily_stats"] > 0
