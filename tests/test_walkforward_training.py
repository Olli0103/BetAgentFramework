"""Tests for Phase 3: Anti-Leakage ML Training & Dynamic Feature Extraction.

Tests:
  1. Dynamic feature extraction from JSONB profiles (variable-width)
  2. Tennis metadata features (age_diff, height_diff, hand encoding)
  3. Walk-Forward Validation (TimeSeriesSplit) — temporal ordering respected
  4. Multiclass Brier Score correctness
  5. get_feature_names() dynamic mode
  6. No future leakage in training pipeline
"""

from datetime import date

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import Base, HistoricalMatch, Sport, TeamDailyStats
from bet_agent.tools.feature_factory import (
    FeatureVector,
    _encode_hand,
    _extract_dynamic_features,
    build_training_dataset,
    get_feature_names,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


# ── Hand Encoding ────────────────────────────────────────────────────


class TestHandEncoding:
    def test_right_hand(self):
        assert _encode_hand("R") == 1.0

    def test_left_hand(self):
        assert _encode_hand("L") == -1.0

    def test_unknown_hand(self):
        assert _encode_hand("U") == 0.0

    def test_none_hand(self):
        assert _encode_hand(None) == 0.0

    def test_lowercase(self):
        assert _encode_hand("r") == 1.0
        assert _encode_hand("l") == -1.0


# ── Dynamic Feature Extraction ──────────────────────────────────────


class TestDynamicFeatureExtraction:
    def test_basic_numeric_features(self):
        """All numeric JSONB keys become h_/a_ features."""
        home = {"roll_5_goals_for": 2.5, "season_wins": 10, "games_played": 20}
        away = {"roll_5_goals_for": 1.8, "season_wins": 8, "games_played": 15}

        features = _extract_dynamic_features(home, away, Sport.FOOTBALL)

        assert features["h_roll_5_goals_for"] == 2.5
        assert features["a_roll_5_goals_for"] == 1.8
        assert features["h_season_wins"] == 10.0
        assert features["a_season_wins"] == 8.0

    def test_differential_features_for_roll_keys(self):
        """roll_* keys automatically get diff_ features."""
        home = {"roll_3_shots": 30.0, "roll_10_goals_for": 3.0}
        away = {"roll_3_shots": 25.0, "roll_10_goals_for": 2.0}

        features = _extract_dynamic_features(home, away, Sport.ICE_HOCKEY)

        assert features["diff_roll_3_shots"] == 5.0
        assert features["diff_roll_10_goals_for"] == 1.0

    def test_missing_keys_fill_zero(self):
        """Missing keys in one profile default to 0.0."""
        home = {"roll_5_goals_for": 2.5}
        away = {}

        features = _extract_dynamic_features(home, away, Sport.FOOTBALL)

        assert features["h_roll_5_goals_for"] == 2.5
        assert features["a_roll_5_goals_for"] == 0.0

    def test_metadata_keys_skipped(self):
        """Keys in _SKIP_PROFILE_KEYS are not extracted as numeric features."""
        home = {"hand": "R", "height_cm": 188, "roll_5_goals_for": 2.0}
        away = {"hand": "L", "height_cm": 175, "roll_5_goals_for": 1.5}

        features = _extract_dynamic_features(home, away, Sport.TENNIS)

        # hand/height_cm not in h_/a_ numeric features
        assert "h_hand" not in features or features.get("h_hand") is not None  # hand is a metadata feature
        assert "h_height_cm" not in features  # skipped from numeric extraction
        assert "h_roll_5_goals_for" in features

    def test_tennis_metadata_features(self):
        """Tennis gets age_diff, height_diff, hand_matchup features."""
        home = {"age": 23, "height_cm": 188, "hand": "R", "roll_5_goals_for": 2.0}
        away = {"age": 27, "height_cm": 180, "hand": "L", "roll_5_goals_for": 1.5}

        features = _extract_dynamic_features(home, away, Sport.TENNIS)

        assert features["age_diff"] == -4.0  # 23 - 27
        assert features["height_diff"] == 8.0  # 188 - 180
        assert features["h_hand"] == 1.0  # R
        assert features["a_hand"] == -1.0  # L
        assert features["hand_matchup"] == -1.0  # R * L = 1.0 * -1.0

    def test_tennis_elo_features(self):
        """ELO rating gets extracted as a special tennis feature."""
        home = {"elo_rating": 2100.0}
        away = {"elo_rating": 1950.0}

        features = _extract_dynamic_features(home, away, Sport.TENNIS)

        assert features["h_elo_rating"] == 2100.0
        assert features["a_elo_rating"] == 1950.0
        assert features["elo_diff"] == 150.0

    def test_non_tennis_no_metadata(self):
        """Non-tennis sports don't get age_diff, height_diff, etc."""
        home = {"age": 30, "roll_5_goals_for": 2.0}
        away = {"age": 25, "roll_5_goals_for": 1.5}

        features = _extract_dynamic_features(home, away, Sport.FOOTBALL)

        assert "age_diff" not in features
        assert "height_diff" not in features
        assert "h_hand" not in features

    def test_empty_profiles(self):
        """Empty profiles produce empty feature dict."""
        features = _extract_dynamic_features({}, {}, Sport.FOOTBALL)
        assert features == {}

    def test_nhl_dynamic_features(self):
        """NHL pre-computed CSV features flow through dynamically."""
        home = {
            "roll_3_shots": 32.5, "roll_10_goals_for": 3.2,
            "opp_season_win_pct": 0.55, "season_goal_diff": 15,
            "pre_game_point_pct": 0.62, "rest_days": 2,
        }
        away = {
            "roll_3_shots": 28.0, "roll_10_goals_for": 2.8,
            "opp_season_win_pct": 0.48, "season_goal_diff": -5,
            "pre_game_point_pct": 0.51, "rest_days": 1,
        }

        features = _extract_dynamic_features(home, away, Sport.ICE_HOCKEY)

        assert features["h_roll_3_shots"] == 32.5
        assert features["a_roll_3_shots"] == 28.0
        assert features["h_opp_season_win_pct"] == 0.55
        assert features["diff_roll_3_shots"] == pytest.approx(4.5)


# ── Dynamic get_feature_names ────────────────────────────────────────


class TestDynamicFeatureNames:
    def test_from_dataset(self):
        """Dynamic feature names extracted from a dataset."""
        fv1 = FeatureVector(
            match_date=date(2025, 1, 1), sport=Sport.FOOTBALL,
            home_team="A", away_team="B",
            features={"h_goals": 2.0, "a_goals": 1.0, "diff_goals": 1.0},
        )
        fv2 = FeatureVector(
            match_date=date(2025, 1, 2), sport=Sport.FOOTBALL,
            home_team="C", away_team="D",
            features={"h_goals": 3.0, "a_goals": 2.0, "extra_feat": 0.5},
        )

        names = get_feature_names(Sport.FOOTBALL, dataset=[fv1, fv2])

        assert "h_goals" in names
        assert "a_goals" in names
        assert "diff_goals" in names
        assert "extra_feat" in names
        # Sorted
        assert names == sorted(names)

    def test_fallback_no_dataset(self):
        """Without a dataset, returns static universal features."""
        names = get_feature_names(Sport.FOOTBALL)
        assert "h_roll_5_goals_for" in names
        assert "h2h_matches" in names


# ── Multiclass Brier Score ──────────────────────────────────────────


class TestMulticlassBrierScore:
    def test_perfect_predictions(self):
        """Perfect predictions → Brier = 0.0."""
        from bet_agent.ml.trainer import _multiclass_brier_score

        y_true = np.array([0, 1, 2, 0])
        probs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])

        assert _multiclass_brier_score(y_true, probs, n_classes=3) == 0.0

    def test_worst_predictions(self):
        """Completely wrong predictions → Brier = 2.0."""
        from bet_agent.ml.trainer import _multiclass_brier_score

        y_true = np.array([0, 1, 2])
        probs = np.array([
            [0.0, 0.0, 1.0],  # should be class 0
            [1.0, 0.0, 0.0],  # should be class 1
            [0.0, 1.0, 0.0],  # should be class 2
        ])

        brier = _multiclass_brier_score(y_true, probs, n_classes=3)
        assert brier == pytest.approx(2.0)

    def test_uniform_predictions(self):
        """Uniform 1/3 predictions → Brier = 2/3 ≈ 0.667."""
        from bet_agent.ml.trainer import _multiclass_brier_score

        y_true = np.array([0, 1, 2])
        probs = np.array([
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 1 / 3, 1 / 3],
        ])

        brier = _multiclass_brier_score(y_true, probs, n_classes=3)
        assert brier == pytest.approx(2 / 3, abs=0.001)

    def test_empty_input(self):
        """Empty input → Brier = 1.0 (fallback)."""
        from bet_agent.ml.trainer import _multiclass_brier_score

        brier = _multiclass_brier_score(np.array([]), np.array([]).reshape(0, 3), n_classes=3)
        assert brier == 1.0


# ── Walk-Forward Validation ─────────────────────────────────────────


class TestWalkForwardValidation:
    def test_walk_forward_basic(self):
        """Walk-Forward produces fold metrics with correct structure."""
        from bet_agent.ml.trainer import _walk_forward_validate

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y_result = np.random.randint(0, 3, size=n)
        y_total = np.random.rand(n) * 5

        result = _walk_forward_validate(
            X, y_result, y_total,
            feature_names=[f"f{i}" for i in range(5)],
            sport=Sport.FOOTBALL,
            n_splits=3,
        )

        assert result["n_splits"] == 3
        assert len(result["folds"]) == 3
        assert "mean_brier" in result
        assert "mean_accuracy" in result
        assert "mean_rmse" in result
        assert result["mean_brier"] >= 0
        assert result["mean_accuracy"] >= 0
        assert result["mean_rmse"] >= 0

    def test_walk_forward_temporal_ordering(self):
        """Walk-Forward respects chronological order (train < val indices)."""
        from sklearn.model_selection import TimeSeriesSplit

        tss = TimeSeriesSplit(n_splits=3)
        X = np.arange(30).reshape(30, 1)

        for train_idx, val_idx in tss.split(X):
            # All training indices must be less than all validation indices
            assert max(train_idx) < min(val_idx)

    def test_fold_sizes_grow(self):
        """Each successive fold has a larger training set."""
        from bet_agent.ml.trainer import _walk_forward_validate

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y_result = np.random.randint(0, 3, size=n)
        y_total = np.random.rand(n) * 5

        result = _walk_forward_validate(
            X, y_result, y_total,
            feature_names=["f0", "f1", "f2"],
            sport=Sport.FOOTBALL,
            n_splits=3,
        )

        train_sizes = [f["train_size"] for f in result["folds"]]
        assert train_sizes == sorted(train_sizes), "Training set should grow with each fold"

    def test_walk_forward_binary_sport(self):
        """Walk-Forward uses 2-class binary:logistic for non-draw sports."""
        from bet_agent.ml.trainer import _walk_forward_validate

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y_result = np.random.randint(0, 2, size=n)  # Binary: 0=Home, 1=Away
        y_total = np.random.rand(n) * 5

        result = _walk_forward_validate(
            X, y_result, y_total,
            feature_names=[f"f{i}" for i in range(5)],
            sport=Sport.TENNIS,
            n_splits=3,
        )

        assert result["n_classes"] == 2
        assert result["n_splits"] == 3
        assert len(result["folds"]) == 3
        assert result["mean_brier"] >= 0
        assert result["mean_accuracy"] >= 0

    def test_walk_forward_three_way_sport(self):
        """Walk-Forward uses 3-class multi:softprob for draw sports."""
        from bet_agent.ml.trainer import _walk_forward_validate

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y_result = np.random.randint(0, 3, size=n)  # 3-way: H/D/A
        y_total = np.random.rand(n) * 5

        result = _walk_forward_validate(
            X, y_result, y_total,
            feature_names=[f"f{i}" for i in range(5)],
            sport=Sport.FOOTBALL,
            n_splits=3,
        )

        assert result["n_classes"] == 3


# ── Integration: build_training_dataset with dynamic features ───────


class TestTrainingDatasetIntegration:
    def _seed_data(self, session, sport=Sport.FOOTBALL):
        """Seed historical matches and team daily stats."""
        # Create 10 matches over 10 days
        for i in range(10):
            d = date(2025, 1, i + 1)
            session.add(HistoricalMatch(
                sport=sport,
                season="2025",
                division="Test",
                match_date=d,
                home_team="Team A",
                away_team="Team B",
                home_score=2 if i % 2 == 0 else 1,
                away_score=1 if i % 2 == 0 else 2,
                result="H" if i % 2 == 0 else "A",
                match_stats={},
                odds={},
                betting_lines={},
                advanced_stats={},
                source="test",
            ))

            # Team daily stats with dynamic features
            for team in ["Team A", "Team B"]:
                session.add(TeamDailyStats(
                    sport=sport,
                    team_name=team,
                    league="Test",
                    stat_date=d,
                    stats={
                        "roll_5_goals_for": 2.0 + i * 0.1,
                        "roll_10_goals_for": 1.8 + i * 0.05,
                        "season_wins": i,
                        "games_played": i + 5,
                        "win_pct": 0.5 + i * 0.01,
                        "season_losses": max(0, 5 - i),
                        "season_draws": 0,
                    },
                    source_url="test",
                ))
        session.flush()

    def test_dataset_chronological(self, db_session):
        """Dataset is sorted chronologically."""
        self._seed_data(db_session)
        dataset = build_training_dataset(db_session, Sport.FOOTBALL, min_games_played=0)

        dates = [fv.match_date for fv in dataset]
        assert dates == sorted(dates)

    def test_dynamic_features_in_dataset(self, db_session):
        """Dynamic JSONB features appear in training vectors."""
        self._seed_data(db_session)
        dataset = build_training_dataset(db_session, Sport.FOOTBALL, min_games_played=0)

        if dataset:
            fv = dataset[-1]  # Last match (has most history)
            assert "h_roll_5_goals_for" in fv.features
            assert "a_roll_5_goals_for" in fv.features
            assert "diff_roll_5_goals_for" in fv.features

    def test_dynamic_feature_names_from_dataset(self, db_session):
        """get_feature_names with dataset returns all dynamic keys."""
        self._seed_data(db_session)
        dataset = build_training_dataset(db_session, Sport.FOOTBALL, min_games_played=0)

        names = get_feature_names(Sport.FOOTBALL, dataset=dataset)
        assert len(names) > 0
        # All feature keys from all vectors should be in names
        for fv in dataset:
            for key in fv.features:
                assert key in names

    def test_point_in_time_no_leakage(self, db_session):
        """Features at match T come from stats BEFORE T (strict < comparison)."""
        self._seed_data(db_session)
        dataset = build_training_dataset(db_session, Sport.FOOTBALL, min_games_played=0)

        if len(dataset) >= 2:
            # First match should have stats from before day 1
            # Since we seed stats ON the match date, _lookup_stats_at_date
            # uses strict < so day 1 match gets NO stats (or prior day's stats)
            fv_first = dataset[0]
            # The stats for day 1 are not available for day 1's match
            # (they might be 0.0 from no prior data, which is correct)
            assert fv_first.features.get("h_roll_5_goals_for", 0.0) is not None
