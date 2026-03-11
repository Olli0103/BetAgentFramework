"""XGBoost Training Pipeline for sports betting predictions.

Trains two model types per sport:
  1. Match Winner Classifier (1X2) — XGBClassifier with softmax
  2. Totals Regressor (Over/Under) — XGBRegressor for expected total score

Model artifacts include: serialized model, feature names, metadata, and version.
The Auditor agent can call evaluate_model_performance() to check Brier Score.

Golden Rule #1: NO LLM MATH.  All training is deterministic scikit-learn + XGBoost.
Golden Rule #2: STATEFUL MEMORY.  Model metrics go to the model_metrics table.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import os

import numpy as np

from bet_agent.db.models import LedgerType, ModelMetrics, PlacedBet, Sport

logger = logging.getLogger(__name__)

# Reserve CPU cores for the Telegram bot and async agents during training.
# n_jobs=-1 would starve the entire system while XGBoost runs.
_TRAINING_JOBS = max(1, (os.cpu_count() or 4) - 2)

# Default model storage directory
_MODEL_DIR = Path("models")

# Brier Score threshold for the Auditor (above this = model needs retraining).
# 0.25 = coin-flip baseline (worthless). A model must beat the bookmaker's
# implied probabilities, which typically sit around 0.18-0.21 Brier.
BRIER_THRESHOLD = 0.21


@dataclass
class ModelArtifact:
    """Metadata and reference for a trained model."""

    model_name: str
    sport: Sport
    model_type: str  # "match_winner" or "over_under"
    version: str
    feature_names: list[str]
    trained_at: str
    train_samples: int
    train_accuracy: float | None = None
    train_rmse: float | None = None
    hyperparams: dict = field(default_factory=dict)
    file_path: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sport"] = self.sport.value
        return d


# ── Training ─────────────────────────────────────────────────────────


def train_match_winner(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    sport: Sport,
    model_dir: Path | None = None,
    hyperparams: dict | None = None,
) -> ModelArtifact:
    """Train an XGBoost classifier for 1X2 match result prediction.

    Args:
        features: (n_samples, n_features) array.
        labels: (n_samples,) array with values in {0=Home, 1=Draw, 2=Away}.
        feature_names: Ordered list of feature column names.
        sport: Sport enum for model naming.
        model_dir: Where to save the artifact. Defaults to ./models/.
        hyperparams: Optional XGBoost hyperparameters override.

    Returns:
        ModelArtifact with metadata and file path.
    """
    from xgboost import XGBClassifier

    if model_dir is None:
        model_dir = _MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": _TRAINING_JOBS,
    }
    if hyperparams:
        params.update(hyperparams)

    clf = XGBClassifier(**params)
    clf.fit(features, labels)

    # Training accuracy
    preds = clf.predict(features)
    train_acc = float(np.mean(preds == labels))

    # Version hash (features + data shape + timestamp)
    version = _make_version(sport, "match_winner", features.shape)

    # Save
    model_name = f"xgb_{sport.value}_match_winner"
    model_path = model_dir / f"{model_name}_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Save metadata
    artifact = ModelArtifact(
        model_name=model_name,
        sport=sport,
        model_type="match_winner",
        version=version,
        feature_names=feature_names,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_samples=features.shape[0],
        train_accuracy=round(train_acc, 4),
        hyperparams=params,
        file_path=str(model_path),
    )

    meta_path = model_dir / f"{model_name}_{version}.json"
    with open(meta_path, "w") as f:
        json.dump(artifact.to_dict(), f, indent=2)

    logger.info(
        "Trained %s: %d samples, accuracy=%.4f, saved to %s",
        model_name, features.shape[0], train_acc, model_path,
    )
    return artifact


def train_over_under(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    sport: Sport,
    model_dir: Path | None = None,
    hyperparams: dict | None = None,
) -> ModelArtifact:
    """Train an XGBoost regressor for total goals/points prediction.

    Args:
        features: (n_samples, n_features) array.
        targets: (n_samples,) array with total goals/points.
        feature_names: Ordered list of feature column names.
        sport: Sport enum for model naming.
        model_dir: Where to save the artifact.
        hyperparams: Optional XGBoost hyperparameters override.

    Returns:
        ModelArtifact with metadata and file path.
    """
    from xgboost import XGBRegressor

    if model_dir is None:
        model_dir = _MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": _TRAINING_JOBS,
    }
    if hyperparams:
        params.update(hyperparams)

    reg = XGBRegressor(**params)
    reg.fit(features, targets)

    # Training RMSE
    preds = reg.predict(features)
    train_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    version = _make_version(sport, "over_under", features.shape)

    model_name = f"xgb_{sport.value}_over_under"
    model_path = model_dir / f"{model_name}_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(reg, f)

    artifact = ModelArtifact(
        model_name=model_name,
        sport=sport,
        model_type="over_under",
        version=version,
        feature_names=feature_names,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_samples=features.shape[0],
        train_rmse=round(train_rmse, 4),
        hyperparams=params,
        file_path=str(model_path),
    )

    meta_path = model_dir / f"{model_name}_{version}.json"
    with open(meta_path, "w") as f:
        json.dump(artifact.to_dict(), f, indent=2)

    logger.info(
        "Trained %s: %d samples, RMSE=%.4f, saved to %s",
        model_name, features.shape[0], train_rmse, model_path,
    )
    return artifact


# ── Loading & Inference ──────────────────────────────────────────────


@functools.lru_cache(maxsize=8)
def load_model(model_path: str | Path) -> object:
    """Load a pickled model artifact from disk.

    Cached with LRU (maxsize=8) to avoid repeated unpickling when the
    same model is used for multiple predictions in a single pipeline run.
    The cache key is the string path, so callers should use consistent
    path representations (str or resolved Path).
    """
    model_path = str(model_path)  # Normalize for cache key
    with open(model_path, "rb") as f:
        return pickle.load(f)


def find_latest_model(
    sport: Sport,
    model_type: str,
    model_dir: Path | None = None,
) -> ModelArtifact | None:
    """Find the most recently trained model for a sport + type.

    Scans the model directory for matching JSON metadata files.
    """
    if model_dir is None:
        model_dir = _MODEL_DIR

    if not model_dir.exists():
        return None

    pattern = f"xgb_{sport.value}_{model_type}_*.json"
    meta_files = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not meta_files:
        return None

    with open(meta_files[0]) as f:
        data = json.load(f)

    return ModelArtifact(
        model_name=data["model_name"],
        sport=Sport(data["sport"]),
        model_type=data["model_type"],
        version=data["version"],
        feature_names=data["feature_names"],
        trained_at=data["trained_at"],
        train_samples=data["train_samples"],
        train_accuracy=data.get("train_accuracy"),
        train_rmse=data.get("train_rmse"),
        hyperparams=data.get("hyperparams", {}),
        file_path=data["file_path"],
    )


def predict(model_path: str | Path, features: np.ndarray) -> np.ndarray:
    """Run inference on a loaded model.

    For classifiers, returns shape (n_samples, n_classes) probabilities.
    For regressors, returns shape (n_samples,) predicted values.
    """
    model = load_model(model_path)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)
    else:
        return model.predict(features)


def predict_match_winner(
    model_path: str | Path,
    features: np.ndarray,
) -> dict[str, float]:
    """Predict 1X2 probabilities for a single match.

    Args:
        model_path: Path to the trained classifier.
        features: (1, n_features) array.

    Returns:
        {"home": p, "draw": p, "away": p}
    """
    probs = predict(model_path, features)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    return {
        "home": float(probs[0, 0]),
        "draw": float(probs[0, 1]),
        "away": float(probs[0, 2]),
    }


def predict_total(
    model_path: str | Path,
    features: np.ndarray,
) -> float:
    """Predict expected total goals/points for a single match."""
    pred = predict(model_path, features)
    return float(pred[0]) if pred.ndim >= 1 else float(pred)


# ── Training Pipeline (end-to-end) ──────────────────────────────────


def run_training_pipeline(
    session,
    sport: Sport,
    season: str | None = None,
    model_dir: Path | None = None,
    min_games: int = 5,
) -> dict[str, ModelArtifact]:
    """Full training pipeline: extract features → train models → save.

    Args:
        session: SQLAlchemy session.
        sport: Sport to train for.
        season: Optional season filter.
        model_dir: Model storage directory.
        min_games: Minimum games before including a team.

    Returns:
        {"match_winner": artifact, "over_under": artifact}
    """
    from bet_agent.tools.feature_factory import build_training_dataset, get_feature_names

    dataset = build_training_dataset(session, sport, season, min_games_played=min_games)
    if len(dataset) < 20:
        logger.warning("Insufficient data for %s: only %d samples", sport.value, len(dataset))
        return {}

    feature_names = get_feature_names(sport)

    # Build numpy arrays
    X = np.zeros((len(dataset), len(feature_names)))
    y_result = np.zeros(len(dataset), dtype=int)
    y_total = np.zeros(len(dataset))

    result_map = {"H": 0, "D": 1, "A": 2}

    for i, fv in enumerate(dataset):
        for j, fname in enumerate(feature_names):
            X[i, j] = fv.features.get(fname, 0.0)
        y_result[i] = result_map.get(fv.target_result or "D", 1)
        y_total[i] = float(fv.target_total_goals or 0)

    results = {}
    results["match_winner"] = train_match_winner(X, y_result, feature_names, sport, model_dir)
    results["over_under"] = train_over_under(X, y_total, feature_names, sport, model_dir)

    return results


# ── Auditor Hook: Model Performance Evaluation ──────────────────────


def evaluate_model_performance(
    session,
    model_name: str,
    eval_date: date | None = None,
    ledger_type: LedgerType = LedgerType.PAPER,
) -> dict:
    """Evaluate a model's prediction performance.

    Called by the Auditor Agent to check if a model is still beating
    the Brier Score threshold.

    Computes:
      - Brier Score (calibration)
      - ROI percentage
      - Win/Loss record
      - Whether the model passes the threshold

    Args:
        session: SQLAlchemy session.
        model_name: Name of the model to evaluate.
        eval_date: Date to evaluate up to (defaults to today).
        ledger_type: Which ledger to evaluate (PAPER or REAL).

    Returns:
        Dict with performance metrics and pass/fail status.
    """
    from sqlalchemy import select

    if eval_date is None:
        eval_date = date.today()

    # Get all resolved bets from this model
    bets = session.execute(
        select(PlacedBet).where(
            PlacedBet.ledger_type == ledger_type,
            PlacedBet.placed_at <= datetime.combine(eval_date, datetime.max.time(), tzinfo=timezone.utc),
            PlacedBet.status.in_(["won", "lost"]),
        )
    ).scalars().all()

    if not bets:
        return {
            "model_name": model_name,
            "brier_score": None,
            "roi_pct": 0.0,
            "total_bets": 0,
            "record_win": 0,
            "record_loss": 0,
            "passes_threshold": False,
            "reason": "no_resolved_bets",
        }

    # Brier Score: binary (predicted_prob - outcome)^2
    # NOTE: For proper multi-class Brier on 3-way markets, use
    # auditor_metrics.evaluate_model_performance() which reconstructs
    # the full probability vector from sibling predictions.
    brier_sum = 0.0
    total_stake = 0.0
    total_pnl = 0.0
    wins = 0
    losses = 0

    for bet in bets:
        outcome = 1.0 if bet.status.value == "won" else 0.0
        prob = float(bet.model_prob)
        brier_sum += (prob - outcome) ** 2

        total_stake += float(bet.stake_eur)
        total_pnl += float(bet.pnl_eur or 0)

        if bet.status.value == "won":
            wins += 1
        else:
            losses += 1

    n = len(bets)
    brier_score = round(brier_sum / n, 4) if n > 0 else None
    roi_pct = round((total_pnl / total_stake) * 100, 2) if total_stake > 0 else 0.0

    passes = brier_score is not None and brier_score < BRIER_THRESHOLD

    result = {
        "model_name": model_name,
        "eval_date": str(eval_date),
        "brier_score": brier_score,
        "roi_pct": roi_pct,
        "total_bets": n,
        "record_win": wins,
        "record_loss": losses,
        "passes_threshold": passes,
        "threshold": BRIER_THRESHOLD,
    }

    # Persist to model_metrics table
    session.add(ModelMetrics(
        model_name=model_name,
        date=eval_date,
        brier_score=Decimal(str(brier_score)) if brier_score is not None else Decimal("1.0"),
        roi_pct=Decimal(str(roi_pct)),
        total_bets=n,
        record_win=wins,
        record_loss=losses,
        ledger_type=ledger_type,
    ))

    logger.info(
        "Model %s eval: brier=%.4f, roi=%.2f%%, %d/%d W/L, passes=%s",
        model_name, brier_score or 0, roi_pct, wins, losses, passes,
    )
    return result


# ── Helpers ──────────────────────────────────────────────────────────


def _make_version(sport: Sport, model_type: str, shape: tuple) -> str:
    """Generate a short version hash from sport, type, and data shape."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    raw = f"{sport.value}_{model_type}_{shape}_{ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
