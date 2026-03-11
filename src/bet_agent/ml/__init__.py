"""Machine learning pipeline for sports betting predictions."""

from bet_agent.ml.trainer import (
    ModelArtifact,
    evaluate_model_performance,
    load_model,
    predict,
    train_match_winner,
    train_over_under,
)

__all__ = [
    "ModelArtifact",
    "evaluate_model_performance",
    "load_model",
    "predict",
    "train_match_winner",
    "train_over_under",
]
