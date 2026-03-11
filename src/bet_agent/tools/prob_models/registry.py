"""Sport → probability model dispatcher.

Every sport model implements a common interface:
- match_outcome_probs(**kwargs) -> dict with P(home), P(draw), P(away)
- over_under_prob(**kwargs) -> float
- live_update(pre_match_prob, live_score, live_time, live_stats) -> float
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@runtime_checkable
class SportModel(Protocol):
    """Interface all sport probability models must implement."""

    sport: str

    def match_outcome_probs(self, **kwargs) -> dict[str, float]:
        """Return {"home": p, "draw": p, "away": p}."""
        ...

    def over_under_prob(self, **kwargs) -> float:
        """Return P(over) for a given line."""
        ...

    def live_update(
        self,
        pre_match_prob: float,
        live_score: tuple[int, int],
        live_time: float,
        live_stats: dict | None = None,
    ) -> float:
        """Update pre-match probability with live match state."""
        ...


_REGISTRY: dict[str, SportModel] = {}


def register(model: SportModel) -> SportModel:
    """Register a sport model in the global registry."""
    _REGISTRY[model.sport] = model
    return model


def get_model(sport: str) -> SportModel:
    """Look up the probability model for a sport.

    Raises KeyError if the sport is not registered.
    """
    if sport not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"No probability model registered for sport '{sport}'. "
            f"Available: {available}"
        )
    return _REGISTRY[sport]


def available_sports() -> list[str]:
    """Return list of all registered sport names."""
    return sorted(_REGISTRY.keys())
