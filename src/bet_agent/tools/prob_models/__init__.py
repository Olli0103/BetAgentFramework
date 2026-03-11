"""Sport-specific probability models.

Importing this package auto-registers all sport models in the registry.
"""

# Auto-register all sport models on import
from bet_agent.tools.prob_models import (  # noqa: F401
    american_football,
    basketball,
    darts,
    football,
    ice_hockey,
    tennis,
)
