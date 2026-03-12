"""Sport key matching for the Odds API integration.

The `sports_mapping` in agents.yaml maps internal sport names to external
API sport keys. Entries can be:
  - Exact: ``basketball_nba``
  - Wildcard: ``tennis_atp*`` (matches tennis_atp_french_open, tennis_atp_dubai, etc.)

The Scout Agent calls ``match_sport_keys()`` to expand wildcards against
the live list of available sport keys from the Odds API ``/sports`` endpoint.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "agents.yaml"

# ── Event family prefixes for common sport groupings ─────────────────

SPORT_FAMILIES: dict[str, list[str]] = {
    "tennis": ["tennis_atp", "tennis_wta", "tennis_itf"],
    "football": ["soccer_"],
    "american_football": ["americanfootball_"],
    "basketball": ["basketball_"],
    "ice_hockey": ["icehockey_"],
    "darts": ["darts_"],
}


def load_sports_mapping(config_path: Path | str = _CONFIG_PATH) -> dict[str, list[str]]:
    """Load the sports_mapping from agents.yaml.

    Returns:
        Dict mapping internal sport name → list of API sport key patterns.
        Patterns may contain wildcards (e.g. ``tennis_atp*``).
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("agents.yaml not found at %s", path)
        return {}

    raw = yaml.safe_load(path.read_text())
    odds_cfg = (
        raw.get("agents", [{}])[1]  # scout agent
        .get("odds_api", {})
        .get("sports_mapping", {})
    )

    result: dict[str, list[str]] = {}
    for sport, keys_str in odds_cfg.items():
        if isinstance(keys_str, str):
            result[sport] = [k.strip() for k in keys_str.split(",") if k.strip()]
        elif isinstance(keys_str, list):
            result[sport] = keys_str
    return result


def match_sport_keys(
    patterns: list[str],
    available_keys: list[str],
) -> list[str]:
    """Expand wildcard patterns against available sport keys.

    Args:
        patterns: Sport key patterns from config (e.g. ["tennis_atp*", "tennis_wta*"])
        available_keys: Live sport keys from the Odds API /sports endpoint.

    Returns:
        Sorted deduplicated list of matched sport keys.

    Examples:
        >>> match_sport_keys(["tennis_atp*"], ["tennis_atp_french_open", "tennis_atp_dubai", "basketball_nba"])
        ['tennis_atp_dubai', 'tennis_atp_french_open']
        >>> match_sport_keys(["basketball_nba"], ["basketball_nba", "basketball_euroleague"])
        ['basketball_nba']
    """
    matched: set[str] = set()
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            # Wildcard pattern — expand against available keys
            for key in available_keys:
                if fnmatch.fnmatch(key, pattern):
                    matched.add(key)
        else:
            # Exact match — only include if available
            if pattern in available_keys:
                matched.add(pattern)

    result = sorted(matched)
    logger.info(
        "Sport key expansion: %d patterns → %d matched keys",
        len(patterns), len(result),
    )
    return result


def get_sport_keys_for(
    sport: str,
    available_keys: list[str],
    config_path: Path | str = _CONFIG_PATH,
) -> list[str]:
    """Get expanded sport keys for a given internal sport name.

    Convenience function combining config load + pattern expansion.

    Args:
        sport: Internal sport name (e.g. "tennis", "basketball")
        available_keys: Live sport keys from the API
        config_path: Path to agents.yaml

    Returns:
        List of matched API sport keys.
    """
    mapping = load_sports_mapping(config_path)
    patterns = mapping.get(sport, [])
    if not patterns:
        logger.warning("No sport key patterns for '%s' in config", sport)
        return []
    return match_sport_keys(patterns, available_keys)
