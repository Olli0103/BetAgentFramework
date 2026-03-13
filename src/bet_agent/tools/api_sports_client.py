"""API-Sports Client — rate-limited integration for all sport APIs.

Wraps api-sports.io endpoints (football, basketball, hockey, NFL, NBA)
with per-sport daily budget tracking and sport-specific stat parsers.

Auth: x-apisports-key header (NOT Authorization: Bearer).
Rate limits: football Pro 500/day, all others Free 100/day.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

_AGENTS_YAML = Path(__file__).resolve().parents[3] / "config" / "agents.yaml"

# Default daily limits per sport (from agents.yaml, overridden at init)
_DEFAULT_LIMITS: dict[str, int] = {
    "football": 500,
    "basketball": 100,
    "nba": 100,
    "hockey": 100,
    "nfl": 100,
}

# Sport enum value → API sport key in agents.yaml
_SPORT_TO_API: dict[str, str] = {
    "football": "football",
    "basketball": "basketball",
    "ice_hockey": "hockey",
    "american_football": "nfl",
}


@dataclass
class RateBudget:
    """Track daily API call budget per sport."""
    used: int = 0
    limit: int = 100
    reset_date: str = ""

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


def _load_api_config() -> dict[str, dict]:
    """Load API-Sports config from agents.yaml."""
    if not _AGENTS_YAML.exists():
        return {}
    try:
        raw = yaml.safe_load(_AGENTS_YAML.read_text())
        agents = raw.get("agents", [])
        for agent in agents:
            if agent.get("id") == "scout":
                return agent.get("api_sports", {}).get("apis", {})
        return {}
    except Exception:
        return {}


class APISportsClient:
    """Rate-limited client for api-sports.io endpoints.

    Supports football, basketball, hockey, NFL, and NBA APIs.
    Each sport has its own subdomain, rate limit, and endpoint set.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("API_SPORTS_KEY", "")
        self._apis = _load_api_config()
        self._budgets: dict[str, RateBudget] = {}
        self._today = date.today().isoformat()

        # Initialize budgets from config
        for sport_key, config in self._apis.items():
            limit = config.get("daily_limit", 100)
            self._budgets[sport_key] = RateBudget(
                limit=limit, reset_date=self._today
            )

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _reset_if_new_day(self, sport_key: str) -> None:
        """Reset budget counter if the day has changed."""
        today = date.today().isoformat()
        budget = self._budgets.get(sport_key)
        if budget and budget.reset_date != today:
            budget.used = 0
            budget.reset_date = today

    def _check_budget(self, sport_key: str) -> bool:
        """Check if we have remaining budget for this sport."""
        self._reset_if_new_day(sport_key)
        budget = self._budgets.get(sport_key)
        if not budget:
            return False
        return budget.remaining > 0

    def _increment_count(self, sport_key: str) -> None:
        """Increment the daily usage counter."""
        budget = self._budgets.get(sport_key)
        if budget:
            budget.used += 1

    def remaining_budget(self, sport_key: str) -> int:
        """Return remaining API calls for this sport today."""
        self._reset_if_new_day(sport_key)
        budget = self._budgets.get(sport_key)
        return budget.remaining if budget else 0

    def _get_base_url(self, sport_key: str) -> str | None:
        """Get base URL for a sport's API."""
        config = self._apis.get(sport_key)
        return config.get("base_url") if config else None

    def _api_key_for_sport(self, sport_key: str) -> str | None:
        """Resolve the sport's API → internal config key."""
        return _SPORT_TO_API.get(sport_key, sport_key)

    def _request(
        self,
        sport_key: str,
        endpoint: str,
        params: dict | None = None,
    ) -> dict | None:
        """Make an authenticated API request with budget enforcement.

        Returns parsed JSON response or None on failure.
        """
        if not self._api_key:
            logger.warning("API_SPORTS_KEY not set — cannot call API")
            return None

        if not self._check_budget(sport_key):
            logger.warning(
                "API budget exhausted for %s (%d/%d today)",
                sport_key,
                self._budgets.get(sport_key, RateBudget()).used,
                self._budgets.get(sport_key, RateBudget()).limit,
            )
            return None

        base_url = self._get_base_url(sport_key)
        if not base_url:
            logger.warning("No base URL configured for sport: %s", sport_key)
            return None

        url = f"{base_url}/{endpoint}"
        headers = {"x-apisports-key": self._api_key}

        try:
            resp = requests.get(url, headers=headers, params=params or {}, timeout=15)
            self._increment_count(sport_key)

            if resp.status_code == 429:
                logger.warning("Rate limited by API-Sports for %s", sport_key)
                return None

            resp.raise_for_status()
            data = resp.json()

            # API-Sports wraps all responses in {"response": [...], "errors": {...}}
            errors = data.get("errors")
            if errors and isinstance(errors, dict) and errors:
                logger.warning("API-Sports error for %s/%s: %s", sport_key, endpoint, errors)
                return None

            return data

        except requests.RequestException as exc:
            logger.warning("API-Sports request failed: %s/%s — %s", sport_key, endpoint, exc)
            return None

    def _resolve_sport_key(self, sport: str) -> str | None:
        """Map a Sport enum value to the api-sports config key."""
        api_key = _SPORT_TO_API.get(sport)
        if api_key and api_key in self._apis:
            return api_key
        # Try direct match (e.g. "basketball" → "basketball")
        if sport in self._apis:
            return sport
        return None

    # ── Public API methods ────────────────────────────────────────────

    def fetch_fixtures_by_date(
        self, sport: str, date_str: str, league_id: int | None = None,
    ) -> list[dict]:
        """Fetch all fixtures for a sport on a given date.

        Args:
            sport: Sport enum value (e.g. "football", "ice_hockey").
            date_str: Date in YYYY-MM-DD format.
            league_id: Optional league/season filter.

        Returns:
            List of fixture dicts from the API response.
        """
        sport_key = self._resolve_sport_key(sport)
        if not sport_key:
            return []

        # Endpoint varies by sport
        endpoint_map = {
            "football": "fixtures",
            "basketball": "games",
            "hockey": "games",
            "nfl": "games",
            "nba": "games",
        }
        endpoint = endpoint_map.get(sport_key, "fixtures")
        params: dict = {"date": date_str}
        if league_id is not None:
            params["league"] = str(league_id)

        data = self._request(sport_key, endpoint, params)
        if not data:
            return []

        results = data.get("response", [])
        logger.info(
            "API-Sports: %d fixtures for %s on %s", len(results), sport_key, date_str
        )
        return results

    def fetch_fixture_stats(
        self, sport: str, fixture_id: int,
    ) -> dict | None:
        """Fetch detailed statistics for a specific fixture.

        Args:
            sport: Sport enum value.
            fixture_id: API-Sports fixture/game ID.

        Returns:
            Parsed stats dict or None.
        """
        sport_key = self._resolve_sport_key(sport)
        if not sport_key:
            return None

        # Football uses "fixtures/statistics", others use "statistics" or "games/statistics"
        if sport_key == "football":
            endpoint = "fixtures/statistics"
            params = {"fixture": str(fixture_id)}
        else:
            endpoint = "games/statistics" if sport_key != "nfl" else "games"
            params = {"id": str(fixture_id)}

        data = self._request(sport_key, endpoint, params)
        if not data:
            return None

        response = data.get("response", [])
        if not response:
            return None

        # Parse through sport-specific parser
        parsers = {
            "football": parse_football_stats,
            "basketball": parse_basketball_stats,
            "hockey": parse_hockey_stats,
            "nfl": parse_nfl_stats,
        }
        parser = parsers.get(sport_key)
        if parser:
            return parser(response)
        return {"raw": response}

    def fetch_standings(
        self, sport: str, league_id: int, season: int | str,
    ) -> list[dict]:
        """Fetch league standings.

        Args:
            sport: Sport enum value.
            league_id: League ID.
            season: Season year or string.

        Returns:
            List of standings entries.
        """
        sport_key = self._resolve_sport_key(sport)
        if not sport_key:
            return []

        data = self._request(sport_key, "standings", {
            "league": str(league_id),
            "season": str(season),
        })
        if not data:
            return []

        return data.get("response", [])

    def fetch_injuries(self, sport: str, fixture_id: int) -> list[dict]:
        """Fetch injury reports for a fixture (football and NFL only)."""
        sport_key = self._resolve_sport_key(sport)
        if not sport_key:
            return []

        if sport_key not in ("football", "nfl"):
            return []

        data = self._request(sport_key, "injuries", {"fixture": str(fixture_id)})
        if not data:
            return []

        return data.get("response", [])

    def get_budget_summary(self) -> dict[str, dict]:
        """Return budget status for all configured sports."""
        summary = {}
        for sport_key in self._apis:
            self._reset_if_new_day(sport_key)
            budget = self._budgets.get(sport_key, RateBudget())
            summary[sport_key] = {
                "used": budget.used,
                "limit": budget.limit,
                "remaining": budget.remaining,
            }
        return summary


# ── Sport-specific stat parsers ────────────────────────────────────────


def parse_football_stats(response: list[dict]) -> dict:
    """Parse football fixture statistics into a flat dict for match_stats JSONB.

    API response is a list of team stats:
    [{"team": {...}, "statistics": [{"type": "Shots on Goal", "value": 5}, ...]}]
    """
    result: dict = {}
    prefixes = {0: "home_", 1: "away_"}
    stat_mapping = {
        "shots on goal": "sot",
        "shots off goal": "shots_off",
        "total shots": "shots",
        "blocked shots": "blocked_shots",
        "shots insidebox": "shots_inside",
        "shots outsidebox": "shots_outside",
        "fouls": "fouls",
        "corner kicks": "corners",
        "offsides": "offsides",
        "ball possession": "possession",
        "yellow cards": "yellow_cards",
        "red cards": "red_cards",
        "goalkeeper saves": "gk_saves",
        "total passes": "passes",
        "passes accurate": "passes_accurate",
        "passes %": "pass_pct",
        "expected_goals": "xg",
    }

    for idx, team_data in enumerate(response[:2]):
        prefix = prefixes.get(idx, f"team{idx}_")
        stats = team_data.get("statistics", [])
        for stat in stats:
            stat_type = (stat.get("type") or "").lower()
            value = stat.get("value")
            mapped = stat_mapping.get(stat_type)
            if mapped:
                # Handle percentage strings like "55%"
                if isinstance(value, str) and value.endswith("%"):
                    try:
                        value = float(value.rstrip("%"))
                    except ValueError:
                        continue
                result[f"{prefix}{mapped}"] = value

    return result


def parse_basketball_stats(response: list[dict]) -> dict:
    """Parse basketball game statistics."""
    result: dict = {}

    for item in response:
        # Basketball API returns game-level data
        if isinstance(item, dict):
            scores = item.get("scores", {})
            if isinstance(scores, dict):
                home = scores.get("home", {})
                away = scores.get("away", {})
                if isinstance(home, dict):
                    for q in ["quarter_1", "quarter_2", "quarter_3", "quarter_4", "over_time", "total"]:
                        val = home.get(q)
                        if val is not None:
                            short = q.replace("quarter_", "q").replace("over_time", "ot").replace("total", "pts")
                            result[f"{short}_home"] = val
                if isinstance(away, dict):
                    for q in ["quarter_1", "quarter_2", "quarter_3", "quarter_4", "over_time", "total"]:
                        val = away.get(q)
                        if val is not None:
                            short = q.replace("quarter_", "q").replace("over_time", "ot").replace("total", "pts")
                            result[f"{short}_away"] = val

    return result


def parse_hockey_stats(response: list[dict]) -> dict:
    """Parse hockey game statistics."""
    result: dict = {}

    for item in response:
        if isinstance(item, dict):
            teams = item.get("teams", {})
            scores = item.get("scores", {})

            if isinstance(scores, dict):
                home_scores = scores.get("home")
                away_scores = scores.get("away")
                if home_scores is not None:
                    result["goals_home"] = home_scores
                if away_scores is not None:
                    result["goals_away"] = away_scores

            # Period scores
            periods = item.get("periods", {})
            if isinstance(periods, dict):
                for period_key in ["first", "second", "third", "overtime"]:
                    period = periods.get(period_key)
                    if isinstance(period, dict):
                        result[f"p{period_key[:1]}_home"] = period.get("home")
                        result[f"p{period_key[:1]}_away"] = period.get("away")

    return result


def parse_nfl_stats(response: list[dict]) -> dict:
    """Parse NFL game statistics."""
    result: dict = {}

    for item in response:
        if isinstance(item, dict):
            scores = item.get("scores", {})
            if isinstance(scores, dict):
                home = scores.get("home", {})
                away = scores.get("away", {})
                if isinstance(home, dict):
                    result["pts_home"] = home.get("total")
                    for q in ["quarter_1", "quarter_2", "quarter_3", "quarter_4", "overtime"]:
                        val = home.get(q)
                        if val is not None:
                            short = q.replace("quarter_", "q").replace("overtime", "ot")
                            result[f"{short}_home"] = val
                if isinstance(away, dict):
                    result["pts_away"] = away.get("total")
                    for q in ["quarter_1", "quarter_2", "quarter_3", "quarter_4", "overtime"]:
                        val = away.get(q)
                        if val is not None:
                            short = q.replace("quarter_", "q").replace("overtime", "ot")
                            result[f"{short}_away"] = val

    return result
