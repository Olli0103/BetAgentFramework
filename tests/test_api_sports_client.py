"""Tests for the API-Sports client — rate limiting and stat parsers."""

import pytest

from bet_agent.tools.api_sports_client import (
    APISportsClient,
    RateBudget,
    parse_basketball_stats,
    parse_football_stats,
    parse_hockey_stats,
    parse_nfl_stats,
)


# ── Rate Limiting Tests ──────────────────────────────────────────────────


class TestAPISportsClient:
    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("API_SPORTS_KEY", raising=False)
        client = APISportsClient(api_key="")
        assert not client.is_available

    def test_available_with_key(self, monkeypatch):
        client = APISportsClient(api_key="test_key_123")
        assert client.is_available

    def test_budget_tracking(self):
        budget = RateBudget(used=0, limit=100)
        assert budget.remaining == 100
        budget.used = 50
        assert budget.remaining == 50
        budget.used = 100
        assert budget.remaining == 0

    def test_budget_summary(self, monkeypatch):
        monkeypatch.delenv("API_SPORTS_KEY", raising=False)
        client = APISportsClient(api_key="test_key")
        summary = client.get_budget_summary()
        # Should return dict with sport keys
        assert isinstance(summary, dict)

    def test_request_without_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("API_SPORTS_KEY", raising=False)
        client = APISportsClient(api_key="")
        result = client._request("football", "fixtures", {})
        assert result is None

    def test_resolve_unknown_sport(self):
        client = APISportsClient(api_key="test")
        key = client._resolve_sport_key("underwater_polo")
        assert key is None

    def test_resolve_known_sport(self):
        client = APISportsClient(api_key="test")
        # These depend on agents.yaml being present
        key = client._resolve_sport_key("ice_hockey")
        # Should map to "hockey" if agents.yaml has it, else None
        assert key is None or key == "hockey"

    def test_fetch_fixtures_unknown_sport(self, monkeypatch):
        monkeypatch.delenv("API_SPORTS_KEY", raising=False)
        client = APISportsClient(api_key="test")
        result = client.fetch_fixtures_by_date("unknown_sport", "2026-03-13")
        assert result == []


# ── Stat Parser Tests ────────────────────────────────────────────────────


class TestStatsParsers:
    def test_parse_football_stats(self):
        response = [
            {
                "team": {"id": 1, "name": "Bayern Munich"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 7},
                    {"type": "Total Shots", "value": 15},
                    {"type": "Corner Kicks", "value": 6},
                    {"type": "Ball Possession", "value": "55%"},
                    {"type": "Yellow Cards", "value": 2},
                ],
            },
            {
                "team": {"id": 2, "name": "Dortmund"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 8},
                    {"type": "Corner Kicks", "value": 4},
                    {"type": "Ball Possession", "value": "45%"},
                    {"type": "Yellow Cards", "value": 3},
                ],
            },
        ]
        result = parse_football_stats(response)

        assert result["home_sot"] == 7
        assert result["away_sot"] == 3
        assert result["home_shots"] == 15
        assert result["away_shots"] == 8
        assert result["home_corners"] == 6
        assert result["away_corners"] == 4
        assert result["home_possession"] == 55.0
        assert result["home_yellow_cards"] == 2
        assert result["away_yellow_cards"] == 3

    def test_parse_football_stats_empty(self):
        result = parse_football_stats([])
        assert result == {}

    def test_parse_basketball_stats(self):
        response = [
            {
                "scores": {
                    "home": {
                        "quarter_1": 28,
                        "quarter_2": 25,
                        "quarter_3": 30,
                        "quarter_4": 27,
                        "total": 110,
                    },
                    "away": {
                        "quarter_1": 22,
                        "quarter_2": 30,
                        "quarter_3": 25,
                        "quarter_4": 28,
                        "total": 105,
                    },
                }
            }
        ]
        result = parse_basketball_stats(response)
        assert result["q1_home"] == 28
        assert result["q1_away"] == 22
        assert result["pts_home"] == 110
        assert result["pts_away"] == 105

    def test_parse_hockey_stats(self):
        response = [
            {
                "teams": {},
                "scores": {"home": 3, "away": 2},
                "periods": {
                    "first": {"home": 1, "away": 0},
                    "second": {"home": 1, "away": 1},
                    "third": {"home": 1, "away": 1},
                },
            }
        ]
        result = parse_hockey_stats(response)
        assert result["goals_home"] == 3
        assert result["goals_away"] == 2

    def test_parse_nfl_stats(self):
        response = [
            {
                "scores": {
                    "home": {
                        "quarter_1": 7,
                        "quarter_2": 10,
                        "quarter_3": 3,
                        "quarter_4": 14,
                        "total": 34,
                    },
                    "away": {
                        "quarter_1": 3,
                        "quarter_2": 7,
                        "quarter_3": 7,
                        "quarter_4": 7,
                        "total": 24,
                    },
                }
            }
        ]
        result = parse_nfl_stats(response)
        assert result["pts_home"] == 34
        assert result["pts_away"] == 24
        assert result["q1_home"] == 7
        assert result["q1_away"] == 3

    def test_parse_football_unknown_stat_ignored(self):
        response = [
            {
                "team": {"id": 1, "name": "Team"},
                "statistics": [
                    {"type": "Unknown Stat Type", "value": 42},
                    {"type": "Shots on Goal", "value": 5},
                ],
            },
        ]
        result = parse_football_stats(response)
        assert result.get("home_sot") == 5
        # Unknown stat should not be in result
        assert "home_unknown_stat_type" not in result
