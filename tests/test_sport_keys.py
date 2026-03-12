"""Tests for sport key wildcard matching."""

import pytest

from bet_agent.tools.sport_keys import match_sport_keys


class TestMatchSportKeys:
    def test_exact_match(self):
        result = match_sport_keys(
            ["basketball_nba"],
            ["basketball_nba", "basketball_euroleague"],
        )
        assert result == ["basketball_nba"]

    def test_exact_not_available(self):
        result = match_sport_keys(
            ["basketball_nba"],
            ["basketball_euroleague"],
        )
        assert result == []

    def test_wildcard_tennis_atp(self):
        available = [
            "tennis_atp_french_open",
            "tennis_atp_us_open",
            "tennis_atp_wimbledon",
            "tennis_atp_dubai",
            "tennis_atp_indian_wells",
            "tennis_wta_rome",
            "basketball_nba",
        ]
        result = match_sport_keys(["tennis_atp*"], available)
        assert len(result) == 5
        assert "tennis_atp_dubai" in result
        assert "tennis_atp_indian_wells" in result
        assert "tennis_wta_rome" not in result

    def test_wildcard_tennis_wta(self):
        available = [
            "tennis_atp_french_open",
            "tennis_wta_rome",
            "tennis_wta_madrid",
        ]
        result = match_sport_keys(["tennis_wta*"], available)
        assert result == ["tennis_wta_madrid", "tennis_wta_rome"]

    def test_mixed_exact_and_wildcard(self):
        available = [
            "tennis_atp_dubai",
            "tennis_atp_rome",
            "basketball_nba",
        ]
        result = match_sport_keys(
            ["tennis_atp*", "basketball_nba"],
            available,
        )
        assert "tennis_atp_dubai" in result
        assert "tennis_atp_rome" in result
        assert "basketball_nba" in result

    def test_no_duplicates(self):
        available = ["tennis_atp_dubai"]
        result = match_sport_keys(
            ["tennis_atp*", "tennis_atp_dubai"],
            available,
        )
        assert result == ["tennis_atp_dubai"]

    def test_empty_patterns(self):
        assert match_sport_keys([], ["basketball_nba"]) == []

    def test_empty_available(self):
        assert match_sport_keys(["tennis_atp*"], []) == []
