"""American football probability model using point differential + normal distribution.

Models expected scoring via power ratings and home advantage.
Standard deviation of NFL game margins ≈ 13.5 points.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from bet_agent.tools.prob_models.registry import register

_DIFF_STD = 13.5       # Empirical std dev of NFL point differentials
_HOME_ADVANTAGE = 2.5   # Points (NFL home advantage, declining over years)
_AVG_TOTAL = 44.0       # Average NFL total points
_TOTAL_STD = 13.0       # Std dev of total points


class AmericanFootballModel:
    sport = "american_football"
    TOTAL_MINUTES = 60.0  # 4 × 15 min quarters

    def match_outcome_probs(
        self,
        home_power_rtg: float,
        away_power_rtg: float,
        home_advantage: float = _HOME_ADVANTAGE,
    ) -> dict[str, float]:
        """P(home win), P(away win) from power ratings.

        Power ratings: expected point differential vs average team.
        E.g., +5 means team scores 5 more points than average opponent.
        """
        expected_diff = (home_power_rtg - away_power_rtg) + home_advantage
        p_home = float(norm.cdf(expected_diff / _DIFF_STD))
        # Ties are extremely rare in NFL (< 0.5%), fold into draw
        return {"home": p_home, "draw": 0.0, "away": 1.0 - p_home}

    def spread_prob(
        self,
        home_power_rtg: float,
        away_power_rtg: float,
        spread: float,
        home_advantage: float = _HOME_ADVANTAGE,
    ) -> float:
        """P(home team covers the spread).

        spread: negative means home is favored (e.g., -7.5).
        """
        expected_diff = (home_power_rtg - away_power_rtg) + home_advantage
        return float(norm.cdf((expected_diff + spread) / _DIFF_STD))

    def over_under_prob(
        self,
        home_power_rtg: float = 0.0,
        away_power_rtg: float = 0.0,
        line: float = 44.5,
        expected_total: float | None = None,
    ) -> float:
        """P(total points > line).

        If expected_total is provided, use it directly.
        Otherwise estimate from power ratings.
        """
        if expected_total is None:
            # Rough: average total + sum of ratings (offensive teams → more points)
            expected_total = _AVG_TOTAL + (home_power_rtg + away_power_rtg) * 0.3
        return float(1.0 - norm.cdf(line, loc=expected_total, scale=_TOTAL_STD))

    def live_update(
        self,
        pre_match_prob: float,
        live_score: tuple[int, int],
        live_time: float,
        live_stats: dict | None = None,
    ) -> float:
        """Update win probability with live game state.

        Args:
            live_time: Minutes elapsed (0-60).
            live_stats: Optional dict with "home_power_rtg", "away_power_rtg".
        """
        home_pts, away_pts = live_score
        remaining = max(0.0, self.TOTAL_MINUTES - live_time)

        if remaining <= 0.0:
            if home_pts > away_pts:
                return 1.0
            elif home_pts < away_pts:
                return 0.0
            else:
                return 0.5  # OT

        current_diff = home_pts - away_pts
        remaining_frac = remaining / self.TOTAL_MINUTES

        # Expected remaining differential from pre-match edge
        pre_match_edge = (pre_match_prob - 0.5) * 2.0 * _DIFF_STD
        expected_remaining_diff = pre_match_edge * remaining_frac

        # Scale std by sqrt of remaining fraction
        scaled_std = _DIFF_STD * np.sqrt(remaining_frac)

        total_expected_diff = current_diff + expected_remaining_diff

        if scaled_std < 0.01:
            return 1.0 if total_expected_diff > 0 else 0.0

        p_home = float(norm.cdf(total_expected_diff / scaled_std))
        return float(np.clip(p_home, 0.0, 1.0))


register(AmericanFootballModel())
