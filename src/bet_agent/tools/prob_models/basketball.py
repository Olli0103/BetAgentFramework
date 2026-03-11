"""Basketball probability model using pace-adjusted efficiency ratings.

High-scoring sport — Central Limit Theorem applies, so we model the
point differential as a normal distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from bet_agent.tools.prob_models.registry import register

# Average possessions per 48-minute NBA game
_DEFAULT_PACE = 100.0
# Standard deviation of point differential (empirically ~12 pts for NBA)
_DIFF_STD = 12.0
_HOME_ADVANTAGE = 3.0  # Points


class BasketballModel:
    sport = "basketball"
    TOTAL_MINUTES = 48.0  # NBA regulation (adjust for college/FIBA)

    def match_outcome_probs(
        self,
        home_off_rtg: float,
        home_def_rtg: float,
        away_off_rtg: float,
        away_def_rtg: float,
        pace: float = _DEFAULT_PACE,
        league_avg_rtg: float = 110.0,
    ) -> dict[str, float]:
        """P(home win), P(away win) from offensive/defensive ratings.

        Ratings are points per 100 possessions.
        No draw in basketball (OT until winner).
        """
        # Expected points
        home_pts = (home_off_rtg + away_def_rtg) / 2.0 * pace / 100.0
        away_pts = (away_off_rtg + home_def_rtg) / 2.0 * pace / 100.0

        # Apply home advantage
        diff = (home_pts - away_pts) + _HOME_ADVANTAGE

        # P(home wins) = P(diff > 0) using normal distribution
        p_home = float(norm.cdf(diff / _DIFF_STD))

        return {"home": p_home, "draw": 0.0, "away": 1.0 - p_home}

    def expected_total(
        self,
        home_off_rtg: float,
        home_def_rtg: float,
        away_off_rtg: float,
        away_def_rtg: float,
        pace: float = _DEFAULT_PACE,
    ) -> float:
        """Expected total points in the game."""
        home_pts = (home_off_rtg + away_def_rtg) / 2.0 * pace / 100.0
        away_pts = (away_off_rtg + home_def_rtg) / 2.0 * pace / 100.0
        return home_pts + away_pts

    def over_under_prob(
        self,
        home_off_rtg: float = 110.0,
        home_def_rtg: float = 110.0,
        away_off_rtg: float = 110.0,
        away_def_rtg: float = 110.0,
        pace: float = _DEFAULT_PACE,
        line: float = 220.5,
        total_std: float = 12.0,
    ) -> float:
        """P(total points > line)."""
        expected = self.expected_total(
            home_off_rtg, home_def_rtg, away_off_rtg, away_def_rtg, pace
        )
        return float(1.0 - norm.cdf(line, loc=expected, scale=total_std))

    def spread_prob(
        self,
        home_off_rtg: float,
        home_def_rtg: float,
        away_off_rtg: float,
        away_def_rtg: float,
        spread: float,
        pace: float = _DEFAULT_PACE,
    ) -> float:
        """P(home team covers the spread).

        spread: negative means home is favored (e.g., -5.5).
        """
        home_pts = (home_off_rtg + away_def_rtg) / 2.0 * pace / 100.0
        away_pts = (away_off_rtg + home_def_rtg) / 2.0 * pace / 100.0
        diff = (home_pts - away_pts) + _HOME_ADVANTAGE

        # Home covers if actual_diff > -spread (for negative spread)
        return float(norm.cdf((diff + spread) / _DIFF_STD))

    def live_update(
        self,
        pre_match_prob: float,
        live_score: tuple[int, int],
        live_time: float,
        live_stats: dict | None = None,
    ) -> float:
        """Update win probability with live game state.

        Args:
            live_time: Minutes elapsed (0-48, 48+ for OT).
            live_stats: Optional dict with "pace", "home_off_rtg", etc.
        """
        home_pts, away_pts = live_score
        remaining = max(0.0, self.TOTAL_MINUTES - live_time)

        if remaining <= 0.0:
            # Regulation over
            if home_pts > away_pts:
                return 1.0
            elif home_pts < away_pts:
                return 0.0
            else:
                return 0.5  # OT — coin flip approximation

        current_diff = home_pts - away_pts

        # Scale std by sqrt of remaining fraction
        remaining_frac = remaining / self.TOTAL_MINUTES
        scaled_std = _DIFF_STD * np.sqrt(remaining_frac)

        # Expected additional differential from pre-match edge,
        # scaled to remaining time
        pre_match_edge = (pre_match_prob - 0.5) * 2.0 * _DIFF_STD
        expected_remaining_diff = pre_match_edge * remaining_frac

        # P(home wins) = P(current_diff + remaining_diff > 0)
        total_expected_diff = current_diff + expected_remaining_diff

        if scaled_std < 0.01:
            return 1.0 if total_expected_diff > 0 else 0.0

        p_home = float(norm.cdf(total_expected_diff / scaled_std))
        return float(np.clip(p_home, 0.0, 1.0))


register(BasketballModel())
