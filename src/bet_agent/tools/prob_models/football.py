"""Football (soccer) probability model using Poisson distribution + xG.

Handles: Match Winner, Over/Under, BTTS, and live Bayesian updates.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from bet_agent.tools.prob_models.registry import register

_MAX_GOALS = 10  # Truncate Poisson summation at 10 goals per side


class FootballModel:
    sport = "football"
    TOTAL_MINUTES = 90.0

    def match_outcome_probs(
        self, home_xg: float, away_xg: float
    ) -> dict[str, float]:
        """Calculate P(home win), P(draw), P(away win) from expected goals.

        Uses independent Poisson distributions for each team's goal count.
        """
        home_xg = max(home_xg, 0.0)
        away_xg = max(away_xg, 0.0)
        home_pmf = poisson.pmf(np.arange(_MAX_GOALS + 1), home_xg)
        away_pmf = poisson.pmf(np.arange(_MAX_GOALS + 1), away_xg)

        # Score matrix: prob[i][j] = P(home scores i AND away scores j)
        score_matrix = np.outer(home_pmf, away_pmf)

        p_home = float(np.sum(np.tril(score_matrix, k=-1)))  # home > away
        p_away = float(np.sum(np.triu(score_matrix, k=1)))   # away > home
        p_draw = float(np.sum(np.diag(score_matrix)))

        return {"home": p_home, "draw": p_draw, "away": p_away}

    def over_under_prob(
        self, home_xg: float, away_xg: float, line: float = 2.5
    ) -> float:
        """P(total goals > line). Default line is 2.5."""
        home_xg = max(home_xg, 0.0)
        away_xg = max(away_xg, 0.0)
        total_xg = home_xg + away_xg
        # P(over) = 1 - P(X <= floor(line))
        p_under_or_equal = poisson.cdf(int(line), total_xg)
        return float(1.0 - p_under_or_equal)

    def btts_prob(self, home_xg: float, away_xg: float) -> float:
        """P(both teams score at least 1 goal)."""
        home_xg = max(home_xg, 0.0)
        away_xg = max(away_xg, 0.0)
        p_home_scores = 1.0 - poisson.pmf(0, home_xg)
        p_away_scores = 1.0 - poisson.pmf(0, away_xg)
        return float(p_home_scores * p_away_scores)

    def live_update(
        self,
        pre_match_prob: float,
        live_score: tuple[int, int],
        live_time: float,
        live_stats: dict | None = None,
    ) -> float:
        """Bayesian-Poisson update: adjust win probability with live state.

        Args:
            pre_match_prob: Pre-match model probability of the home team winning.
            live_score: (home_goals, away_goals) at current time.
            live_time: Minutes elapsed (0-90+).
            live_stats: Optional dict with "home_xg" and "away_xg" for live xG.

        Returns:
            Updated probability of home team winning.
        """
        home_goals, away_goals = live_score
        remaining = max(0.0, self.TOTAL_MINUTES - live_time) / self.TOTAL_MINUTES

        if remaining <= 0.0:
            # Match is over — deterministic result
            if home_goals > away_goals:
                return 1.0
            elif home_goals < away_goals:
                return 0.0
            else:
                return 0.0  # Draw — home doesn't "win"

        # Derive expected remaining goals from live xG or from pre-match prob
        # Cap per-90 rates to avoid numerical explosion early in the match
        _MAX_RATE = 5.0  # No team realistically averages > 5 xG per 90

        if live_stats and "home_xg" in live_stats and "away_xg" in live_stats:
            elapsed_fraction = live_time / self.TOTAL_MINUTES
            if elapsed_fraction > 0.05:  # Need at least ~5 min for reliable rate
                rate_home = min(max(live_stats["home_xg"], 0.0) / elapsed_fraction, _MAX_RATE)
                rate_away = min(max(live_stats["away_xg"], 0.0) / elapsed_fraction, _MAX_RATE)
            else:
                # Too early — use raw live xG as full-match estimate
                rate_home = max(live_stats.get("home_xg", 1.3), 0.0)
                rate_away = max(live_stats.get("away_xg", 1.1), 0.0)
        else:
            # Estimate from pre-match probability (rough inverse)
            rate_home = max(0.1, -np.log(1.0 - min(pre_match_prob, 0.99)) * 1.5)
            rate_away = max(0.1, rate_home * 0.8)

        # Expected remaining goals
        lambda_home_rem = rate_home * remaining
        lambda_away_rem = rate_away * remaining

        # P(home wins) = P(home_goals + X > away_goals + Y) where X,Y ~ Poisson
        p_home_win = 0.0
        for extra_h in range(_MAX_GOALS + 1):
            for extra_a in range(_MAX_GOALS + 1):
                total_h = home_goals + extra_h
                total_a = away_goals + extra_a
                if total_h > total_a:
                    p_home_win += (
                        poisson.pmf(extra_h, lambda_home_rem)
                        * poisson.pmf(extra_a, lambda_away_rem)
                    )

        return float(np.clip(p_home_win, 0.0, 1.0))


register(FootballModel())
