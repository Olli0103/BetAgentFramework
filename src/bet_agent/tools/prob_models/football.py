"""Football (soccer) probability model using Dixon-Coles corrected Poisson + xG.

Handles: Match Winner, Over/Under, BTTS, and live Bayesian updates.

The Dixon-Coles correction (1997) adjusts the independent Poisson assumption
for low-scoring outcomes (0:0, 1:0, 0:1, 1:1) where independence breaks down.
Without it, the model systematically underestimates draw probability.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from bet_agent.tools.prob_models.registry import register

_MAX_GOALS = 10  # Truncate Poisson summation at 10 goals per side

# Default Dixon-Coles rho parameter.
# Estimated from large European league datasets (Dixon & Coles, 1997).
# Negative rho increases P(0:0) and P(1:1), decreases P(1:0) and P(0:1).
# Typical fitted values range from -0.13 to -0.08.
_DEFAULT_RHO = -0.13


def _dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
    rho: float,
) -> float:
    """Dixon-Coles correction factor tau for low-scoring outcomes.

    Adjusts the joint probability P(X=x, Y=y) = P(X=x) * P(Y=y) * tau(x,y)
    for scores (0,0), (1,0), (0,1), (1,1). All other scores get tau = 1.0.
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_xg * away_xg * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + away_xg * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + home_xg * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    else:
        return 1.0


class FootballModel:
    sport = "football"
    TOTAL_MINUTES = 90.0

    # Pre-match xG defaults used as fallback when no live stats available.
    # Based on European top-5 league averages (home ~1.45, away ~1.15).
    DEFAULT_HOME_XG = 1.45
    DEFAULT_AWAY_XG = 1.15

    def match_outcome_probs(
        self,
        home_xg: float,
        away_xg: float,
        rho: float = _DEFAULT_RHO,
    ) -> dict[str, float]:
        """Calculate P(home win), P(draw), P(away win) from expected goals.

        Uses Dixon-Coles corrected Poisson: adjusts independent Poisson
        for low-scoring outcomes where the independence assumption breaks.
        """
        home_xg = max(home_xg, 0.0)
        away_xg = max(away_xg, 0.0)
        home_pmf = poisson.pmf(np.arange(_MAX_GOALS + 1), home_xg)
        away_pmf = poisson.pmf(np.arange(_MAX_GOALS + 1), away_xg)

        # Score matrix with Dixon-Coles correction
        score_matrix = np.outer(home_pmf, away_pmf)
        for i in range(min(2, _MAX_GOALS + 1)):
            for j in range(min(2, _MAX_GOALS + 1)):
                score_matrix[i, j] *= _dixon_coles_tau(i, j, home_xg, away_xg, rho)

        # Re-normalize (tau adjustments break exact sum-to-1)
        total = score_matrix.sum()
        if total > 0:
            score_matrix /= total

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
        pre_match_home_xg: float | None = None,
        pre_match_away_xg: float | None = None,
    ) -> float:
        """Bayesian-Poisson update: adjust win probability with live state.

        Uses a principled xG-based approach:
        - If live xG is available, extrapolate the observed rate to remaining time
        - If only pre-match xG is available, scale to remaining time directly
        - Blends live rate with pre-match prior (weighted by elapsed fraction)
          to stabilize early-match estimates

        Args:
            pre_match_prob: Pre-match model probability of the home team winning.
            live_score: (home_goals, away_goals) at current time.
            live_time: Minutes elapsed (0-90+).
            live_stats: Optional dict with "home_xg" and "away_xg" for live xG.
            pre_match_home_xg: Pre-match expected goals for home team.
            pre_match_away_xg: Pre-match expected goals for away team.

        Returns:
            Updated probability of home team winning.
        """
        home_goals, away_goals = live_score
        remaining = max(0.0, self.TOTAL_MINUTES - live_time) / self.TOTAL_MINUTES

        if remaining <= 0.0:
            if home_goals > away_goals:
                return 1.0
            elif home_goals < away_goals:
                return 0.0
            else:
                return 0.0  # Draw — home doesn't "win"

        # Pre-match xG (use provided values or league-average defaults)
        pm_home_xg = pre_match_home_xg if pre_match_home_xg is not None else self.DEFAULT_HOME_XG
        pm_away_xg = pre_match_away_xg if pre_match_away_xg is not None else self.DEFAULT_AWAY_XG

        # Cap per-90 rates to avoid numerical explosion
        _MAX_RATE = 5.0

        elapsed_fraction = live_time / self.TOTAL_MINUTES

        if live_stats and "home_xg" in live_stats and "away_xg" in live_stats:
            if elapsed_fraction > 0.05:
                # Extrapolate observed live xG rate to full-match rate
                live_rate_home = min(max(live_stats["home_xg"], 0.0) / elapsed_fraction, _MAX_RATE)
                live_rate_away = min(max(live_stats["away_xg"], 0.0) / elapsed_fraction, _MAX_RATE)

                # Blend: weight live rate by elapsed fraction, pre-match by remaining
                # Early in match → trust pre-match more; late → trust live rate more
                w_live = elapsed_fraction
                rate_home = w_live * live_rate_home + (1 - w_live) * pm_home_xg
                rate_away = w_live * live_rate_away + (1 - w_live) * pm_away_xg
            else:
                # Too early for reliable live rate — use pre-match xG
                rate_home = pm_home_xg
                rate_away = pm_away_xg
        else:
            # No live xG — scale pre-match xG to remaining time
            rate_home = pm_home_xg
            rate_away = pm_away_xg

        # Expected remaining goals = per-90 rate × remaining fraction
        lambda_home_rem = max(rate_home * remaining, 0.01)
        lambda_away_rem = max(rate_away * remaining, 0.01)

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
