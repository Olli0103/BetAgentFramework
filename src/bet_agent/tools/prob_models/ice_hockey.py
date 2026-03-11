"""Ice hockey probability model using Poisson distribution.

Higher-scoring sport (~5-6 total goals avg), 3 periods of 20 min each.
Handles overtime/shootout probability.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from bet_agent.tools.prob_models.registry import register

_MAX_GOALS = 12  # Higher ceiling for hockey


class IceHockeyModel:
    sport = "ice_hockey"
    TOTAL_MINUTES = 60.0  # 3 × 20 min regulation
    OT_PROB_HOME = 0.52   # Slight home advantage in OT/shootout

    def match_outcome_probs(
        self, home_xg: float, away_xg: float
    ) -> dict[str, float]:
        """P(home win), P(draw), P(away win) in regulation.

        Note: Most hockey markets are "including OT" — use
        match_outcome_probs_with_ot() for moneyline markets.
        """
        home_pmf = poisson.pmf(np.arange(_MAX_GOALS + 1), home_xg)
        away_pmf = poisson.pmf(np.arange(_MAX_GOALS + 1), away_xg)
        score_matrix = np.outer(home_pmf, away_pmf)

        p_home = float(np.sum(np.tril(score_matrix, k=-1)))
        p_away = float(np.sum(np.triu(score_matrix, k=1)))
        p_draw = float(np.sum(np.diag(score_matrix)))

        return {"home": p_home, "draw": p_draw, "away": p_away}

    def match_outcome_probs_with_ot(
        self, home_xg: float, away_xg: float
    ) -> dict[str, float]:
        """Including OT/shootout — no draw possible (moneyline market)."""
        reg = self.match_outcome_probs(home_xg, away_xg)
        # Distribute draw probability to home/away via OT
        p_home = reg["home"] + reg["draw"] * self.OT_PROB_HOME
        p_away = reg["away"] + reg["draw"] * (1.0 - self.OT_PROB_HOME)
        return {"home": p_home, "draw": 0.0, "away": p_away}

    def over_under_prob(
        self, home_xg: float, away_xg: float, line: float = 5.5
    ) -> float:
        """P(total goals > line). Default line is 5.5 for hockey."""
        total_xg = home_xg + away_xg
        return float(1.0 - poisson.cdf(int(line), total_xg))

    def live_update(
        self,
        pre_match_prob: float,
        live_score: tuple[int, int],
        live_time: float,
        live_stats: dict | None = None,
    ) -> float:
        """Bayesian-Poisson update for live ice hockey.

        Args:
            live_time: Minutes elapsed (0-60 regulation, 60+ for OT).
        """
        home_goals, away_goals = live_score
        remaining = max(0.0, self.TOTAL_MINUTES - live_time) / self.TOTAL_MINUTES

        if remaining <= 0.0:
            if home_goals > away_goals:
                return 1.0
            elif home_goals < away_goals:
                return 0.0
            else:
                return self.OT_PROB_HOME  # Regulation tie → OT

        # Estimate per-60 rates
        if live_stats and "home_xg" in live_stats and "away_xg" in live_stats:
            elapsed_frac = live_time / self.TOTAL_MINUTES
            if elapsed_frac > 0:
                rate_home = live_stats["home_xg"] / elapsed_frac
                rate_away = live_stats["away_xg"] / elapsed_frac
            else:
                rate_home = 2.8
                rate_away = 2.6
        else:
            rate_home = max(0.1, -np.log(1.0 - min(pre_match_prob, 0.99)) * 2.5)
            rate_away = max(0.1, rate_home * 0.85)

        lambda_home_rem = rate_home * remaining
        lambda_away_rem = rate_away * remaining

        p_home_win = 0.0
        p_tie = 0.0
        for extra_h in range(_MAX_GOALS + 1):
            for extra_a in range(_MAX_GOALS + 1):
                total_h = home_goals + extra_h
                total_a = away_goals + extra_a
                joint_p = (
                    poisson.pmf(extra_h, lambda_home_rem)
                    * poisson.pmf(extra_a, lambda_away_rem)
                )
                if total_h > total_a:
                    p_home_win += joint_p
                elif total_h == total_a:
                    p_tie += joint_p

        # Ties go to OT — home wins OT with OT_PROB_HOME
        p_home_win += p_tie * self.OT_PROB_HOME
        return float(np.clip(p_home_win, 0.0, 1.0))


register(IceHockeyModel())
