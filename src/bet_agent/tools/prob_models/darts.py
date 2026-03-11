"""Darts probability model using leg/set hierarchy.

Models: P(win leg) → P(win set of legs) → P(win match of sets).
Supports various PDC/BDO formats (best-of-N legs, sets-of-legs).
"""

from __future__ import annotations

from functools import lru_cache

from bet_agent.tools.prob_models.registry import register


class DartsModel:
    sport = "darts"

    @staticmethod
    @lru_cache(maxsize=2048)
    def _p_best_of_n(
        p_win_unit: float, target: int, a_wins: int = 0, b_wins: int = 0
    ) -> float:
        """Generic best-of-N model: P(player A wins) needing `target` wins.

        Works for both legs-in-a-set and sets-in-a-match.
        """
        if a_wins >= target:
            return 1.0
        if b_wins >= target:
            return 0.0

        p = p_win_unit * DartsModel._p_best_of_n(p_win_unit, target, a_wins + 1, b_wins)
        p += (1 - p_win_unit) * DartsModel._p_best_of_n(p_win_unit, target, a_wins, b_wins + 1)
        return p

    def match_win_prob_legs(
        self,
        p_leg_home: float,
        legs_to_win: int,
        home_legs: int = 0,
        away_legs: int = 0,
    ) -> float:
        """P(home wins) in a best-of-N legs format (no sets)."""
        return self._p_best_of_n(p_leg_home, legs_to_win, home_legs, away_legs)

    def match_win_prob_sets(
        self,
        p_leg_home: float,
        legs_per_set: int,
        sets_to_win: int,
        home_sets: int = 0,
        away_sets: int = 0,
    ) -> float:
        """P(home wins) in a sets format (e.g., World Championship).

        Each set is best-of-`legs_per_set` legs, match is best-of-N sets.
        """
        legs_to_win_set = (legs_per_set + 1) // 2
        p_win_set = self._p_best_of_n(p_leg_home, legs_to_win_set)
        return self._p_best_of_n(p_win_set, sets_to_win, home_sets, away_sets)

    def match_outcome_probs(
        self,
        p_leg_home: float = 0.55,
        p_leg_away: float = 0.45,
        legs_to_win: int = 6,
        **kwargs,
    ) -> dict[str, float]:
        """P(home), P(away). No draws in darts.

        Uses p_leg_home directly (p_leg_away is 1 - p_leg_home in head-to-head).
        """
        p_home = self.match_win_prob_legs(p_leg_home, legs_to_win)
        return {"home": p_home, "draw": 0.0, "away": 1.0 - p_home}

    def over_under_prob(
        self,
        p_leg_home: float = 0.55,
        legs_to_win: int = 6,
        line: float = 9.5,
        **kwargs,
    ) -> float:
        """P(total legs > line).

        Approximation: expected total legs and std from simulation-like approach.
        """
        # Min legs = legs_to_win, max = 2 * legs_to_win - 1
        min_legs = legs_to_win
        max_legs = 2 * legs_to_win - 1
        expected = (min_legs + max_legs) / 2.0
        std = (max_legs - min_legs) / 4.0  # Rough approximation

        from scipy.stats import norm
        return float(1.0 - norm.cdf(line, loc=expected, scale=max(std, 0.5)))

    def live_update(
        self,
        pre_match_prob: float,
        live_score: tuple[int, int],
        live_time: float,
        live_stats: dict | None = None,
    ) -> float:
        """Update win probability with live leg/set score.

        Args:
            live_score: (legs_won_home, legs_won_away) or (sets_home, sets_away).
            live_time: Not used (darts is untimed).
            live_stats: Optional dict with "p_leg_home", "legs_to_win",
                        "format" ("legs" or "sets"), "legs_per_set", "sets_to_win".
        """
        a_wins, b_wins = live_score

        p_leg_home = 0.55
        legs_to_win = 6
        fmt = "legs"
        legs_per_set = 5
        sets_to_win = 4

        if live_stats:
            p_leg_home = live_stats.get("p_leg_home", p_leg_home)
            legs_to_win = live_stats.get("legs_to_win", legs_to_win)
            fmt = live_stats.get("format", fmt)
            legs_per_set = live_stats.get("legs_per_set", legs_per_set)
            sets_to_win = live_stats.get("sets_to_win", sets_to_win)

        if fmt == "sets":
            return self.match_win_prob_sets(
                p_leg_home, legs_per_set, sets_to_win,
                home_sets=a_wins, away_sets=b_wins,
            )
        else:
            return self.match_win_prob_legs(
                p_leg_home, legs_to_win,
                home_legs=a_wins, away_legs=b_wins,
            )


register(DartsModel())
