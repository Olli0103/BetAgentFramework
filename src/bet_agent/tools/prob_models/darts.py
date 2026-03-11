"""Darts probability model using leg/set hierarchy.

Models: P(win leg) → P(win set of legs) → P(win match of sets).
Supports various PDC/BDO formats (best-of-N legs, sets-of-legs).
Throw advantage alternates between legs/sets.
"""

from __future__ import annotations

from functools import lru_cache

from bet_agent.tools.prob_models.registry import register

_ROUND = 4  # decimal places for float rounding (cache-friendly)


class DartsModel:
    sport = "darts"

    @staticmethod
    @lru_cache(maxsize=4096)
    def _p_best_of_n_alt(
        p_a_throw: float, p_b_throw: float,
        target: int, a_wins: int = 0, b_wins: int = 0,
        a_throwing: bool = True,
    ) -> float:
        """P(player A wins) best-of-N with alternating throw advantage.

        Args:
            p_a_throw: P(A wins leg when A throws first).
            p_b_throw: P(B wins leg when B throws first).
            target: Legs/sets needed to win.
            a_throwing: Whether A throws first in the current leg.
        """
        if a_wins >= target:
            return 1.0
        if b_wins >= target:
            return 0.0

        if a_throwing:
            p_a_wins = p_a_throw
        else:
            p_a_wins = 1.0 - p_b_throw

        p = p_a_wins * DartsModel._p_best_of_n_alt(
            p_a_throw, p_b_throw, target, a_wins + 1, b_wins, not a_throwing,
        )
        p += (1 - p_a_wins) * DartsModel._p_best_of_n_alt(
            p_a_throw, p_b_throw, target, a_wins, b_wins + 1, not a_throwing,
        )
        return p

    def match_win_prob_legs(
        self,
        p_leg_home: float,
        p_leg_away: float | None = None,
        legs_to_win: int = 6,
        home_legs: int = 0,
        away_legs: int = 0,
        home_throwing: bool = True,
    ) -> float:
        """P(home wins) in a best-of-N legs format (no sets).

        Args:
            p_leg_home: P(home wins leg when home throws first).
            p_leg_away: P(away wins leg when away throws first).
                        Defaults to 1 - p_leg_home (no throw advantage).
        """
        if p_leg_away is None:
            p_leg_away = 1.0 - p_leg_home
        p_leg_home = round(p_leg_home, _ROUND)
        p_leg_away = round(p_leg_away, _ROUND)
        return self._p_best_of_n_alt(
            p_leg_home, p_leg_away, legs_to_win,
            home_legs, away_legs, home_throwing,
        )

    @staticmethod
    @lru_cache(maxsize=2048)
    def _p_match_of_sets(
        p_set_home_throws: float, p_set_away_throws: float,
        target: int, home_sets: int = 0, away_sets: int = 0,
        home_throwing: bool = True,
    ) -> float:
        """P(home wins match) in sets format with alternating first-throw."""
        if home_sets >= target:
            return 1.0
        if away_sets >= target:
            return 0.0

        p_home_wins_set = p_set_home_throws if home_throwing else p_set_away_throws

        p = p_home_wins_set * DartsModel._p_match_of_sets(
            p_set_home_throws, p_set_away_throws, target,
            home_sets + 1, away_sets, not home_throwing,
        )
        p += (1 - p_home_wins_set) * DartsModel._p_match_of_sets(
            p_set_home_throws, p_set_away_throws, target,
            home_sets, away_sets + 1, not home_throwing,
        )
        return p

    def match_win_prob_sets(
        self,
        p_leg_home: float,
        p_leg_away: float | None = None,
        legs_per_set: int = 5,
        sets_to_win: int = 4,
        home_sets: int = 0,
        away_sets: int = 0,
        home_throwing_first: bool = True,
    ) -> float:
        """P(home wins) in a sets format (e.g., World Championship).

        Each set is best-of-`legs_per_set` legs, match is best-of-N sets.
        Throw advantage alternates at both leg and set level.

        Args:
            p_leg_home: P(home wins leg when home throws first).
            p_leg_away: P(away wins leg when away throws first).
                        Defaults to 1 - p_leg_home (no throw advantage).
        """
        if p_leg_away is None:
            p_leg_away = 1.0 - p_leg_home
        p_leg_home = round(p_leg_home, _ROUND)
        p_leg_away = round(p_leg_away, _ROUND)

        legs_target = (legs_per_set + 1) // 2

        # P(home wins set) depends on who throws first in the set
        p_set_home_throws = self._p_best_of_n_alt(
            p_leg_home, p_leg_away, legs_target, a_throwing=True,
        )
        p_set_away_throws = self._p_best_of_n_alt(
            p_leg_home, p_leg_away, legs_target, a_throwing=False,
        )

        return self._p_match_of_sets(
            round(p_set_home_throws, _ROUND),
            round(p_set_away_throws, _ROUND),
            sets_to_win, home_sets, away_sets, home_throwing_first,
        )

    def match_outcome_probs(
        self,
        p_leg_home: float = 0.55,
        p_leg_away: float | None = None,
        legs_to_win: int = 6,
        **kwargs,
    ) -> dict[str, float]:
        """P(home), P(away). No draws in darts.

        Args:
            p_leg_home: P(home wins leg when home throws first).
            p_leg_away: P(away wins leg when away throws first).
                        Defaults to 1 - p_leg_home (no throw advantage).
        """
        p_home = self.match_win_prob_legs(p_leg_home, p_leg_away, legs_to_win)
        return {"home": p_home, "draw": 0.0, "away": 1.0 - p_home}

    def over_under_prob(
        self,
        p_leg_home: float = 0.55,
        p_leg_away: float | None = None,
        legs_to_win: int = 6,
        line: float = 9.5,
        **kwargs,
    ) -> float:
        """P(total legs > line).

        Uses normal approximation with expected legs derived from match
        competitiveness.
        """
        min_legs = legs_to_win
        max_legs = 2 * legs_to_win - 1
        # More competitive matches → more legs expected
        p_home = self.match_win_prob_legs(p_leg_home, p_leg_away, legs_to_win)
        competitiveness = 1.0 - abs(p_home - 0.5) * 2  # 0..1
        expected = min_legs + (max_legs - min_legs) * competitiveness * 0.5
        std = max((max_legs - min_legs) / 4.0, 0.5)

        from scipy.stats import norm

        return float(1.0 - norm.cdf(line, loc=expected, scale=std))

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
            live_stats: Optional dict with "p_leg_home", "p_leg_away",
                        "legs_to_win", "format" ("legs" or "sets"),
                        "legs_per_set", "sets_to_win", "home_throwing".
        """
        a_wins, b_wins = live_score

        p_leg_home = 0.55
        p_leg_away: float | None = None
        legs_to_win = 6
        fmt = "legs"
        legs_per_set = 5
        sets_to_win = 4
        home_throwing = True

        if live_stats:
            p_leg_home = live_stats.get("p_leg_home", p_leg_home)
            p_leg_away = live_stats.get("p_leg_away", p_leg_away)
            legs_to_win = live_stats.get("legs_to_win", legs_to_win)
            fmt = live_stats.get("format", fmt)
            legs_per_set = live_stats.get("legs_per_set", legs_per_set)
            sets_to_win = live_stats.get("sets_to_win", sets_to_win)
            home_throwing = live_stats.get("home_throwing", home_throwing)

        if fmt == "sets":
            return self.match_win_prob_sets(
                p_leg_home, p_leg_away, legs_per_set, sets_to_win,
                home_sets=a_wins, away_sets=b_wins,
                home_throwing_first=home_throwing,
            )
        else:
            return self.match_win_prob_legs(
                p_leg_home, p_leg_away, legs_to_win,
                home_legs=a_wins, away_legs=b_wins,
                home_throwing=home_throwing,
            )


register(DartsModel())
