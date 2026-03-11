"""Tennis probability model using hierarchical point → game → set → match.

Recursive calculation building up from the probability of winning a
single point on serve to the probability of winning the entire match.
Supports best-of-3 and best-of-5 formats.
"""

from __future__ import annotations

from functools import lru_cache

from bet_agent.tools.prob_models.registry import register


class TennisModel:
    sport = "tennis"

    @staticmethod
    @lru_cache(maxsize=4096)
    def _p_game(p_point: float, server_points: int = 0, returner_points: int = 0) -> float:
        """P(server wins game) from current point score.

        Points: 0, 1(15), 2(30), 3(40). Deuce at 3-3.
        """
        if server_points >= 4 and server_points - returner_points >= 2:
            return 1.0
        if returner_points >= 4 and returner_points - server_points >= 2:
            return 0.0

        # Deuce logic
        if server_points >= 3 and returner_points >= 3:
            # At deuce or ad, model as repeated deuce points
            # P(win from deuce) = p^2 / (p^2 + (1-p)^2)
            p = p_point
            return (p * p) / (p * p + (1 - p) * (1 - p))

        if server_points >= 4 or returner_points >= 4:
            # Shouldn't reach here, but safety
            return 0.5

        # Recursive: server wins next point with prob p_point
        p_win = p_point * TennisModel._p_game(p_point, server_points + 1, returner_points)
        p_win += (1 - p_point) * TennisModel._p_game(p_point, server_points, returner_points + 1)
        return p_win

    @staticmethod
    @lru_cache(maxsize=4096)
    def _p_tiebreak(p_serve_a: float, p_serve_b: float, a: int = 0, b: int = 0, a_serving: bool = True) -> float:
        """P(player A wins tiebreak) from current tiebreak score.

        First to 7, win by 2. Service alternates every 2 points (after first).
        """
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0

        # At "deuce" in tiebreak (both >= 6, scores equal) → closed-form
        if a >= 6 and b >= 6 and a == b:
            # Alternating serve every 2 points; approximate with average
            p_a_pt = p_serve_a * 0.5 + (1.0 - p_serve_b) * 0.5
            return (p_a_pt * p_a_pt) / (p_a_pt * p_a_pt + (1 - p_a_pt) * (1 - p_a_pt))

        # Determine who serves this point
        total_points = a + b
        if total_points == 0:
            is_a_serving = a_serving
        else:
            # After first point, alternate every 2 points
            is_a_serving = ((total_points - 1) // 2) % 2 == (0 if a_serving else 1)

        if is_a_serving:
            p_a_wins_point = p_serve_a
        else:
            p_a_wins_point = 1.0 - p_serve_b

        p = p_a_wins_point * TennisModel._p_tiebreak(p_serve_a, p_serve_b, a + 1, b, a_serving)
        p += (1 - p_a_wins_point) * TennisModel._p_tiebreak(p_serve_a, p_serve_b, a, b + 1, a_serving)
        return p

    @staticmethod
    @lru_cache(maxsize=4096)
    def _p_set(p_serve_a: float, p_serve_b: float, a_games: int = 0, b_games: int = 0, a_serving: bool = True) -> float:
        """P(player A wins set) from current game score.

        Standard set: first to 6, tiebreak at 6-6.
        """
        if a_games >= 6 and a_games - b_games >= 2:
            return 1.0
        if b_games >= 6 and b_games - a_games >= 2:
            return 0.0

        if a_games == 6 and b_games == 6:
            return TennisModel._p_tiebreak(p_serve_a, p_serve_b, a_serving=a_serving)

        # Current game: who's serving?
        if a_serving:
            p_a_wins_game = TennisModel._p_game(p_serve_a)
        else:
            p_a_wins_game = 1.0 - TennisModel._p_game(p_serve_b)

        p = p_a_wins_game * TennisModel._p_set(p_serve_a, p_serve_b, a_games + 1, b_games, not a_serving)
        p += (1 - p_a_wins_game) * TennisModel._p_set(p_serve_a, p_serve_b, a_games, b_games + 1, not a_serving)
        return p

    @staticmethod
    @lru_cache(maxsize=1024)
    def _p_match(p_serve_a: float, p_serve_b: float, a_sets: int = 0, b_sets: int = 0, best_of: int = 3, a_serving: bool = True) -> float:
        """P(player A wins match) from current set score."""
        sets_to_win = (best_of + 1) // 2
        if a_sets >= sets_to_win:
            return 1.0
        if b_sets >= sets_to_win:
            return 0.0

        p_a_wins_set = TennisModel._p_set(p_serve_a, p_serve_b, a_serving=a_serving)

        p = p_a_wins_set * TennisModel._p_match(p_serve_a, p_serve_b, a_sets + 1, b_sets, best_of, a_serving)
        p += (1 - p_a_wins_set) * TennisModel._p_match(p_serve_a, p_serve_b, a_sets, b_sets + 1, best_of, a_serving)
        return p

    def match_outcome_probs(
        self,
        p_serve_home: float = 0.65,
        p_serve_away: float = 0.62,
        best_of: int = 3,
    ) -> dict[str, float]:
        """P(player A wins), P(player B wins). No draws in tennis."""
        p_home = self._p_match(p_serve_home, p_serve_away, best_of=best_of)
        return {"home": p_home, "draw": 0.0, "away": 1.0 - p_home}

    def over_under_prob(
        self,
        p_serve_home: float = 0.65,
        p_serve_away: float = 0.62,
        line: float = 22.5,
        best_of: int = 3,
    ) -> float:
        """P(total games > line). Approximation using expected games."""
        # Average games per set ≈ 9-10 for competitive matches
        # This is a simplification; a full model would simulate
        p_match = self._p_match(p_serve_home, p_serve_away, best_of=best_of)
        avg_sets = best_of * 0.7  # rough approximation
        avg_games_per_set = 10.0  # competitive average
        expected_games = avg_sets * avg_games_per_set
        std_games = 4.0

        from scipy.stats import norm
        return float(1.0 - norm.cdf(line, loc=expected_games, scale=std_games))

    def live_update(
        self,
        pre_match_prob: float,
        live_score: tuple[int, int],
        live_time: float,
        live_stats: dict | None = None,
    ) -> float:
        """Update match win probability with live set score.

        Args:
            live_score: (sets_won_A, sets_won_B).
            live_time: Not directly used (tennis is untimed), but can
                       represent elapsed sets or games.
            live_stats: Optional dict with "p_serve_home", "p_serve_away",
                        "best_of", "a_serving".
        """
        sets_a, sets_b = live_score

        p_serve_home = 0.65
        p_serve_away = 0.62
        best_of = 3
        a_serving = True

        if live_stats:
            p_serve_home = live_stats.get("p_serve_home", p_serve_home)
            p_serve_away = live_stats.get("p_serve_away", p_serve_away)
            best_of = live_stats.get("best_of", best_of)
            a_serving = live_stats.get("a_serving", a_serving)

        return self._p_match(
            p_serve_home, p_serve_away,
            a_sets=sets_a, b_sets=sets_b,
            best_of=best_of, a_serving=a_serving,
        )


register(TennisModel())
