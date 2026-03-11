"""NHL historical data ingester — Pre-computed Feature Mapping to Daily Profiles.

CSV format with 120+ columns including rolling averages and opponent stats.
TWO rows per game (one per team) — merged into a single HistoricalMatch.

PHASE 2 ENHANCEMENT:
  Maps pre-calculated rolling features (roll_3_*, roll_10_*, roll_30_*) from
  the CSV directly into TeamDailyStats JSONB profiles. These features are
  already point-in-time in the source data (computed before each game).

PROFILE STRUCTURE (TeamDailyStats JSONB):
  Rolling features:  roll_3_shots, roll_10_goals_for, roll_30_win_pct, ...
  Season stats:      season_win_pct, season_goals_for, season_goal_diff, ...
  Context:           rest_days, pre_game_point_pct
  Opponent context:  opp_roll_3_goals_for, opp_season_win_pct, opp_rest_days

Also supports web ingestion from hockey-statistics.com/data/.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from bet_agent.db.models import HistoricalMatch, Sport
from bet_agent.ingest.base import (
    BaseIngester,
    parse_date,
    read_csv,
    safe_float,
    safe_int,
)

logger = logging.getLogger(__name__)

# Columns to extract into match_stats (per-team, game-level)
MATCH_STAT_COLS = [
    "shots", "power_play_goals", "power_play_opportunities",
    "faceoff_win_pct", "hits", "blocked_shots", "pim",
    "giveaways", "takeaways",
]

# Rolling average prefixes — these are PRE-COMPUTED in the CSV
ROLL_PREFIXES = ["roll_3", "roll_10", "roll_30"]
ROLL_STAT_COLS = [
    "shots", "power_play_goals", "power_play_opportunities",
    "faceoff_win_pct", "hits", "blocked_shots", "pim",
    "giveaways", "takeaways", "goals_for", "goals_against",
]

# Season-level columns
SEASON_COLS = [
    "season_games_played", "season_wins", "season_goals_for",
    "season_goals_against", "season_win_pct", "season_goal_diff",
]

# Opponent-level columns
OPP_COLS = [
    "opp_season_wins", "opp_season_goals_for",
    "opp_season_goals_against", "opp_season_win_pct",
    "opp_season_goal_diff", "opp_rest_days",
    "opp_pre_game_point_pct",
]

# Differential columns (computed from home perspective)
DIFF_COLS = [
    "rest_diff", "win_pct_diff", "goal_diff_diff",
    "record_wins_diff", "record_losses_diff",
    "roll_3_goal_diff", "opp_roll_3_goal_diff",
    "roll_10_goal_diff", "opp_roll_10_goal_diff",
    "roll_30_goal_diff", "opp_roll_30_goal_diff",
]


class NHLIngester(BaseIngester):
    sport = Sport.ICE_HOCKEY
    source_name = "nhl_historical_csv"

    def parse_file(self, file_path: Path) -> list[dict]:
        """Parse CSV and merge two-rows-per-game into one dict per game."""
        raw_rows = read_csv(file_path)

        # Group by game_id
        games: dict[str, list[dict]] = {}
        for row in raw_rows:
            gid = (row.get("game_id") or "").strip()
            if not gid:
                continue
            games.setdefault(gid, []).append(row)

        merged: list[dict] = []
        for gid, rows in games.items():
            if len(rows) < 2:
                logger.warning("Game %s has only %d rows, skipping", gid, len(rows))
                continue

            # Identify home and away
            home_row = next((r for r in rows if r.get("is_home", "").strip() in ("1", "True", "true", "TRUE")), None)
            away_row = next((r for r in rows if r.get("is_home", "").strip() in ("0", "False", "false", "FALSE")), None)

            if not home_row or not away_row:
                home_row, away_row = rows[0], rows[1]

            merged.append({
                "game_id": gid,
                "home": home_row,
                "away": away_row,
            })

        return merged

    def row_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        home = row["home"]
        away = row["away"]

        match_date = parse_date(
            home["date"],
            formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"],
        )
        season = (home.get("season") or "").strip() or str(match_date.year)

        home_goals = safe_int(home.get("goals_for"))
        away_goals = safe_int(away.get("goals_for"))
        result = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")

        # Match stats (game-level)
        match_stats: dict = {
            "game_id": row["game_id"],
            "venue": (home.get("venue") or "").strip(),
            "attendance": safe_int(home.get("attendance")),
        }

        for prefix, team_row in [("home", home), ("away", away)]:
            for col in MATCH_STAT_COLS:
                val = safe_float(team_row.get(col))
                if val is not None:
                    match_stats[f"{prefix}_{col}"] = val

        # Betting lines
        betting_lines: dict = {}
        for key, col in [
            ("spread", "spread"),
            ("over_under", "over_under"),
            ("favorite_moneyline", "favorite_moneyline"),
        ]:
            val = safe_float(home.get(col))
            if val is not None:
                betting_lines[key] = val

        # Advanced stats: rolling averages, season stats, opponent stats
        advanced_stats: dict = {}

        for prefix, team_row in [("home", home), ("away", away)]:
            # Season-level stats
            for col in SEASON_COLS:
                val = safe_float(team_row.get(col))
                if val is not None:
                    advanced_stats[f"{prefix}_{col}"] = val

            # Pre-game metrics
            val = safe_float(team_row.get("pre_game_point_pct"))
            if val is not None:
                advanced_stats[f"{prefix}_pre_game_point_pct"] = val

            val = safe_int(team_row.get("rest_days"), default=-1)
            if val >= 0:
                advanced_stats[f"{prefix}_rest_days"] = val

            # Rolling averages (pre-computed in CSV)
            for roll in ROLL_PREFIXES:
                for stat in ROLL_STAT_COLS:
                    col_name = f"{roll}_{stat}"
                    val = safe_float(team_row.get(col_name))
                    if val is not None:
                        advanced_stats[f"{prefix}_{col_name}"] = val

            # Opponent rolling averages (opp_roll_*)
            for roll in ROLL_PREFIXES:
                for stat in ROLL_STAT_COLS:
                    col_name = f"opp_{roll}_{stat}"
                    val = safe_float(team_row.get(col_name))
                    if val is not None:
                        advanced_stats[f"{prefix}_{col_name}"] = val

            # Opponent season stats
            for col in OPP_COLS:
                val = safe_float(team_row.get(col))
                if val is not None:
                    advanced_stats[f"{prefix}_{col}"] = val

        # Differential columns (from home perspective)
        for col in DIFF_COLS:
            val = safe_float(home.get(col))
            if val is not None:
                advanced_stats[col] = val

        return HistoricalMatch(
            sport=Sport.ICE_HOCKEY,
            season=season,
            division="NHL",
            match_date=match_date,
            home_team=(home.get("team_name") or "").strip(),
            away_team=(away.get("team_name") or "").strip(),
            home_score=home_goals,
            away_score=away_goals,
            result=result,
            match_stats=match_stats,
            odds={},
            betting_lines=betting_lines,
            advanced_stats=advanced_stats,
            source=self.source_name,
            source_file=source_file,
        )

    # ── Profile-building ingestion override ─────────────────────────

    def ingest_file(self, session: Session, file_path: Path) -> int:
        """Parse file, build team profiles, then upsert matches.

        For each game:
          1. Extract pre-computed rolling features for home & away teams
          2. Write team profiles to TeamDailyStats JSONB
          3. Upsert the HistoricalMatch (via base class enforcement)

        The CSV rolling features are already point-in-time (computed before
        each game in the source data), so no additional anti-leakage logic
        is needed — we just map them faithfully.
        """
        self._ensure_resolver(session)

        rows = self.parse_file(file_path)
        if not rows:
            return 0

        # Sort chronologically for deterministic processing
        rows.sort(key=lambda r: r["home"].get("date", "1970-01-01"))

        count = 0
        for row in rows:
            try:
                home_raw = row["home"]
                away_raw = row["away"]

                match_date = parse_date(
                    home_raw["date"],
                    formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"],
                )

                # Resolve team names through Ironclad
                home_team = self.resolve_name((home_raw.get("team_name") or "").strip())
                away_team = self.resolve_name((away_raw.get("team_name") or "").strip())

                # ── Build profiles from pre-computed CSV features ───
                home_profile = self._extract_team_profile(home_raw)
                away_profile = self._extract_team_profile(away_raw)

                # Write profiles to TeamDailyStats
                self._upsert_daily_stat(session, home_team, match_date, home_profile, "NHL")
                self._upsert_daily_stat(session, away_team, match_date, away_profile, "NHL")

                # ── Upsert HistoricalMatch (with name normalization) ──
                model = self.row_to_model(row, str(file_path))
                model = self._normalize_model(model)
                self._upsert(session, model)
                count += 1

            except Exception as exc:
                logger.warning(
                    "Skipping game %s: %s", row.get("game_id", "?"), exc,
                )

        logger.info(
            "Ingested %d/%d games from %s (with team profiles)",
            count, len(rows), file_path.name,
        )
        return count

    @staticmethod
    def _extract_team_profile(team_row: dict) -> dict:
        """Extract all pre-computed features from a CSV row into a profile dict.

        These features were pre-computed BEFORE the game in the source data,
        so they are already point-in-time (no look-ahead leakage).

        Captures ALL roll_*, opp_roll_*, season_*, opp_season_* columns
        dynamically, plus context features (rest_days, pre_game_point_pct).
        This is intentionally greedy — we want every pre-computed feature
        the CSV offers, not just a hardcoded subset.
        """
        profile: dict = {}

        # Skip these — they are game-level outcomes, not pre-game features
        _SKIP_KEYS = {"game_id", "date", "team_name", "is_home", "goals_for",
                      "venue", "attendance", "season"}

        for col, val_str in team_row.items():
            if col in _SKIP_KEYS or not val_str:
                continue

            # Capture all rolling, season, and opponent features
            is_profile_col = (
                col.startswith("roll_")
                or col.startswith("opp_roll_")
                or col.startswith("opp_season_")
                or col.startswith("season_")
                or col in ("pre_game_point_pct", "rest_days")
            )

            if is_profile_col:
                if col == "rest_days":
                    val = safe_int(val_str, default=-1)
                    if val >= 0:
                        profile[col] = val
                else:
                    val = safe_float(val_str)
                    if val is not None:
                        profile[col] = val

        return profile


def fetch_hockey_statistics(
    season: str = "2025-2026",
    output_dir: Path | None = None,
) -> Path | None:
    """Download CSV data from hockey-statistics.com.

    Args:
        season: Season string like "2025-2026".
        output_dir: Directory to save downloaded CSV. Defaults to cwd.

    Returns:
        Path to downloaded file, or None on failure.
    """
    import requests

    url = f"https://hockey-statistics.com/data/csv/{season}.csv"
    logger.info("Fetching NHL data from %s", url)

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Failed to fetch hockey-statistics.com: %s", exc)
        return None

    if output_dir is None:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"nhl_{season}.csv"
    out_path.write_bytes(resp.content)
    logger.info("Saved NHL data to %s (%d bytes)", out_path, len(resp.content))
    return out_path
