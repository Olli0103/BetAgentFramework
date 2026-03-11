"""NBA historical data ingester — Point-in-Time Profiling + Robust Season Parsing.

CSV columns: season, date, regular, playoffs, away, home, score_away,
score_home, q1-q4_away/home, ot_away/home, whos_favored, spread, total,
moneyline_away, moneyline_home, h2_spread, h2_total, id_spread, id_total

PHASE 2 ENHANCEMENT:
  1. Robust season parsing: handles integer (2024), string ("2024"), float
     ("2024.0"), and empty values without crashing.
  2. Extracts box-score data (quarter scores) into match_stats JSONB.
  3. Builds O(1) rolling-stats buffers per team for Point-in-Time profiles.
  4. Chronological sorting + 4-step anti-leakage pipeline.

PROFILE STRUCTURE (TeamDailyStats JSONB):
  Rolling 5-game:   roll_5_points_for, roll_5_points_against, roll_5_win_pct,
                    roll_5_total_points, roll_5_margin
  Rolling 10-game:  roll_10_* (same keys)
  Season counters:  season_wins, season_losses, games_played, win_pct
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
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

# Rolling window sizes
_ROLL_WINDOWS = (5, 10)

# Per-match stats to track in the rolling buffer
_BUFFER_STATS = ("points_for", "points_against", "margin", "total_points")


def _parse_season(raw: str | None, match_date=None) -> str:
    """Robustly parse season from CSV value.

    Handles:
      - Integer: 2024 → "2024"
      - Float string: "2024.0" → "2024"
      - Plain string: "2024" → "2024"
      - Empty/None → derived from match_date year
    """
    if not raw or not str(raw).strip():
        if match_date is not None:
            return str(match_date.year)
        return "unknown"

    s = str(raw).strip()

    # Handle float-like strings: "2024.0" → "2024"
    try:
        val = float(s)
        return str(int(val))
    except (ValueError, TypeError):
        pass

    return s


# ── Team State Buffer ──────────────────────────────────────────────


class _TeamBuffer:
    """Per-team match history buffer for O(1) rolling stat computation.

    CRITICAL: Call snapshot() BEFORE update() for each match.
    """

    __slots__ = ("matches", "season_wins", "season_losses", "total_played")

    def __init__(self) -> None:
        self.matches: deque[dict] = deque(maxlen=max(_ROLL_WINDOWS))
        self.season_wins: int = 0
        self.season_losses: int = 0
        self.total_played: int = 0

    def snapshot(self) -> dict:
        """Compute point-in-time profile from PAST matches only."""
        profile: dict = {
            "games_played": self.total_played,
            "season_wins": self.season_wins,
            "season_losses": self.season_losses,
        }

        if self.total_played > 0:
            profile["win_pct"] = round(self.season_wins / self.total_played, 4)
        else:
            profile["win_pct"] = 0.0

        if not self.matches:
            return profile

        for window in _ROLL_WINDOWS:
            recent = list(self.matches)[-window:]
            n = len(recent)
            if n == 0:
                continue

            for stat in _BUFFER_STATS:
                vals = [m.get(stat) for m in recent if m.get(stat) is not None]
                if vals:
                    profile[f"roll_{window}_{stat}"] = round(sum(vals) / len(vals), 4)

            wins = sum(1 for m in recent if m.get("won"))
            profile[f"roll_{window}_win_pct"] = round(wins / n, 4)

        return profile

    def update(self, match_stats: dict, won: bool) -> None:
        """Add current match to buffer (call AFTER snapshot)."""
        match_stats["won"] = won
        self.matches.append(match_stats)
        self.total_played += 1
        if won:
            self.season_wins += 1
        else:
            self.season_losses += 1


# ── NBA Ingester ───────────────────────────────────────────────────


class NBAIngester(BaseIngester):
    sport = Sport.BASKETBALL
    source_name = "nba_historical_csv"

    def __init__(self) -> None:
        super().__init__()
        self._buffers: dict[str, _TeamBuffer] = defaultdict(_TeamBuffer)

    def parse_file(self, file_path: Path) -> list[dict]:
        rows = read_csv(file_path)
        return [
            r for r in rows
            if r.get("home") and r.get("away") and r.get("date")
        ]

    # ── Chronological ingestion with profile building ───────────────

    def ingest_file(self, session: Session, file_path: Path) -> int:
        """Parse, sort chronologically, build profiles, then upsert.

        ANTI-LEAKAGE: For each match:
          1. Snapshot profiles from PAST matches only
          2. Write profiles to TeamDailyStats
          3. Upsert HistoricalMatch (with box-score data)
          4. Update buffers with THIS match's stats
        """
        self._ensure_resolver(session)

        rows = self.parse_file(file_path)
        if not rows:
            return 0

        # Sort chronologically — the FOUNDATION of anti-leakage
        rows.sort(key=lambda r: self._extract_date_key(r))

        count = 0
        for row in rows:
            try:
                count += self._ingest_row(session, row, str(file_path))
            except Exception as exc:
                logger.warning("Skipping row in %s: %s", file_path.name, exc)

        logger.info(
            "Ingested %d/%d rows from %s (with profiles)",
            count, len(rows), file_path.name,
        )
        return count

    def _extract_date_key(self, row: dict) -> str:
        """Extract a sortable date string from a row."""
        raw = (row.get("date") or "1970-01-01").strip()
        try:
            d = parse_date(raw, formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"])
            return d.isoformat()
        except ValueError:
            return raw

    def _ingest_row(self, session: Session, row: dict, source_file: str) -> int:
        """Ingest a single CSV row with profile building."""
        match_date = parse_date(
            row["date"],
            formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"],
        )

        # Resolve through Ironclad
        home_team = self.resolve_name(row["home"].strip())
        away_team = self.resolve_name(row["away"].strip())

        home_score = safe_int(row.get("score_home"))
        away_score = safe_int(row.get("score_away"))
        home_won = home_score > away_score
        away_won = away_score > home_score

        # Per-team match stats for buffer
        margin_home = home_score - away_score
        total = home_score + away_score

        home_match_data = {
            "points_for": home_score,
            "points_against": away_score,
            "margin": margin_home,
            "total_points": total,
        }
        away_match_data = {
            "points_for": away_score,
            "points_against": home_score,
            "margin": -margin_home,
            "total_points": total,
        }

        # ── STEP 1: Snapshot profiles BEFORE this match (NO LEAKAGE) ──
        home_profile = self._buffers[home_team].snapshot()
        away_profile = self._buffers[away_team].snapshot()

        # ── STEP 2: Write profiles to TeamDailyStats ──────────────────
        self._upsert_daily_stat(session, home_team, match_date, home_profile, "NBA")
        self._upsert_daily_stat(session, away_team, match_date, away_profile, "NBA")

        # ── STEP 3: Build and upsert HistoricalMatch ──────────────────
        model = self.row_to_model(row, source_file)
        model = self._normalize_model(model)
        self._upsert(session, model)

        # ── STEP 4: Update buffers WITH this match (for future use) ───
        self._buffers[home_team].update(home_match_data, home_won)
        self._buffers[away_team].update(away_match_data, away_won)

        return 1

    def row_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        match_date = parse_date(
            row["date"],
            formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"],
        )

        # Robust season parsing — handles int, float string, empty
        season = _parse_season(row.get("season"), match_date)

        is_regular = row.get("regular", "").strip()
        is_playoffs = row.get("playoffs", "").strip()

        # Box-score data (quarter scores + overtime)
        match_stats: dict = {}
        if is_regular:
            match_stats["regular"] = is_regular
        if is_playoffs:
            match_stats["playoffs"] = is_playoffs

        for q in ["q1", "q2", "q3", "q4"]:
            for side in ["away", "home"]:
                val = safe_int(row.get(f"{q}_{side}"), default=-1)
                if val >= 0:
                    match_stats[f"{q}_{side}"] = val
        for side in ["away", "home"]:
            val = safe_int(row.get(f"ot_{side}"), default=-1)
            if val >= 0:
                match_stats[f"ot_{side}"] = val

        home_score = safe_int(row.get("score_home"))
        away_score = safe_int(row.get("score_away"))
        result = "H" if home_score > away_score else ("A" if away_score > home_score else "D")

        # Betting lines
        betting_lines: dict = {}
        for key, col in [
            ("spread", "spread"),
            ("total", "total"),
            ("h2_spread", "h2_spread"),
            ("h2_total", "h2_total"),
        ]:
            val = safe_float(row.get(col))
            if val is not None:
                betting_lines[key] = val

        # Odds/moneylines
        odds: dict = {}
        for key, col in [
            ("moneyline_away", "moneyline_away"),
            ("moneyline_home", "moneyline_home"),
        ]:
            val = safe_float(row.get(col))
            if val is not None:
                odds[key] = val

        if row.get("whos_favored", "").strip():
            betting_lines["favored"] = row["whos_favored"].strip()

        # IDs for spread/total
        advanced_stats: dict = {}
        for key, col in [("id_spread", "id_spread"), ("id_total", "id_total")]:
            val = row.get(col, "").strip()
            if val:
                advanced_stats[key] = val

        return HistoricalMatch(
            sport=Sport.BASKETBALL,
            season=season,
            division="NBA",
            match_date=match_date,
            home_team=row["home"].strip(),
            away_team=row["away"].strip(),
            home_score=home_score,
            away_score=away_score,
            result=result,
            match_stats=match_stats,
            odds=odds,
            betting_lines=betting_lines,
            advanced_stats=advanced_stats,
            source=self.source_name,
            source_file=source_file,
        )
