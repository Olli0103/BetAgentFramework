"""NFL historical data ingester — Point-in-Time Profiling + Closing Line Extraction.

XLSX format with columns: Date, Home Team, Away Team, Home Score, Away Score,
Overtime?, Playoff Game?, Neutral Venue?,
Home Odds Open/Min/Max/Close, Away Odds Open/Min/Max/Close,
Home Line Open/Min/Max/Close, Away Line Open/Min/Max/Close,
Home Line Odds Open/Min/Max/Close, Away Line Odds Open/Min/Max/Close,
Total Score Open/Min/Max/Close,
Total Score Over/Under Open/Min/Max/Close, Notes

PHASE 2 ENHANCEMENT:
  1. Extracts Closing Lines (Home Odds Close, Away Odds Close, Home Line Close)
     into match_stats for later CLV (Closing Line Value) calculations.
  2. Builds O(1) rolling-stats buffers per team for Point-in-Time profiles.
  3. Chronological sorting + 4-step anti-leakage pipeline.

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
from bet_agent.ingest.base import BaseIngester, parse_date, safe_float, safe_int

logger = logging.getLogger(__name__)

# Rolling window sizes
_ROLL_WINDOWS = (5, 10)

# Per-match stats to track in the rolling buffer
_BUFFER_STATS = ("points_for", "points_against", "margin", "total_points")

# Closing line keys — CRITICAL for CLV analysis
_CLOSING_LINE_KEYS = {
    "home_odds_close": "Home Odds Close",
    "away_odds_close": "Away Odds Close",
    "home_line_close": "Home Line Close",
    "away_line_close": "Away Line Close",
    "home_line_odds_close": "Home Line Odds Close",
    "away_line_odds_close": "Away Line Odds Close",
    "total_close": "Total Score Close",
    "total_over_close": "Total Score Over Close",
    "total_under_close": "Total Score Under Close",
}


def _read_xlsx(file_path: Path) -> list[dict]:
    """Read an XLSX file into a list of dicts using openpyxl."""
    from openpyxl import load_workbook

    wb = load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return []

    # First row is header
    header = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(rows[0])]
    result = []
    for row in rows[1:]:
        d = {}
        for i, val in enumerate(row):
            if i < len(header):
                d[header[i]] = str(val).strip() if val is not None else ""
        result.append(d)
    return result


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


# ── NFL Ingester ───────────────────────────────────────────────────


class NFLIngester(BaseIngester):
    sport = Sport.AMERICAN_FOOTBALL
    source_name = "nfl_historical_xlsx"

    def __init__(self, default_season: str = "unknown") -> None:
        super().__init__()
        self._default_season = default_season
        self._buffers: dict[str, _TeamBuffer] = defaultdict(_TeamBuffer)
        # Track last-seen season per team for reset detection
        self._team_season: dict[str, str] = {}

    def parse_file(self, file_path: Path) -> list[dict]:
        rows = _read_xlsx(file_path)
        return [
            r for r in rows
            if r.get("Home Team") and r.get("Away Team") and r.get("Date")
        ]

    def ingest_directory(self, session, dir_path, glob="*.xlsx"):
        return super().ingest_directory(session, dir_path, glob=glob)

    # ── Chronological ingestion with profile building ───────────────

    def ingest_file(self, session: Session, file_path: Path) -> int:
        """Parse, sort chronologically, build profiles, then upsert.

        ANTI-LEAKAGE: For each match:
          1. Snapshot profiles from PAST matches only
          2. Write profiles to TeamDailyStats
          3. Upsert HistoricalMatch (with Closing Lines in match_stats)
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
            "Ingested %d/%d rows from %s (with profiles + closing lines)",
            count, len(rows), file_path.name,
        )
        return count

    def _extract_date_key(self, row: dict) -> str:
        """Extract a sortable date string from a row."""
        raw = (row.get("Date") or "1970-01-01").strip()
        try:
            d = parse_date(
                raw,
                formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"],
            )
            return d.isoformat()
        except ValueError:
            return raw

    def _ingest_row(self, session: Session, row: dict, source_file: str) -> int:
        """Ingest a single XLSX row with profile building + closing lines."""
        match_date = parse_date(
            row["Date"],
            formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"],
        )

        # Resolve through Ironclad
        home_team = self.resolve_name(row["Home Team"].strip())
        away_team = self.resolve_name(row["Away Team"].strip())

        # Validate scores — skip rows with missing/empty scores (avoid phantom 0-0)
        raw_home_score = (row.get("Home Score") or "").strip()
        raw_away_score = (row.get("Away Score") or "").strip()
        if not raw_home_score or not raw_away_score:
            logger.debug("Skipping row with missing score: %s vs %s", home_team, away_team)
            return 0

        home_score = safe_int(raw_home_score)
        away_score = safe_int(raw_away_score)
        home_won = home_score > away_score
        away_won = away_score > home_score

        # ── Season-reset detection ────────────────────────────────────
        # NFL season: Aug-Feb. Derive season key from match_date.
        season_key = self._derive_season("", match_date)
        for team in (home_team, away_team):
            prev = self._team_season.get(team)
            if prev is not None and prev != season_key:
                self._buffers[team] = _TeamBuffer()
                logger.debug("Season reset for %s: %s → %s", team, prev, season_key)
            self._team_season[team] = season_key

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
        self._upsert_daily_stat(session, home_team, match_date, home_profile, "NFL")
        self._upsert_daily_stat(session, away_team, match_date, away_profile, "NFL")

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
            row["Date"],
            formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"],
        )

        season = self._derive_season(source_file, match_date)
        home_score = safe_int(row.get("Home Score"))
        away_score = safe_int(row.get("Away Score"))
        result = "H" if home_score > away_score else ("A" if away_score > home_score else "D")

        # Match metadata + CLOSING LINES (critical for CLV)
        match_stats: dict = {}
        if row.get("Overtime?") and row["Overtime?"].lower() in ("yes", "true", "1"):
            match_stats["overtime"] = True
        if row.get("Playoff Game?") and row["Playoff Game?"].lower() in ("yes", "true", "1"):
            match_stats["playoff"] = True
        if row.get("Neutral Venue?") and row["Neutral Venue?"].lower() in ("yes", "true", "1"):
            match_stats["neutral_venue"] = True
        if row.get("Notes"):
            match_stats["notes"] = row["Notes"]

        # ── CLOSING LINES — HFT Gold for CLV analysis ─────────────
        for key, col in _CLOSING_LINE_KEYS.items():
            val = safe_float(row.get(col))
            if val is not None:
                match_stats[key] = val

        # Moneyline odds (open/min/max/close for each side)
        odds: dict = {}
        for side in ["Home", "Away"]:
            for variant in ["Open", "Min", "Max", "Close"]:
                col = f"{side} Odds {variant}"
                val = safe_float(row.get(col))
                if val is not None:
                    odds[f"{side.lower()}_odds_{variant.lower()}"] = val

        # Spread / line data
        betting_lines: dict = {}
        for side in ["Home", "Away"]:
            for variant in ["Open", "Min", "Max", "Close"]:
                # Point spread
                col = f"{side} Line {variant}"
                val = safe_float(row.get(col))
                if val is not None:
                    betting_lines[f"{side.lower()}_line_{variant.lower()}"] = val

                # Spread odds (juice)
                col = f"{side} Line Odds {variant}"
                val = safe_float(row.get(col))
                if val is not None:
                    betting_lines[f"{side.lower()}_line_odds_{variant.lower()}"] = val

        # Total score lines
        for variant in ["Open", "Min", "Max", "Close"]:
            col = f"Total Score {variant}"
            val = safe_float(row.get(col))
            if val is not None:
                betting_lines[f"total_{variant.lower()}"] = val

            # Over/under odds
            for ou in ["Over", "Under"]:
                col = f"Total Score {ou} {variant}"
                val = safe_float(row.get(col))
                if val is not None:
                    betting_lines[f"total_{ou.lower()}_{variant.lower()}"] = val

        return HistoricalMatch(
            sport=Sport.AMERICAN_FOOTBALL,
            season=season,
            division="NFL",
            match_date=match_date,
            home_team=row["Home Team"].strip(),
            away_team=row["Away Team"].strip(),
            home_score=home_score,
            away_score=away_score,
            result=result,
            match_stats=match_stats,
            odds=odds,
            betting_lines=betting_lines,
            advanced_stats={},
            source=self.source_name,
            source_file=source_file,
        )

    def _derive_season(self, source_file: str, match_date) -> str:
        """NFL season: Aug-Feb spans two calendar years."""
        import re
        name = Path(source_file).stem
        m = re.search(r"(20\d{2})", name)
        if m:
            return m.group(1)
        year = match_date.year
        month = match_date.month
        if month <= 6:
            return str(year - 1)
        return str(year)
