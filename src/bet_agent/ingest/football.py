"""Football (soccer) historical data ingester — Point-in-Time Profiling.

Source: football-data.co.uk CSV format.
Columns: Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR,
         HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR,
         B365H, B365D, B365A, BWH, BWD, BWA, ... (multi-bookmaker odds)

PHASE 2 ENHANCEMENT:
  Builds O(1) rolling-stats buffers per team. For each match:
    1. Snapshot profile from PAST matches only (no leakage)
    2. Write profile to TeamDailyStats JSONB
    3. Upsert HistoricalMatch (with Ironclad name normalization)
    4. Update buffer with THIS match's stats

PROFILE STRUCTURE (TeamDailyStats JSONB):
  Rolling 5-match:   roll_5_goals_for, roll_5_goals_against, roll_5_shots,
                     roll_5_shots_target, roll_5_corners, roll_5_fouls,
                     roll_5_win_pct, roll_5_total_goals
  Rolling 10-match:  roll_10_* (same keys)
  Rolling 20-match:  roll_20_* (same keys)
  Season counters:   season_wins, season_draws, season_losses,
                     games_played, win_pct
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

# Map division codes to league names
DIVISION_MAP = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "I1": "Serie A",
    "I2": "Serie B",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Jupiler Pro League",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
    "G1": "Super League Greece",
    "SC0": "Scottish Premiership",
}

# Bookmaker odds column prefixes
BOOKMAKER_ODDS = {
    "b365": ("B365H", "B365D", "B365A"),
    "bw": ("BWH", "BWD", "BWA"),
    "gb": ("GBH", "GBD", "GBA"),
    "iw": ("IWH", "IWD", "IWA"),
    "lb": ("LBH", "LBD", "LBA"),
    "sb": ("SBH", "SBD", "SBA"),
    "wh": ("WHH", "WHD", "WHA"),
    "sj": ("SJH", "SJD", "SJA"),
    "vc": ("VCH", "VCD", "VCA"),
    "bs": ("BSH", "BSD", "BSA"),
}

# Rolling window sizes for profile computation
_ROLL_WINDOWS = (5, 10, 20)

# Per-match stats to track in the rolling buffer
_BUFFER_STATS = ("goals_for", "goals_against", "shots", "shots_target",
                 "corners", "fouls", "yellows", "reds")


# ── Team State Buffer ──────────────────────────────────────────────


class _TeamBuffer:
    """Per-team match history buffer for O(1) rolling stat computation.

    Maintains a deque of recent match stats and season-level counters.
    CRITICAL: Call snapshot() BEFORE update() for each match.
    """

    __slots__ = ("matches", "season_wins", "season_draws", "season_losses", "total_played")

    def __init__(self) -> None:
        self.matches: deque[dict] = deque(maxlen=max(_ROLL_WINDOWS))
        self.season_wins: int = 0
        self.season_draws: int = 0
        self.season_losses: int = 0
        self.total_played: int = 0

    def snapshot(self) -> dict:
        """Compute point-in-time profile from PAST matches only."""
        profile: dict = {
            "games_played": self.total_played,
            "season_wins": self.season_wins,
            "season_draws": self.season_draws,
            "season_losses": self.season_losses,
        }

        if self.total_played > 0:
            profile["win_pct"] = round(self.season_wins / self.total_played, 4)
        else:
            profile["win_pct"] = 0.0

        if not self.matches:
            return profile

        # Rolling averages for each window size
        for window in _ROLL_WINDOWS:
            recent = list(self.matches)[-window:]
            n = len(recent)
            if n == 0:
                continue

            for stat in _BUFFER_STATS:
                vals = [m.get(stat) for m in recent if m.get(stat) is not None]
                if vals:
                    profile[f"roll_{window}_{stat}"] = round(sum(vals) / len(vals), 4)

            # Total goals (for + against)
            gf_vals = [m.get("goals_for", 0) for m in recent]
            ga_vals = [m.get("goals_against", 0) for m in recent]
            profile[f"roll_{window}_total_goals"] = round(
                (sum(gf_vals) + sum(ga_vals)) / n, 4
            )

            # Win percentage within the window
            wins = sum(1 for m in recent if m.get("result") == "W")
            profile[f"roll_{window}_win_pct"] = round(wins / n, 4)

        return profile

    def update(self, match_stats: dict, result: str) -> None:
        """Add current match to buffer (call AFTER snapshot)."""
        match_stats["result"] = result
        self.matches.append(match_stats)
        self.total_played += 1
        if result == "W":
            self.season_wins += 1
        elif result == "D":
            self.season_draws += 1
        else:
            self.season_losses += 1

    def reset_season(self) -> None:
        """Reset season counters (call when season changes)."""
        self.season_wins = 0
        self.season_draws = 0
        self.season_losses = 0
        self.total_played = 0


# ── Football Ingester ──────────────────────────────────────────────


class FootballIngester(BaseIngester):
    sport = Sport.FOOTBALL
    source_name = "football_data_co_uk"

    def __init__(self, default_season: str = "unknown") -> None:
        super().__init__()
        self._default_season = default_season
        # Per-team buffers keyed by canonical name
        self._buffers: dict[str, _TeamBuffer] = defaultdict(_TeamBuffer)

    def parse_file(self, file_path: Path) -> list[dict]:
        rows = read_csv(file_path)
        # Filter out rows with missing essential fields
        return [
            r for r in rows
            if r.get("HomeTeam") and r.get("AwayTeam") and r.get("Date")
        ]

    # ── Chronological ingestion with profile building ───────────────

    def ingest_file(self, session: Session, file_path: Path) -> int:
        """Parse, sort chronologically, build profiles, then upsert.

        ANTI-LEAKAGE: For each match:
          1. Snapshot profiles from PAST matches only
          2. Write profiles to TeamDailyStats
          3. Upsert HistoricalMatch
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
                logger.warning(
                    "Skipping row in %s: %s", file_path.name, exc,
                )

        logger.info(
            "Ingested %d/%d rows from %s (with profiles)",
            count, len(rows), file_path.name,
        )
        return count

    def _extract_date_key(self, row: dict) -> str:
        """Extract a sortable date string from a row."""
        raw = (row.get("Date") or "1970-01-01").strip()
        # Normalize to YYYY-MM-DD for sorting
        try:
            d = parse_date(raw)
            return d.isoformat()
        except ValueError:
            return raw

    def _ingest_row(self, session: Session, row: dict, source_file: str) -> int:
        """Ingest a single CSV row with full profile building."""
        match_date = parse_date(row["Date"])

        # Resolve through Ironclad
        home_team = self.resolve_name(row["HomeTeam"].strip())
        away_team = self.resolve_name(row["AwayTeam"].strip())

        # Extract per-team match stats for the buffer
        home_goals = safe_int(row.get("FTHG"))
        away_goals = safe_int(row.get("FTAG"))
        result_code = (row.get("FTR") or "").strip() or "U"

        home_match_data = {
            "goals_for": home_goals,
            "goals_against": away_goals,
            "shots": safe_int(row.get("HS")),
            "shots_target": safe_int(row.get("HST")),
            "corners": safe_int(row.get("HC")),
            "fouls": safe_int(row.get("HF")),
            "yellows": safe_int(row.get("HY")),
            "reds": safe_int(row.get("HR")),
        }
        away_match_data = {
            "goals_for": away_goals,
            "goals_against": home_goals,
            "shots": safe_int(row.get("AS")),
            "shots_target": safe_int(row.get("AST")),
            "corners": safe_int(row.get("AC")),
            "fouls": safe_int(row.get("AF")),
            "yellows": safe_int(row.get("AY")),
            "reds": safe_int(row.get("AR")),
        }

        # Map FTR → per-team result
        home_result = "W" if result_code == "H" else ("D" if result_code == "D" else "L")
        away_result = "W" if result_code == "A" else ("D" if result_code == "D" else "L")

        # ── STEP 1: Snapshot profiles BEFORE this match (NO LEAKAGE) ──
        home_profile = self._buffers[home_team].snapshot()
        away_profile = self._buffers[away_team].snapshot()

        # ── STEP 2: Write profiles to TeamDailyStats ──────────────────
        div = (row.get("Div") or "").strip()
        league = DIVISION_MAP.get(div, div or "unknown")

        self._upsert_daily_stat(session, home_team, match_date, home_profile, league)
        self._upsert_daily_stat(session, away_team, match_date, away_profile, league)

        # ── STEP 3: Build and upsert HistoricalMatch ──────────────────
        model = self.row_to_model(row, source_file)
        model = self._normalize_model(model)
        self._upsert(session, model)

        # ── STEP 4: Update buffers WITH this match (for future use) ───
        self._buffers[home_team].update(home_match_data, home_result)
        self._buffers[away_team].update(away_match_data, away_result)

        return 1

    def row_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        div = (row.get("Div") or "").strip()
        match_date = parse_date(row["Date"])

        # Derive season from filename or date
        season = self._derive_season(source_file, match_date)

        # Match stats (extended: shots on target, corners, fouls, cards)
        match_stats: dict = {}
        for key, col in [
            ("ht_home", "HTHG"), ("ht_away", "HTAG"), ("ht_result", "HTR"),
            ("home_shots", "HS"), ("away_shots", "AS"),
            ("home_shots_target", "HST"), ("away_shots_target", "AST"),
            ("home_fouls", "HF"), ("away_fouls", "AF"),
            ("home_corners", "HC"), ("away_corners", "AC"),
            ("home_yellows", "HY"), ("away_yellows", "AY"),
            ("home_reds", "HR"), ("away_reds", "AR"),
        ]:
            val = row.get(col)
            if val and val.strip():
                match_stats[key] = safe_int(val) if key != "ht_result" else val.strip()

        # Multi-bookmaker odds
        odds: dict = {}
        for book, (h_col, d_col, a_col) in BOOKMAKER_ODDS.items():
            h = safe_float(row.get(h_col))
            d = safe_float(row.get(d_col))
            a = safe_float(row.get(a_col))
            if h is not None:
                odds[f"{book}_home"] = h
            if d is not None:
                odds[f"{book}_draw"] = d
            if a is not None:
                odds[f"{book}_away"] = a

        # Aggregate odds columns
        for key, col in [
            ("bb_max_home", "BbMxH"), ("bb_avg_home", "BbAvH"),
            ("bb_max_draw", "BbMxD"), ("bb_avg_draw", "BbAvD"),
            ("bb_max_away", "BbMxA"), ("bb_avg_away", "BbAvA"),
        ]:
            val = safe_float(row.get(col))
            if val is not None:
                odds[key] = val

        # Over/under and Asian handicap lines
        betting_lines: dict = {}
        for key, col in [
            ("bb_ou_count", "BbOU"),
            ("bb_max_over25", "BbMx>2.5"), ("bb_avg_over25", "BbAv>2.5"),
            ("bb_max_under25", "BbMx<2.5"), ("bb_avg_under25", "BbAv<2.5"),
            ("bb_ah_count", "BbAH"), ("bb_ah_line", "BbAHh"),
            ("bb_max_ah_home", "BbMxAHH"), ("bb_avg_ah_home", "BbAvAHH"),
            ("bb_max_ah_away", "BbMxAHA"), ("bb_avg_ah_away", "BbAvAHA"),
        ]:
            val = safe_float(row.get(col))
            if val is not None:
                betting_lines[key] = val

        return HistoricalMatch(
            sport=Sport.FOOTBALL,
            season=season,
            division=div or "unknown",
            match_date=match_date,
            home_team=row["HomeTeam"].strip(),
            away_team=row["AwayTeam"].strip(),
            home_score=safe_int(row.get("FTHG")),
            away_score=safe_int(row.get("FTAG")),
            result=(row.get("FTR") or "").strip() or "U",
            match_stats=match_stats,
            odds=odds,
            betting_lines=betting_lines,
            advanced_stats={},
            source=self.source_name,
            source_file=source_file,
        )

    def _derive_season(self, source_file: str, match_date) -> str:
        """Try to extract season from filename like 'E0_2324.csv' or '2023-24'."""
        import re
        name = Path(source_file).stem
        # Pattern: 2324 or 2223 (two-digit year pairs)
        m = re.search(r"(\d{2})(\d{2})", name)
        if m:
            y1, y2 = m.group(1), m.group(2)
            return f"20{y1}-{y2}"
        # Pattern: 2023-24 or 2023_24
        m = re.search(r"(20\d{2})[-_](\d{2})", name)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        # Fallback: derive from match date
        year = match_date.year
        month = match_date.month
        if month >= 7:
            return f"{year}-{(year + 1) % 100:02d}"
        else:
            return f"{year - 1}-{year % 100:02d}"
