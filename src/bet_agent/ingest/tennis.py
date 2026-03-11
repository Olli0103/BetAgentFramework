"""Tennis historical data ingester — Data Fusion + Point-in-Time Profiling.

Supports three ingestion modes:
1. Sackmann CSV: Rich match stats (aces, serve %, break points) + player metadata
   → Builds rolling player profiles in TeamDailyStats JSONB
2. XLSX Odds CSV (tennis-data.co.uk): Historical closing odds from 5+ bookmakers
   → Fuses odds into existing HistoricalMatch records
3. Combined flow: Ingest Sackmann first, then fuse odds on top

CRITICAL ANTI-LEAKAGE:
  Tennis has no home/away. Source files list Winner first. We randomize the
  assignment (50/50) to eliminate positional bias.

  Profile stats for match T are computed from matches T-1, T-2, ... T-N ONLY.
  The current match's stats are added to the buffer AFTER profile snapshot.
  This is the STRICT chronological ordering that prevents look-ahead bias.

PROFILE STRUCTURE (TeamDailyStats JSONB):
  Static metadata:  hand, height_cm, age_years, ioc, rank, rank_points
  Rolling 10-match: roll_10_ace_avg, roll_10_1stIn_pct, roll_10_1stWon_pct,
                    roll_10_2ndWon_pct, roll_10_bpSaved_pct, roll_10_minutes_avg,
                    roll_10_win_pct
  Surface splits:   surface_{hard|clay|grass|carpet}_win_pct
  Volume:           games_played
"""

from __future__ import annotations

import logging
import random
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

# Bookmaker columns in the XLSX format
XLSX_BOOK_COLS = {
    "b365": ("B365W", "B365L"),
    "ex": ("EXW", "EXL"),
    "lb": ("LBW", "LBL"),
    "ps": ("PSW", "PSL"),
    "sj": ("SJW", "SJL"),
}

# In-match stat columns from Sackmann CSV (w_ and l_ prefixed)
_SERVE_STATS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]

# Rolling window for player profiles
_PROFILE_WINDOW = 10


# ── Player State Buffer ─────────────────────────────────────────────


class _PlayerBuffer:
    """Per-player match history buffer for rolling stat computation.

    Maintains a deque of recent match stats and surface-level win tracking.
    CRITICAL: Only call snapshot() BEFORE updating with the current match.
    """

    __slots__ = ("matches", "surface_wins", "surface_total", "total_wins", "total_played")

    def __init__(self) -> None:
        self.matches: deque[dict] = deque(maxlen=_PROFILE_WINDOW)
        self.surface_wins: dict[str, int] = defaultdict(int)
        self.surface_total: dict[str, int] = defaultdict(int)
        self.total_wins: int = 0
        self.total_played: int = 0

    def snapshot(self, metadata: dict) -> dict:
        """Compute point-in-time profile from PAST matches only.

        Returns a dict suitable for TeamDailyStats JSONB.
        """
        profile: dict = {}

        # Static metadata (updates each match — age changes!)
        for key in ("hand", "height_cm", "age_years", "ioc", "rank", "rank_points"):
            if metadata.get(key) is not None:
                profile[key] = metadata[key]

        profile["games_played"] = self.total_played

        if not self.matches:
            return profile

        n = len(self.matches)

        # Rolling averages
        for stat in ("ace", "df", "svpt", "svgms", "minutes"):
            vals = [m.get(stat) for m in self.matches if m.get(stat) is not None]
            if vals:
                profile[f"roll_{_PROFILE_WINDOW}_{stat}_avg"] = round(sum(vals) / len(vals), 2)

        # Rolling percentages (computed from raw counts for accuracy)
        # Keys are stored lowercase in the buffer (stat.lower() in update)
        total_svpt = sum(m.get("svpt", 0) for m in self.matches)
        total_1stIn = sum(m.get("1stin", 0) for m in self.matches)
        total_1stWon = sum(m.get("1stwon", 0) for m in self.matches)
        total_2ndWon = sum(m.get("2ndwon", 0) for m in self.matches)
        total_bpSaved = sum(m.get("bpsaved", 0) for m in self.matches)
        total_bpFaced = sum(m.get("bpfaced", 0) for m in self.matches)

        if total_svpt > 0:
            profile[f"roll_{_PROFILE_WINDOW}_1stIn_pct"] = round(total_1stIn / total_svpt, 4)
        if total_1stIn > 0:
            profile[f"roll_{_PROFILE_WINDOW}_1stWon_pct"] = round(total_1stWon / total_1stIn, 4)
        second_serves = total_svpt - total_1stIn
        if second_serves > 0:
            profile[f"roll_{_PROFILE_WINDOW}_2ndWon_pct"] = round(total_2ndWon / second_serves, 4)
        if total_bpFaced > 0:
            profile[f"roll_{_PROFILE_WINDOW}_bpSaved_pct"] = round(total_bpSaved / total_bpFaced, 4)

        # Rolling win percentage
        recent_wins = sum(1 for m in self.matches if m.get("won"))
        profile[f"roll_{_PROFILE_WINDOW}_win_pct"] = round(recent_wins / n, 4)

        # Surface-specific win percentages (lifetime, not windowed)
        for surface in ("Hard", "Clay", "Grass", "Carpet"):
            total = self.surface_total.get(surface, 0)
            if total > 0:
                wins = self.surface_wins.get(surface, 0)
                profile[f"surface_{surface.lower()}_win_pct"] = round(wins / total, 4)

        return profile

    def update(self, match_stats: dict, surface: str, won: bool) -> None:
        """Add current match to the buffer (call AFTER snapshot)."""
        self.matches.append(match_stats)
        self.total_played += 1
        if won:
            self.total_wins += 1
            if surface:
                self.surface_wins[surface] += 1
        if surface:
            self.surface_total[surface] += 1


# ── Tennis Ingester ─────────────────────────────────────────────────


class TennisIngester(BaseIngester):
    sport = Sport.TENNIS
    source_name = "tennis_fusion"

    def __init__(self, tour: str = "ATP") -> None:
        """Args: tour: 'ATP' or 'WTA'."""
        super().__init__()
        self.tour = tour.upper()
        # Player buffers for rolling stats — keyed by canonical name
        self._buffers: dict[str, _PlayerBuffer] = defaultdict(_PlayerBuffer)

    def parse_file(self, file_path: Path) -> list[dict]:
        suffix = file_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return self._parse_xlsx(file_path)
        elif suffix == ".csv":
            return self._parse_csv(file_path)
        else:
            logger.warning("Unknown file type: %s", file_path)
            return []

    def _parse_xlsx(self, file_path: Path) -> list[dict]:
        from openpyxl import load_workbook

        wb = load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()

        if not rows:
            return []

        header = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(rows[0])]
        result = []
        for row in rows[1:]:
            d = {"_format": "xlsx"}
            for i, val in enumerate(row):
                if i < len(header):
                    d[header[i]] = str(val).strip() if val is not None else ""
            if d.get("Winner") and d.get("Loser") and d.get("Date"):
                result.append(d)
        return result

    def _parse_csv(self, file_path: Path) -> list[dict]:
        rows = read_csv(file_path)
        for r in rows:
            r["_format"] = "sackmann"
        return [
            r for r in rows
            if r.get("winner_name") and r.get("loser_name")
        ]

    # ── Chronological ingestion with profile building ───────────────

    def ingest_file(self, session: Session, file_path: Path) -> int:
        """Parse, sort chronologically, build profiles, then upsert.

        ANTI-LEAKAGE: For each match:
          1. Snapshot profiles from PAST matches only
          2. Write profiles to TeamDailyStats
          3. Upsert HistoricalMatch
          4. Update buffers with THIS match's stats (for future matches)
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
                fmt = row.get("_format", "xlsx")
                if fmt == "sackmann":
                    count += self._ingest_sackmann_row(session, row, str(file_path))
                else:
                    count += self._ingest_xlsx_row(session, row, str(file_path))
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
        fmt = row.get("_format", "xlsx")
        if fmt == "sackmann":
            return (row.get("tourney_date") or "19700101").strip()
        # XLSX: try multiple date formats
        return (row.get("Date") or "1970-01-01").strip()

    def _ingest_sackmann_row(self, session: Session, row: dict, source_file: str) -> int:
        """Ingest a single Sackmann CSV row with full profile building."""
        date_str = (row.get("tourney_date") or "").strip()
        match_date = parse_date(date_str, formats=["%Y%m%d", "%Y-%m-%d"])

        raw_winner = row["winner_name"].strip()
        raw_loser = row["loser_name"].strip()

        # Resolve through Ironclad
        winner = self.resolve_name(raw_winner)
        loser = self.resolve_name(raw_loser)

        surface = (row.get("surface") or "").strip()
        score = (row.get("score") or "").strip()

        # ── Extract in-match stats ──────────────────────────────────
        w_stats: dict = {}
        l_stats: dict = {}
        for stat in _SERVE_STATS:
            w_val = safe_int(row.get(f"w_{stat}"), default=-1)
            l_val = safe_int(row.get(f"l_{stat}"), default=-1)
            if w_val >= 0:
                w_stats[stat.lower()] = w_val
            if l_val >= 0:
                l_stats[stat.lower()] = l_val

        minutes = safe_int(row.get("minutes"), default=-1)
        if minutes > 0:
            w_stats["minutes"] = minutes
            l_stats["minutes"] = minutes

        w_stats["won"] = True
        l_stats["won"] = False

        # ── Extract metadata ────────────────────────────────────────
        w_meta: dict = {}
        l_meta: dict = {}

        for prefix, meta in [("winner", w_meta), ("loser", l_meta)]:
            hand = (row.get(f"{prefix}_hand") or "").strip()
            if hand:
                meta["hand"] = hand
            ht = safe_float(row.get(f"{prefix}_ht"))
            if ht is not None:
                meta["height_cm"] = ht
            age = safe_float(row.get(f"{prefix}_age"))
            if age is not None:
                meta["age_years"] = age
            ioc = (row.get(f"{prefix}_ioc") or "").strip()
            if ioc:
                meta["ioc"] = ioc
            rank = safe_int(row.get(f"{prefix}_rank"), default=-1)
            if rank > 0:
                meta["rank"] = rank
            rank_pts = safe_int(row.get(f"{prefix}_rank_points"), default=-1)
            if rank_pts > 0:
                meta["rank_points"] = rank_pts

        # ── STEP 1: Snapshot profiles BEFORE this match (NO LEAKAGE) ──
        w_profile = self._buffers[winner].snapshot(w_meta)
        l_profile = self._buffers[loser].snapshot(l_meta)

        # ── STEP 2: Write profiles to TeamDailyStats ────────────────
        tourney = (row.get("tourney_name") or "").strip()
        league = f"{self.tour}_{tourney}" if tourney else self.tour

        self._upsert_daily_stat(session, winner, match_date, w_profile, league)
        self._upsert_daily_stat(session, loser, match_date, l_profile, league)

        # ── STEP 3: Build and upsert HistoricalMatch ────────────────
        model = self._sackmann_to_model(row, source_file)
        model = self._normalize_model(model)
        self._upsert(session, model)

        # ── STEP 4: Update buffers WITH this match (for future use) ──
        self._buffers[winner].update(w_stats, surface, won=True)
        self._buffers[loser].update(l_stats, surface, won=False)

        return 1

    def _ingest_xlsx_row(self, session: Session, row: dict, source_file: str) -> int:
        """Ingest a single XLSX row (odds data, limited stats)."""
        match_date = parse_date(
            row["Date"],
            formats=["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"],
        )

        raw_winner = row["Winner"].strip()
        raw_loser = row["Loser"].strip()

        winner = self.resolve_name(raw_winner)
        loser = self.resolve_name(raw_loser)

        # XLSX has limited stats — build minimal profile snapshots
        w_meta: dict = {}
        l_meta: dict = {}
        for key, col in [("rank", "WRank"), ("rank_points", "WPts")]:
            val = safe_int(row.get(col), default=-1)
            if val > 0:
                w_meta[key] = val
        for key, col in [("rank", "LRank"), ("rank_points", "LPts")]:
            val = safe_int(row.get(col), default=-1)
            if val > 0:
                l_meta[key] = val

        # Snapshot and write profiles
        w_profile = self._buffers[winner].snapshot(w_meta)
        l_profile = self._buffers[loser].snapshot(l_meta)

        location = (row.get("Location") or row.get("Tournament") or "").strip()
        league = f"{self.tour}_{location}" if location else self.tour

        self._upsert_daily_stat(session, winner, match_date, w_profile, league)
        self._upsert_daily_stat(session, loser, match_date, l_profile, league)

        # Build and upsert HistoricalMatch
        model = self._xlsx_to_model(row, source_file)
        model = self._normalize_model(model)
        self._upsert(session, model)

        # Update buffers (minimal stats for XLSX)
        surface = (row.get("Surface") or "").strip()
        self._buffers[winner].update({"won": True}, surface, won=True)
        self._buffers[loser].update({"won": False}, surface, won=False)

        return 1

    # ── Odds Fusion ─────────────────────────────────────────────────

    def fuse_odds_file(self, session: Session, file_path: Path) -> int:
        """Merge odds from an XLSX file into existing HistoricalMatch records.

        Matches are joined by date + canonical player names. This allows
        ingesting Sackmann stats first, then layering odds on top.

        Returns:
            Number of matches updated with odds data.
        """
        self._ensure_resolver(session)

        rows = self._parse_xlsx(file_path) if file_path.suffix.lower() in (".xlsx", ".xls") \
            else self._parse_csv(file_path)

        fused = 0
        for row in rows:
            try:
                fmt = row.get("_format", "xlsx")
                if fmt == "sackmann":
                    continue  # Sackmann CSVs don't have odds

                match_date = parse_date(
                    row["Date"],
                    formats=["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"],
                )
                winner = self.resolve_name(row["Winner"].strip())
                loser = self.resolve_name(row["Loser"].strip())

                # Extract odds
                odds_winner: dict = {}
                odds_loser: dict = {}
                for book, (w_col, l_col) in XLSX_BOOK_COLS.items():
                    w = safe_float(row.get(w_col))
                    l = safe_float(row.get(l_col))
                    if w is not None:
                        odds_winner[book] = w
                    if l is not None:
                        odds_loser[book] = l
                for key, col in [("max", "MaxW"), ("avg", "AvgW")]:
                    val = safe_float(row.get(col))
                    if val is not None:
                        odds_winner[key] = val
                for key, col in [("max", "MaxL"), ("avg", "AvgL")]:
                    val = safe_float(row.get(col))
                    if val is not None:
                        odds_loser[key] = val

                if not odds_winner and not odds_loser:
                    continue

                # Find the existing match (could be stored as H or A)
                from sqlalchemy import select, or_, and_
                from bet_agent.db.models import HistoricalMatch as HM

                existing = session.execute(
                    select(HM).where(
                        HM.sport == Sport.TENNIS,
                        HM.match_date == match_date,
                        or_(
                            and_(HM.home_team == winner, HM.away_team == loser),
                            and_(HM.home_team == loser, HM.away_team == winner),
                        ),
                    )
                ).scalar_one_or_none()

                if existing:
                    # Map odds to home/away based on actual assignment
                    if existing.match_stats.get("actual_winner") == winner:
                        is_winner_home = existing.home_team == winner
                    else:
                        is_winner_home = existing.home_team == loser

                    if is_winner_home:
                        odds = {f"{k}_home": v for k, v in odds_winner.items()}
                        odds.update({f"{k}_away": v for k, v in odds_loser.items()})
                    else:
                        odds = {f"{k}_home": v for k, v in odds_loser.items()}
                        odds.update({f"{k}_away": v for k, v in odds_winner.items()})

                    existing.odds = {**existing.odds, **odds}
                    fused += 1
                else:
                    logger.debug(
                        "No match found for odds fusion: %s vs %s on %s",
                        winner, loser, match_date,
                    )
            except Exception as exc:
                logger.warning("Skipping odds row: %s", exc)

        logger.info("Fused odds into %d/%d matches from %s", fused, len(rows), file_path.name)
        return fused

    # ── row_to_model implementations (kept for backward compat) ────

    def row_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        fmt = row.get("_format", "xlsx")
        if fmt == "sackmann":
            return self._sackmann_to_model(row, source_file)
        return self._xlsx_to_model(row, source_file)

    def _xlsx_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        match_date = parse_date(
            row["Date"],
            formats=["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"],
        )

        winner = row["Winner"].strip()
        loser = row["Loser"].strip()

        # Match stats
        match_stats: dict = {
            "tournament": (row.get("Tournament") or row.get("Location") or "").strip(),
            "surface": (row.get("Surface") or "").strip(),
            "court": (row.get("Court") or "").strip(),
            "round": (row.get("Round") or "").strip(),
            "best_of": safe_int(row.get("Best of"), default=3),
        }

        # Series/Tier
        series = (row.get("Series") or row.get("Tier") or "").strip()
        if series:
            match_stats["series"] = series

        # Set scores
        wsets = safe_int(row.get("Wsets"))
        lsets = safe_int(row.get("Lsets"))
        match_stats["winner_sets"] = wsets
        match_stats["loser_sets"] = lsets

        for i in range(1, 6):
            wval = row.get(f"W{i}", "").strip()
            lval = row.get(f"L{i}", "").strip()
            if wval:
                match_stats[f"set{i}_winner"] = safe_int(wval)
            if lval:
                match_stats[f"set{i}_loser"] = safe_int(lval)

        if row.get("Comment", "").strip():
            match_stats["comment"] = row["Comment"].strip()

        # Rankings
        advanced_stats: dict = {}
        for key, col in [
            ("winner_rank", "WRank"), ("loser_rank", "LRank"),
            ("winner_pts", "WPts"), ("loser_pts", "LPts"),
        ]:
            val = safe_int(row.get(col), default=-1)
            if val >= 0:
                advanced_stats[key] = val

        # Odds (stored relative to winner/loser, mapped to home/away below)
        odds_winner: dict = {}
        odds_loser: dict = {}
        for book, (w_col, l_col) in XLSX_BOOK_COLS.items():
            w = safe_float(row.get(w_col))
            l = safe_float(row.get(l_col))
            if w is not None:
                odds_winner[f"{book}"] = w
            if l is not None:
                odds_loser[f"{book}"] = l

        for key, col in [("max", "MaxW"), ("avg", "AvgW")]:
            val = safe_float(row.get(col))
            if val is not None:
                odds_winner[key] = val
        for key, col in [("max", "MaxL"), ("avg", "AvgL")]:
            val = safe_float(row.get(col))
            if val is not None:
                odds_loser[key] = val

        location = (row.get("Location") or row.get("Tournament") or "").strip()
        division = f"{self.tour}_{location}" if location else self.tour

        # CRITICAL: Randomize home/away assignment to prevent positional bias.
        if random.random() < 0.5:
            home, away = winner, loser
            home_score, away_score = wsets, lsets
            result = "H"
            odds = {f"{k}_home": v for k, v in odds_winner.items()}
            odds.update({f"{k}_away": v for k, v in odds_loser.items()})
        else:
            home, away = loser, winner
            home_score, away_score = lsets, wsets
            result = "A"
            odds = {f"{k}_home": v for k, v in odds_loser.items()}
            odds.update({f"{k}_away": v for k, v in odds_winner.items()})

        match_stats["actual_winner"] = winner
        match_stats["actual_loser"] = loser

        return HistoricalMatch(
            sport=Sport.TENNIS,
            season=str(match_date.year),
            division=division,
            match_date=match_date,
            home_team=home,
            away_team=away,
            home_score=home_score,
            away_score=away_score,
            result=result,
            match_stats=match_stats,
            odds=odds,
            betting_lines={},
            advanced_stats=advanced_stats,
            source="tennis_xlsx",
            source_file=source_file,
        )

    def _sackmann_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        """Parse Sackmann tennis_atp / tennis_wta CSV format."""
        date_str = (row.get("tourney_date") or "").strip()
        match_date = parse_date(date_str, formats=["%Y%m%d", "%Y-%m-%d"])

        winner = row["winner_name"].strip()
        loser = row["loser_name"].strip()
        score = (row.get("score") or "").strip()

        match_stats: dict = {
            "tournament": (row.get("tourney_name") or "").strip(),
            "tourney_id": (row.get("tourney_id") or "").strip(),
            "surface": (row.get("surface") or "").strip(),
            "round": (row.get("round") or "").strip(),
            "best_of": safe_int(row.get("best_of"), default=3),
            "score_raw": score,
        }

        if row.get("tourney_level", "").strip():
            match_stats["tourney_level"] = row["tourney_level"].strip()
        if row.get("draw_size", "").strip():
            match_stats["draw_size"] = safe_int(row.get("draw_size"))

        minutes = safe_int(row.get("minutes"), default=-1)
        if minutes > 0:
            match_stats["minutes"] = minutes

        wsets, lsets = _parse_score_sets(score)
        match_stats["winner_sets"] = wsets
        match_stats["loser_sets"] = lsets

        # Serve/return stats
        advanced_stats: dict = {}
        for prefix, side in [("w", "winner"), ("l", "loser")]:
            for stat in _SERVE_STATS:
                col = f"{prefix}_{stat}"
                val = safe_int(row.get(col), default=-1)
                if val >= 0:
                    advanced_stats[f"{side}_{stat.lower()}"] = val

        # Rankings
        for key, col in [
            ("winner_rank", "winner_rank"), ("loser_rank", "loser_rank"),
            ("winner_rank_points", "winner_rank_points"),
            ("loser_rank_points", "loser_rank_points"),
        ]:
            val = safe_int(row.get(col), default=-1)
            if val >= 0:
                advanced_stats[key] = val

        # Player info (metadata for profiles)
        for prefix, side in [("winner", "winner"), ("loser", "loser")]:
            for attr in ["hand", "ht", "ioc"]:
                col = f"{prefix}_{attr}"
                val = (row.get(col) or "").strip()
                if val:
                    advanced_stats[f"{side}_{attr}"] = val
            age = safe_float(row.get(f"{prefix}_age"))
            if age is not None:
                advanced_stats[f"{side}_age"] = age

        tourney_name = (row.get("tourney_name") or "").strip()
        division = f"{self.tour}_{tourney_name}" if tourney_name else self.tour

        match_stats["actual_winner"] = winner
        match_stats["actual_loser"] = loser

        if random.random() < 0.5:
            home, away = winner, loser
            home_score, away_score = wsets, lsets
            result = "H"
        else:
            home, away = loser, winner
            home_score, away_score = lsets, wsets
            result = "A"

        return HistoricalMatch(
            sport=Sport.TENNIS,
            season=str(match_date.year),
            division=division,
            match_date=match_date,
            home_team=home,
            away_team=away,
            home_score=home_score,
            away_score=away_score,
            result=result,
            match_stats=match_stats,
            odds={},
            betting_lines={},
            advanced_stats=advanced_stats,
            source="sackmann_" + self.tour.lower(),
            source_file=source_file,
        )


def _parse_score_sets(score: str) -> tuple[int, int]:
    """Count sets won from a tennis score string like '6-4 7-6(3) 6-3'."""
    wsets = 0
    lsets = 0
    for part in score.split():
        part = part.split("(")[0]
        parts = part.split("-")
        if len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                if a > b:
                    wsets += 1
                elif b > a:
                    lsets += 1
            except ValueError:
                continue
    return wsets, lsets


def fetch_sackmann_repo(
    tour: str = "atp",
    output_dir: Path | None = None,
    years: list[int] | None = None,
) -> list[Path]:
    """Download match CSVs from Jeff Sackmann's GitHub repos.

    Args:
        tour: 'atp' or 'wta'.
        output_dir: Directory to save CSVs. Defaults to cwd.
        years: List of years to download. Defaults to 2020-2026.

    Returns:
        List of downloaded file paths.
    """
    import requests

    if years is None:
        years = list(range(2020, 2027))

    if output_dir is None:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master"
    downloaded: list[Path] = []

    for year in years:
        filename = f"{tour}_matches_{year}.csv"
        url = f"{base_url}/{filename}"
        logger.info("Fetching %s", url)

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                logger.info("No data for %s %d (404)", tour, year)
                continue
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", url, exc)
            continue

        out_path = output_dir / filename
        out_path.write_bytes(resp.content)
        downloaded.append(out_path)
        logger.info("Saved %s (%d bytes)", out_path, len(resp.content))

    return downloaded
