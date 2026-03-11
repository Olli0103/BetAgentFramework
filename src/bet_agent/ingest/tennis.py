"""Tennis historical data ingester.

Supports two formats:
1. ATP/WTA XLSX: Location, Tournament, Date, Series/Tier, Court, Surface,
   Round, Best of, Winner, Loser, WRank, LRank, WPts, LPts,
   W1-W5, L1-L5, Wsets, Lsets, Comment, B365W, B365L, ...
2. Sackmann CSV (tennis_atp / tennis_wta GitHub repos):
   tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date,
   match_num, winner_*, loser_*, score, best_of, round, minutes,
   w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, ...

CRITICAL: Tennis has no home/away. Source files list Winner first (or higher
ranked player first). If we naively assign Winner=home, the ML model learns
a fake "home advantage" bias because home_team always wins in training data.
We randomize the assignment: each match has a 50/50 chance of winner being
home or away, eliminating the positional bias and forcing the model to rely
on actual features (serve %, Elo, surface stats).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

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


class TennisIngester(BaseIngester):
    sport = Sport.TENNIS
    source_name = "tennis_xlsx"

    def __init__(self, tour: str = "ATP") -> None:
        """Args: tour: 'ATP' or 'WTA'."""
        self.tour = tour.upper()

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

        for key, col in [
            ("max", "MaxW"), ("avg", "AvgW"),
        ]:
            val = safe_float(row.get(col))
            if val is not None:
                odds_winner[key] = val
        for key, col in [
            ("max", "MaxL"), ("avg", "AvgL"),
        ]:
            val = safe_float(row.get(col))
            if val is not None:
                odds_loser[key] = val

        location = (row.get("Location") or row.get("Tournament") or "").strip()
        division = f"{self.tour}_{location}" if location else self.tour

        # CRITICAL: Randomize home/away assignment to prevent positional bias.
        # Tennis has no home team — source files list winner first, which would
        # create a fake "home advantage" if always mapped to home_team.
        if random.random() < 0.5:
            home, away = winner, loser
            home_score, away_score = wsets, lsets
            result = "H"  # home = winner
            odds = {f"{k}_home": v for k, v in odds_winner.items()}
            odds.update({f"{k}_away": v for k, v in odds_loser.items()})
        else:
            home, away = loser, winner
            home_score, away_score = lsets, wsets
            result = "A"  # away = winner
            odds = {f"{k}_home": v for k, v in odds_loser.items()}
            odds.update({f"{k}_away": v for k, v in odds_winner.items()})

        # Store original winner in match_stats so we never lose information
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
        # Date: tourney_date is YYYYMMDD
        date_str = (row.get("tourney_date") or "").strip()
        match_date = parse_date(date_str, formats=["%Y%m%d", "%Y-%m-%d"])

        winner = row["winner_name"].strip()
        loser = row["loser_name"].strip()
        score = (row.get("score") or "").strip()

        # Parse set scores from score string like "6-4 7-6(3) 6-3"
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

        # Count sets won from score string
        wsets, lsets = _parse_score_sets(score)
        match_stats["winner_sets"] = wsets
        match_stats["loser_sets"] = lsets

        # Serve/return stats (gold for our probability models)
        advanced_stats: dict = {}
        for prefix, side in [("w", "winner"), ("l", "loser")]:
            for stat in ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon",
                         "SvGms", "bpSaved", "bpFaced"]:
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

        # Player info
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

        # CRITICAL: Randomize home/away assignment (see _xlsx_to_model docstring)
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
        # Remove tiebreak notation
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
