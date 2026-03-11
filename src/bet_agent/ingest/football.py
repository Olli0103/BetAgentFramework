"""Football (soccer) historical data ingester.

Source: football-data.co.uk CSV format.
Columns: Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR,
         HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR,
         B365H, B365D, B365A, BWH, BWD, BWA, ... (multi-bookmaker odds)
"""

from __future__ import annotations

from pathlib import Path

from bet_agent.db.models import HistoricalMatch, Sport
from bet_agent.ingest.base import (
    BaseIngester,
    parse_date,
    read_csv,
    safe_float,
    safe_int,
)

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


class FootballIngester(BaseIngester):
    sport = Sport.FOOTBALL
    source_name = "football_data_co_uk"

    def __init__(self, default_season: str = "unknown") -> None:
        self._default_season = default_season

    def parse_file(self, file_path: Path) -> list[dict]:
        rows = read_csv(file_path)
        # Filter out rows with missing essential fields
        return [
            r for r in rows
            if r.get("HomeTeam") and r.get("AwayTeam") and r.get("Date")
        ]

    def row_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        div = (row.get("Div") or "").strip()
        match_date = parse_date(row["Date"])

        # Derive season from filename or date
        season = self._derive_season(source_file, match_date)

        # Match stats
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
