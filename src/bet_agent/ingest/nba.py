"""NBA historical data ingester.

CSV columns: season, date, regular, playoffs, away, home, score_away,
score_home, q1-q4_away/home, ot_away/home, whos_favored, spread, total,
moneyline_away, moneyline_home, h2_spread, h2_total, id_spread, id_total
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


class NBAIngester(BaseIngester):
    sport = Sport.BASKETBALL
    source_name = "nba_historical_csv"

    def parse_file(self, file_path: Path) -> list[dict]:
        rows = read_csv(file_path)
        return [
            r for r in rows
            if r.get("home") and r.get("away") and r.get("date")
        ]

    def row_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        match_date = parse_date(
            row["date"],
            formats=["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"],
        )
        season = (row.get("season") or "").strip() or str(match_date.year)

        is_regular = row.get("regular", "").strip()
        is_playoffs = row.get("playoffs", "").strip()

        # Quarter scores
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
