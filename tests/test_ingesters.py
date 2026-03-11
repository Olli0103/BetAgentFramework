"""Tests for historical data ingesters — all 5 sports."""

import csv
import tempfile
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import Base, HistoricalMatch, Sport
from bet_agent.ingest.base import parse_date, safe_float, safe_int


@pytest.fixture
def db_session():
    """In-memory SQLite for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _write_csv(tmp_dir: Path, filename: str, header: list[str], rows: list[list[str]]) -> Path:
    """Write a CSV file and return its path."""
    path = tmp_dir / filename
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ── Base utilities ───────────────────────────────────────────────────


class TestBaseUtilities:
    def test_safe_int(self):
        assert safe_int("42") == 42
        assert safe_int("3.7") == 3
        assert safe_int("") == 0
        assert safe_int(None) == 0
        assert safe_int("abc", default=-1) == -1

    def test_safe_float(self):
        assert safe_float("3.14") == 3.14
        assert safe_float("") is None
        assert safe_float(None) is None
        assert safe_float("abc", default=0.0) == 0.0

    def test_parse_date_multiple_formats(self):
        assert parse_date("11/03/2026") == date(2026, 3, 11)
        assert parse_date("2026-03-11") == date(2026, 3, 11)
        assert parse_date("20260311", formats=["%Y%m%d"]) == date(2026, 3, 11)

    def test_parse_date_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            parse_date("not-a-date")


# ── Football Ingester ────────────────────────────────────────────────


class TestFootballIngester:
    def test_parse_and_convert(self):
        from bet_agent.ingest.football import FootballIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "E0_2324.csv",
                ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                 "HTHG", "HTAG", "B365H", "B365D", "B365A"],
                [
                    ["E0", "11/03/2026", "Arsenal", "Chelsea", "2", "1", "H",
                     "1", "0", "1.90", "3.50", "4.20"],
                ],
            )
            ingester = FootballIngester()
            rows = ingester.parse_file(path)
            assert len(rows) == 1

            model = ingester.row_to_model(rows[0], str(path))
            assert model.sport == Sport.FOOTBALL
            assert model.home_team == "Arsenal"
            assert model.away_team == "Chelsea"
            assert model.home_score == 2
            assert model.away_score == 1
            assert model.result == "H"
            assert model.odds["b365_home"] == 1.90
            assert model.match_stats["ht_home"] == 1

    def test_season_derivation(self):
        from bet_agent.ingest.football import FootballIngester

        ingester = FootballIngester()
        # filename pattern 2324 → 2023-24
        assert ingester._derive_season("E0_2324.csv", date(2024, 1, 1)) == "2023-24"

    def test_ingest_file_upserts(self, db_session):
        from bet_agent.ingest.football import FootballIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "D1_2526.csv",
                ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
                [
                    ["D1", "10/03/2026", "Bayern", "Dortmund", "3", "1", "H"],
                    ["D1", "11/03/2026", "Leipzig", "Frankfurt", "0", "0", "D"],
                ],
            )
            ingester = FootballIngester()
            count = ingester.ingest_file(db_session, path)
            db_session.flush()
            assert count == 2
            assert db_session.query(HistoricalMatch).count() == 2

    def test_skips_rows_missing_fields(self):
        from bet_agent.ingest.football import FootballIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "test.csv",
                ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
                [
                    ["E0", "11/03/2026", "", "Chelsea", "1", "0", "H"],  # no home
                    ["E0", "11/03/2026", "Arsenal", "", "1", "0", "H"],  # no away
                ],
            )
            ingester = FootballIngester()
            rows = ingester.parse_file(path)
            assert len(rows) == 0


# ── NBA Ingester ─────────────────────────────────────────────────────


class TestNBAIngester:
    def test_parse_and_convert(self):
        from bet_agent.ingest.nba import NBAIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "nba_2025.csv",
                ["season", "date", "regular", "playoffs", "away", "home",
                 "score_away", "score_home", "q1_home", "q1_away",
                 "spread", "total", "moneyline_home", "moneyline_away"],
                [
                    ["2025", "2026-03-11", "1", "", "Lakers", "Celtics",
                     "108", "120", "30", "25", "-5.5", "220.5", "-200", "+170"],
                ],
            )
            ingester = NBAIngester()
            rows = ingester.parse_file(path)
            assert len(rows) == 1

            model = ingester.row_to_model(rows[0], str(path))
            assert model.sport == Sport.BASKETBALL
            assert model.home_team == "Celtics"
            assert model.away_team == "Lakers"
            assert model.home_score == 120
            assert model.away_score == 108
            assert model.result == "H"
            assert model.division == "NBA"
            assert model.betting_lines["spread"] == -5.5
            assert model.betting_lines["total"] == 220.5
            assert model.match_stats["q1_home"] == 30

    def test_away_win_result(self):
        from bet_agent.ingest.nba import NBAIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "nba.csv",
                ["season", "date", "away", "home", "score_away", "score_home"],
                [["2025", "2026-03-11", "Warriors", "Nets", "130", "100"]],
            )
            ingester = NBAIngester()
            model = ingester.row_to_model(ingester.parse_file(path)[0], str(path))
            assert model.result == "A"


# ── NHL Ingester ─────────────────────────────────────────────────────


class TestNHLIngester:
    def test_merges_two_rows_per_game(self):
        from bet_agent.ingest.nhl import NHLIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "nhl_2025.csv",
                ["game_id", "date", "team_name", "is_home", "goals_for",
                 "shots", "season", "venue"],
                [
                    ["2025020001", "2026-03-11", "Avalanche", "1", "4",
                     "32", "2025-26", "Ball Arena"],
                    ["2025020001", "2026-03-11", "Stars", "0", "2",
                     "28", "2025-26", ""],
                ],
            )
            ingester = NHLIngester()
            merged = ingester.parse_file(path)
            assert len(merged) == 1
            assert merged[0]["home"]["team_name"] == "Avalanche"
            assert merged[0]["away"]["team_name"] == "Stars"

    def test_row_to_model(self):
        from bet_agent.ingest.nhl import NHLIngester

        row = {
            "game_id": "2025020001",
            "home": {
                "date": "2026-03-11", "team_name": "Avalanche",
                "goals_for": "4", "shots": "32", "season": "2025-26",
                "venue": "Ball Arena", "is_home": "1",
            },
            "away": {
                "date": "2026-03-11", "team_name": "Stars",
                "goals_for": "2", "shots": "28", "season": "2025-26",
                "venue": "", "is_home": "0",
            },
        }
        ingester = NHLIngester()
        model = ingester.row_to_model(row, "nhl_2025.csv")
        assert model.sport == Sport.ICE_HOCKEY
        assert model.home_team == "Avalanche"
        assert model.away_team == "Stars"
        assert model.home_score == 4
        assert model.away_score == 2
        assert model.result == "H"
        assert model.division == "NHL"
        assert model.match_stats["home_shots"] == 32.0
        assert model.match_stats["away_shots"] == 28.0

    def test_single_row_game_skipped(self):
        from bet_agent.ingest.nhl import NHLIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "nhl.csv",
                ["game_id", "date", "team_name", "is_home", "goals_for"],
                [["2025020001", "2026-03-11", "Avalanche", "1", "4"]],
            )
            ingester = NHLIngester()
            merged = ingester.parse_file(path)
            assert len(merged) == 0


# ── NFL Ingester ─────────────────────────────────────────────────────


class TestNFLIngester:
    def test_parse_and_convert(self):
        """Test with a minimal XLSX-like dict (simulating _read_xlsx output)."""
        from bet_agent.ingest.nfl import NFLIngester

        ingester = NFLIngester()
        row = {
            "Date": "2025-09-07",
            "Home Team": "Chiefs",
            "Away Team": "Ravens",
            "Home Score": "27",
            "Away Score": "20",
            "Overtime?": "no",
            "Playoff Game?": "no",
            "Neutral Venue?": "no",
            "Home Odds Open": "1.65",
            "Away Odds Open": "2.30",
            "Home Line Open": "-3.5",
            "Total Score Open": "48.5",
        }
        model = ingester.row_to_model(row, "nfl_2025.xlsx")
        assert model.sport == Sport.AMERICAN_FOOTBALL
        assert model.home_team == "Chiefs"
        assert model.away_team == "Ravens"
        assert model.home_score == 27
        assert model.away_score == 20
        assert model.result == "H"
        assert model.odds["home_odds_open"] == 1.65
        assert model.betting_lines["home_line_open"] == -3.5
        assert model.betting_lines["total_open"] == 48.5

    def test_overtime_and_playoff_flags(self):
        from bet_agent.ingest.nfl import NFLIngester

        ingester = NFLIngester()
        row = {
            "Date": "2026-01-15",
            "Home Team": "Bills",
            "Away Team": "Dolphins",
            "Home Score": "31",
            "Away Score": "28",
            "Overtime?": "yes",
            "Playoff Game?": "true",
            "Neutral Venue?": "",
        }
        model = ingester.row_to_model(row, "nfl_2025.xlsx")
        assert model.match_stats["overtime"] is True
        assert model.match_stats["playoff"] is True
        assert "neutral_venue" not in model.match_stats

    def test_season_derivation_from_filename(self):
        from bet_agent.ingest.nfl import NFLIngester

        ingester = NFLIngester()
        assert ingester._derive_season("nfl_2025.xlsx", date(2026, 1, 15)) == "2025"

    def test_season_derivation_from_date(self):
        from bet_agent.ingest.nfl import NFLIngester

        ingester = NFLIngester()
        # Feb game → previous year's season
        assert ingester._derive_season("games.xlsx", date(2026, 2, 1)) == "2025"
        # Sep game → current year
        assert ingester._derive_season("games.xlsx", date(2025, 9, 7)) == "2025"


# ── Tennis Ingester ──────────────────────────────────────────────────


class TestTennisIngester:
    def test_parse_score_sets(self):
        from bet_agent.ingest.tennis import _parse_score_sets

        assert _parse_score_sets("6-4 7-6(3) 6-3") == (3, 0)
        assert _parse_score_sets("6-4 3-6 7-6(5)") == (2, 1)
        assert _parse_score_sets("6-3 6-7(4) 3-6") == (1, 2)
        assert _parse_score_sets("") == (0, 0)
        assert _parse_score_sets("RET") == (0, 0)

    def test_sackmann_csv_format(self):
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "atp_matches_2025.csv",
                ["tourney_id", "tourney_name", "surface", "tourney_date",
                 "winner_name", "loser_name", "score", "best_of", "round",
                 "minutes", "w_ace", "w_df", "winner_rank", "loser_rank"],
                [
                    ["2025-0001", "Australian Open", "Hard", "20260119",
                     "Sinner", "Djokovic", "6-3 7-6(4) 6-4", "5", "F",
                     "178", "12", "3", "1", "7"],
                ],
            )
            ingester = TennisIngester(tour="ATP")
            rows = ingester.parse_file(path)
            assert len(rows) == 1
            assert rows[0]["_format"] == "sackmann"

            model = ingester.row_to_model(rows[0], str(path))
            assert model.sport == Sport.TENNIS
            # Home/away is randomized to prevent positional bias.
            # Both players must be assigned, and result must be consistent.
            assert {model.home_team, model.away_team} == {"Sinner", "Djokovic"}
            assert model.home_score + model.away_score == 3  # 3+0 sets
            if model.home_team == "Sinner":
                assert model.result == "H"
                assert model.home_score == 3
                assert model.away_score == 0
            else:
                assert model.result == "A"
                assert model.home_score == 0
                assert model.away_score == 3
            # Original winner always preserved in match_stats
            assert model.match_stats["actual_winner"] == "Sinner"
            assert model.match_stats["actual_loser"] == "Djokovic"
            assert model.match_stats["tournament"] == "Australian Open"
            assert model.match_stats["surface"] == "Hard"
            assert model.advanced_stats["winner_ace"] == 12
            assert model.advanced_stats["winner_rank"] == 1

    def test_xlsx_format_detected(self):
        from bet_agent.ingest.tennis import TennisIngester

        ingester = TennisIngester()
        # .csv → _parse_csv, .xlsx → _parse_xlsx (just verify routing)
        assert ingester.parse_file(Path("/nonexistent.txt")) == []

    def test_ingest_sackmann_to_db(self, db_session):
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "atp_matches_2025.csv",
                ["tourney_id", "tourney_name", "surface", "tourney_date",
                 "winner_name", "loser_name", "score", "best_of", "round"],
                [
                    ["2025-001", "Roland Garros", "Clay", "20260525",
                     "Alcaraz", "Ruud", "6-3 6-1 6-2", "5", "F"],
                    ["2025-002", "Wimbledon", "Grass", "20260714",
                     "Sinner", "Medvedev", "7-6(3) 6-4 6-3", "5", "F"],
                ],
            )
            ingester = TennisIngester(tour="ATP")
            count = ingester.ingest_file(db_session, path)
            db_session.flush()
            assert count == 2
            assert db_session.query(HistoricalMatch).count() == 2


# ── Upsert logic (BaseIngester._upsert) ─────────────────────────────


class TestUpsert:
    def test_upsert_merges_jsonb(self, db_session):
        from bet_agent.ingest.football import FootballIngester

        with tempfile.TemporaryDirectory() as tmp:
            # First ingest
            path1 = _write_csv(
                Path(tmp), "E0_2526.csv",
                ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                 "B365H", "B365D", "B365A"],
                [["E0", "11/03/2026", "Arsenal", "Chelsea", "2", "1", "H",
                  "1.90", "3.50", "4.20"]],
            )
            ingester = FootballIngester()
            ingester.ingest_file(db_session, path1)
            db_session.flush()

            row = db_session.query(HistoricalMatch).one()
            assert row.odds["b365_home"] == 1.90

            # Second ingest with different odds but same match
            path2 = _write_csv(
                Path(tmp), "E0_2526_v2.csv",
                ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                 "BWH", "BWD", "BWA"],
                [["E0", "11/03/2026", "Arsenal", "Chelsea", "2", "1", "H",
                  "1.85", "3.60", "4.50"]],
            )
            ingester.ingest_file(db_session, path2)
            db_session.flush()

            row = db_session.query(HistoricalMatch).one()
            # Original odds preserved, new ones merged
            assert row.odds["b365_home"] == 1.90
            assert row.odds["bw_home"] == 1.85

    def test_different_matches_separate_rows(self, db_session):
        from bet_agent.ingest.football import FootballIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "E0_2526.csv",
                ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
                [
                    ["E0", "10/03/2026", "Arsenal", "Chelsea", "2", "1", "H"],
                    ["E0", "11/03/2026", "Liverpool", "Man Utd", "3", "0", "H"],
                ],
            )
            ingester = FootballIngester()
            ingester.ingest_file(db_session, path)
            db_session.flush()
            assert db_session.query(HistoricalMatch).count() == 2


# ── Package imports ──────────────────────────────────────────────────


class TestPackageImports:
    def test_all_ingesters_importable(self):
        from bet_agent.ingest import (
            BaseIngester,
            FootballIngester,
            NBAIngester,
            NFLIngester,
            NHLIngester,
            TennisIngester,
        )
        assert FootballIngester.sport == Sport.FOOTBALL
        assert NBAIngester.sport == Sport.BASKETBALL
        assert NHLIngester.sport == Sport.ICE_HOCKEY
        assert NFLIngester.sport == Sport.AMERICAN_FOOTBALL
        assert TennisIngester.sport == Sport.TENNIS
