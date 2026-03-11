"""Tests for Phase 2: Data Fusion & Point-in-Time Profile Building.

Tests the anti-leakage guarantees: profiles at match T only contain
data from matches T-1, T-2, ..., T-N. Never from T itself.
"""

import csv
import tempfile
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import Base, HistoricalMatch, Sport, TeamDailyStats


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _write_csv(tmp_dir: Path, filename: str, header: list[str], rows: list[list[str]]) -> Path:
    path = tmp_dir / filename
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ── Tennis Profile Building ─────────────────────────────────────────


class TestTennisProfileBuilding:
    """Verify that tennis ingestion builds point-in-time player profiles."""

    def _make_sackmann_csv(self, tmp_dir: Path) -> Path:
        """Create a Sackmann CSV with 3 chronological matches for Sinner."""
        return _write_csv(
            tmp_dir, "atp_matches_2025.csv",
            ["tourney_id", "tourney_name", "surface", "tourney_date",
             "winner_name", "loser_name", "score", "best_of", "round",
             "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
             "w_2ndWon", "w_bpSaved", "w_bpFaced",
             "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
             "l_2ndWon", "l_bpSaved", "l_bpFaced",
             "winner_rank", "loser_rank", "winner_hand", "loser_hand",
             "winner_ht", "loser_ht", "winner_age", "loser_age",
             "winner_ioc", "loser_ioc"],
            [
                # Match 1: Sinner beats Djokovic on Hard (Jan 15)
                ["2025-001", "AO", "Hard", "20250115",
                 "Sinner", "Djokovic", "6-3 7-6(4) 6-4", "5", "F",
                 "178", "12", "3", "80", "55", "42",
                 "15", "8", "10",
                 "5", "4", "75", "50", "35",
                 "10", "6", "8",
                 "1", "7", "R", "R",
                 "188", "188", "23.5", "37.8",
                 "ITA", "SRB"],
                # Match 2: Sinner loses to Alcaraz on Clay (May 25)
                ["2025-002", "RG", "Clay", "20250525",
                 "Alcaraz", "Sinner", "6-3 6-1 6-2", "5", "F",
                 "105", "8", "2", "70", "48", "38",
                 "12", "5", "7",
                 "4", "5", "65", "42", "30",
                 "8", "3", "6",
                 "2", "1", "R", "R",
                 "185", "188", "22.0", "23.8",
                 "ESP", "ITA"],
                # Match 3: Sinner beats Medvedev on Hard (Aug 20)
                ["2025-003", "USO", "Hard", "20250820",
                 "Sinner", "Medvedev", "7-5 6-4 6-3", "5", "F",
                 "145", "10", "2", "85", "60", "45",
                 "14", "7", "9",
                 "6", "3", "78", "52", "38",
                 "12", "5", "7",
                 "1", "4", "R", "R",
                 "188", "196", "24.0", "29.5",
                 "ITA", "RUS"],
            ],
        )

    def test_profiles_created_for_all_players(self, db_session):
        """Each player gets a TeamDailyStats entry per match date."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_sackmann_csv(Path(tmp))
            ingester = TennisIngester(tour="ATP")
            count = ingester.ingest_file(db_session, path)
            db_session.flush()

            assert count == 3

            # All players should have profiles
            profiles = db_session.query(TeamDailyStats).filter_by(
                sport=Sport.TENNIS
            ).all()

            # 3 matches × 2 players each = 6 profiles (some players appear twice)
            player_dates = {(p.team_name, p.stat_date) for p in profiles}
            assert ("Sinner", date(2025, 1, 15)) in player_dates
            assert ("Djokovic", date(2025, 1, 15)) in player_dates
            assert ("Sinner", date(2025, 5, 25)) in player_dates
            assert ("Alcaraz", date(2025, 5, 25)) in player_dates
            assert ("Sinner", date(2025, 8, 20)) in player_dates
            assert ("Medvedev", date(2025, 8, 20)) in player_dates

    def test_first_match_profile_has_no_rolling_stats(self, db_session):
        """First match profile should have metadata but no rolling stats."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_sackmann_csv(Path(tmp))
            ingester = TennisIngester(tour="ATP")
            ingester.ingest_file(db_session, path)
            db_session.flush()

            # Sinner's first match (Jan 15) — no prior data
            profile = db_session.execute(
                select(TeamDailyStats).where(
                    TeamDailyStats.team_name == "Sinner",
                    TeamDailyStats.stat_date == date(2025, 1, 15),
                )
            ).scalar_one()

            stats = profile.stats
            assert stats["games_played"] == 0
            assert stats.get("hand") == "R"
            assert stats.get("height_cm") == 188.0
            # No rolling stats yet
            assert "roll_10_ace_avg" not in stats
            assert "roll_10_1stIn_pct" not in stats

    def test_second_match_profile_uses_only_first_match(self, db_session):
        """Anti-leakage: second match profile only contains first match data."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_sackmann_csv(Path(tmp))
            ingester = TennisIngester(tour="ATP")
            ingester.ingest_file(db_session, path)
            db_session.flush()

            # Sinner's second match (May 25) — should have data from Jan 15 ONLY
            profile = db_session.execute(
                select(TeamDailyStats).where(
                    TeamDailyStats.team_name == "Sinner",
                    TeamDailyStats.stat_date == date(2025, 5, 25),
                )
            ).scalar_one()

            stats = profile.stats
            assert stats["games_played"] == 1

            # Rolling stats from the FIRST match only (where Sinner was LOSER)
            # Sinner lost in match 2, so his stats come from l_ columns of match 2?
            # No — his buffer was updated after match 1 where he WON (w_ columns)
            # Match 1: Sinner won → w_ace=12, w_svpt=80, w_1stIn=55
            assert stats.get("roll_10_ace_avg") == 12.0
            assert stats.get("roll_10_1stIn_pct") == pytest.approx(55 / 80, abs=0.01)

    def test_third_match_profile_uses_first_two(self, db_session):
        """Third match profile aggregates first two matches."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_sackmann_csv(Path(tmp))
            ingester = TennisIngester(tour="ATP")
            ingester.ingest_file(db_session, path)
            db_session.flush()

            # Sinner's third match (Aug 20) — should have data from both prior
            profile = db_session.execute(
                select(TeamDailyStats).where(
                    TeamDailyStats.team_name == "Sinner",
                    TeamDailyStats.stat_date == date(2025, 8, 20),
                )
            ).scalar_one()

            stats = profile.stats
            assert stats["games_played"] == 2
            # Win pct: 1 win (match 1) + 0 wins (match 2, lost) = 0.5
            assert stats.get("roll_10_win_pct") == 0.5

    def test_surface_splits_tracked(self, db_session):
        """Surface-specific win percentages are computed correctly."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_sackmann_csv(Path(tmp))
            ingester = TennisIngester(tour="ATP")
            ingester.ingest_file(db_session, path)
            db_session.flush()

            # Sinner at match 3 (Aug 20): Hard 1-0 (match 1), Clay 0-1 (match 2)
            profile = db_session.execute(
                select(TeamDailyStats).where(
                    TeamDailyStats.team_name == "Sinner",
                    TeamDailyStats.stat_date == date(2025, 8, 20),
                )
            ).scalar_one()

            stats = profile.stats
            assert stats.get("surface_hard_win_pct") == 1.0
            assert stats.get("surface_clay_win_pct") == 0.0

    def test_metadata_updates_with_age(self, db_session):
        """Age and rank update chronologically."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_sackmann_csv(Path(tmp))
            ingester = TennisIngester(tour="ATP")
            ingester.ingest_file(db_session, path)
            db_session.flush()

            # Sinner at match 3 — age should be 24.0 (from match 3 metadata)
            profile = db_session.execute(
                select(TeamDailyStats).where(
                    TeamDailyStats.team_name == "Sinner",
                    TeamDailyStats.stat_date == date(2025, 8, 20),
                )
            ).scalar_one()

            # Note: age comes from the CURRENT match's metadata row
            # (it's a static attribute, not a rolling stat)
            assert profile.stats.get("age_years") == 24.0


# ── NHL Profile Building ───────────────────────────────────────────


class TestNHLProfileBuilding:
    """Verify NHL ingester maps pre-computed features to TeamDailyStats."""

    def test_profiles_created_for_both_teams(self, db_session):
        from bet_agent.ingest.nhl import NHLIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "nhl_2025.csv",
                ["game_id", "date", "team_name", "is_home", "goals_for",
                 "shots", "season", "venue",
                 "roll_3_shots", "roll_10_goals_for", "roll_30_win_pct",
                 "season_win_pct", "rest_days"],
                [
                    ["2025020001", "2026-03-11", "Avalanche", "1", "4",
                     "32", "2025-26", "Ball Arena",
                     "31.5", "3.2", "0.620", "0.580", "2"],
                    ["2025020001", "2026-03-11", "Stars", "0", "2",
                     "28", "2025-26", "",
                     "29.0", "2.8", "0.540", "0.510", "1"],
                ],
            )
            ingester = NHLIngester()
            count = ingester.ingest_file(db_session, path)
            db_session.flush()

            assert count == 1  # 1 game

            profiles = db_session.query(TeamDailyStats).filter_by(
                sport=Sport.ICE_HOCKEY
            ).all()
            assert len(profiles) == 2  # home + away

            teams = {p.team_name for p in profiles}
            assert "Avalanche" in teams
            assert "Stars" in teams

    def test_rolling_features_mapped_correctly(self, db_session):
        from bet_agent.ingest.nhl import NHLIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "nhl_2025.csv",
                ["game_id", "date", "team_name", "is_home", "goals_for",
                 "shots", "season",
                 "roll_3_shots", "roll_10_goals_for", "roll_30_win_pct",
                 "season_win_pct", "rest_days", "pre_game_point_pct"],
                [
                    ["2025020001", "2026-03-11", "Avalanche", "1", "4",
                     "32", "2025-26",
                     "31.5", "3.2", "0.620", "0.580", "2", "0.610"],
                    ["2025020001", "2026-03-11", "Stars", "0", "2",
                     "28", "2025-26",
                     "29.0", "2.8", "0.540", "0.510", "1", "0.520"],
                ],
            )
            ingester = NHLIngester()
            ingester.ingest_file(db_session, path)
            db_session.flush()

            avs = db_session.execute(
                select(TeamDailyStats).where(
                    TeamDailyStats.team_name == "Avalanche",
                )
            ).scalar_one()

            assert avs.stats["roll_3_shots"] == 31.5
            assert avs.stats["roll_10_goals_for"] == 3.2
            assert avs.stats["roll_30_win_pct"] == 0.620
            assert avs.stats["season_win_pct"] == 0.580
            assert avs.stats["rest_days"] == 2
            assert avs.stats["pre_game_point_pct"] == 0.610

    def test_match_also_created(self, db_session):
        """Profile building doesn't break HistoricalMatch creation."""
        from bet_agent.ingest.nhl import NHLIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "nhl_2025.csv",
                ["game_id", "date", "team_name", "is_home", "goals_for",
                 "shots", "season"],
                [
                    ["2025020001", "2026-03-11", "Avalanche", "1", "4",
                     "32", "2025-26"],
                    ["2025020001", "2026-03-11", "Stars", "0", "2",
                     "28", "2025-26"],
                ],
            )
            ingester = NHLIngester()
            ingester.ingest_file(db_session, path)
            db_session.flush()

            matches = db_session.query(HistoricalMatch).all()
            assert len(matches) == 1
            assert matches[0].home_team == "Avalanche"
            assert matches[0].away_team == "Stars"
            assert matches[0].home_score == 4
            assert matches[0].away_score == 2


# ── Anti-Leakage Regression Tests ──────────────────────────────────


class TestAntiLeakage:
    """Verify no future data leaks into current profiles."""

    def test_tennis_no_leakage_in_first_profile(self, db_session):
        """First profile must have zero games_played and no rolling stats."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "atp.csv",
                ["tourney_id", "tourney_name", "surface", "tourney_date",
                 "winner_name", "loser_name", "score", "best_of", "round",
                 "w_ace", "w_svpt", "w_1stIn"],
                [
                    ["2025-001", "T1", "Hard", "20250101",
                     "PlayerA", "PlayerB", "6-3 6-4", "3", "F",
                     "10", "60", "40"],
                ],
            )
            ingester = TennisIngester()
            ingester.ingest_file(db_session, path)
            db_session.flush()

            for player in ["PlayerA", "PlayerB"]:
                profile = db_session.execute(
                    select(TeamDailyStats).where(
                        TeamDailyStats.team_name == player,
                        TeamDailyStats.stat_date == date(2025, 1, 1),
                    )
                ).scalar_one()

                assert profile.stats["games_played"] == 0
                # NO rolling stats should exist for the first match
                rolling_keys = [k for k in profile.stats if k.startswith("roll_")]
                assert len(rolling_keys) == 0, f"Leakage detected: {rolling_keys}"

    def test_tennis_chronological_ordering_enforced(self, db_session):
        """Even if CSV rows are out of order, profiles are chronological."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            # Write matches in REVERSE chronological order
            path = _write_csv(
                Path(tmp), "atp.csv",
                ["tourney_id", "tourney_name", "surface", "tourney_date",
                 "winner_name", "loser_name", "score", "best_of", "round",
                 "w_ace", "w_svpt", "w_1stIn"],
                [
                    # Match 2 comes first in file (later date)
                    ["2025-002", "T2", "Clay", "20250601",
                     "PlayerA", "PlayerC", "6-2 6-3", "3", "F",
                     "8", "55", "38"],
                    # Match 1 comes second in file (earlier date)
                    ["2025-001", "T1", "Hard", "20250101",
                     "PlayerA", "PlayerB", "6-3 6-4", "3", "F",
                     "10", "60", "40"],
                ],
            )
            ingester = TennisIngester()
            ingester.ingest_file(db_session, path)
            db_session.flush()

            # At Jan 1 (first chronologically), PlayerA has 0 games
            p1 = db_session.execute(
                select(TeamDailyStats).where(
                    TeamDailyStats.team_name == "PlayerA",
                    TeamDailyStats.stat_date == date(2025, 1, 1),
                )
            ).scalar_one()
            assert p1.stats["games_played"] == 0

            # At Jun 1 (second chronologically), PlayerA has 1 game
            p2 = db_session.execute(
                select(TeamDailyStats).where(
                    TeamDailyStats.team_name == "PlayerA",
                    TeamDailyStats.stat_date == date(2025, 6, 1),
                )
            ).scalar_one()
            assert p2.stats["games_played"] == 1
            # Rolling stats should exist from match 1
            assert "roll_10_ace_avg" in p2.stats
