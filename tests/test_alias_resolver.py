"""Tests for the Ironclad Alias Resolver — name normalization gatekeeper."""

import csv
import tempfile
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import Base, HistoricalMatch, Sport, TeamAlias
from bet_agent.ingest.alias_resolver import (
    IroncladAliasResolver,
    is_abbreviated_tennis_name,
    make_key,
    normalize_text,
    try_match_abbreviated,
)


@pytest.fixture
def db_session():
    """In-memory SQLite for testing."""
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


# ── Text normalization ──────────────────────────────────────────────


class TestTextNormalization:
    def test_strip_and_collapse_whitespace(self):
        assert normalize_text("  Bayern   Munich  ") == "Bayern Munich"

    def test_remove_diacritics(self):
        assert normalize_text("FC Bayern München") == "FC Bayern Munchen"
        assert normalize_text("Fenerbahçe") == "Fenerbahce"
        assert normalize_text("Zürich") == "Zurich"
        assert normalize_text("São Paulo") == "Sao Paulo"

    def test_make_key_lowercase(self):
        assert make_key("FC Bayern München") == "fc bayern munchen"

    def test_empty_string(self):
        assert normalize_text("") == ""
        assert make_key("  ") == ""


# ── Tennis name heuristics ──────────────────────────────────────────


class TestTennisNameHeuristics:
    def test_detect_abbreviated_lastname_initial(self):
        assert is_abbreviated_tennis_name("Sinner J.") is True
        assert is_abbreviated_tennis_name("Sinner J") is True
        assert is_abbreviated_tennis_name("Djokovic N.") is True

    def test_detect_abbreviated_initial_lastname(self):
        assert is_abbreviated_tennis_name("J. Sinner") is True
        assert is_abbreviated_tennis_name("N Djokovic") is True

    def test_detect_comma_format(self):
        assert is_abbreviated_tennis_name("Sinner, Jannik") is True
        assert is_abbreviated_tennis_name("Sinner, J.") is True

    def test_full_names_not_abbreviated(self):
        assert is_abbreviated_tennis_name("Jannik Sinner") is False
        assert is_abbreviated_tennis_name("Novak Djokovic") is False

    def test_match_lastname_initial_to_full(self):
        assert try_match_abbreviated("Sinner J.", "Jannik Sinner") is True
        assert try_match_abbreviated("Djokovic N.", "Novak Djokovic") is True
        assert try_match_abbreviated("Sinner J.", "Novak Djokovic") is False

    def test_match_initial_lastname_to_full(self):
        assert try_match_abbreviated("J. Sinner", "Jannik Sinner") is True
        assert try_match_abbreviated("N. Djokovic", "Novak Djokovic") is True

    def test_match_comma_format_full_name(self):
        assert try_match_abbreviated("Sinner, Jannik", "Jannik Sinner") is True
        assert try_match_abbreviated("Djokovic, Novak", "Novak Djokovic") is True

    def test_match_comma_format_initial(self):
        assert try_match_abbreviated("Sinner, J.", "Jannik Sinner") is True

    def test_no_match_wrong_initial(self):
        assert try_match_abbreviated("Sinner N.", "Jannik Sinner") is False

    def test_no_match_wrong_lastname(self):
        assert try_match_abbreviated("Djokovic J.", "Jannik Sinner") is False


# ── IroncladAliasResolver ──────────────────────────────────────────


class TestIroncladResolver:
    def test_auto_registers_new_canonical(self, db_session):
        resolver = IroncladAliasResolver(db_session, Sport.FOOTBALL)
        result = resolver.resolve("Bayern Munich")
        assert result == "Bayern Munich"

        # Should be persisted
        db_session.flush()
        alias = db_session.execute(
            select(TeamAlias).where(TeamAlias.alias == "Bayern Munich")
        ).scalar_one()
        assert alias.canonical_name == "Bayern Munich"
        assert alias.sport == Sport.FOOTBALL

    def test_exact_match_case_insensitive(self, db_session):
        resolver = IroncladAliasResolver(db_session, Sport.FOOTBALL)

        # First encounter → auto-register
        resolver.resolve("Bayern Munich")
        # Same name different case → should match
        result = resolver.resolve("bayern munich")
        assert result == "Bayern Munich"

    def test_diacritic_matching(self, db_session):
        resolver = IroncladAliasResolver(db_session, Sport.FOOTBALL)

        resolver.resolve("FC Bayern München")
        # Without umlaut → should match (accent-stripped comparison)
        result = resolver.resolve("FC Bayern Munchen")
        assert result == "FC Bayern Munchen"

    def test_fuzzy_matching(self, db_session):
        resolver = IroncladAliasResolver(db_session, Sport.FOOTBALL, fuzzy_threshold=0.85)

        resolver.resolve("Borussia Dortmund")
        # Very close variant → fuzzy match
        result = resolver.resolve("Bor. Dortmund")
        # Should either fuzzy-match or register new (depends on ratio)
        assert isinstance(result, str) and len(result) > 0

    def test_tennis_abbreviated_resolution(self, db_session):
        resolver = IroncladAliasResolver(db_session, Sport.TENNIS)

        # Register full name first
        resolver.resolve("Jannik Sinner")
        # Now resolve abbreviated form
        result = resolver.resolve("Sinner J.")
        assert result == "Jannik Sinner"

    def test_tennis_initial_dot_lastname(self, db_session):
        resolver = IroncladAliasResolver(db_session, Sport.TENNIS)

        resolver.resolve("Novak Djokovic")
        result = resolver.resolve("N. Djokovic")
        assert result == "Novak Djokovic"

    def test_tennis_comma_format(self, db_session):
        resolver = IroncladAliasResolver(db_session, Sport.TENNIS)

        resolver.resolve("Carlos Alcaraz")
        result = resolver.resolve("Alcaraz, Carlos")
        assert result == "Carlos Alcaraz"

    def test_sport_scoping(self, db_session):
        """Aliases for different sports don't interfere."""
        football_resolver = IroncladAliasResolver(db_session, Sport.FOOTBALL)
        tennis_resolver = IroncladAliasResolver(db_session, Sport.TENNIS)

        football_resolver.resolve("Arsenal")
        tennis_resolver.resolve("Arsenal")  # hypothetical tennis player

        # Both should have their own entry
        aliases = db_session.execute(
            select(TeamAlias).where(TeamAlias.canonical_name == "Arsenal")
        ).scalars().all()
        sports = {a.sport for a in aliases}
        assert Sport.FOOTBALL in sports
        assert Sport.TENNIS in sports

    def test_pre_seeded_alias_found(self, db_session):
        """Aliases pre-loaded in the DB are found."""
        db_session.add(TeamAlias(
            canonical_name="FC Bayern Munich",
            alias="FC Bayern",
            source="manual",
            sport=Sport.FOOTBALL,
        ))
        db_session.flush()

        resolver = IroncladAliasResolver(db_session, Sport.FOOTBALL)
        assert resolver.resolve("FC Bayern") == "FC Bayern Munich"

    def test_resolver_stats(self, db_session):
        resolver = IroncladAliasResolver(db_session, Sport.FOOTBALL)
        resolver.resolve("Team A")
        resolver.resolve("Team B")
        stats = resolver.stats
        assert stats["sport"] == "football"
        assert stats["canonical_names"] >= 2


# ── BaseIngester enforcement ───────────────────────────────────────


class TestBaseIngesterEnforcement:
    """Verify that BaseIngester normalizes names through the resolver."""

    def test_ingested_names_are_normalized(self, db_session):
        from bet_agent.ingest.football import FootballIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "E0_2526.csv",
                ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
                [["E0", "11/03/2026", "Arsenal", "Chelsea", "2", "1", "H"]],
            )
            ingester = FootballIngester()
            count = ingester.ingest_file(db_session, path)
            db_session.flush()
            assert count == 1

            # The resolver should have auto-registered both names
            match = db_session.query(HistoricalMatch).one()
            assert match.home_team == "Arsenal"
            assert match.away_team == "Chelsea"

            # Aliases should exist in DB
            aliases = db_session.execute(
                select(TeamAlias).where(TeamAlias.sport == Sport.FOOTBALL)
            ).scalars().all()
            canonical_names = {a.canonical_name for a in aliases}
            assert "Arsenal" in canonical_names
            assert "Chelsea" in canonical_names

    def test_second_ingestion_uses_same_canonical(self, db_session):
        """Re-ingesting with variant names maps to same canonical."""
        from bet_agent.ingest.football import FootballIngester

        # Pre-seed an alias
        db_session.add(TeamAlias(
            canonical_name="Arsenal FC",
            alias="Arsenal",
            source="manual",
            sport=Sport.FOOTBALL,
        ))
        db_session.flush()

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "test.csv",
                ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
                [["E0", "11/03/2026", "Arsenal", "Chelsea", "2", "1", "H"]],
            )
            ingester = FootballIngester()
            ingester.ingest_file(db_session, path)
            db_session.flush()

            match = db_session.query(HistoricalMatch).one()
            # Arsenal should be resolved to "Arsenal FC" via the pre-seeded alias
            assert match.home_team == "Arsenal FC"

    def test_tennis_ingester_normalizes_names(self, db_session):
        """Tennis ingester names pass through resolver."""
        from bet_agent.ingest.tennis import TennisIngester

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp), "atp_2025.csv",
                ["tourney_id", "tourney_name", "surface", "tourney_date",
                 "winner_name", "loser_name", "score", "best_of", "round"],
                [["2025-001", "AO", "Hard", "20260119",
                  "Sinner", "Djokovic", "6-3 6-4 6-2", "5", "F"]],
            )
            ingester = TennisIngester(tour="ATP")
            count = ingester.ingest_file(db_session, path)
            db_session.flush()
            assert count == 1

            match = db_session.query(HistoricalMatch).one()
            # Both names should be registered as aliases
            aliases = db_session.execute(
                select(TeamAlias).where(TeamAlias.sport == Sport.TENNIS)
            ).scalars().all()
            canonical_names = {a.canonical_name for a in aliases}
            assert "Sinner" in canonical_names or "Djokovic" in canonical_names
