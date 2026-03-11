"""Tests for daily stats pipeline — crawler, parser, and DB upsert."""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import Base, Sport, TeamDailyStats
from bet_agent.tools.data_parser import (
    extract_tables,
    parse_basketball_stats,
    parse_football_stats,
    parse_ice_hockey_stats,
    parse_american_football_stats,
    parse_tennis_stats,
    process_crawl_results,
    upsert_daily_stats,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


# ── HTML table extraction ─────────────────────────────────────────────


class TestTableExtractor:
    def test_simple_table(self):
        html = """
        <table>
            <tr><th>Team</th><th>xG</th><th>xGA</th></tr>
            <tr><td>Arsenal</td><td>2.1</td><td>0.8</td></tr>
            <tr><td>Chelsea</td><td>1.5</td><td>1.2</td></tr>
        </table>
        """
        tables = extract_tables(html)
        assert len(tables) == 1
        assert len(tables[0]) == 3  # header + 2 rows
        assert tables[0][0] == ["Team", "xG", "xGA"]
        assert tables[0][1] == ["Arsenal", "2.1", "0.8"]

    def test_no_tables(self):
        assert extract_tables("<p>No tables here</p>") == []

    def test_multiple_tables(self):
        html = """
        <table><tr><td>A</td></tr></table>
        <table><tr><td>B</td></tr></table>
        """
        tables = extract_tables(html)
        assert len(tables) == 2


# ── Football parser ───────────────────────────────────────────────────


class TestFootballParser:
    def test_parses_xg_table(self):
        html = """
        <table>
            <tr><th>Squad</th><th>xG</th><th>xGA</th><th>Poss</th></tr>
            <tr><td>Arsenal</td><td>65.3</td><td>28.1</td><td>62.5</td></tr>
            <tr><td>Man City</td><td>72.1</td><td>31.2</td><td>67.3</td></tr>
        </table>
        """
        results = parse_football_stats(html, league="Premier League")
        assert len(results) == 2
        team, league, stats = results[0]
        assert team == "Arsenal"
        assert league == "Premier League"
        assert stats["xg"] == 65.3
        assert stats["xga"] == 28.1
        assert stats["possession_pct"] == 62.5

    def test_skips_header_rows(self):
        html = """
        <table>
            <tr><th>Squad</th><th>xG</th></tr>
            <tr><td>Squad</td><td>xG</td></tr>
        </table>
        """
        results = parse_football_stats(html)
        assert len(results) == 0

    def test_no_squad_column(self):
        html = """
        <table>
            <tr><th>Rank</th><th>Points</th></tr>
            <tr><td>1</td><td>90</td></tr>
        </table>
        """
        results = parse_football_stats(html)
        assert len(results) == 0


# ── Basketball parser ─────────────────────────────────────────────────


class TestBasketballParser:
    def test_parses_pace_efficiency(self):
        html = """
        <table>
            <tr><th>Team</th><th>Pace</th><th>ORtg</th><th>DRtg</th><th>FG%</th></tr>
            <tr><td>Boston Celtics*</td><td>100.5</td><td>118.2</td><td>108.5</td><td>.492</td></tr>
        </table>
        """
        results = parse_basketball_stats(html)
        assert len(results) == 1
        team, _, stats = results[0]
        assert team == "Boston Celtics"  # asterisk stripped
        assert stats["pace"] == 100.5
        assert stats["off_rtg"] == 118.2
        assert stats["def_rtg"] == 108.5


# ── Ice hockey parser ─────────────────────────────────────────────────


class TestIceHockeyParser:
    def test_parses_corsi(self):
        html = """
        <table>
            <tr><th>Team</th><th>CF%</th><th>FF%</th><th>PP%</th><th>PK%</th><th>SV%</th></tr>
            <tr><td>Colorado Avalanche</td><td>53.2</td><td>52.8</td><td>24.1</td><td>81.3</td><td>.912</td></tr>
        </table>
        """
        results = parse_ice_hockey_stats(html)
        assert len(results) == 1
        _, _, stats = results[0]
        assert stats["corsi_for_pct"] == 53.2
        assert stats["fenwick_for_pct"] == 52.8
        assert stats["pp_pct"] == 24.1


# ── American football parser ─────────────────────────────────────────


class TestAmericanFootballParser:
    def test_parses_team_stats(self):
        html = """
        <table>
            <tr><th>Tm</th><th>PF</th><th>PA</th><th>Y/P</th><th>TO</th></tr>
            <tr><td>Kansas City Chiefs+</td><td>496</td><td>310</td><td>6.2</td><td>12</td></tr>
        </table>
        """
        results = parse_american_football_stats(html)
        assert len(results) == 1
        team, _, stats = results[0]
        assert team == "Kansas City Chiefs"  # markers stripped
        assert stats["pts_for"] == 496.0
        assert stats["pts_against"] == 310.0
        assert stats["yards_per_play"] == 6.2


# ── Tennis parser ─────────────────────────────────────────────────────


class TestTennisParser:
    def test_parses_serve_stats(self):
        html = """
        <table>
            <tr><th>Player</th><th>Ace%</th><th>1st%</th><th>1stW%</th><th>BPS%</th></tr>
            <tr><td>C. Alcaraz</td><td>9.2</td><td>63.5</td><td>75.1</td><td>62.3</td></tr>
        </table>
        """
        results = parse_tennis_stats(html)
        assert len(results) == 1
        player, _, stats = results[0]
        assert player == "C. Alcaraz"
        assert stats["ace_pct"] == 9.2
        assert stats["first_serve_pct"] == 63.5


# ── DB upsert ─────────────────────────────────────────────────────────


class TestUpsertDailyStats:
    def test_insert_new_row(self, db_session):
        row = upsert_daily_stats(
            session=db_session,
            sport="football",
            team_name="Arsenal",
            league="Premier League",
            stat_date=date(2026, 3, 11),
            stats={"xg": 65.3, "xga": 28.1},
        )
        db_session.flush()

        assert row.sport == Sport.FOOTBALL
        assert row.team_name == "Arsenal"
        assert row.stats["xg"] == 65.3

    def test_upsert_merges_stats(self, db_session):
        upsert_daily_stats(
            session=db_session,
            sport="basketball",
            team_name="Celtics",
            league="NBA",
            stat_date=date(2026, 3, 11),
            stats={"pace": 100.5, "off_rtg": 118.2},
        )
        db_session.flush()

        # Upsert with additional stats
        row = upsert_daily_stats(
            session=db_session,
            sport="basketball",
            team_name="Celtics",
            league="NBA",
            stat_date=date(2026, 3, 11),
            stats={"def_rtg": 108.5, "pace": 101.0},  # pace overrides
        )
        db_session.flush()

        assert row.stats["pace"] == 101.0  # overridden
        assert row.stats["off_rtg"] == 118.2  # preserved
        assert row.stats["def_rtg"] == 108.5  # new

    def test_different_dates_separate_rows(self, db_session):
        upsert_daily_stats(
            db_session, "football", "Arsenal", "PL",
            date(2026, 3, 10), {"xg": 60.0},
        )
        upsert_daily_stats(
            db_session, "football", "Arsenal", "PL",
            date(2026, 3, 11), {"xg": 65.0},
        )
        db_session.flush()

        rows = db_session.query(TeamDailyStats).filter_by(
            team_name="Arsenal",
        ).all()
        assert len(rows) == 2


# ── process_crawl_results ─────────────────────────────────────────────


class TestProcessCrawlResults:
    def test_processes_football_pages(self, db_session):
        pages = [
            {
                "url": "https://fbref.com/en/comps/9/",
                "content": """
                <table>
                    <tr><th>Squad</th><th>xG</th><th>xGA</th></tr>
                    <tr><td>Arsenal</td><td>65.3</td><td>28.1</td></tr>
                    <tr><td>Man City</td><td>72.1</td><td>31.2</td></tr>
                </table>
                """,
            },
        ]
        count = process_crawl_results(
            db_session, "football", pages,
            stat_date=date(2026, 3, 11),
        )
        db_session.flush()
        assert count == 2

        rows = db_session.query(TeamDailyStats).all()
        assert len(rows) == 2

    def test_unknown_sport_returns_zero(self, db_session):
        count = process_crawl_results(db_session, "cricket", [])
        assert count == 0

    def test_empty_pages_returns_zero(self, db_session):
        count = process_crawl_results(db_session, "football", [])
        assert count == 0

    def test_page_without_content_skipped(self, db_session):
        pages = [{"url": "https://example.com", "content": ""}]
        count = process_crawl_results(db_session, "football", pages)
        assert count == 0


# ── Cloudflare crawler ────────────────────────────────────────────────


class TestCrawlerConfig:
    def test_sport_configs_have_required_keys(self):
        from bet_agent.tools.cloudflare_crawler import SPORT_CRAWL_CONFIGS

        for sport, config in SPORT_CRAWL_CONFIGS.items():
            assert "seed_url" in config, f"{sport} missing seed_url"
            assert "max_pages" in config, f"{sport} missing max_pages"
            assert config["max_pages"] <= 100, f"{sport} exceeds 100-page limit"

    def test_total_crawls_within_free_tier(self):
        from bet_agent.tools.cloudflare_crawler import (
            MAX_CRAWLS_PER_DAY,
            SPORT_CRAWL_CONFIGS,
        )
        assert len(SPORT_CRAWL_CONFIGS) <= MAX_CRAWLS_PER_DAY

    def test_missing_credentials_raises(self):
        import os
        from bet_agent.tools.cloudflare_crawler import _get_credentials

        # Ensure env vars are not set
        old_id = os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
        old_token = os.environ.pop("CLOUDFLARE_API_TOKEN", None)
        try:
            with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
                _get_credentials()
        finally:
            if old_id:
                os.environ["CLOUDFLARE_ACCOUNT_ID"] = old_id
            if old_token:
                os.environ["CLOUDFLARE_API_TOKEN"] = old_token

    def test_start_crawl_unknown_sport_no_url(self):
        from bet_agent.tools.cloudflare_crawler import start_crawl

        with pytest.raises(ValueError, match="No seed URL"):
            start_crawl("cricket")


# ── TeamDailyStats model ─────────────────────────────────────────────


class TestTeamDailyStatsModel:
    def test_table_exists(self, db_session):
        """team_daily_stats table should be created."""
        from sqlalchemy import inspect
        inspector = inspect(db_session.bind)
        tables = inspector.get_table_names()
        assert "team_daily_stats" in tables

    def test_repr(self, db_session):
        row = TeamDailyStats(
            sport=Sport.BASKETBALL,
            team_name="Celtics",
            league="NBA",
            stat_date=date(2026, 3, 11),
            stats={"pace": 100.5},
        )
        db_session.add(row)
        db_session.flush()
        assert "Celtics" in repr(row)
        assert "basketball" in repr(row)
