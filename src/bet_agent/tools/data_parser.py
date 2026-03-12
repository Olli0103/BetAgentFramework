"""Data Janitor parsing & upsert logic for daily stats.

Golden Rule #1: NO LLM MATH. This is deterministic HTML/JSON parsing.
Golden Rule #2: STATEFUL MEMORY. All parsed stats go to PostgreSQL.

Takes raw HTML/JSON from the Cloudflare crawler, extracts sport-specific
metrics, and UPSERTs them into the team_daily_stats table.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from html.parser import HTMLParser

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import Sport, TeamDailyStats

logger = logging.getLogger(__name__)


# ── Generic HTML table extractor ──────────────────────────────────────


class _TableExtractor(HTMLParser):
    """Extracts rows from HTML tables as list of lists of strings."""

    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell: str = ""
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in ("td", "th") and self._in_row:
            self._in_cell = True
            self._current_cell = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._current_row:
                self._current_table.append(self._current_row)
        elif tag == "table" and self._in_table:
            self._in_table = False
            if self._current_table:
                self.tables.append(self._current_table)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell += data


def extract_tables(html: str) -> list[list[list[str]]]:
    """Extract all HTML tables as list[table][row][cell]."""
    parser = _TableExtractor()
    parser.feed(html)
    return parser.tables


def _safe_float(val: str) -> float | None:
    """Parse a numeric string, returning None on failure."""
    try:
        cleaned = re.sub(r"[^\d.\-+]", "", val)
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


# ── Sport-specific parsers ────────────────────────────────────────────
# Each parser takes raw HTML content and returns a list of
# (team_name, league, stats_dict) tuples.


def parse_football_stats(
    html: str, league: str = "Unknown",
) -> list[tuple[str, str, dict]]:
    """Parse FBRef-style football stats (xG, possession, etc.).

    Looks for table rows with columns: Squad, xG, xGA, Poss, Sh, SoT, etc.
    """
    results = []
    tables = extract_tables(html)

    for table in tables:
        if not table:
            continue
        header = [h.lower().strip() for h in table[0]]

        # Look for the standard stats table
        squad_idx = _find_col(header, ["squad", "team"])
        xg_idx = _find_col(header, ["xg"])
        xga_idx = _find_col(header, ["xga", "xg against"])
        poss_idx = _find_col(header, ["poss", "possession"])

        if squad_idx is None:
            continue

        for row in table[1:]:
            if len(row) <= squad_idx:
                continue
            team = row[squad_idx].strip()
            if not team or team.lower() in ("squad", "team", ""):
                continue

            stats: dict = {}
            if xg_idx is not None and xg_idx < len(row):
                val = _safe_float(row[xg_idx])
                if val is not None:
                    stats["xg"] = val
            if xga_idx is not None and xga_idx < len(row):
                val = _safe_float(row[xga_idx])
                if val is not None:
                    stats["xga"] = val
            if poss_idx is not None and poss_idx < len(row):
                val = _safe_float(row[poss_idx])
                if val is not None:
                    stats["possession_pct"] = val

            # Extract additional columns if present
            for col_name in ["sh", "sot", "gf", "ga", "gd", "pts"]:
                idx = _find_col(header, [col_name])
                if idx is not None and idx < len(row):
                    val = _safe_float(row[idx])
                    if val is not None:
                        stats[col_name] = val

            if stats:
                results.append((team, league, stats))

    return results


def parse_basketball_stats(
    html: str, league: str = "NBA",
) -> list[tuple[str, str, dict]]:
    """Parse Basketball-Reference style stats (Pace, ORtg, DRtg, etc.)."""
    results = []
    tables = extract_tables(html)

    for table in tables:
        if not table:
            continue
        header = [h.lower().strip() for h in table[0]]

        team_idx = _find_col(header, ["team", "squad"])
        if team_idx is None:
            continue

        for row in table[1:]:
            if len(row) <= team_idx:
                continue
            team = row[team_idx].strip()
            team = re.sub(r"\*$", "", team).strip()  # Remove playoff asterisk
            if not team or team.lower() in ("team", "league average"):
                continue

            stats: dict = {}
            for col_name, keys in [
                ("pace", ["pace"]),
                ("off_rtg", ["ortg", "offrtg", "off rtg"]),
                ("def_rtg", ["drtg", "defrtg", "def rtg"]),
                ("net_rtg", ["nrtg", "netrtg", "net rtg", "mov"]),
                ("fg_pct", ["fg%", "fg pct"]),
                ("three_pct", ["3p%", "3p pct", "3pt%"]),
                ("ft_pct", ["ft%", "ft pct"]),
                ("trb", ["trb", "reb"]),
                ("ast", ["ast"]),
                ("tov", ["tov"]),
            ]:
                idx = _find_col(header, keys)
                if idx is not None and idx < len(row):
                    val = _safe_float(row[idx])
                    if val is not None:
                        stats[col_name] = val

            if stats:
                results.append((team, league, stats))

    return results


def parse_ice_hockey_stats(
    html: str, league: str = "NHL",
) -> list[tuple[str, str, dict]]:
    """Parse Hockey-Reference style stats (Corsi, Fenwick, SV%, etc.)."""
    results = []
    tables = extract_tables(html)

    for table in tables:
        if not table:
            continue
        header = [h.lower().strip() for h in table[0]]

        team_idx = _find_col(header, ["team", "squad"])
        if team_idx is None:
            continue

        for row in table[1:]:
            if len(row) <= team_idx:
                continue
            team = row[team_idx].strip()
            team = re.sub(r"\*$", "", team).strip()
            if not team or team.lower() in ("team", "league average"):
                continue

            stats: dict = {}
            for col_name, keys in [
                ("corsi_for_pct", ["cf%", "corsi%", "corsi for%"]),
                ("fenwick_for_pct", ["ff%", "fenwick%"]),
                ("pp_pct", ["pp%"]),
                ("pk_pct", ["pk%"]),
                ("sv_pct", ["sv%", "save%"]),
                ("sh_pct", ["sh%", "shooting%"]),
                ("gf", ["gf"]),
                ("ga", ["ga"]),
                ("pts", ["pts"]),
                ("so_pct", ["so%"]),
            ]:
                idx = _find_col(header, keys)
                if idx is not None and idx < len(row):
                    val = _safe_float(row[idx])
                    if val is not None:
                        stats[col_name] = val

            if stats:
                results.append((team, league, stats))

    return results


def parse_american_football_stats(
    html: str, league: str = "NFL",
) -> list[tuple[str, str, dict]]:
    """Parse Pro-Football-Reference style stats (EPA, yards, etc.)."""
    results = []
    tables = extract_tables(html)

    for table in tables:
        if not table:
            continue
        header = [h.lower().strip() for h in table[0]]

        team_idx = _find_col(header, ["tm", "team"])
        if team_idx is None:
            continue

        for row in table[1:]:
            if len(row) <= team_idx:
                continue
            team = row[team_idx].strip()
            team = re.sub(r"\*|\+$", "", team).strip()
            if not team or team.lower() in ("tm", "team", "avg total", "league total"):
                continue

            stats: dict = {}
            for col_name, keys in [
                ("pts_for", ["pf", "pts"]),
                ("pts_against", ["pa"]),
                ("yards_per_play", ["y/p"]),
                ("pass_yds", ["pyds", "pass yds"]),
                ("rush_yds", ["ryds", "rush yds"]),
                ("turnovers", ["to", "turnovers"]),
                ("takeaways", ["takeaways"]),
            ]:
                idx = _find_col(header, keys)
                if idx is not None and idx < len(row):
                    val = _safe_float(row[idx])
                    if val is not None:
                        stats[col_name] = val

            if stats:
                results.append((team, league, stats))

    return results


def parse_tennis_stats(
    html: str, league: str = "ATP",
) -> list[tuple[str, str, dict]]:
    """Parse Tennis Abstract style player stats.

    Tennis uses player names instead of team names in team_daily_stats.
    """
    results = []
    tables = extract_tables(html)

    for table in tables:
        if not table:
            continue
        header = [h.lower().strip() for h in table[0]]

        player_idx = _find_col(header, ["player", "name"])
        if player_idx is None:
            continue

        for row in table[1:]:
            if len(row) <= player_idx:
                continue
            player = row[player_idx].strip()
            if not player or player.lower() in ("player", "name"):
                continue

            stats: dict = {}
            for col_name, keys in [
                ("ace_pct", ["ace%", "aces%"]),
                ("first_serve_pct", ["1st%", "1stin%", "1st serve%"]),
                ("first_serve_won_pct", ["1stw%", "1st won%"]),
                ("second_serve_won_pct", ["2ndw%", "2nd won%"]),
                ("bp_saved_pct", ["bps%", "bp saved%"]),
                ("return_pts_won_pct", ["rpw%", "ret pts won%"]),
                ("service_games_won_pct", ["sg%", "svc games%"]),
            ]:
                idx = _find_col(header, keys)
                if idx is not None and idx < len(row):
                    val = _safe_float(row[idx])
                    if val is not None:
                        stats[col_name] = val

            if stats:
                results.append((player, league, stats))

    return results


def _find_col(header: list[str], candidates: list[str]) -> int | None:
    """Find column index matching any candidate name."""
    for i, h in enumerate(header):
        if h in candidates:
            return i
    return None


# ── Parser registry ───────────────────────────────────────────────────

SPORT_PARSERS = {
    "football": parse_football_stats,
    "basketball": parse_basketball_stats,
    "ice_hockey": parse_ice_hockey_stats,
    "american_football": parse_american_football_stats,
    "tennis": parse_tennis_stats,
}


# ── Upsert logic ─────────────────────────────────────────────────────


def upsert_daily_stats(
    session: Session,
    sport: str,
    team_name: str,
    league: str,
    stat_date: date,
    stats: dict,
    source_url: str | None = None,
    crawl_job_id: str | None = None,
) -> TeamDailyStats:
    """Insert or update a daily stats row.

    If a row for (sport, team_name, stat_date) exists, merges the new
    stats into the existing JSONB (new keys override, existing keys preserved).

    Args:
        session: SQLAlchemy session (caller manages transaction).
        sport: Sport string (must match Sport enum value).
        team_name: Canonical team/player name.
        league: League identifier.
        stat_date: Date the stats apply to.
        stats: Sport-specific stats dict.
        source_url: URL the data was crawled from.
        crawl_job_id: Cloudflare crawl job ID for traceability.

    Returns:
        The upserted TeamDailyStats row.
    """
    sport_enum = Sport(sport)

    existing = session.execute(
        select(TeamDailyStats).where(
            TeamDailyStats.sport == sport_enum,
            TeamDailyStats.team_name == team_name,
            TeamDailyStats.stat_date == stat_date,
        )
    ).scalar_one_or_none()

    if existing:
        # Merge: new stats override existing keys
        merged = {**existing.stats, **stats}
        existing.stats = merged
        existing.updated_at = datetime.now(timezone.utc)
        if source_url:
            existing.source_url = source_url
        if crawl_job_id:
            existing.crawl_job_id = crawl_job_id
        logger.debug("Updated stats for %s/%s on %s", sport, team_name, stat_date)
        return existing
    else:
        row = TeamDailyStats(
            sport=sport_enum,
            team_name=team_name,
            league=league,
            stat_date=stat_date,
            stats=stats,
            source_url=source_url,
            crawl_job_id=crawl_job_id,
        )
        session.add(row)
        logger.debug("Inserted stats for %s/%s on %s", sport, team_name, stat_date)
        return row


def process_crawl_results(
    session: Session,
    sport: str,
    pages: list[dict],
    crawl_job_id: str | None = None,
    stat_date: date | None = None,
) -> int:
    """Parse crawl results and upsert all extracted stats.

    This is the main entry point the Data Janitor calls after the
    Scout hands over raw crawl data.

    Args:
        session: SQLAlchemy session.
        sport: Sport identifier.
        pages: List of page dicts from CrawlResult.pages.
               Each should have "content" or "html" or "markdown" key.
        crawl_job_id: For traceability.
        stat_date: Override date (defaults to today UTC).

    Returns:
        Number of team/player stat rows upserted.
    """
    if stat_date is None:
        stat_date = datetime.now(timezone.utc).date()

    parser = SPORT_PARSERS.get(sport)
    if parser is None:
        logger.warning("No parser registered for sport: %s", sport)
        return 0

    # Ironclad gate: resolve all team names through the alias resolver
    resolver = None
    try:
        from bet_agent.ingest.alias_resolver import IroncladAliasResolver

        sport_enum = Sport(sport)
        resolver = IroncladAliasResolver(session, sport_enum)
    except (ValueError, ImportError) as exc:
        logger.warning(
            "Could not init alias resolver for %s: %s — raw names will be used",
            sport, exc,
        )

    total_upserted = 0

    for page in pages:
        html = page.get("content") or page.get("html") or page.get("markdown", "")
        url = page.get("url", "")

        if not html:
            continue

        try:
            parsed = parser(html)
        except Exception as exc:
            logger.error("Failed to parse page %s for %s: %s", url, sport, exc)
            continue

        for team_name, league, stats in parsed:
            try:
                canonical_name = resolver.resolve(team_name) if resolver else team_name
                upsert_daily_stats(
                    session=session,
                    sport=sport,
                    team_name=canonical_name,
                    league=league,
                    stat_date=stat_date,
                    stats=stats,
                    source_url=url,
                    crawl_job_id=crawl_job_id,
                )
                total_upserted += 1
            except Exception as exc:
                logger.error(
                    "Failed to upsert %s/%s: %s", sport, team_name, exc,
                )

    logger.info(
        "Processed %d pages for %s, upserted %d stat rows",
        len(pages), sport, total_upserted,
    )
    return total_upserted
