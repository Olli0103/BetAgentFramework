"""Base ingester class for historical data imports.

Golden Rule #1: NO LLM MATH. All parsing is deterministic Python.
Golden Rule #2: STATEFUL MEMORY. All data goes to PostgreSQL.
Golden Rule #3: NO RAW STRINGS. Every name passes through IroncladAliasResolver.
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import HistoricalMatch, Sport, TeamDailyStats

logger = logging.getLogger(__name__)


class BaseIngester(ABC):
    """Base class for sport-specific historical data ingesters.

    ENFORCEMENT: After each row_to_model() call, home_team and away_team
    are normalized through the IroncladAliasResolver. No subclass can
    write raw strings to the database — the base class intercepts and
    resolves every name before the upsert.
    """

    sport: Sport
    source_name: str  # e.g. "football_data_co_uk", "sackmann_atp"

    def __init__(self) -> None:
        self._resolver = None

    @abstractmethod
    def parse_file(self, file_path: Path) -> list[dict]:
        """Parse a CSV/XLSX file into a list of row dicts.

        Each dict must contain the keys expected by `row_to_model()`.
        """

    @abstractmethod
    def row_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        """Convert a parsed row dict to a HistoricalMatch instance."""

    def _ensure_resolver(self, session: Session) -> None:
        """Lazily initialize the IroncladAliasResolver on first use."""
        if self._resolver is None:
            from bet_agent.ingest.alias_resolver import IroncladAliasResolver

            self._resolver = IroncladAliasResolver(session, self.sport)

    def resolve_name(self, raw_name: str) -> str:
        """Resolve a team/player name through the alias resolver.

        Can be called by subclasses for additional name fields beyond
        home_team / away_team (e.g. match_stats["actual_winner"]).
        Raises RuntimeError if called before ingest_file (no resolver).
        """
        if self._resolver is None:
            raise RuntimeError(
                "resolve_name() called before resolver initialization. "
                "Call ingest_file() first, or use _ensure_resolver()."
            )
        return self._resolver.resolve(raw_name)

    def _normalize_model(self, model: HistoricalMatch) -> HistoricalMatch:
        """Enforce name normalization on a HistoricalMatch.

        This is the IRONCLAD GATE — called after every row_to_model().
        No raw string gets past this point.
        """
        assert self._resolver is not None, "Resolver must be initialized"
        model.home_team = self._resolver.resolve(model.home_team)
        model.away_team = self._resolver.resolve(model.away_team)
        return model

    def ingest_file(self, session: Session, file_path: Path) -> int:
        """Parse a file and upsert all rows into the database.

        Args:
            session: SQLAlchemy session (caller manages transaction).
            file_path: Path to the CSV/XLSX file.

        Returns:
            Number of rows upserted.
        """
        self._ensure_resolver(session)

        rows = self.parse_file(file_path)
        count = 0

        for row in rows:
            try:
                model = self.row_to_model(row, str(file_path))
                model = self._normalize_model(model)
                self._upsert(session, model)
                count += 1
            except Exception as exc:
                logger.warning(
                    "Skipping row in %s: %s (data: %s)",
                    file_path.name, exc, _truncate(row),
                )

        logger.info(
            "Ingested %d/%d rows from %s (%s)",
            count, len(rows), file_path.name, self.source_name,
        )
        return count

    def _upsert(self, session: Session, model: HistoricalMatch) -> None:
        """Insert or update a historical match row."""
        existing = session.execute(
            select(HistoricalMatch).where(
                HistoricalMatch.sport == model.sport,
                HistoricalMatch.home_team == model.home_team,
                HistoricalMatch.away_team == model.away_team,
                HistoricalMatch.match_date == model.match_date,
                HistoricalMatch.division == model.division,
            )
        ).scalar_one_or_none()

        if existing:
            existing.home_score = model.home_score
            existing.away_score = model.away_score
            existing.result = model.result
            existing.match_stats = {**existing.match_stats, **model.match_stats}
            existing.odds = {**existing.odds, **model.odds}
            existing.betting_lines = {**existing.betting_lines, **model.betting_lines}
            existing.advanced_stats = {**existing.advanced_stats, **model.advanced_stats}
            existing.source = model.source
            if model.source_file:
                existing.source_file = model.source_file
        else:
            session.add(model)

    def _upsert_daily_stat(
        self,
        session: Session,
        team: str,
        stat_date: date,
        stats: dict,
        league: str = "unknown",
    ) -> None:
        """Insert or merge a TeamDailyStats row for this sport.

        Used by subclass ingesters that build point-in-time profiles
        during ingestion (Tennis, NHL).
        """
        existing = session.execute(
            select(TeamDailyStats).where(
                TeamDailyStats.sport == self.sport,
                TeamDailyStats.team_name == team,
                TeamDailyStats.stat_date == stat_date,
            )
        ).scalar_one_or_none()

        if existing:
            existing.stats = {**existing.stats, **stats}
        else:
            session.add(TeamDailyStats(
                sport=self.sport,
                team_name=team,
                league=league,
                stat_date=stat_date,
                stats=stats,
                source_url="ingester_profile",
            ))

    def ingest_directory(self, session: Session, dir_path: Path, glob: str = "*.csv") -> int:
        """Ingest all matching files from a directory.

        Returns:
            Total rows upserted across all files.
        """
        total = 0
        files = sorted(dir_path.glob(glob))
        if not files:
            logger.warning("No files matching %s in %s", glob, dir_path)
            return 0

        for f in files:
            total += self.ingest_file(session, f)

        logger.info(
            "Ingested %d total rows from %d files in %s",
            total, len(files), dir_path,
        )
        return total


def read_csv(file_path: Path, encoding: str = "utf-8") -> list[dict]:
    """Read a CSV file into a list of dicts (header row = keys)."""
    with open(file_path, newline="", encoding=encoding, errors="replace") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_int(val: str | None, default: int = 0) -> int:
    """Parse an int from a string, returning default on failure."""
    if not val or not val.strip():
        return default
    try:
        return int(float(val.strip()))
    except (ValueError, TypeError):
        return default


def safe_float(val: str | None, default: float | None = None) -> float | None:
    """Parse a float from a string, returning default on failure.

    Rejects NaN and Inf to prevent feature vector poisoning.
    """
    if not val or not val.strip():
        return default
    try:
        result = float(val.strip())
        if result != result or result == float("inf") or result == float("-inf"):
            return default
        return result
    except (ValueError, TypeError):
        return default


def parse_date(val: str, formats: list[str] | None = None) -> date:
    """Parse a date string, trying multiple formats."""
    from datetime import datetime as dt

    if formats is None:
        formats = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"]

    for fmt in formats:
        try:
            return dt.strptime(val.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: '{val}'")


def _truncate(row: dict, max_keys: int = 5) -> str:
    """Truncate a row dict for logging."""
    items = list(row.items())[:max_keys]
    s = ", ".join(f"{k}={v}" for k, v in items)
    if len(row) > max_keys:
        s += f", ... ({len(row)} total keys)"
    return s
