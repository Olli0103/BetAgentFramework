"""Base ingester class for historical data imports.

Golden Rule #1: NO LLM MATH. All parsing is deterministic Python.
Golden Rule #2: STATEFUL MEMORY. All data goes to PostgreSQL.
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import HistoricalMatch, Sport

logger = logging.getLogger(__name__)


class BaseIngester(ABC):
    """Base class for sport-specific historical data ingesters."""

    sport: Sport
    source_name: str  # e.g. "football_data_co_uk", "sackmann_atp"

    @abstractmethod
    def parse_file(self, file_path: Path) -> list[dict]:
        """Parse a CSV/XLSX file into a list of row dicts.

        Each dict must contain the keys expected by `row_to_model()`.
        """

    @abstractmethod
    def row_to_model(self, row: dict, source_file: str) -> HistoricalMatch:
        """Convert a parsed row dict to a HistoricalMatch instance."""

    def ingest_file(self, session: Session, file_path: Path) -> int:
        """Parse a file and upsert all rows into the database.

        Args:
            session: SQLAlchemy session (caller manages transaction).
            file_path: Path to the CSV/XLSX file.

        Returns:
            Number of rows upserted.
        """
        rows = self.parse_file(file_path)
        count = 0

        for row in rows:
            try:
                model = self.row_to_model(row, str(file_path))
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
    """Parse a float from a string, returning default on failure."""
    if not val or not val.strip():
        return default
    try:
        return float(val.strip())
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
