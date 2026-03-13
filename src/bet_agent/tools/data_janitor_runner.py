"""Data Janitor Runner — scheduled job entrypoints and CLI commands.

Provides the three scheduled jobs for the active Data Janitor:
  - Intra-day scan (every 30 min): detect gaps, trigger targeted backfill
  - Nightly backfill (03:30 UTC): full backfill + open gap sweep
  - Weekly reconcile (Sunday 02:00 UTC): deep reconcile + coverage audit

Also provides CLI entrypoints for manual invocation:
  betagent-gap-scan
  betagent-backfill
  betagent-reconcile
  betagent-coverage
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta

from bet_agent.tools.backfill_engine import (
    BackfillResult,
    ReconcileResult,
    backfill_day,
    backfill_open_gaps,
    reconcile_open_results,
)
from bet_agent.tools.coverage_engine import (
    CoverageReport,
    GapReport,
    check_quality_gate,
    coverage_report,
    scan_gaps,
)
from bet_agent.tools.fixture_seeder import (
    SeedResult,
    seed_fixtures_for_window,
    seed_today_window,
)

logger = logging.getLogger(__name__)


# ── Database session helper ──────────────────────────────────────────────


def _get_session():
    """Create a database session from environment config.

    Expects DATABASE_URL environment variable or .env file.
    """
    import os

    from dotenv import load_dotenv
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    load_dotenv()
    url = os.environ.get("DATABASE_URL", "sqlite:///betagent.db")
    engine = create_engine(url)
    return Session(engine)


# ── Scheduled job functions ──────────────────────────────────────────────


def run_intraday_scan(session) -> dict:
    """30-minute scan: seed fixtures, detect gaps, trigger targeted backfill.

    Called by the scheduler every 30 minutes. First seeds any missing
    fixtures for the operational window, then scans for data gaps and
    attempts to fill them from API sources.

    Returns:
        Summary dict with seeding, gap scan, and backfill results.
    """
    logger.info("=== Intra-day scan starting ===")

    today = date.today()

    # 0. Seed missing fixtures for today's operational window
    seed_result = seed_today_window(session)

    # 1. Scan for gaps in today's matches
    gap_report = scan_gaps(session, target_date=today, lookback_days=1)

    result = {
        "job": "intraday_scan",
        "seeding": seed_result.to_dict(),
        "gaps": gap_report.to_dict(),
        "backfill": None,
    }

    if gap_report.total_gaps > 0:
        # Targeted backfill for today only
        backfill_result = backfill_day(session, today)
        result["backfill"] = backfill_result.to_dict()

    session.commit()

    logger.info(
        "=== Intra-day scan complete: %d seeded, %d gaps found, %s filled ===",
        seed_result.fixtures_inserted,
        gap_report.total_gaps,
        result["backfill"]["total_filled"] if result["backfill"] else 0,
    )

    return result


def run_nightly_backfill(session) -> dict:
    """Nightly backfill: seed fixtures, backfill yesterday + open gaps.

    Called at 03:30 UTC, runs before the Auditor's morning audit at 05:00 UTC.
    Seeds fixtures for the upcoming window first, then ensures maximum data
    coverage before settlement and metrics evaluation.

    Returns:
        Summary dict with seeding, backfill, and coverage results.
    """
    logger.info("=== Nightly backfill starting ===")

    yesterday = date.today() - timedelta(days=1)

    # 0. Seed fixtures for today's operational window (upcoming matches)
    seed_result = seed_today_window(session)

    # 1. Backfill yesterday specifically
    yesterday_result = backfill_day(session, yesterday)

    # 2. Sweep all open gaps from the last 7 days
    open_result = backfill_open_gaps(session, max_days_back=7)

    # 3. Generate coverage report
    report = coverage_report(session, lookback_days=7)

    session.commit()

    result = {
        "job": "nightly_backfill",
        "seeding": seed_result.to_dict(),
        "yesterday_backfill": yesterday_result.to_dict(),
        "open_gaps_backfill": open_result.to_dict(),
        "coverage": report.to_dict(),
    }

    logger.info(
        "=== Nightly backfill complete: %d seeded, yesterday=%d, open_gaps=%d filled ===",
        seed_result.fixtures_inserted,
        yesterday_result.total_filled, open_result.total_filled,
    )

    return result


def run_weekly_reconcile(session) -> dict:
    """Weekly deep reconcile: resolve all unmatched, full coverage audit.

    Called every Sunday at 02:00 UTC. Performs:
    1. Deep result reconciliation for all unresolved matches
    2. Full 30-day coverage report
    3. Quality gate check per sport
    4. Alias audit (detect issues)

    Returns:
        Summary dict with all results.
    """
    logger.info("=== Weekly reconcile starting ===")

    # 1. Reconcile all open results
    reconcile_result = reconcile_open_results(session)

    # 2. Full 30-day coverage report
    report = coverage_report(session, lookback_days=30)

    # 3. Quality gate checks per sport
    from bet_agent.db.models import Sport
    gate_results = {}
    for sport in Sport:
        gate = check_quality_gate(session, sport.value)
        gate_results[sport.value] = gate.to_dict()

    session.commit()

    result = {
        "job": "weekly_reconcile",
        "reconcile": reconcile_result.to_dict(),
        "coverage_30d": report.to_dict(),
        "quality_gates": gate_results,
    }

    # Log alerts for failed gates
    for sport_val, gate_dict in gate_results.items():
        if not gate_dict["passed"]:
            logger.warning(
                "QUALITY GATE FAILED for %s: %s",
                sport_val, "; ".join(gate_dict["violations"]),
            )

    logger.info(
        "=== Weekly reconcile complete: %d/%d resolved, %d sports checked ===",
        reconcile_result.matches_resolved,
        reconcile_result.matches_checked,
        len(gate_results),
    )

    return result


def run_coverage_report_job(session, output_format: str = "json") -> dict | str:
    """Generate and return a coverage report.

    Args:
        session: SQLAlchemy session.
        output_format: "json" or "text".

    Returns:
        Coverage report as dict or formatted text.
    """
    report = coverage_report(session, lookback_days=30)
    data = report.to_dict()

    if output_format == "text":
        lines = [f"Coverage Report ({data['report_date']})"]
        lines.append("=" * 50)
        for sport_kpis in data.get("sports", []):
            lines.append(f"\n{sport_kpis['sport'].upper()}")
            lines.append(f"  Total matches:     {sport_kpis['total_matches']}")
            lines.append(f"  Final scores:      {sport_kpis['pct_final_score']:.1%}")
            lines.append(f"  Halftime splits:   {sport_kpis['pct_halftime_splits']:.1%}")
            lines.append(f"  Key features:      {sport_kpis['pct_key_features']:.1%}")
            lines.append(f"  Canonical mapped:  {sport_kpis['pct_canonical_mapped']:.1%}")
            lines.append(f"  Stale matches:     {sport_kpis['stale_count']}")
        return "\n".join(lines)

    return data


# ── CLI entrypoints ──────────────────────────────────────────────────────


def cli_gap_scan():
    """CLI: Scan for data gaps."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scan for data gaps")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--sports", type=str, nargs="*", help="Sports to scan")
    parser.add_argument("--lookback", type=int, default=7, help="Days to look back")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    session = _get_session()
    try:
        report = scan_gaps(session, target_date=target, sports=args.sports, lookback_days=args.lookback)
        print(json.dumps(report.to_dict(), indent=2))
    finally:
        session.close()


def cli_backfill_day():
    """CLI: Backfill missing data for a specific day."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Backfill data for a day")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--sports", type=str, nargs="*", help="Sports to backfill")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)

    session = _get_session()
    try:
        result = backfill_day(session, target, sports=args.sports)
        session.commit()
        print(json.dumps(result.to_dict(), indent=2))
    finally:
        session.close()


def cli_reconcile_open_results():
    """CLI: Reconcile all unresolved match results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    session = _get_session()
    try:
        result = reconcile_open_results(session)
        session.commit()
        print(json.dumps(result.to_dict(), indent=2))
    finally:
        session.close()


def cli_seed_fixtures():
    """CLI: Seed fixtures for the operational window."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Seed fixtures for operational window")
    parser.add_argument("--sports", type=str, nargs="*", help="Sports to seed")
    parser.add_argument(
        "--start", type=str, default=None,
        help="Window start (ISO datetime). Defaults to today 07:00 UTC.",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Window end (ISO datetime). Defaults to tomorrow 07:00 UTC.",
    )
    args = parser.parse_args()

    session = _get_session()
    try:
        if args.start and args.end:
            from datetime import datetime, timezone
            start = datetime.fromisoformat(args.start)
            end = datetime.fromisoformat(args.end)
            result = seed_fixtures_for_window(session, start, end, sports=args.sports)
        else:
            result = seed_today_window(session, sports=args.sports)
        session.commit()
        print(json.dumps(result.to_dict(), indent=2))
    finally:
        session.close()


def cli_coverage_report():
    """CLI: Generate coverage report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate coverage report")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    args = parser.parse_args()

    session = _get_session()
    try:
        output = run_coverage_report_job(session, output_format=args.format)
        if isinstance(output, str):
            print(output)
        else:
            print(json.dumps(output, indent=2))
    finally:
        session.close()
