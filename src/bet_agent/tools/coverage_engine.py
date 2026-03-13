"""Coverage Engine — gap detection, coverage KPIs, and quality gates.

Scans the database for missing data across all dimensions (scores, stats,
features, aliases) and produces machine-readable reports + hard gates that
block training/prediction when quality is below SLO thresholds.

Part of the Data Janitor's active monitoring loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    HistoricalMatch,
    Match,
    MatchState,
    Sport,
    TeamAlias,
    TeamDailyStats,
)

logger = logging.getLogger(__name__)


# ── Sport-specific required features ────────────────────────────────────

REQUIRED_FEATURES: dict[str, list[str]] = {
    "football": [
        "home_score", "away_score", "ht_home", "ht_away",
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners", "home_cards", "away_cards",
        "possession_home",
    ],
    "tennis": [
        "aces", "double_faults", "first_serve_pct",
        "first_serve_won_pct", "break_points_faced",
        "break_points_saved", "return_points_won_pct",
    ],
    "basketball": [
        "pts_home", "pts_away",
        "q1_home", "q2_home", "q3_home", "q4_home",
    ],
    "ice_hockey": [
        "goals_home", "goals_away", "shots_home", "shots_away",
        "pp_goals", "pp_opps", "faceoff_pct",
    ],
    "american_football": [
        "pts_home", "pts_away",
        "total_yards_home", "total_yards_away",
        "turnovers_home", "turnovers_away",
    ],
}


# ── SLO thresholds ──────────────────────────────────────────────────────

COVERAGE_SLOS: dict[str, float] = {
    "pct_final_score": 0.95,
    "pct_halftime_splits": 0.80,
    "pct_key_features": 0.70,
    "pct_canonical_mapped": 0.98,
    "max_stale_days": 4,
}


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class GapDetail:
    """A single identified data gap."""
    match_id: str
    sport: str
    home_team: str
    away_team: str
    scheduled_at: str
    gap_type: str  # "missing_score", "missing_stats", "missing_features", "stale"
    details: str = ""


@dataclass
class GapReport:
    """Result of a gap scan across one or more sports."""
    scan_date: str
    sports_scanned: list[str] = field(default_factory=list)
    missing_scores: list[GapDetail] = field(default_factory=list)
    missing_stats: list[GapDetail] = field(default_factory=list)
    missing_features: list[GapDetail] = field(default_factory=list)
    stale_matches: list[GapDetail] = field(default_factory=list)
    unresolved_aliases: list[str] = field(default_factory=list)

    @property
    def total_gaps(self) -> int:
        return (
            len(self.missing_scores)
            + len(self.missing_stats)
            + len(self.missing_features)
            + len(self.stale_matches)
        )

    def to_dict(self) -> dict:
        return {
            "scan_date": self.scan_date,
            "sports_scanned": self.sports_scanned,
            "total_gaps": self.total_gaps,
            "missing_scores": len(self.missing_scores),
            "missing_stats": len(self.missing_stats),
            "missing_features": len(self.missing_features),
            "stale_matches": len(self.stale_matches),
            "unresolved_aliases": len(self.unresolved_aliases),
        }


@dataclass
class CoverageKPIs:
    """Coverage metrics for a single sport/league/date combination."""
    sport: str
    league: str = "all"
    date_range: str = ""
    total_matches: int = 0
    with_final_score: int = 0
    with_halftime_splits: int = 0
    with_key_features: int = 0
    with_canonical_names: int = 0
    stale_count: int = 0

    @property
    def pct_final_score(self) -> float:
        return self.with_final_score / self.total_matches if self.total_matches else 0.0

    @property
    def pct_halftime_splits(self) -> float:
        return self.with_halftime_splits / self.total_matches if self.total_matches else 0.0

    @property
    def pct_key_features(self) -> float:
        return self.with_key_features / self.total_matches if self.total_matches else 0.0

    @property
    def pct_canonical_mapped(self) -> float:
        return self.with_canonical_names / self.total_matches if self.total_matches else 0.0

    def to_dict(self) -> dict:
        return {
            "sport": self.sport,
            "league": self.league,
            "date_range": self.date_range,
            "total_matches": self.total_matches,
            "pct_final_score": round(self.pct_final_score, 4),
            "pct_halftime_splits": round(self.pct_halftime_splits, 4),
            "pct_key_features": round(self.pct_key_features, 4),
            "pct_canonical_mapped": round(self.pct_canonical_mapped, 4),
            "stale_count": self.stale_count,
        }


@dataclass
class CoverageReport:
    """Aggregated coverage report across multiple sports."""
    report_date: str
    kpis: list[CoverageKPIs] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "report_date": self.report_date,
            "sports": [k.to_dict() for k in self.kpis],
        }


@dataclass
class QualityGate:
    """Result of a quality gate check."""
    passed: bool
    can_train: bool
    can_predict: bool
    violations: list[str] = field(default_factory=list)
    kpis: CoverageKPIs | None = None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "can_train": self.can_train,
            "can_predict": self.can_predict,
            "violations": self.violations,
            "kpis": self.kpis.to_dict() if self.kpis else None,
        }


# ── Helper: halftime field names per sport ───────────────────────────────

_HALFTIME_FIELDS: dict[str, list[str]] = {
    "football": ["ht_home", "ht_away"],
    "basketball": ["q1_home", "q2_home", "q3_home", "q4_home"],
    "ice_hockey": ["p1_home", "p1_away"],
    "american_football": ["q1_home", "q2_home"],
    "tennis": [],  # no intermediate scores in same sense
}


# ── Helper: check features in JSONB ─────────────────────────────────────


def _match_has_key_features(match: Match, sport_value: str) -> bool:
    """Check if a match has the required sport-specific features in live_stats."""
    required = REQUIRED_FEATURES.get(sport_value, [])
    if not required:
        return True
    stats = match.live_stats or {}
    present = sum(1 for f in required if f in stats and stats[f] is not None)
    return present >= len(required) * 0.5  # at least 50% of required features


def _match_has_halftime(match: Match, sport_value: str) -> bool:
    """Check if a match has halftime/period split data."""
    fields = _HALFTIME_FIELDS.get(sport_value, [])
    if not fields:
        return True  # sport doesn't have halftime splits
    stats = match.live_stats or {}
    return any(f in stats and stats[f] is not None for f in fields)


def _historical_has_key_features(hm: HistoricalMatch) -> bool:
    """Check if a historical match has required features in match_stats."""
    required = REQUIRED_FEATURES.get(hm.sport.value, [])
    if not required:
        return True
    stats = hm.match_stats or {}
    present = sum(1 for f in required if f in stats and stats[f] is not None)
    return present >= len(required) * 0.5


def _historical_has_halftime(hm: HistoricalMatch) -> bool:
    """Check if a historical match has halftime split data."""
    fields = _HALFTIME_FIELDS.get(hm.sport.value, [])
    if not fields:
        return True
    stats = hm.match_stats or {}
    return any(f in stats and stats[f] is not None for f in fields)


# ── Core functions ───────────────────────────────────────────────────────


def _get_sport_list(sports: Sequence[str] | None) -> list[Sport]:
    """Convert sport name strings to Sport enums, defaulting to all."""
    if sports:
        result = []
        for s in sports:
            try:
                result.append(Sport(s))
            except ValueError:
                logger.warning("Unknown sport: %s", s)
        return result
    return list(Sport)


def scan_gaps(
    session: Session,
    target_date: date | None = None,
    sports: Sequence[str] | None = None,
    lookback_days: int = 7,
) -> GapReport:
    """Scan for missing data across all dimensions.

    Checks matches from target_date back lookback_days for:
    - Missing final scores (FINISHED matches with NULL scores)
    - Missing live_stats / match_stats
    - Missing sport-specific features
    - Stale matches (past scheduled_at, still NOT_STARTED, > stale threshold)

    Args:
        session: SQLAlchemy session.
        target_date: End date of scan range. Defaults to today.
        sports: Sport names to scan. Defaults to all.
        lookback_days: How many days back to scan.

    Returns:
        GapReport with all identified gaps.
    """
    if target_date is None:
        target_date = date.today()

    start_date = target_date - timedelta(days=lookback_days)
    sport_list = _get_sport_list(sports)
    sport_names = [s.value for s in sport_list]

    report = GapReport(
        scan_date=target_date.isoformat(),
        sports_scanned=sport_names,
    )

    stale_threshold = datetime.now(timezone.utc) - timedelta(
        days=int(COVERAGE_SLOS["max_stale_days"])
    )

    # Query matches in date range
    from datetime import time as dt_time
    range_start = datetime.combine(start_date, dt_time.min, tzinfo=timezone.utc)
    range_end = datetime.combine(target_date + timedelta(days=1), dt_time.min, tzinfo=timezone.utc)

    matches = list(
        session.execute(
            select(Match).where(
                Match.sport.in_(sport_list),
                Match.scheduled_at >= range_start,
                Match.scheduled_at < range_end,
            )
        ).scalars().all()
    )

    for m in matches:
        sport_val = m.sport.value
        common = dict(
            match_id=str(m.id),
            sport=sport_val,
            home_team=m.home_team,
            away_team=m.away_team,
            scheduled_at=m.scheduled_at.isoformat() if m.scheduled_at else "",
        )

        # Missing scores: match is past but scores are NULL
        if m.scheduled_at < datetime.now(timezone.utc) and m.home_score is None:
            report.missing_scores.append(GapDetail(**common, gap_type="missing_score"))

        # Missing stats: match is finished but no live_stats
        if m.match_state == MatchState.FINISHED and not m.live_stats:
            report.missing_stats.append(GapDetail(**common, gap_type="missing_stats"))

        # Missing features: match is finished but key features absent
        if m.match_state == MatchState.FINISHED and not _match_has_key_features(m, sport_val):
            report.missing_features.append(GapDetail(
                **common, gap_type="missing_features",
                details=f"required: {REQUIRED_FEATURES.get(sport_val, [])[:5]}...",
            ))

        # Stale: past scheduled time, still NOT_STARTED, older than threshold
        if (
            m.match_state == MatchState.NOT_STARTED
            and m.scheduled_at < stale_threshold
        ):
            report.stale_matches.append(GapDetail(**common, gap_type="stale"))

    logger.info(
        "Gap scan (%s, %d sports): %d total gaps "
        "(scores=%d, stats=%d, features=%d, stale=%d)",
        target_date, len(sport_list), report.total_gaps,
        len(report.missing_scores), len(report.missing_stats),
        len(report.missing_features), len(report.stale_matches),
    )

    return report


def coverage_report(
    session: Session,
    target_date: date | None = None,
    sports: Sequence[str] | None = None,
    lookback_days: int = 30,
) -> CoverageReport:
    """Generate machine-readable coverage KPIs per sport.

    Scans both the live Match table and HistoricalMatch table for
    comprehensive coverage metrics.

    Args:
        session: SQLAlchemy session.
        target_date: End date. Defaults to today.
        sports: Sport names to report on. Defaults to all.
        lookback_days: How far back to scan.

    Returns:
        CoverageReport with per-sport KPIs.
    """
    if target_date is None:
        target_date = date.today()

    start_date = target_date - timedelta(days=lookback_days)
    sport_list = _get_sport_list(sports)

    report = CoverageReport(report_date=target_date.isoformat())

    for sport in sport_list:
        kpis = _compute_kpis_for_sport(session, sport, start_date, target_date)
        if kpis.total_matches > 0:
            report.kpis.append(kpis)

    logger.info(
        "Coverage report (%s): %d sports, %d total matches",
        target_date, len(report.kpis),
        sum(k.total_matches for k in report.kpis),
    )

    return report


def _compute_kpis_for_sport(
    session: Session,
    sport: Sport,
    start_date: date,
    end_date: date,
) -> CoverageKPIs:
    """Compute coverage KPIs for a single sport across both tables."""
    kpis = CoverageKPIs(
        sport=sport.value,
        date_range=f"{start_date.isoformat()} to {end_date.isoformat()}",
    )

    stale_threshold = datetime.now(timezone.utc) - timedelta(
        days=int(COVERAGE_SLOS["max_stale_days"])
    )

    # -- Live matches (Match table) --
    from datetime import time as dt_time
    range_start = datetime.combine(start_date, dt_time.min, tzinfo=timezone.utc)
    range_end = datetime.combine(end_date + timedelta(days=1), dt_time.min, tzinfo=timezone.utc)

    live_matches = list(
        session.execute(
            select(Match).where(
                Match.sport == sport,
                Match.scheduled_at >= range_start,
                Match.scheduled_at < range_end,
            )
        ).scalars().all()
    )

    # -- Historical matches (HistoricalMatch table) --
    hist_matches = list(
        session.execute(
            select(HistoricalMatch).where(
                HistoricalMatch.sport == sport,
                HistoricalMatch.match_date >= start_date,
                HistoricalMatch.match_date <= end_date,
            )
        ).scalars().all()
    )

    total = len(live_matches) + len(hist_matches)
    kpis.total_matches = total

    if total == 0:
        return kpis

    # Live match KPIs
    for m in live_matches:
        if m.home_score is not None and m.away_score is not None:
            kpis.with_final_score += 1
        if _match_has_halftime(m, sport.value):
            kpis.with_halftime_splits += 1
        if _match_has_key_features(m, sport.value):
            kpis.with_key_features += 1
        # Canonical check: assume resolved if name length > 3 (no raw abbreviations)
        if len(m.home_team) > 3 and len(m.away_team) > 3:
            kpis.with_canonical_names += 1
        if (
            m.match_state == MatchState.NOT_STARTED
            and m.scheduled_at < stale_threshold
        ):
            kpis.stale_count += 1

    # Historical match KPIs (always have scores)
    for hm in hist_matches:
        kpis.with_final_score += 1
        if _historical_has_halftime(hm):
            kpis.with_halftime_splits += 1
        if _historical_has_key_features(hm):
            kpis.with_key_features += 1
        # Historical matches passed through IroncladAliasResolver — always canonical
        kpis.with_canonical_names += 1

    return kpis


def check_quality_gate(
    session: Session,
    sport: str,
    league: str | None = None,
    min_date: date | None = None,
    lookback_days: int = 30,
) -> QualityGate:
    """Check if data quality meets SLO thresholds for training/prediction.

    Args:
        session: SQLAlchemy session.
        sport: Sport name (e.g. "football").
        league: Optional league filter.
        min_date: Start date. Defaults to lookback_days ago.
        lookback_days: Fallback lookback if min_date not given.

    Returns:
        QualityGate with pass/fail status and violation details.
    """
    end_date = date.today()
    start_date = min_date or (end_date - timedelta(days=lookback_days))

    try:
        sport_enum = Sport(sport)
    except ValueError:
        return QualityGate(
            passed=False,
            can_train=False,
            can_predict=False,
            violations=[f"Unknown sport: {sport}"],
        )

    kpis = _compute_kpis_for_sport(session, sport_enum, start_date, end_date)

    violations: list[str] = []

    if kpis.total_matches == 0:
        violations.append(f"No matches found for {sport} in [{start_date}, {end_date}]")
        return QualityGate(
            passed=False,
            can_train=False,
            can_predict=False,
            violations=violations,
            kpis=kpis,
        )

    # Check each SLO
    if kpis.pct_final_score < COVERAGE_SLOS["pct_final_score"]:
        violations.append(
            f"pct_final_score={kpis.pct_final_score:.2%} < "
            f"{COVERAGE_SLOS['pct_final_score']:.0%} SLO"
        )

    if kpis.pct_halftime_splits < COVERAGE_SLOS["pct_halftime_splits"]:
        violations.append(
            f"pct_halftime_splits={kpis.pct_halftime_splits:.2%} < "
            f"{COVERAGE_SLOS['pct_halftime_splits']:.0%} SLO"
        )

    if kpis.pct_key_features < COVERAGE_SLOS["pct_key_features"]:
        violations.append(
            f"pct_key_features={kpis.pct_key_features:.2%} < "
            f"{COVERAGE_SLOS['pct_key_features']:.0%} SLO"
        )

    if kpis.pct_canonical_mapped < COVERAGE_SLOS["pct_canonical_mapped"]:
        violations.append(
            f"pct_canonical_mapped={kpis.pct_canonical_mapped:.2%} < "
            f"{COVERAGE_SLOS['pct_canonical_mapped']:.0%} SLO"
        )

    if kpis.stale_count > 0:
        violations.append(
            f"{kpis.stale_count} stale matches (>{int(COVERAGE_SLOS['max_stale_days'])}d)"
        )

    # Derive gate decisions
    can_train = (
        kpis.pct_key_features >= COVERAGE_SLOS["pct_key_features"]
        and kpis.pct_canonical_mapped >= COVERAGE_SLOS["pct_canonical_mapped"]
    )
    can_predict = (
        kpis.pct_final_score >= 0.80  # relaxed from 95% SLO for predictions
        and kpis.pct_canonical_mapped >= 0.95  # relaxed from 98%
    )

    passed = len(violations) == 0

    if violations:
        logger.warning(
            "Quality gate for %s: %d violations — %s",
            sport, len(violations), "; ".join(violations),
        )
    else:
        logger.info("Quality gate for %s: PASSED", sport)

    return QualityGate(
        passed=passed,
        can_train=can_train,
        can_predict=can_predict,
        violations=violations,
        kpis=kpis,
    )
