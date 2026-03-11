"""Devil's Advocate Veto Engine — qualitative risk assessment.

Takes PENDING predictions and searches for qualitative risks (injuries,
lineup changes, fatigue, weather) before approving or vetoing.

Uses Tier 1 LLM (via configurable search backend) to analyze news results
and make APPROVE/VETO decisions.

Golden Rule: When in doubt, VETO.  A missed bet costs nothing.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    Match,
    Prediction,
    PredictionStatus,
)

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class VetoResult:
    """Result of a qualitative veto check."""

    prediction_id: object  # UUID
    decision: str  # "APPROVE" or "VETO"
    reason: str
    risk_factors: list[str] = field(default_factory=list)
    news_snippets: list[str] = field(default_factory=list)


# ── Search backend protocol ──────────────────────────────────────────


class NewsSearchBackend(Protocol):
    """Protocol for pluggable news search backends (Tavily, Cloudflare, Google)."""

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return list of dicts with 'title', 'snippet', 'url' keys."""
        ...


class DefaultNewsSearch:
    """Stub search backend — returns empty results when no API is configured.

    In production, replace with TavilySearch, CloudflareSearch, or GoogleSearch.
    """

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return []


# ── Risk keyword analysis ────────────────────────────────────────────

_RISK_KEYWORDS = [
    r"\binjur(?:y|ed|ies)\b",
    r"\bout\b.*\b(?:game|match|lineup)\b",
    r"\bsuspend(?:ed|sion)\b",
    r"\bfatigu(?:e|ed)\b",
    r"\brotation|rest(?:ed|ing)\b",
    r"\bmanager(?:ial)?\s+change\b",
    r"\bsack(?:ed|ing)\b",
    r"\bweather\s+(?:warning|delay|alert)\b",
    r"\bpostpon(?:ed|ement)\b",
    r"\bwithdra(?:wn|wal)\b",
    r"\bread\s+card\b",
    r"\bhamstring|knee|ankle|concussion|ACL\b",
    r"\bdoubtful|questionable|day-to-day\b",
    r"\blineup\s+change\b",
]

_RISK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _RISK_KEYWORDS]


def _extract_risk_factors(snippets: list[str]) -> list[str]:
    """Scan news snippets for risk-related keywords."""
    factors: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        for pattern in _RISK_PATTERNS:
            match = pattern.search(snippet)
            if match and match.group(0).lower() not in seen:
                factors.append(match.group(0))
                seen.add(match.group(0).lower())
    return factors


# ── Core veto logic ──────────────────────────────────────────────────


def _build_search_queries(match: Match) -> list[str]:
    """Build targeted search queries for a match's qualitative risks."""
    teams = [match.home_team, match.away_team]
    queries = []
    for team in teams:
        queries.append(f"{team} injury news today")
        queries.append(f"{team} lineup changes {date.today()}")
    # General match query
    queries.append(f"{match.home_team} vs {match.away_team} preview news")
    return queries


def veto_check(
    session: Session,
    prediction: Prediction,
    search_backend: NewsSearchBackend | None = None,
    risk_threshold: int = 2,
) -> VetoResult:
    """Run a qualitative veto check on a single prediction.

    Args:
        session: SQLAlchemy session.
        prediction: The PENDING prediction to check.
        search_backend: Pluggable news search (defaults to stub).
        risk_threshold: Number of risk factors to trigger automatic VETO.

    Returns:
        VetoResult with decision and reasoning.
    """
    if search_backend is None:
        search_backend = DefaultNewsSearch()

    match = prediction.match
    if match is None:
        match = session.get(Match, prediction.match_id)

    if match is None:
        return VetoResult(
            prediction_id=prediction.id,
            decision="VETO",
            reason="Match not found in database",
        )

    # Collect news snippets
    all_snippets: list[str] = []
    all_news: list[dict] = []

    queries = _build_search_queries(match)
    for query in queries:
        try:
            results = search_backend.search(query, max_results=3)
            for r in results:
                snippet = r.get("snippet", r.get("title", ""))
                if snippet:
                    all_snippets.append(snippet)
                    all_news.append(r)
        except Exception as exc:
            logger.warning("Search failed for query '%s': %s", query, exc)

    # Extract risk factors from snippets
    risk_factors = _extract_risk_factors(all_snippets)

    # Decision logic
    if len(risk_factors) >= risk_threshold:
        decision = "VETO"
        reason = (
            f"Found {len(risk_factors)} risk factor(s) for "
            f"{match.home_team} vs {match.away_team}: "
            + ", ".join(risk_factors[:5])
        )
    elif len(risk_factors) > 0:
        decision = "APPROVE"
        reason = (
            f"Minor risk factors found ({len(risk_factors)}), "
            f"but below threshold: {', '.join(risk_factors)}"
        )
    else:
        decision = "APPROVE"
        reason = f"No risk factors found for {match.home_team} vs {match.away_team}"

    news_snippet_texts = [
        s.get("snippet", s.get("title", ""))[:200]
        for s in all_news[:5]
    ]

    return VetoResult(
        prediction_id=prediction.id,
        decision=decision,
        reason=reason,
        risk_factors=risk_factors,
        news_snippets=news_snippet_texts,
    )


def apply_veto_result(
    session: Session,
    prediction: Prediction,
    result: VetoResult,
) -> None:
    """Apply a VetoResult to a prediction in the database.

    Updates status to APPROVED or VETOED and stores veto_reason.
    """
    if result.decision == "VETO":
        prediction.status = PredictionStatus.VETOED
        prediction.veto_reason = result.reason
    else:
        prediction.status = PredictionStatus.APPROVED
        prediction.veto_reason = None


# ── Batch processing ─────────────────────────────────────────────────


def run_veto_checks(
    session: Session,
    predictions: list[Prediction] | None = None,
    search_backend: NewsSearchBackend | None = None,
    risk_threshold: int = 2,
) -> list[VetoResult]:
    """Run veto checks on a batch of PENDING predictions.

    If no predictions are provided, queries all PENDING predictions for today.

    Returns:
        List of VetoResult objects (one per prediction).
    """
    if predictions is None:
        today = date.today()
        day_start = datetime.combine(today, time.min, tzinfo=timezone.utc)
        day_end = datetime.combine(today, time.max, tzinfo=timezone.utc)

        predictions = list(
            session.execute(
                select(Prediction)
                .join(Match)
                .where(
                    Prediction.status == PredictionStatus.PENDING,
                    Match.scheduled_at >= day_start,
                    Match.scheduled_at <= day_end,
                )
            ).scalars().all()
        )

    results: list[VetoResult] = []
    for pred in predictions:
        result = veto_check(session, pred, search_backend, risk_threshold)
        apply_veto_result(session, pred, result)
        results.append(result)

    session.flush()
    logger.info(
        "Veto check complete: %d approved, %d vetoed out of %d",
        sum(1 for r in results if r.decision == "APPROVE"),
        sum(1 for r in results if r.decision == "VETO"),
        len(results),
    )
    return results
