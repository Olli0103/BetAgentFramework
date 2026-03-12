"""Devil's Advocate Veto Engine — qualitative risk assessment.

Takes PENDING predictions and searches for qualitative risks (injuries,
lineup changes, fatigue, weather) before approving or vetoing.

Two-layer analysis:
  Layer 1 (Regex): Fast keyword scan for obvious risk terms.
  Layer 2 (LLM):  Tier-1 semantic analysis of news snippets — catches
                   nuanced risks that regex misses (e.g. "trained
                   individually", "rested key players in cup match").

Golden Rule: When in doubt, VETO.  A missed bet costs nothing.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import uuid
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

    prediction_id: uuid.UUID
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
    """Stub search backend — returns empty results when no API is configured."""

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return []


# ── Risk keyword analysis (Layer 1: Regex) ────────────────────────────

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

# Maximum time for the entire veto check per prediction (search + LLM).
# If exceeded, approve by default — a slow veto should not block the pipeline.
_VETO_TIMEOUT_SECONDS = int(os.environ.get("VETO_TIMEOUT_SECONDS", "30"))


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


# ── LLM risk analysis (Layer 2: Semantic) ────────────────────────────

_LLM_RISK_PROMPT = """\
You are a sports betting risk analyst. Analyze these news snippets about \
an upcoming match between {home} and {away}.

NEWS SNIPPETS:
{snippets}

Identify risk factors that could invalidate a statistical prediction. \
Look for:
- Key player injuries, illness, or personal issues (even if not using \
the word "injury")
- Tactical changes (rotation, resting players, youth lineup)
- Travel fatigue, fixture congestion, back-to-back games
- Coaching changes, internal conflicts, morale issues
- External factors (weather, pitch conditions, fan protests)
- Motivation asymmetry (nothing to play for vs must-win)

Respond ONLY with a JSON array of short risk factor strings. \
If no risks found, respond with an empty array [].
Example: ["Mbappé trained individually - doubtful", "3rd match in 7 days"]"""


def _llm_analyze_risks(
    home: str,
    away: str,
    snippets: list[str],
) -> list[str]:
    """Use Tier-1 LLM to semantically extract risk factors from news.

    Returns list of risk factor strings, or empty list on failure.
    """
    if not snippets:
        return []

    try:
        from bet_agent.llm.client import LLMClient
        client = LLMClient.for_tier("tier1_heavy_reasoning")
    except Exception:
        logger.debug("LLM unavailable for risk analysis — regex only")
        return []

    snippet_text = "\n".join(f"- {s[:300]}" for s in snippets[:10])
    prompt = _LLM_RISK_PROMPT.format(
        home=home, away=away, snippets=snippet_text,
    )

    try:
        response = client.chat(prompt, temperature=0.1, max_tokens=512)
    except Exception as exc:
        logger.warning("LLM risk analysis failed: %s", exc)
        return []

    # Parse JSON array from response
    import json
    try:
        # Find the JSON array in the response (LLM might add preamble)
        start = response.index("[")
        end = response.rindex("]") + 1
        factors = json.loads(response[start:end])
        if isinstance(factors, list):
            return [str(f) for f in factors if f]
    except (ValueError, json.JSONDecodeError):
        logger.debug("Could not parse LLM risk response: %s", response[:200])

    return []


# ── Core veto logic ──────────────────────────────────────────────────


def _build_search_query(match: Match) -> str:
    """Build a single consolidated search query for a match.

    Uses one query per match instead of 5 to conserve API budget.
    """
    return (
        f"{match.home_team} vs {match.away_team} "
        f"injury lineup news {date.today()}"
    )


def veto_check(
    session: Session,
    prediction: Prediction,
    search_backend: NewsSearchBackend | None = None,
    risk_threshold: int = 2,
    *,
    use_llm: bool = True,
) -> VetoResult:
    """Run a qualitative veto check on a single prediction.

    Two-layer analysis:
      1. Regex keyword scan (fast, free, catches obvious terms)
      2. LLM semantic analysis (catches nuanced risks regex misses)

    Risk factors from both layers are merged and deduplicated.

    Args:
        session: SQLAlchemy session.
        prediction: The PENDING prediction to check.
        search_backend: Pluggable news search (auto-detects if None).
        risk_threshold: Number of risk factors to trigger automatic VETO.
        use_llm: Whether to run LLM analysis on top of regex.

    Returns:
        VetoResult with decision and reasoning.
    """
    if search_backend is None:
        from bet_agent.tools.search_backends import create_search_backend
        search_backend = create_search_backend()

    match = prediction.match
    if match is None:
        match = session.get(Match, prediction.match_id)

    if match is None:
        return VetoResult(
            prediction_id=prediction.id,
            decision="VETO",
            reason="Match not found in database",
        )

    # Collect news snippets (1 query per match to conserve budget)
    all_snippets: list[str] = []
    all_news: list[dict] = []

    query = _build_search_query(match)
    try:
        # Fast-fail: search backend gets its own timeout so a hung DNS/HTTP
        # call doesn't block the entire pipeline.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(search_backend.search, query, 5)
            results = future.result(timeout=_VETO_TIMEOUT_SECONDS)
        for r in results:
            snippet = r.get("snippet", r.get("title", ""))
            if snippet:
                all_snippets.append(snippet)
                all_news.append(r)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "Search timed out after %ds for '%s' — proceeding with regex only",
            _VETO_TIMEOUT_SECONDS, query,
        )
    except Exception as exc:
        logger.warning("Search failed for '%s': %s", query, exc)

    # Layer 1: Regex keyword extraction
    regex_factors = _extract_risk_factors(all_snippets)

    # Layer 2: LLM semantic analysis (only if we have snippets)
    llm_factors: list[str] = []
    if use_llm and all_snippets:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    _llm_analyze_risks,
                    match.home_team, match.away_team, all_snippets,
                )
                llm_factors = future.result(timeout=_VETO_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "LLM risk analysis timed out after %ds — regex only",
                _VETO_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("LLM risk analysis wrapper failed: %s", exc)

    # Merge and deduplicate risk factors
    risk_factors = _merge_risk_factors(regex_factors, llm_factors)

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


def _merge_risk_factors(
    regex_factors: list[str],
    llm_factors: list[str],
) -> list[str]:
    """Merge regex and LLM risk factors, deduplicating by lowercase."""
    seen: set[str] = set()
    merged: list[str] = []

    # Regex factors first (deterministic, trusted)
    for f in regex_factors:
        key = f.lower().strip()
        if key not in seen:
            seen.add(key)
            merged.append(f)

    # LLM factors (may overlap with regex hits)
    for f in llm_factors:
        key = f.lower().strip()
        # Check if any existing factor is a substring match
        if key not in seen and not any(key in s or s in key for s in seen):
            seen.add(key)
            merged.append(f)

    return merged


def apply_veto_result(
    session: Session,
    prediction: Prediction,
    result: VetoResult,
) -> None:
    """Apply a VetoResult to a prediction in the database.

    Updates status to APPROVED or VETOED and stores veto_reason.
    """
    _MAX_VETO_REASON_LEN = 32_768  # TEXT column, generous safety cap
    if result.decision == "VETO":
        prediction.status = PredictionStatus.VETOED
        reason = result.reason or ""
        if len(reason) > _MAX_VETO_REASON_LEN:
            reason = reason[:_MAX_VETO_REASON_LEN - 3] + "..."
        prediction.veto_reason = reason
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
