"""Tests for the Devil's Advocate veto engine."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from bet_agent.tools.veto_engine import (
    VetoResult,
    _extract_risk_factors,
    _merge_risk_factors,
    veto_check,
)


# ── Risk factor extraction (regex layer) ─────────────────────────────


def test_extract_no_risks():
    snippets = ["Great weather for the match today", "Both teams in good form"]
    assert _extract_risk_factors(snippets) == []


def test_extract_injury_risk():
    snippets = ["Star striker injured in training"]
    factors = _extract_risk_factors(snippets)
    assert any("injur" in f.lower() for f in factors)


def test_extract_suspension_risk():
    snippets = ["Midfielder suspended for two games after red card"]
    factors = _extract_risk_factors(snippets)
    assert any("suspend" in f.lower() for f in factors)


def test_extract_multiple_risks():
    snippets = [
        "Key player injured, doubtful for weekend",
        "Weather warning issued for the stadium area",
    ]
    factors = _extract_risk_factors(snippets)
    assert len(factors) >= 2


def test_extract_deduplicates():
    snippets = [
        "Player injured in training",
        "Same player injured again in warm-up",
    ]
    factors = _extract_risk_factors(snippets)
    # "injured" should appear only once
    lower_factors = [f.lower() for f in factors]
    assert lower_factors.count("injured") == 1


# ── Risk factor merging (regex + LLM) ────────────────────────────────


def test_merge_no_overlap():
    regex = ["injured"]
    llm = ["3rd match in 7 days"]
    merged = _merge_risk_factors(regex, llm)
    assert len(merged) == 2
    assert merged[0] == "injured"  # regex first


def test_merge_exact_duplicate():
    regex = ["injured"]
    llm = ["injured"]
    merged = _merge_risk_factors(regex, llm)
    assert len(merged) == 1


def test_merge_substring_duplicate():
    regex = ["injured"]
    llm = ["star player injured in training"]
    merged = _merge_risk_factors(regex, llm)
    # "injured" is substring of "star player injured in training" → dedup
    assert len(merged) == 1


def test_merge_empty_inputs():
    assert _merge_risk_factors([], []) == []
    assert _merge_risk_factors(["a"], []) == ["a"]
    assert _merge_risk_factors([], ["b"]) == ["b"]


# ── Veto check logic ────────────────────────────────────────────────


def _make_prediction_and_match():
    """Create mock prediction and match objects."""
    match = MagicMock()
    match.home_team = "Bayern München"
    match.away_team = "Borussia Dortmund"

    pred = MagicMock()
    pred.id = uuid.uuid4()
    pred.match = match
    pred.match_id = uuid.uuid4()

    return pred, match


def test_veto_check_no_match():
    pred = MagicMock()
    pred.id = uuid.uuid4()
    pred.match = None
    pred.match_id = uuid.uuid4()

    session = MagicMock()
    session.get.return_value = None

    result = veto_check(session, pred, search_backend=MagicMock(search=lambda *a, **kw: []))
    assert result.decision == "VETO"
    assert "not found" in result.reason.lower()


def test_veto_check_no_risks_approves():
    pred, match = _make_prediction_and_match()
    stub = MagicMock()
    stub.search.return_value = [
        {"title": "Match preview", "snippet": "Both teams at full strength"},
    ]

    result = veto_check(
        MagicMock(), pred, search_backend=stub, use_llm=False,
    )
    assert result.decision == "APPROVE"
    assert result.prediction_id == pred.id


def test_veto_check_vetoes_on_multiple_risks():
    pred, match = _make_prediction_and_match()
    stub = MagicMock()
    stub.search.return_value = [
        {"snippet": "Star striker injured, suspended defender, doubtful goalkeeper"},
    ]

    result = veto_check(
        MagicMock(), pred, search_backend=stub,
        risk_threshold=2, use_llm=False,
    )
    assert result.decision == "VETO"
    assert len(result.risk_factors) >= 2


def test_veto_check_approves_below_threshold():
    pred, match = _make_prediction_and_match()
    stub = MagicMock()
    stub.search.return_value = [
        {"snippet": "Midfielder has minor fatigue but expected to play"},
    ]

    result = veto_check(
        MagicMock(), pred, search_backend=stub,
        risk_threshold=3, use_llm=False,
    )
    assert result.decision == "APPROVE"
    assert "minor" in result.reason.lower() or "below threshold" in result.reason.lower()


def test_veto_check_search_failure_still_approves():
    """If search fails, no risk factors found → APPROVE."""
    pred, match = _make_prediction_and_match()
    stub = MagicMock()
    stub.search.side_effect = Exception("API down")

    result = veto_check(MagicMock(), pred, search_backend=stub, use_llm=False)
    assert result.decision == "APPROVE"
    assert len(result.risk_factors) == 0


def test_veto_check_with_llm_layer():
    """LLM layer adds risk factors beyond regex."""
    pred, match = _make_prediction_and_match()
    stub = MagicMock()
    stub.search.return_value = [
        {"snippet": "The team trained individually today, unusual for match day"},
    ]

    with patch("bet_agent.tools.veto_engine._llm_analyze_risks") as mock_llm:
        mock_llm.return_value = ["trained individually - doubtful", "unusual pre-match routine"]
        result = veto_check(
            MagicMock(), pred, search_backend=stub,
            risk_threshold=2, use_llm=True,
        )

    assert result.decision == "VETO"
    assert len(result.risk_factors) >= 2
