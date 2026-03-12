"""Tests for search backends (Tavily, Brave, Cascading fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bet_agent.tools.search_backends import (
    BraveSearch,
    CascadingSearch,
    DefaultNewsSearch,
    TavilySearch,
    _budget_remaining,
    _get_monthly_usage,
    _increment_usage,
    budget_remaining,
    create_search_backend,
    get_monthly_usage,
)


# ── DefaultNewsSearch ────────────────────────────────────────────────


def test_default_search_returns_empty():
    backend = DefaultNewsSearch()
    assert backend.search("any query") == []


# ── Budget tracking ──────────────────────────────────────────────────


def test_budget_tracking_increment(tmp_path: Path):
    usage_file = tmp_path / "test_usage.json"

    assert _get_monthly_usage(usage_file) == 0
    assert _budget_remaining(usage_file, 100) == 100

    count = _increment_usage(usage_file)
    assert count == 1
    assert _get_monthly_usage(usage_file) == 1
    assert _budget_remaining(usage_file, 100) == 99


def test_budget_exhaustion(tmp_path: Path):
    usage_file = tmp_path / "test_usage.json"
    # Simulate exhausted budget
    from datetime import date
    month_key = date.today().strftime("%Y-%m")
    usage_file.write_text(json.dumps({month_key: 1000}))

    assert _budget_remaining(usage_file, 1000) == 0


def test_budget_corrupted_file(tmp_path: Path):
    usage_file = tmp_path / "bad_usage.json"
    usage_file.write_text("not valid json{{{")

    assert _get_monthly_usage(usage_file) == 0


# ── TavilySearch ─────────────────────────────────────────────────────


def test_tavily_no_api_key():
    backend = TavilySearch(api_key="")
    assert not backend.available
    assert backend.search("test") == []


def test_tavily_backward_compat_budget_remaining(tmp_path: Path):
    """Convenience wrappers delegate to generic functions correctly."""
    usage_file = tmp_path / "t.json"
    with patch("bet_agent.tools.search_backends._TAVILY_USAGE_FILE", usage_file):
        assert get_monthly_usage() == 0
        assert budget_remaining() == 1000


def test_tavily_available_with_key_and_budget(tmp_path: Path):
    with patch("bet_agent.tools.search_backends._TAVILY_USAGE_FILE", tmp_path / "t.json"):
        backend = TavilySearch(api_key="test-key")
        assert backend.available


def test_tavily_search_success(tmp_path: Path):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"title": "Player injured", "content": "Star striker out for 3 weeks", "url": "https://example.com/1"},
            {"title": "Team news", "content": "Lineup announced", "url": "https://example.com/2"},
        ]
    }

    with patch("bet_agent.tools.search_backends._TAVILY_USAGE_FILE", tmp_path / "t.json"), \
         patch("requests.post", return_value=mock_resp):
        backend = TavilySearch(api_key="test-key")
        results = backend.search("test query")

    assert len(results) == 2
    assert results[0]["title"] == "Player injured"
    assert results[0]["snippet"] == "Star striker out for 3 weeks"


def test_tavily_search_api_failure(tmp_path: Path):
    with patch("bet_agent.tools.search_backends._TAVILY_USAGE_FILE", tmp_path / "t.json"), \
         patch("requests.post", side_effect=Exception("timeout")):
        backend = TavilySearch(api_key="test-key")
        results = backend.search("test query")

    assert results == []


# ── BraveSearch ──────────────────────────────────────────────────────


def test_brave_no_api_key():
    backend = BraveSearch(api_key="")
    assert not backend.available
    assert backend.search("test") == []


def test_brave_available_with_key_and_budget(tmp_path: Path):
    with patch("bet_agent.tools.search_backends._BRAVE_USAGE_FILE", tmp_path / "b.json"):
        backend = BraveSearch(api_key="test-key")
        assert backend.available


def test_brave_search_success(tmp_path: Path):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "web": {
            "results": [
                {"title": "Match preview", "description": "Key player returns", "url": "https://example.com/3"},
            ]
        }
    }

    with patch("bet_agent.tools.search_backends._BRAVE_USAGE_FILE", tmp_path / "b.json"), \
         patch("requests.get", return_value=mock_resp):
        backend = BraveSearch(api_key="test-key")
        results = backend.search("test query")

    assert len(results) == 1
    assert results[0]["title"] == "Match preview"
    assert results[0]["snippet"] == "Key player returns"


def test_brave_search_api_failure(tmp_path: Path):
    with patch("bet_agent.tools.search_backends._BRAVE_USAGE_FILE", tmp_path / "b.json"), \
         patch("requests.get", side_effect=Exception("connection error")):
        backend = BraveSearch(api_key="test-key")
        results = backend.search("test query")

    assert results == []


def test_brave_budget_exhausted(tmp_path: Path):
    usage_file = tmp_path / "b.json"
    from datetime import date
    month_key = date.today().strftime("%Y-%m")
    usage_file.write_text(json.dumps({month_key: 2000}))

    with patch("bet_agent.tools.search_backends._BRAVE_USAGE_FILE", usage_file):
        backend = BraveSearch(api_key="test-key")
        assert not backend.available
        assert backend.search("test") == []


# ── CascadingSearch ──────────────────────────────────────────────────


def test_cascading_uses_first_available():
    """First backend with results wins."""
    b1 = MagicMock()
    b1.available = True
    b1.search.return_value = [{"title": "from b1", "snippet": "s1", "url": "u1"}]

    b2 = MagicMock()
    b2.available = True

    cascade = CascadingSearch([b1, b2])
    results = cascade.search("test")

    assert len(results) == 1
    assert results[0]["title"] == "from b1"
    b2.search.assert_not_called()


def test_cascading_skips_unavailable():
    """Skips backends where available=False."""
    b1 = MagicMock()
    b1.available = False

    b2 = MagicMock()
    b2.available = True
    b2.search.return_value = [{"title": "from b2", "snippet": "s2", "url": "u2"}]

    cascade = CascadingSearch([b1, b2])
    results = cascade.search("test")

    assert results[0]["title"] == "from b2"
    b1.search.assert_not_called()


def test_cascading_falls_through_on_empty():
    """Falls to next backend when current returns empty results."""
    b1 = MagicMock()
    b1.available = True
    b1.search.return_value = []

    b2 = MagicMock()
    b2.available = True
    b2.search.return_value = [{"title": "from b2", "snippet": "s2", "url": "u2"}]

    cascade = CascadingSearch([b1, b2])
    results = cascade.search("test")

    assert results[0]["title"] == "from b2"


def test_cascading_all_empty():
    """Returns empty when all backends return nothing."""
    cascade = CascadingSearch([DefaultNewsSearch()])
    assert cascade.search("test") == []


def test_cascading_stub_has_no_available_attr():
    """DefaultNewsSearch has no .available — works with CascadingSearch."""
    stub = DefaultNewsSearch()
    assert not hasattr(stub, "available")

    cascade = CascadingSearch([stub])
    assert cascade.search("test") == []


# ── Factory ──────────────────────────────────────────────────────────


def test_factory_no_keys():
    with patch.dict("os.environ", {}, clear=True):
        backend = create_search_backend()
        assert isinstance(backend, CascadingSearch)
        assert len(backend._backends) == 1  # only DefaultNewsSearch
        assert isinstance(backend._backends[0], DefaultNewsSearch)


def test_factory_tavily_only():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "tk"}, clear=True):
        backend = create_search_backend()
        assert isinstance(backend, CascadingSearch)
        assert len(backend._backends) == 2
        assert isinstance(backend._backends[0], TavilySearch)
        assert isinstance(backend._backends[1], DefaultNewsSearch)


def test_factory_brave_only():
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "bk"}, clear=True):
        backend = create_search_backend()
        assert isinstance(backend, CascadingSearch)
        assert len(backend._backends) == 2
        assert isinstance(backend._backends[0], BraveSearch)
        assert isinstance(backend._backends[1], DefaultNewsSearch)


def test_factory_both_keys():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "tk", "BRAVE_SEARCH_API_KEY": "bk"}, clear=True):
        backend = create_search_backend()
        assert isinstance(backend, CascadingSearch)
        assert len(backend._backends) == 3
        assert isinstance(backend._backends[0], TavilySearch)
        assert isinstance(backend._backends[1], BraveSearch)
        assert isinstance(backend._backends[2], DefaultNewsSearch)
