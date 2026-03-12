"""Tests for search backends (Reddit, Tavily, Brave, Cascading fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bet_agent.tools.search_backends import (
    BraveSearch,
    CascadingSearch,
    DefaultNewsSearch,
    RedditRSSSearch,
    TavilySearch,
    _budget_remaining,
    _get_monthly_usage,
    _get_reddit_oauth_token,
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


# ── RedditRSSSearch ──────────────────────────────────────────────────


def test_reddit_always_available():
    backend = RedditRSSSearch()
    assert backend.available is True


def test_reddit_has_oauth_false():
    with patch.dict("os.environ", {}, clear=True):
        backend = RedditRSSSearch()
        assert backend.has_oauth is False


def test_reddit_has_oauth_true():
    with patch.dict("os.environ", {"REDDIT_CLIENT_ID": "id", "REDDIT_CLIENT_SECRET": "secret"}):
        backend = RedditRSSSearch()
        assert backend.has_oauth is True


def test_reddit_oauth_token_no_creds():
    """Without credentials, _get_reddit_oauth_token returns None."""
    import bet_agent.tools.search_backends as mod
    mod._reddit_token = None
    mod._reddit_token_expires = 0.0
    with patch.dict("os.environ", {}, clear=True):
        assert _get_reddit_oauth_token() is None


def test_reddit_oauth_token_success():
    """With credentials, token is fetched and cached."""
    import bet_agent.tools.search_backends as mod
    mod._reddit_token = None
    mod._reddit_token_expires = 0.0

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"access_token": "abc123", "expires_in": 3600}

    with patch.dict("os.environ", {"REDDIT_CLIENT_ID": "id", "REDDIT_CLIENT_SECRET": "secret"}), \
         patch("requests.post", return_value=mock_resp):
        token = _get_reddit_oauth_token()

    assert token == "abc123"
    assert mod._reddit_token == "abc123"


def test_reddit_oauth_search_uses_oauth_url():
    """When OAuth token is available, search uses oauth.reddit.com."""
    import bet_agent.tools.search_backends as mod
    mod._reddit_token = "test_token"
    mod._reddit_token_expires = float("inf")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": {"children": []}}

    with patch.dict("os.environ", {"REDDIT_CLIENT_ID": "id", "REDDIT_CLIENT_SECRET": "secret"}), \
         patch("requests.get", return_value=mock_resp) as mock_get:
        backend = RedditRSSSearch(subreddits=["nfl"])
        backend.search("test")

    call_url = mock_get.call_args[0][0]
    assert "oauth.reddit.com" in call_url
    auth_header = mock_get.call_args[1]["headers"]["Authorization"]
    assert auth_header == "Bearer test_token"


def test_reddit_403_handled():
    """403 response is handled gracefully (endpoint blocked)."""
    import bet_agent.tools.search_backends as mod
    mod._reddit_token = None
    mod._reddit_token_expires = 0.0

    mock_resp = MagicMock()
    mock_resp.status_code = 403

    with patch.dict("os.environ", {}, clear=True), \
         patch("requests.get", return_value=mock_resp):
        backend = RedditRSSSearch(subreddits=["nfl"])
        results = backend.search("test")

    assert results == []


def test_reddit_pick_subreddits_nfl():
    backend = RedditRSSSearch()
    subs = backend._pick_subreddits("NFL Chiefs injury report")
    assert "nfl" in subs
    assert "fantasyfootball" in subs


def test_reddit_pick_subreddits_bundesliga():
    backend = RedditRSSSearch()
    subs = backend._pick_subreddits("Bundesliga Bayern München lineup")
    assert "Bundesliga" in subs


def test_reddit_pick_subreddits_nba():
    backend = RedditRSSSearch()
    subs = backend._pick_subreddits("NBA Lakers injury")
    assert "nba" in subs


def test_reddit_pick_subreddits_fallback():
    """Generic query with no sport keywords falls back to default subs."""
    backend = RedditRSSSearch()
    subs = backend._pick_subreddits("some random query")
    assert len(subs) > 0  # Should return some defaults


def test_reddit_subreddits_for_sport():
    assert "nfl" in RedditRSSSearch.subreddits_for_sport("american_football")
    assert "soccer" in RedditRSSSearch.subreddits_for_sport("football")
    assert "nba" in RedditRSSSearch.subreddits_for_sport("basketball")
    assert "hockey" in RedditRSSSearch.subreddits_for_sport("ice_hockey")
    assert "tennis" in RedditRSSSearch.subreddits_for_sport("tennis")


def test_reddit_search_success():
    """Mocked Reddit JSON response is parsed correctly."""
    import bet_agent.tools.search_backends as mod
    mod._reddit_token = None
    mod._reddit_token_expires = 0.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "children": [
                {
                    "data": {
                        "title": "Mahomes questionable for Sunday",
                        "selftext": "Per Schefter, Mahomes has ankle issues...",
                        "permalink": "/r/nfl/comments/abc123/mahomes/",
                    }
                },
            ]
        }
    }

    with patch.dict("os.environ", {}, clear=True), \
         patch("requests.get", return_value=mock_resp):
        backend = RedditRSSSearch(subreddits=["nfl"])
        results = backend.search("Mahomes injury")

    assert len(results) == 1
    assert results[0]["title"] == "Mahomes questionable for Sunday"
    assert "Schefter" in results[0]["snippet"]
    assert results[0]["url"] == "https://www.reddit.com/r/nfl/comments/abc123/mahomes/"
    assert results[0]["source"] == "reddit/r/nfl"


def test_reddit_search_rate_limited():
    """429 response is handled gracefully."""
    import bet_agent.tools.search_backends as mod
    mod._reddit_token = None
    mod._reddit_token_expires = 0.0

    mock_resp = MagicMock()
    mock_resp.status_code = 429

    with patch.dict("os.environ", {}, clear=True), \
         patch("requests.get", return_value=mock_resp):
        backend = RedditRSSSearch(subreddits=["nfl"])
        results = backend.search("test")

    assert results == []


def test_reddit_search_network_error():
    """Network errors are handled gracefully."""
    import bet_agent.tools.search_backends as mod
    mod._reddit_token = None
    mod._reddit_token_expires = 0.0

    with patch.dict("os.environ", {}, clear=True), \
         patch("requests.get", side_effect=Exception("connection refused")):
        backend = RedditRSSSearch(subreddits=["nfl"])
        results = backend.search("test")

    assert results == []


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
    """Reddit is always first, stub is always last."""
    with patch.dict("os.environ", {}, clear=True):
        backend = create_search_backend()
        assert isinstance(backend, CascadingSearch)
        assert len(backend._backends) == 2  # Reddit + DefaultNewsSearch
        assert isinstance(backend._backends[0], RedditRSSSearch)
        assert isinstance(backend._backends[1], DefaultNewsSearch)


def test_factory_tavily_only():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "tk"}, clear=True):
        backend = create_search_backend()
        assert isinstance(backend, CascadingSearch)
        assert len(backend._backends) == 3
        assert isinstance(backend._backends[0], RedditRSSSearch)
        assert isinstance(backend._backends[1], TavilySearch)
        assert isinstance(backend._backends[2], DefaultNewsSearch)


def test_factory_brave_only():
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "bk"}, clear=True):
        backend = create_search_backend()
        assert isinstance(backend, CascadingSearch)
        assert len(backend._backends) == 3
        assert isinstance(backend._backends[0], RedditRSSSearch)
        assert isinstance(backend._backends[1], BraveSearch)
        assert isinstance(backend._backends[2], DefaultNewsSearch)


def test_factory_both_keys():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "tk", "BRAVE_SEARCH_API_KEY": "bk"}, clear=True):
        backend = create_search_backend()
        assert isinstance(backend, CascadingSearch)
        assert len(backend._backends) == 4
        assert isinstance(backend._backends[0], RedditRSSSearch)
        assert isinstance(backend._backends[1], TavilySearch)
        assert isinstance(backend._backends[2], BraveSearch)
        assert isinstance(backend._backends[3], DefaultNewsSearch)
