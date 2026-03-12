"""Search backends for the Devil's Advocate veto engine.

Provides pluggable news search implementations with cascading fallback:
  Priority 1: RedditRSSSearch — free, no API key, uses /r/{sub}/.rss feeds
                                 with local keyword filtering
  Priority 2: TavilySearch    — best quality, 1000 req/month free tier
  Priority 3: BraveSearch     — good quality, 2000 req/month free tier
  Priority 4: DefaultNewsSearch — stub (regex-only veto path)

Reddit uses public Atom/RSS feeds (/r/{sub}/.rss) which:
  - Need NO API key, NO OAuth, NO account
  - Have generous rate limits (public RSS, no documented cap)
  - Are stable (RSS is a web standard, unlike Reddit's JSON API)
  - Return the latest ~25 posts per subreddit

Optional: Set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET for OAuth-powered
  search (60 req/min) which adds full-text search capability on top of RSS.

Budget management:
  Each paid backend tracks its own monthly usage via file-based counters.
  When a backend's budget is exhausted, the cascading search falls through
  to the next available backend automatically.

  Combined budget: ~3000 searches/month → ~100/day → plenty of headroom.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import threading
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Monthly request budgets
_TAVILY_MONTHLY_BUDGET = 1000
_BRAVE_MONTHLY_BUDGET = 2000
_USAGE_DIR = Path(os.environ.get("SEARCH_USAGE_DIR", "/tmp"))
_TAVILY_USAGE_FILE = _USAGE_DIR / "tavily_usage.json"
_BRAVE_USAGE_FILE = _USAGE_DIR / "brave_usage.json"


# ── Budget tracking ──────────────────────────────────────────────────


def _load_usage(usage_file: Path) -> dict:
    """Load {month: count} from a usage file."""
    if usage_file.exists():
        try:
            return json.loads(usage_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_usage(usage: dict, usage_file: Path) -> None:
    try:
        usage_file.parent.mkdir(parents=True, exist_ok=True)
        usage_file.write_text(json.dumps(usage))
    except OSError as exc:
        logger.warning("Could not save usage file %s: %s", usage_file, exc)


def _increment_usage(usage_file: Path) -> int:
    """Increment this month's counter and return the new total."""
    month_key = date.today().strftime("%Y-%m")
    usage = _load_usage(usage_file)
    count = usage.get(month_key, 0) + 1
    usage[month_key] = count
    _save_usage(usage, usage_file)
    return count


def _get_monthly_usage(usage_file: Path) -> int:
    """Return the current month's request count for a backend."""
    month_key = date.today().strftime("%Y-%m")
    return _load_usage(usage_file).get(month_key, 0)


def _budget_remaining(usage_file: Path, budget: int) -> int:
    """Return how many requests remain this month for a backend."""
    return max(0, budget - _get_monthly_usage(usage_file))


# ── Convenience wrappers (Tavily — backward compat) ──────────────────

def get_monthly_usage() -> int:
    """Return Tavily's current month request count."""
    return _get_monthly_usage(_TAVILY_USAGE_FILE)


def budget_remaining() -> int:
    """Return how many Tavily requests remain this month."""
    return _budget_remaining(_TAVILY_USAGE_FILE, _TAVILY_MONTHLY_BUDGET)


# ── Reddit search backend ─────────────────────────────────────────────

# Subreddits per sport — curated for injury news, analysis, and discussion
_REDDIT_SPORT_SUBS: dict[str, list[str]] = {
    "football": ["soccer", "Bundesliga", "PremierLeague", "LaLiga", "SerieA", "ChampionsLeague"],
    "american_football": ["nfl", "fantasyfootball"],
    "basketball": ["nba", "NBAdiscussion"],
    "ice_hockey": ["hockey", "nhl"],
    "tennis": ["tennis"],
}

# Flattened list of all sport subreddits for general queries
_ALL_SPORT_SUBS = sorted({sub for subs in _REDDIT_SPORT_SUBS.values() for sub in subs})

# Reddit OAuth token cache (thread-safe)
_reddit_token_lock = threading.Lock()
_reddit_token: str | None = None
_reddit_token_expires: float = 0.0


def _get_reddit_oauth_token() -> str | None:
    """Obtain a Reddit OAuth token using client credentials (app-only).

    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.
    Returns None if credentials are not configured.
    Token is cached and auto-refreshed when expired.
    """
    global _reddit_token, _reddit_token_expires  # noqa: PLW0603

    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None

    with _reddit_token_lock:
        if _reddit_token and time.monotonic() < _reddit_token_expires:
            return _reddit_token

        import requests as _requests

        try:
            resp = _requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(client_id, client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": "BetAgent/1.0 (sports research bot)"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            _reddit_token = data["access_token"]
            # Refresh 60s before expiry
            _reddit_token_expires = time.monotonic() + data.get("expires_in", 3600) - 60
            logger.info("Reddit OAuth token acquired (expires in %ds)", data.get("expires_in", 3600))
            return _reddit_token
        except Exception as exc:
            logger.warning("Reddit OAuth token request failed: %s", exc)
            return None


class RedditRSSSearch:
    """News backend using Reddit's public RSS/Atom feeds.

    Primary mode: Fetches ``/r/{sub}/.rss`` (Atom XML) and filters
    locally by query keywords. No API key, no OAuth, no account needed.

    Optional OAuth mode: If REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET are
    set, also tries ``oauth.reddit.com/r/{sub}/search.json`` for
    full-text search (60 req/min). RSS results are tried first.
    """

    # Atom namespace used in Reddit RSS feeds
    _ATOM_NS = "{http://www.w3.org/2005/Atom}"

    def __init__(self, subreddits: list[str] | None = None) -> None:
        self._subreddits = subreddits or _ALL_SPORT_SUBS

    @property
    def available(self) -> bool:
        return True  # Always available — RSS needs no credentials

    @property
    def has_oauth(self) -> bool:
        """Return True if Reddit OAuth credentials are configured."""
        return bool(
            os.environ.get("REDDIT_CLIENT_ID", "")
            and os.environ.get("REDDIT_CLIENT_SECRET", "")
        )

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search Reddit for recent posts matching the query.

        Strategy:
          1. Fetch RSS feeds and filter locally by query keywords
          2. If OAuth is configured and RSS didn't find enough, try JSON search
        """
        results = self._search_rss(query, max_results)

        # If RSS found enough, return early
        if len(results) >= max_results:
            return results[:max_results]

        # Try OAuth JSON search as supplement if configured
        if self.has_oauth:
            json_results = self._search_json_oauth(query, max_results - len(results))
            # Deduplicate by URL
            seen_urls = {r["url"] for r in results}
            for r in json_results:
                if r["url"] not in seen_urls:
                    results.append(r)
                    seen_urls.add(r["url"])
                    if len(results) >= max_results:
                        break

        return results[:max_results]

    def _search_rss(self, query: str, max_results: int) -> list[dict]:
        """Fetch RSS feeds and filter posts by query keywords."""
        import requests as _requests

        results: list[dict] = []
        subs_to_search = self._pick_subreddits(query)
        keywords = self._extract_keywords(query)

        for sub in subs_to_search[:3]:
            try:
                resp = _requests.get(
                    f"https://www.reddit.com/r/{sub}/.rss",
                    headers={"User-Agent": "BetAgent/1.0 (sports research bot)"},
                    timeout=8,
                )
                if resp.status_code in (429, 403):
                    logger.debug("Reddit RSS returned %d for r/%s — skipping", resp.status_code, sub)
                    continue
                resp.raise_for_status()
            except Exception as exc:
                logger.debug("Reddit RSS fetch failed for r/%s: %s", sub, exc)
                continue

            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError as exc:
                logger.debug("Reddit RSS parse failed for r/%s: %s", sub, exc)
                continue

            for entry in root.findall(f"{self._ATOM_NS}entry"):
                title_el = entry.find(f"{self._ATOM_NS}title")
                title = title_el.text if title_el is not None and title_el.text else ""

                content_el = entry.find(f"{self._ATOM_NS}content")
                content = content_el.text if content_el is not None and content_el.text else ""
                # Strip HTML tags for a clean snippet
                snippet = re.sub(r"<[^>]+>", "", content)[:300] if content else title

                link_el = entry.find(f"{self._ATOM_NS}link")
                url = link_el.get("href", "") if link_el is not None else ""

                # Filter: at least one keyword must appear in title or content
                text_lower = f"{title} {snippet}".lower()
                if keywords and not any(kw in text_lower for kw in keywords):
                    continue

                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "source": f"reddit/r/{sub}",
                })

                if len(results) >= max_results:
                    return results

        return results

    def _search_json_oauth(self, query: str, max_results: int) -> list[dict]:
        """Search via OAuth JSON API (requires credentials)."""
        import requests as _requests

        token = _get_reddit_oauth_token()
        if not token:
            return []

        results: list[dict] = []
        subs_to_search = self._pick_subreddits(query)

        for sub in subs_to_search[:3]:
            try:
                resp = _requests.get(
                    f"https://oauth.reddit.com/r/{sub}/search.json",
                    params={
                        "q": query,
                        "restrict_sr": "on",
                        "sort": "new",
                        "t": "week",
                        "limit": max_results,
                    },
                    headers={
                        "Authorization": f"Bearer {token}",
                        "User-Agent": "BetAgent/1.0 (sports research bot)",
                    },
                    timeout=8,
                )
                if resp.status_code in (429, 403):
                    logger.debug("Reddit OAuth search %d for r/%s — skipping", resp.status_code, sub)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.debug("Reddit OAuth search failed for r/%s: %s", sub, exc)
                continue

            for post in data.get("data", {}).get("children", []):
                pd = post.get("data", {})
                title = pd.get("title", "")
                selftext = pd.get("selftext", "")
                snippet = selftext[:300] if selftext else title
                permalink = pd.get("permalink", "")
                url = f"https://www.reddit.com{permalink}" if permalink else ""

                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "source": f"reddit/r/{sub}",
                })

                if len(results) >= max_results:
                    return results

        return results

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """Extract meaningful keywords from query (lowercase, 3+ chars)."""
        stopwords = {"the", "and", "for", "are", "but", "not", "you", "all",
                     "can", "has", "her", "was", "one", "our", "out", "with"}
        words = query.lower().split()
        return [w for w in words if len(w) >= 3 and w not in stopwords]

    def _pick_subreddits(self, query: str) -> list[str]:
        """Pick the most relevant subreddits based on query keywords."""
        q_lower = query.lower()
        picked: list[str] = []

        for sport, subs in _REDDIT_SPORT_SUBS.items():
            sport_keywords = sport.replace("_", " ").split()
            if any(kw in q_lower for kw in sport_keywords):
                picked.extend(subs)

        # Sport-specific keyword matching
        keyword_map = {
            "nfl": ["nfl", "fantasyfootball"],
            "nba": ["nba", "NBAdiscussion"],
            "nhl": ["hockey", "nhl"],
            "bundesliga": ["Bundesliga"],
            "premier league": ["PremierLeague"],
            "la liga": ["LaLiga"],
            "champions league": ["ChampionsLeague"],
            "serie a": ["SerieA"],
        }
        for keyword, subs in keyword_map.items():
            if keyword in q_lower:
                picked.extend(subs)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for s in picked:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique if unique else self._subreddits[:3]

    @staticmethod
    def subreddits_for_sport(sport: str) -> list[str]:
        """Return subreddit list for a given sport key."""
        return _REDDIT_SPORT_SUBS.get(sport, _ALL_SPORT_SUBS[:3])


# ── Stub backend ─────────────────────────────────────────────────────


class DefaultNewsSearch:
    """Stub search backend — returns empty results when no API is configured."""

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return []


# ── Tavily backend ───────────────────────────────────────────────────


class TavilySearch:
    """Production news search backend using Tavily Search API.

    Requires TAVILY_API_KEY environment variable.

    Budget-aware: refuses to search when monthly budget is exhausted,
    falling back to empty results (regex-only veto path).
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "TAVILY_API_KEY not set — TavilySearch will return empty results"
            )

    @property
    def available(self) -> bool:
        return bool(self._api_key) and _budget_remaining(
            _TAVILY_USAGE_FILE, _TAVILY_MONTHLY_BUDGET,
        ) > 0

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search for recent news using Tavily API.

        Returns list of dicts with 'title', 'snippet', 'url' keys.
        Returns empty list if API key is missing or budget exhausted.
        """
        if not self._api_key:
            return []

        remaining = _budget_remaining(_TAVILY_USAGE_FILE, _TAVILY_MONTHLY_BUDGET)
        if remaining <= 0:
            logger.warning(
                "Tavily monthly budget exhausted (%d/%d) — skipping search",
                _get_monthly_usage(_TAVILY_USAGE_FILE), _TAVILY_MONTHLY_BUDGET,
            )
            return []

        import requests

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                    "topic": "news",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Tavily search failed for '%s': %s", query, exc)
            return []

        count = _increment_usage(_TAVILY_USAGE_FILE)
        logger.debug(
            "Tavily search OK (%d/%d this month): %s",
            count, _TAVILY_MONTHLY_BUDGET, query,
        )

        results: list[dict] = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "url": item.get("url", ""),
            })

        return results


# ── Brave Search backend ─────────────────────────────────────────────


class BraveSearch:
    """News search backend using Brave Search API.

    Free tier: 2000 queries/month (no credit card required).
    Requires BRAVE_SEARCH_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "BRAVE_SEARCH_API_KEY not set — BraveSearch will return empty results"
            )

    @property
    def available(self) -> bool:
        return bool(self._api_key) and _budget_remaining(
            _BRAVE_USAGE_FILE, _BRAVE_MONTHLY_BUDGET,
        ) > 0

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search for recent news using Brave Search API.

        Returns list of dicts with 'title', 'snippet', 'url' keys.
        """
        if not self._api_key:
            return []

        remaining = _budget_remaining(_BRAVE_USAGE_FILE, _BRAVE_MONTHLY_BUDGET)
        if remaining <= 0:
            logger.warning(
                "Brave monthly budget exhausted (%d/%d) — skipping search",
                _get_monthly_usage(_BRAVE_USAGE_FILE), _BRAVE_MONTHLY_BUDGET,
            )
            return []

        import requests

        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
                params={
                    "q": query,
                    "count": max_results,
                    "search_lang": "en",
                    "freshness": "pd",  # past day — most relevant for sports
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Brave search failed for '%s': %s", query, exc)
            return []

        count = _increment_usage(_BRAVE_USAGE_FILE)
        logger.debug(
            "Brave search OK (%d/%d this month): %s",
            count, _BRAVE_MONTHLY_BUDGET, query,
        )

        results: list[dict] = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("description", ""),
                "url": item.get("url", ""),
            })

        return results


# ── Cascading search ─────────────────────────────────────────────────


class CascadingSearch:
    """Tries multiple search backends in priority order.

    Falls through to the next backend when:
      - Current backend has no API key
      - Current backend's monthly budget is exhausted
      - Current backend's API call fails

    Priority: Reddit (free) → Tavily (best quality) → Brave → empty.
    """

    def __init__(self, backends: list) -> None:
        self._backends = backends

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        for backend in self._backends:
            if hasattr(backend, "available") and not backend.available:
                continue
            results = backend.search(query, max_results)
            if results:
                return results
        return []


# ── Factory ──────────────────────────────────────────────────────────


def create_search_backend() -> CascadingSearch:
    """Create a cascading search backend: Reddit → Tavily → Brave → empty stub.

    Reddit is first (free; set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET
    for OAuth 60 req/min, otherwise ~10 req/min unauthenticated).
    Paid backends follow as fallback when Reddit returns no results.
    """
    backends: list = []

    # Reddit RSS — always available, no API key needed.
    # Uses /r/{sub}/.rss feeds with local keyword filtering.
    # Optionally enhanced with OAuth JSON search if credentials are set.
    backends.append(RedditRSSSearch())

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        backends.append(TavilySearch(tavily_key))

    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if brave_key:
        backends.append(BraveSearch(brave_key))

    # Always include stub as final fallback
    backends.append(DefaultNewsSearch())

    names = [type(b).__name__ for b in backends[:-1]]
    logger.info("Search fallback chain: %s → regex-only", " → ".join(names))

    return CascadingSearch(backends)
