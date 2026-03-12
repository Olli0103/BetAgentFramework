"""Search backends for the Devil's Advocate veto engine.

Provides pluggable news search implementations with cascading fallback:
  Priority 1: RedditSearch   — free with OAuth (~60 req/min), or
                                unauthenticated (~10 req/min, may be unreliable)
  Priority 2: TavilySearch   — best quality, 1000 req/month free tier
  Priority 3: BraveSearch    — good quality, 2000 req/month free tier
  Priority 4: DefaultNewsSearch — stub (regex-only veto path)

Reddit OAuth setup (recommended):
  1. Create a Reddit "script" app at https://www.reddit.com/prefs/apps/
  2. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars
  3. This gives 60 req/min (vs 10 req/min unauthenticated)

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
import time
import threading
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
    """News backend using Reddit's JSON search endpoints.

    Two modes of operation:
      - **OAuth** (recommended): Set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET.
        Uses ``oauth.reddit.com`` → 60 req/min, reliable.
      - **Unauthenticated fallback**: No credentials needed.
        Uses ``www.reddit.com/.json`` → ~10 req/min, may be throttled.

    Searches sport subreddits via ``/r/{sub}/search.json`` with
    ``restrict_sr=on`` and ``sort=new``.
    """

    def __init__(self, subreddits: list[str] | None = None) -> None:
        self._subreddits = subreddits or _ALL_SPORT_SUBS

    @property
    def available(self) -> bool:
        return True  # Always available — unauthenticated fallback exists

    @property
    def has_oauth(self) -> bool:
        """Return True if Reddit OAuth credentials are configured."""
        return bool(
            os.environ.get("REDDIT_CLIENT_ID", "")
            and os.environ.get("REDDIT_CLIENT_SECRET", "")
        )

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search Reddit for recent posts matching the query."""
        import requests as _requests

        results: list[dict] = []
        subs_to_search = self._pick_subreddits(query)

        token = _get_reddit_oauth_token()
        if token:
            base_url = "https://oauth.reddit.com"
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": "BetAgent/1.0 (sports research bot)",
            }
        else:
            base_url = "https://www.reddit.com"
            headers = {"User-Agent": "BetAgent/1.0 (sports research bot)"}
            if self.has_oauth:
                logger.warning("Reddit OAuth configured but token fetch failed — using unauthenticated")

        for sub in subs_to_search[:3]:  # Cap at 3 subs to stay fast
            try:
                resp = _requests.get(
                    f"{base_url}/r/{sub}/search.json",
                    params={
                        "q": query,
                        "restrict_sr": "on",
                        "sort": "new",
                        "t": "week",
                        "limit": max_results,
                    },
                    headers=headers,
                    timeout=8,
                )
                if resp.status_code == 429:
                    logger.debug("Reddit rate-limited on r/%s — skipping", sub)
                    continue
                if resp.status_code == 403:
                    logger.warning("Reddit returned 403 for r/%s — endpoint may be blocked", sub)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.debug("Reddit search failed for r/%s: %s", sub, exc)
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

    # Reddit RSS — always available, no API key needed
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
