"""Cloudflare Browser Rendering /crawl endpoint integration.

Golden Rule #1: NO LLM MATH. This is a deterministic HTTP tool.

Batches daily stats gathering into a single crawl job per sport,
respecting the free-tier limit of 5 crawls/day and 100 pages/crawl.
Uses render=false for fast HTML-only fetches (free during beta).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# ── Free-tier guardrails ──────────────────────────────────────────────

MAX_CRAWLS_PER_DAY = 5
MAX_PAGES_PER_CRAWL = 100
POLL_INTERVAL_SECONDS = 10
MAX_POLL_ATTEMPTS = 120  # 20 minutes max wait

# ── Seed URL registry (curated per sport) ─────────────────────────────
# Each sport maps to a list of seed URLs plus include-patterns that
# scope the crawl to the stats pages we actually need.

SPORT_CRAWL_CONFIGS: dict[str, dict] = {
    "american_football": {
        "seed_url": "https://www.pro-football-reference.com",
        "include_patterns": [
            "/years/*/",
            "/teams/*/",
        ],
        "max_pages": 20,
        "description": "NFL team stats, standings, schedule",
    },
    "basketball": {
        "seed_url": "https://www.basketball-reference.com",
        "include_patterns": [
            "/leagues/NBA_*",
            "/teams/*/",
        ],
        "max_pages": 20,
        "description": "NBA team stats, standings, pace/efficiency",
    },
    "ice_hockey": {
        "seed_url": "https://www.hockey-reference.com",
        "include_patterns": [
            "/leagues/NHL_*",
            "/teams/*/",
        ],
        "max_pages": 20,
        "description": "NHL team stats, Corsi, Fenwick, special teams",
    },
    "football": {
        "seed_url": "https://fbref.com",
        "include_patterns": [
            "/en/comps/*/",
            "/en/squads/*/",
        ],
        "max_pages": 25,
        "description": "Football (soccer) xG, squad stats, top-5 leagues",
    },
    "tennis": {
        "seed_url": "https://www.tennisabstract.com",
        "include_patterns": [
            "/cgi-bin/player-classic.cgi*",
            "/reports/*",
        ],
        "max_pages": 15,
        "description": "Tennis player serve/return stats, recent form",
    },
}


@dataclass(frozen=True)
class CrawlResult:
    """Result of a single Cloudflare crawl job."""

    sport: str
    job_id: str
    status: str  # "completed", "running", "cancelled_*"
    pages: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _get_credentials() -> tuple[str, str]:
    """Load Cloudflare credentials from environment.

    Returns:
        (account_id, api_token) tuple.

    Raises:
        RuntimeError: If credentials are not configured.
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment "
            "variables are required for the Cloudflare crawler."
        )
    return account_id, api_token


def _base_url(account_id: str) -> str:
    return (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/browser-rendering/crawl"
    )


def start_crawl(
    sport: str,
    *,
    seed_url: str | None = None,
    include_patterns: list[str] | None = None,
    max_pages: int | None = None,
    depth: int = 2,
    render: bool = False,
) -> str:
    """Start an async crawl job on Cloudflare.

    Args:
        sport: Sport identifier for logging/tracking.
        seed_url: Override the default seed URL for this sport.
        include_patterns: Override URL include patterns.
        max_pages: Override max pages (capped at MAX_PAGES_PER_CRAWL).
        depth: Max link-follow depth from seed URL (default 2).
        render: Whether to render JS (costs browser minutes). Default False.

    Returns:
        The crawl job ID.

    Raises:
        RuntimeError: On missing credentials.
        ValueError: If sport has no configured crawl config and no seed_url.
        requests.HTTPError: On API errors.
    """
    config = SPORT_CRAWL_CONFIGS.get(sport, {})
    url = seed_url or config.get("seed_url")
    if not url:
        raise ValueError(
            f"No seed URL configured for sport '{sport}'. "
            f"Available: {list(SPORT_CRAWL_CONFIGS.keys())}"
        )

    account_id, api_token = _get_credentials()

    patterns = include_patterns or config.get("include_patterns", [])
    pages = min(max_pages or config.get("max_pages", 20), MAX_PAGES_PER_CRAWL)

    payload: dict = {
        "url": url,
        "limit": pages,
        "depth": depth,
        "render": render,
        "formats": ["html"],
        "source": "links",
        "rejectResourceTypes": ["image", "media", "font"],
    }
    if patterns:
        payload["includePatterns"] = patterns

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    logger.info("Starting crawl for %s: %s (max %d pages)", sport, url, pages)

    resp = requests.post(
        _base_url(account_id),
        json=payload,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    job_id = data.get("result", {}).get("id") or data.get("id", "unknown")
    logger.info("Crawl job started for %s: job_id=%s", sport, job_id)
    return job_id


def poll_crawl(job_id: str) -> CrawlResult:
    """Poll a crawl job until completion.

    Args:
        job_id: The Cloudflare crawl job ID.

    Returns:
        CrawlResult with pages and status.
    """
    account_id, api_token = _get_credentials()
    headers = {"Authorization": f"Bearer {api_token}"}
    url = f"{_base_url(account_id)}/{job_id}"

    for attempt in range(MAX_POLL_ATTEMPTS):
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("result", data)
        status = result.get("status", "unknown")

        if status == "running":
            logger.debug(
                "Crawl %s still running (attempt %d/%d)",
                job_id, attempt + 1, MAX_POLL_ATTEMPTS,
            )
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        # Terminal state — collect all pages
        pages = []
        cursor = None
        while True:
            page_url = url
            params: dict = {}
            if cursor:
                params["cursor"] = cursor

            page_resp = requests.get(
                page_url, headers=headers, params=params, timeout=30,
            )
            page_resp.raise_for_status()
            page_data = page_resp.json()

            page_result = page_data.get("result", page_data)
            page_items = page_result.get("pages", page_result.get("data", []))
            if isinstance(page_items, list):
                pages.extend(page_items)

            cursor = page_result.get("cursor")
            if not cursor:
                break

        return CrawlResult(
            sport="",  # caller fills this in
            job_id=job_id,
            status=status,
            pages=pages,
        )

    return CrawlResult(
        sport="",
        job_id=job_id,
        status="timeout_polling",
        errors=[f"Exceeded {MAX_POLL_ATTEMPTS} poll attempts"],
    )


def run_morning_crawl(
    sports: list[str] | None = None,
) -> list[CrawlResult]:
    """Execute the daily morning crawl for all configured sports.

    This is the main entry point for the Scout Agent's daily schedule.
    Launches one crawl per sport (max 5 to respect free tier), polls
    each to completion, and returns results.

    Args:
        sports: List of sports to crawl. Defaults to all configured sports.

    Returns:
        List of CrawlResult, one per sport.
    """
    if sports is None:
        sports = list(SPORT_CRAWL_CONFIGS.keys())

    if len(sports) > MAX_CRAWLS_PER_DAY:
        logger.warning(
            "Requested %d crawls but free tier allows %d/day. Truncating.",
            len(sports), MAX_CRAWLS_PER_DAY,
        )
        sports = sports[:MAX_CRAWLS_PER_DAY]

    # Phase 1: Start all crawl jobs
    jobs: list[tuple[str, str]] = []
    for sport in sports:
        try:
            job_id = start_crawl(sport)
            jobs.append((sport, job_id))
        except Exception as exc:
            logger.error("Failed to start crawl for %s: %s", sport, exc)

    # Phase 2: Poll all jobs to completion
    results: list[CrawlResult] = []
    for sport, job_id in jobs:
        try:
            result = poll_crawl(job_id)
            # Fill in sport since poll_crawl doesn't know it
            result = CrawlResult(
                sport=sport,
                job_id=result.job_id,
                status=result.status,
                pages=result.pages,
                errors=result.errors,
            )
            results.append(result)
            logger.info(
                "Crawl for %s complete: %d pages, status=%s",
                sport, len(result.pages), result.status,
            )
        except Exception as exc:
            logger.error("Failed to poll crawl for %s: %s", sport, exc)
            results.append(CrawlResult(
                sport=sport,
                job_id=job_id,
                status="error",
                errors=[str(exc)],
            ))

    return results
