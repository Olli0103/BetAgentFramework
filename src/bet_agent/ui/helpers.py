"""Display helpers for the WebUI and Telegram formatters.

Resolves abbreviated team names (e.g. "cha", "bos") to full canonical
names (e.g. "Charlotte Hornets", "Boston Celtics") via the team_aliases table.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import Match, Sport, TeamAlias

logger = logging.getLogger(__name__)

# Sport emoji mapping
SPORT_EMOJI: dict[str, str] = {
    "football": "\u26bd",
    "tennis": "\U0001f3be",
    "ice_hockey": "\U0001f3d2",
    "basketball": "\U0001f3c0",
    "american_football": "\U0001f3c8",
    "darts": "\U0001f3af",
}


def resolve_display_name(session: Session, raw_name: str, sport: Sport | None = None) -> str:
    """Resolve a raw/abbreviated team name to its canonical display name.

    Lookup order:
      1. Exact match in team_aliases.alias (case-insensitive)
      2. Exact match in team_aliases.canonical_name
      3. Fallback: return raw_name as-is

    Args:
        session: SQLAlchemy session.
        raw_name: The raw name as stored in the DB.
        sport: Optional sport filter for more specific matching.

    Returns:
        The canonical name, or raw_name if no alias found.
    """
    if not raw_name or not raw_name.strip():
        return raw_name

    key = raw_name.strip().lower()

    # Search aliases (case-insensitive via func.lower)
    from sqlalchemy import func

    query = select(TeamAlias.canonical_name).where(
        func.lower(TeamAlias.alias) == key
    )
    if sport is not None:
        query = query.where(
            (TeamAlias.sport == sport) | (TeamAlias.sport.is_(None))
        )

    result = session.execute(query).scalar_one_or_none()
    if result:
        return result

    return raw_name


def get_match_display(session: Session, match: Match) -> dict:
    """Build a display-friendly match dict with resolved team names.

    Returns:
        Dict with keys: home, away, sport, sport_emoji, league, kickoff, match_id
    """
    home = resolve_display_name(session, match.home_team, match.sport)
    away = resolve_display_name(session, match.away_team, match.sport)
    emoji = SPORT_EMOJI.get(match.sport.value, "\U0001f3c6")

    kickoff = ""
    if match.scheduled_at:
        kickoff = match.scheduled_at.strftime("%H:%M")

    return {
        "home": home,
        "away": away,
        "sport": match.sport.value,
        "sport_emoji": emoji,
        "league": match.league or "",
        "kickoff": kickoff,
        "match_id": match.id,
        "vs": f"{home} vs {away}",
    }
