"""Historical data ingesters for all supported sports.

Each ingester reads CSV/XLSX files or external sources and bulk-inserts
into the historical_matches table. All team/player names are normalized
through the IroncladAliasResolver before reaching the database.
"""

from bet_agent.ingest.alias_resolver import IroncladAliasResolver
from bet_agent.ingest.base import BaseIngester
from bet_agent.ingest.football import FootballIngester
from bet_agent.ingest.nba import NBAIngester
from bet_agent.ingest.nhl import NHLIngester
from bet_agent.ingest.nfl import NFLIngester
from bet_agent.ingest.tennis import TennisIngester

__all__ = [
    "BaseIngester",
    "IroncladAliasResolver",
    "FootballIngester",
    "NBAIngester",
    "NHLIngester",
    "NFLIngester",
    "TennisIngester",
]
