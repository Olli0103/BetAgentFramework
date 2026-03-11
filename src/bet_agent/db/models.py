"""SQLAlchemy ORM models for the BetAgent Multi-Agent System.

6 tables: matches, odds_markets, bankroll_ledger, placed_bets,
model_metrics, team_aliases.
"""

import enum
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ──────────────────────────────────────────────────────────────


class Sport(str, enum.Enum):
    FOOTBALL = "football"
    TENNIS = "tennis"
    ICE_HOCKEY = "ice_hockey"
    BASKETBALL = "basketball"
    DARTS = "darts"
    AMERICAN_FOOTBALL = "american_football"


class MarketType(str, enum.Enum):
    MATCH_WINNER = "match_winner"
    OVER_UNDER = "over_under"
    BTTS = "btts"
    SPREAD = "spread"


class LedgerType(str, enum.Enum):
    REAL = "real"
    PAPER = "paper"


class BetStatus(str, enum.Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    VOID = "void"
    PUSHED_TO_HUMAN = "pushed_to_human"


class MatchState(str, enum.Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BREAK = "break"
    FINISHED = "finished"


# ── Helpers ────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


# ── Tables ─────────────────────────────────────────────────────────────


class Match(Base):
    """A sporting event (pre-match or live)."""

    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    sport: Mapped[Sport] = mapped_column(
        Enum(Sport, native_enum=False), nullable=False, index=True
    )
    league: Mapped[str] = mapped_column(String(128), nullable=False)
    home_team: Mapped[str] = mapped_column(String(128), nullable=False)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False)
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Live state
    is_live: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    match_state: Mapped[MatchState] = mapped_column(
        Enum(MatchState, native_enum=False), nullable=False, default=MatchState.NOT_STARTED
    )
    match_period: Mapped[str | None] = mapped_column(String(32), nullable=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    live_stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    odds: Mapped[list["OddsMarket"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )
    bets: Mapped[list["PlacedBet"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "sport", "league", "home_team", "away_team", "scheduled_at",
            name="uq_match_identity",
        ),
    )

    def __repr__(self) -> str:
        return f"<Match {self.home_team} vs {self.away_team} ({self.sport.value})>"


class OddsMarket(Base):
    """Scraped odds from a sportsbook for a specific market."""

    __tablename__ = "odds_markets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    sportsbook: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[MarketType] = mapped_column(
        Enum(MarketType, native_enum=False), nullable=False
    )
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    odds_decimal: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    is_live: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    # Relationships
    match: Mapped["Match"] = relationship(back_populates="odds")

    __table_args__ = (
        Index("ix_odds_match_market", "match_id", "market_type"),
    )

    def __repr__(self) -> str:
        return f"<OddsMarket {self.sportsbook} {self.selection}@{self.odds_decimal}>"


class BankrollLedger(Base):
    """Current bankroll balance for Real and Paper trading."""

    __tablename__ = "bankroll_ledger"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    ledger_type: Mapped[LedgerType] = mapped_column(
        Enum(LedgerType, native_enum=False), nullable=False, unique=True
    )
    balance: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=Decimal("0.00")
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<BankrollLedger {self.ledger_type.value}: {self.balance} EUR>"


class PlacedBet(Base):
    """A bet placed (or pushed to human) by the system."""

    __tablename__ = "placed_bets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    ledger_type: Mapped[LedgerType] = mapped_column(
        Enum(LedgerType, native_enum=False), nullable=False, index=True
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    market_type: Mapped[MarketType] = mapped_column(
        Enum(MarketType, native_enum=False), nullable=False
    )
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    odds_at_placement: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    stake_eur: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    model_prob: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    ev_at_placement: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)

    # Parlay support
    is_parlay: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    parlay_group_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    # Status & resolution
    status: Mapped[BetStatus] = mapped_column(
        Enum(BetStatus, native_enum=False), default=BetStatus.PENDING,
        nullable=False, index=True,
    )
    is_live_bet: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    placed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pnl_eur: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)

    # Relationships
    match: Mapped["Match"] = relationship(back_populates="bets")

    def __repr__(self) -> str:
        return f"<PlacedBet {self.selection}@{self.odds_at_placement} {self.stake_eur}EUR>"


class ModelMetrics(Base):
    """Daily evaluation metrics for each prediction model."""

    __tablename__ = "model_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    brier_score: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    roi_pct: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    total_bets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    record_win: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    record_loss: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ledger_type: Mapped[LedgerType] = mapped_column(
        Enum(LedgerType, native_enum=False), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("model_name", "date", "ledger_type", name="uq_model_date_ledger"),
    )

    def __repr__(self) -> str:
        return f"<ModelMetrics {self.model_name} {self.date} brier={self.brier_score}>"


class TeamAlias(Base):
    """Fuzzy name resolution: maps sportsbook-specific names to canonical names."""

    __tablename__ = "team_aliases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    canonical_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    alias: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)

    def __repr__(self) -> str:
        return f"<TeamAlias '{self.alias}' -> '{self.canonical_name}'>"
