"""SQLAlchemy ORM models for the BetAgent Multi-Agent System.

9 tables: matches, odds_markets, bankroll_ledger, placed_bets,
model_metrics, team_aliases, team_daily_stats, historical_matches, predictions.
"""

import enum
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Use JSONB on PostgreSQL (binary, indexable) with JSON fallback on SQLite (tests).
JSONB = JSON().with_variant(PG_JSONB, "postgresql")


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
    PLACED = "placed"  # Human confirmed at sportsbook, stake deducted
    WON = "won"
    LOST = "lost"
    VOID = "void"
    PUSHED_TO_HUMAN = "pushed_to_human"


class MatchState(str, enum.Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BREAK = "break"
    FINISHED = "finished"


class PredictionStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    VETOED = "vetoed"
    PLACED = "placed"


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
    live_stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now(), nullable=False
    )

    # Relationships
    odds: Mapped[list["OddsMarket"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )
    bets: Mapped[list["PlacedBet"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
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
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False, index=True
    )

    # Relationships
    match: Mapped["Match"] = relationship(back_populates="odds")

    __table_args__ = (
        Index("ix_odds_match_market", "match_id", "market_type"),
        CheckConstraint("odds_decimal > 1.0", name="ck_odds_decimal_valid"),
    )

    @staticmethod
    def normalize_odds_on_set(target, value, oldvalue, initiator):
        """Auto-normalize American odds to decimal before DB write."""
        if value is None:
            return value
        fval = float(value)
        if fval <= -100 or fval >= 99.0:
            from bet_agent.tools.odds import american_to_decimal
            return Decimal(str(american_to_decimal(fval)))
        return value

    def __repr__(self) -> str:
        return f"<OddsMarket {self.sportsbook} {self.selection}@{self.odds_decimal}>"


# Auto-normalize odds on attribute set (catches all creation paths)
from sqlalchemy import event as _sa_event  # noqa: E402
_sa_event.listen(OddsMarket.odds_decimal, "set", OddsMarket.normalize_odds_on_set, retval=True)


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
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now(), nullable=False
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
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False, index=True
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
    """Sport-scoped name resolution: maps variant names to canonical names.

    Every alias is scoped to a sport so that e.g. a tennis player "Sinner"
    and a hypothetical hockey team "Sinner" can coexist without collision.
    """

    __tablename__ = "team_aliases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    canonical_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    alias: Mapped[str] = mapped_column(String(128), nullable=False)
    sport: Mapped[Sport | None] = mapped_column(
        Enum(Sport, native_enum=False), nullable=True, index=True
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint("alias", "sport", name="uq_alias_sport"),
        Index("ix_alias_sport", "alias", "sport"),
    )

    def __repr__(self) -> str:
        sport_str = self.sport.value if self.sport else "global"
        return f"<TeamAlias '{self.alias}' -> '{self.canonical_name}' ({sport_str})>"


class TeamDailyStats(Base):
    """Daily rolling statistics scraped from public sources.

    One row per team per date per sport. The stats JSONB column holds
    sport-specific metrics:
      - football: xG, xGA, possession, shots, passes
      - basketball: pace, off_rtg, def_rtg, net_rtg, FG%, 3P%
      - ice_hockey: corsi_for%, fenwick_for%, pp%, pk%, sv%
      - american_football: off_epa, def_epa, yards_per_play, turnovers
      - tennis: ace%, 1st_serve%, bp_saved% (stored per player, not team)
    """

    __tablename__ = "team_daily_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    sport: Mapped[Sport] = mapped_column(
        Enum(Sport, native_enum=False), nullable=False, index=True
    )
    team_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(128), nullable=False)
    stat_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Sport-specific stats stored as JSONB for flexibility
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Crawl metadata
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    crawl_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "sport", "team_name", "stat_date",
            name="uq_team_daily_stat",
        ),
        Index("ix_daily_stats_sport_date", "sport", "stat_date"),
    )

    def __repr__(self) -> str:
        return f"<TeamDailyStats {self.team_name} ({self.sport.value}) {self.stat_date}>"


class HistoricalMatch(Base):
    """Bulk-imported historical match data from CSV/XLSX/external sources.

    Uses JSONB columns for sport-specific data to accommodate widely
    different schemas across sports:
      - football: shots, corners, cards, halftime scores + multi-book odds
      - basketball: quarter scores, spread, total, moneylines
      - ice_hockey: 120+ columns of rolling stats, Corsi, Fenwick, PP/PK
      - american_football: detailed odds lines (open/min/max/close)
      - tennis: set scores, rankings, surface, round + multi-book odds
    """

    __tablename__ = "historical_matches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    sport: Mapped[Sport] = mapped_column(
        Enum(Sport, native_enum=False), nullable=False, index=True
    )
    season: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    division: Mapped[str] = mapped_column(String(64), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Teams (for tennis: winner=home, loser=away)
    home_team: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Final result
    home_score: Mapped[int] = mapped_column(Integer, nullable=False)
    away_score: Mapped[int] = mapped_column(Integer, nullable=False)
    result: Mapped[str] = mapped_column(String(8), nullable=False)  # H/D/A or W/L

    # Sport-specific match stats (JSONB)
    # football: {ht_home, ht_away, ht_result, home_shots, away_shots, ...}
    # basketball: {q1_home, q2_home, ..., ot_home, regular, playoffs}
    # ice_hockey: {shots, pp_goals, pp_opps, faceoff_pct, hits, ...}
    # american_football: {overtime, playoff, neutral_venue}
    # tennis: {surface, round, best_of, w1-w5, l1-l5, wsets, lsets, ...}
    match_stats: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Odds from multiple bookmakers (JSONB)
    # football: {b365_home, b365_draw, b365_away, bw_home, ...}
    # basketball: {moneyline_away, moneyline_home}
    # tennis: {b365_winner, b365_loser, ps_winner, ps_loser, ...}
    odds: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Betting lines — spreads, totals, moneylines (JSONB)
    # basketball: {spread, total, h2_spread, h2_total}
    # ice_hockey: {spread, over_under, favorite_moneyline}
    # american_football: {home_line_open/min/max/close, total_open/...}
    betting_lines: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Rolling/advanced stats — mainly NHL (JSONB)
    # ice_hockey: {roll_3_*, roll_10_*, roll_30_*, opp_*, rest_days, ...}
    # tennis: {winner_rank, loser_rank, winner_pts, loser_pts}
    advanced_stats: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Import metadata
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_file: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "sport", "home_team", "away_team", "match_date", "division",
            name="uq_historical_match",
        ),
        Index("ix_hist_sport_date", "sport", "match_date"),
        Index("ix_hist_season", "sport", "season"),
    )


class Prediction(Base):
    """ML/analytical model prediction for a specific match and market.

    Tracks a prediction through the pipeline: pending → approved → placed
    (or vetoed by the Devil's Advocate).
    """

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    market_type: Mapped[MarketType] = mapped_column(
        Enum(MarketType, native_enum=False), nullable=False
    )
    selection: Mapped[str] = mapped_column(String(64), nullable=False)

    # Probabilities and EV
    model_prob: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    implied_prob: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    prob_edge: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    ev: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)

    # Model metadata
    model_source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="analytical"
    )  # "xgboost" or "analytical"

    # Veto / line-shopping metadata
    veto_reason: Mapped[str | None] = mapped_column(String(4096), nullable=True)
    best_odds: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    best_sportsbook: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Pipeline status
    status: Mapped[PredictionStatus] = mapped_column(
        Enum(PredictionStatus, native_enum=False),
        default=PredictionStatus.PENDING,
        nullable=False, index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )

    # Relationships
    match: Mapped["Match"] = relationship(back_populates="predictions")

    __table_args__ = (
        UniqueConstraint(
            "match_id", "model_name", "market_type", "selection",
            name="uq_prediction_identity",
        ),
        Index("ix_pred_status", "status"),
    )

    def __repr__(self) -> str:
        status_str = self.status.value if self.status is not None else "no_status"
        return f"<Prediction {self.selection} ev={self.ev} ({status_str})>"
