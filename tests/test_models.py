"""Tests for database models — uses in-memory SQLite for speed."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bet_agent.db.models import (
    BankrollLedger,
    Base,
    BetStatus,
    LedgerType,
    MarketType,
    Match,
    MatchState,
    ModelMetrics,
    OddsMarket,
    PlacedBet,
    Sport,
    TeamAlias,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


# ── Table creation ─────────────────────────────────────────────────────


class TestTableCreation:
    def test_all_tables_exist(self, db_session: Session):
        tables = Base.metadata.tables.keys()
        expected = {
            "matches",
            "odds_markets",
            "bankroll_ledger",
            "placed_bets",
            "model_metrics",
            "team_aliases",
            "team_daily_stats",
        }
        assert expected == set(tables)


# ── Match ──────────────────────────────────────────────────────────────


class TestMatch:
    def _make_match(self, **overrides) -> Match:
        defaults = dict(
            sport=Sport.FOOTBALL,
            league="Bundesliga",
            home_team="Bayern München",
            away_team="Borussia Dortmund",
            scheduled_at=datetime(2026, 3, 15, 15, 30, tzinfo=timezone.utc),
        )
        defaults.update(overrides)
        return Match(**defaults)

    def test_insert_and_query(self, db_session: Session):
        match = self._make_match()
        db_session.add(match)
        db_session.commit()

        result = db_session.query(Match).one()
        assert result.sport == Sport.FOOTBALL
        assert result.home_team == "Bayern München"
        assert result.is_live is False
        assert result.match_state == MatchState.NOT_STARTED

    def test_all_sports(self, db_session: Session):
        for i, sport in enumerate(Sport):
            m = self._make_match(
                sport=sport,
                league=f"Test {sport.value}",
                home_team=f"Home {i}",
                away_team=f"Away {i}",
            )
            db_session.add(m)
        db_session.commit()
        assert db_session.query(Match).count() == len(Sport)

    def test_live_state(self, db_session: Session):
        match = self._make_match(
            is_live=True,
            match_state=MatchState.IN_PROGRESS,
            match_period="first_half",
            home_score=1,
            away_score=0,
            live_stats={"home_xg": 1.2, "away_xg": 0.5},
        )
        db_session.add(match)
        db_session.commit()

        result = db_session.query(Match).one()
        assert result.is_live is True
        assert result.home_score == 1
        assert result.live_stats["home_xg"] == 1.2


# ── OddsMarket ─────────────────────────────────────────────────────────


class TestOddsMarket:
    def test_insert_with_match_fk(self, db_session: Session):
        match = Match(
            sport=Sport.TENNIS,
            league="ATP",
            home_team="Zverev",
            away_team="Djokovic",
            scheduled_at=datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc),
        )
        db_session.add(match)
        db_session.flush()

        odds = OddsMarket(
            match_id=match.id,
            sportsbook="tipico",
            market_type=MarketType.MATCH_WINNER,
            selection="home",
            odds_decimal=Decimal("2.10"),
        )
        db_session.add(odds)
        db_session.commit()

        result = db_session.query(OddsMarket).one()
        assert result.match_id == match.id
        assert result.odds_decimal == Decimal("2.10")
        assert result.is_live is False


# ── BankrollLedger ─────────────────────────────────────────────────────


class TestBankrollLedger:
    def test_real_and_paper_ledgers(self, db_session: Session):
        real = BankrollLedger(ledger_type=LedgerType.REAL, balance=Decimal("500.00"))
        paper = BankrollLedger(ledger_type=LedgerType.PAPER, balance=Decimal("10000.00"))
        db_session.add_all([real, paper])
        db_session.commit()

        assert db_session.query(BankrollLedger).count() == 2
        real_result = (
            db_session.query(BankrollLedger)
            .filter_by(ledger_type=LedgerType.REAL)
            .one()
        )
        assert real_result.balance == Decimal("500.00")


# ── PlacedBet ──────────────────────────────────────────────────────────


class TestPlacedBet:
    def test_single_bet(self, db_session: Session):
        match = Match(
            sport=Sport.ICE_HOCKEY,
            league="DEL",
            home_team="Eisbären Berlin",
            away_team="Adler Mannheim",
            scheduled_at=datetime(2026, 3, 20, 19, 0, tzinfo=timezone.utc),
        )
        db_session.add(match)
        db_session.flush()

        bet = PlacedBet(
            ledger_type=LedgerType.PAPER,
            match_id=match.id,
            market_type=MarketType.OVER_UNDER,
            selection="over_5.5",
            odds_at_placement=Decimal("1.85"),
            stake_eur=Decimal("10.00"),
            model_prob=Decimal("0.58"),
            ev_at_placement=Decimal("0.073"),
        )
        db_session.add(bet)
        db_session.commit()

        result = db_session.query(PlacedBet).one()
        assert result.status == BetStatus.PENDING
        assert result.is_parlay is False
        assert result.is_live_bet is False

    def test_parlay_bet(self, db_session: Session):
        match = Match(
            sport=Sport.FOOTBALL,
            league="Premier League",
            home_team="Arsenal",
            away_team="Chelsea",
            scheduled_at=datetime(2026, 4, 1, 16, 0, tzinfo=timezone.utc),
        )
        db_session.add(match)
        db_session.flush()

        parlay_id = uuid.uuid4()
        bet = PlacedBet(
            ledger_type=LedgerType.REAL,
            match_id=match.id,
            market_type=MarketType.MATCH_WINNER,
            selection="home",
            odds_at_placement=Decimal("2.50"),
            stake_eur=Decimal("1.00"),  # Moonshot cap
            model_prob=Decimal("0.45"),
            ev_at_placement=Decimal("0.125"),
            is_parlay=True,
            parlay_group_id=parlay_id,
        )
        db_session.add(bet)
        db_session.commit()

        result = db_session.query(PlacedBet).one()
        assert result.is_parlay is True
        assert result.parlay_group_id == parlay_id
        assert result.stake_eur == Decimal("1.00")


# ── ModelMetrics ───────────────────────────────────────────────────────


class TestModelMetrics:
    def test_insert_metrics(self, db_session: Session):
        metrics = ModelMetrics(
            model_name="xgboost_match_winner_v1",
            date=datetime(2026, 3, 10).date(),
            brier_score=Decimal("0.21"),
            roi_pct=Decimal("3.5"),
            total_bets=50,
            record_win=28,
            record_loss=22,
            ledger_type=LedgerType.PAPER,
        )
        db_session.add(metrics)
        db_session.commit()

        result = db_session.query(ModelMetrics).one()
        assert result.brier_score == Decimal("0.21")
        assert result.record_win == 28


# ── TeamAlias ──────────────────────────────────────────────────────────


class TestTeamAlias:
    def test_alias_resolution(self, db_session: Session):
        aliases = [
            TeamAlias(canonical_name="Bayern München", alias="FC Bayern", source="tipico"),
            TeamAlias(
                canonical_name="Bayern München", alias="Bayern Munich", source="bet365"
            ),
        ]
        db_session.add_all(aliases)
        db_session.commit()

        results = (
            db_session.query(TeamAlias)
            .filter_by(canonical_name="Bayern München")
            .all()
        )
        assert len(results) == 2
        alias_names = {r.alias for r in results}
        assert alias_names == {"FC Bayern", "Bayern Munich"}
