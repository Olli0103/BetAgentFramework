"""Tests for Phase 5: Command Center — Master Analysis, Bot Auth, Broadcasting.

Covers:
  - master_analysis.py read-only query tools
  - telegram_bot.py whitelist authentication
  - notifier.py syndicate broadcasting
  - mark_bet_placed_by_user synchronization
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bet_agent.db.models import (
    BankrollLedger,
    Base,
    BetStatus,
    LedgerType,
    MarketType,
    Match,
    MatchState,
    ModelMetrics,
    PlacedBet,
    Prediction,
    PredictionStatus,
    Sport,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _yesterday():
    return date.today() - timedelta(days=1)


def _add_match(
    session,
    sport=Sport.FOOTBALL,
    home="FC Bayern",
    away="BVB Dortmund",
    state=MatchState.NOT_STARTED,
    scheduled=None,
    home_score=None,
    away_score=None,
):
    if scheduled is None:
        scheduled = datetime.combine(
            date.today(), datetime.min.time(), tzinfo=timezone.utc
        )
    m = Match(
        sport=sport,
        league="Bundesliga",
        home_team=home,
        away_team=away,
        scheduled_at=scheduled,
        match_state=state,
        home_score=home_score,
        away_score=away_score,
    )
    session.add(m)
    session.flush()
    return m


def _add_bet(
    session,
    match,
    selection="home",
    market_type=MarketType.MATCH_WINNER,
    odds=Decimal("2.10"),
    stake=Decimal("10.00"),
    model_prob=Decimal("0.55"),
    ledger=LedgerType.REAL,
    status=BetStatus.PENDING,
    pnl=None,
    resolved_at=None,
):
    b = PlacedBet(
        match_id=match.id,
        ledger_type=ledger,
        market_type=market_type,
        selection=selection,
        odds_at_placement=odds,
        stake_eur=stake,
        model_prob=model_prob,
        ev_at_placement=Decimal("0.08"),
        status=status,
        pnl_eur=pnl,
        resolved_at=resolved_at,
    )
    session.add(b)
    session.flush()
    return b


def _add_prediction(
    session,
    match,
    selection="home",
    model_name="analytical_football",
    status=PredictionStatus.PLACED,
    veto_reason=None,
):
    p = Prediction(
        match_id=match.id,
        model_name=model_name,
        market_type=MarketType.MATCH_WINNER,
        selection=selection,
        model_prob=Decimal("0.55"),
        implied_prob=Decimal("0.4762"),
        prob_edge=Decimal("0.0738"),
        ev=Decimal("0.08"),
        model_source="analytical",
        status=status,
        veto_reason=veto_reason,
    )
    session.add(p)
    session.flush()
    return p


def _add_bankroll(session, ledger=LedgerType.REAL, balance=Decimal("1000.00")):
    b = BankrollLedger(ledger_type=ledger, balance=balance)
    session.add(b)
    session.flush()
    return b


def _add_metrics(session, model_name="analytical_football", brier=0.19, roi=3.5, bets=30):
    m = ModelMetrics(
        model_name=model_name,
        date=date.today(),
        brier_score=Decimal(str(brier)),
        roi_pct=Decimal(str(roi)),
        total_bets=bets,
        record_win=int(bets * 0.6),
        record_loss=bets - int(bets * 0.6),
        ledger_type=LedgerType.REAL,
    )
    session.add(m)
    session.flush()
    return m


# ══════════════════════════════════════════════════════════════════════
# MASTER ANALYSIS — Read-only tools
# ══════════════════════════════════════════════════════════════════════


class TestPortfolioSummary:
    """Tests for fetch_portfolio_summary."""

    def test_empty_portfolio(self, db_session):
        from bet_agent.tools.master_analysis import fetch_portfolio_summary

        summary = fetch_portfolio_summary(db_session)
        assert summary.real_balance == Decimal("0.00")
        assert summary.paper_balance == Decimal("0.00")
        assert summary.pending_bets_count == 0

    def test_with_balances_and_bets(self, db_session):
        from bet_agent.tools.master_analysis import fetch_portfolio_summary

        _add_bankroll(db_session, LedgerType.REAL, Decimal("500.00"))
        _add_bankroll(db_session, LedgerType.PAPER, Decimal("1000.00"))

        match = _add_match(db_session)
        _add_bet(db_session, match, ledger=LedgerType.REAL, status=BetStatus.PENDING)
        _add_bet(db_session, match, ledger=LedgerType.PAPER, status=BetStatus.PENDING,
                 selection="away")

        summary = fetch_portfolio_summary(db_session)

        assert summary.real_balance == Decimal("500.00")
        assert summary.paper_balance == Decimal("1000.00")
        assert summary.pending_bets_count == 2
        assert summary.total_exposure_real == Decimal("10.00")
        assert summary.total_exposure_paper == Decimal("10.00")

    def test_sport_exposure_breakdown(self, db_session):
        from bet_agent.tools.master_analysis import fetch_portfolio_summary

        m1 = _add_match(db_session, sport=Sport.FOOTBALL)
        m2 = _add_match(db_session, sport=Sport.ICE_HOCKEY, home="Oilers", away="Flames")

        _add_bet(db_session, m1, stake=Decimal("20.00"))
        _add_bet(db_session, m2, stake=Decimal("15.00"), selection="away")

        summary = fetch_portfolio_summary(db_session)

        assert "football" in summary.sport_exposure
        assert "ice_hockey" in summary.sport_exposure
        assert summary.sport_exposure["football"] == Decimal("20.00")
        assert summary.sport_exposure["ice_hockey"] == Decimal("15.00")

    def test_seven_day_win_rate(self, db_session):
        from bet_agent.tools.master_analysis import fetch_portfolio_summary

        match = _add_match(db_session)
        now = datetime.now(timezone.utc)

        # 3 wins, 2 losses in the last 7 days
        for i in range(3):
            _add_bet(db_session, match, status=BetStatus.WON,
                     pnl=Decimal("10.00"), resolved_at=now,
                     selection=f"win_{i}")
        for i in range(2):
            _add_bet(db_session, match, status=BetStatus.LOST,
                     pnl=Decimal("-10.00"), resolved_at=now,
                     selection=f"loss_{i}")

        summary = fetch_portfolio_summary(db_session)
        assert summary.win_rate_7d == 60.0
        assert summary.total_bets_7d == 5


class TestModelHealth:
    """Tests for fetch_model_health."""

    def test_no_metrics(self, db_session):
        from bet_agent.tools.master_analysis import fetch_model_health

        assert fetch_model_health(db_session) == []

    def test_returns_health_report(self, db_session):
        from bet_agent.tools.master_analysis import fetch_model_health

        _add_metrics(db_session, brier=0.19, roi=3.5)

        reports = fetch_model_health(db_session)
        assert len(reports) == 1
        assert reports[0].model_name == "analytical_football"
        assert reports[0].latest_brier == 0.19
        assert reports[0].latest_roi == 3.5
        assert not reports[0].is_degraded

    def test_filter_by_sport(self, db_session):
        from bet_agent.tools.master_analysis import fetch_model_health

        _add_metrics(db_session, model_name="analytical_football")
        _add_metrics(db_session, model_name="xgboost_ice_hockey", brier=0.21, roi=2.0)

        hockey = fetch_model_health(db_session, sport="ice_hockey")
        assert len(hockey) == 1
        assert "ice_hockey" in hockey[0].model_name

    def test_degraded_model_flagged(self, db_session):
        from bet_agent.tools.master_analysis import fetch_model_health

        _add_metrics(db_session, brier=0.28, roi=-8.0, bets=55)

        reports = fetch_model_health(db_session)
        assert reports[0].is_degraded


class TestVetoExplanation:
    """Tests for explain_veto_reason."""

    def test_approved_prediction(self, db_session):
        from bet_agent.tools.master_analysis import explain_veto_reason

        match = _add_match(db_session)
        _add_prediction(db_session, match, status=PredictionStatus.APPROVED)

        explanations = explain_veto_reason(db_session, match.id)
        assert len(explanations) == 1
        assert explanations[0].veto_reason is None
        assert explanations[0].prediction_status == "approved"

    def test_vetoed_prediction(self, db_session):
        from bet_agent.tools.master_analysis import explain_veto_reason

        match = _add_match(db_session)
        _add_prediction(
            db_session, match,
            status=PredictionStatus.VETOED,
            veto_reason="Key player injured (MCL tear, confirmed by team)",
        )

        explanations = explain_veto_reason(db_session, match.id)
        assert len(explanations) == 1
        assert "injured" in explanations[0].veto_reason

    def test_unknown_match_id(self, db_session):
        from bet_agent.tools.master_analysis import explain_veto_reason

        result = explain_veto_reason(db_session, uuid.uuid4())
        assert result == []


class TestRecentActivity:
    """Tests for fetch_recent_activity."""

    def test_no_activity(self, db_session):
        from bet_agent.tools.master_analysis import fetch_recent_activity

        activity = fetch_recent_activity(db_session)
        assert activity.predictions_today == 0
        assert activity.settled_today == 0

    def test_counts_todays_predictions(self, db_session):
        from bet_agent.tools.master_analysis import fetch_recent_activity

        match = _add_match(db_session)
        _add_prediction(db_session, match, status=PredictionStatus.APPROVED)
        _add_prediction(db_session, match, selection="away", status=PredictionStatus.VETOED,
                        veto_reason="bad weather")

        activity = fetch_recent_activity(db_session)
        assert activity.predictions_today == 2
        assert activity.approved_today == 1
        assert activity.vetoed_today == 1


class TestSportExposure:
    """Tests for fetch_sport_exposure."""

    def test_no_pending_bets(self, db_session):
        from bet_agent.tools.master_analysis import fetch_sport_exposure

        assert fetch_sport_exposure(db_session) == []

    def test_groups_by_sport(self, db_session):
        from bet_agent.tools.master_analysis import fetch_sport_exposure

        m1 = _add_match(db_session, sport=Sport.FOOTBALL)
        m2 = _add_match(db_session, sport=Sport.TENNIS, home="Zverev", away="Alcaraz")

        _add_bet(db_session, m1, stake=Decimal("25.00"))
        _add_bet(db_session, m2, stake=Decimal("15.00"), selection="away")

        exposure = fetch_sport_exposure(db_session)
        assert len(exposure) == 2
        # Sorted by stake descending
        assert exposure[0].sport == "football"
        assert exposure[0].total_stake == Decimal("25.00")


class TestPnLTimeseries:
    """Tests for fetch_pnl_timeseries."""

    def test_no_data(self, db_session):
        from bet_agent.tools.master_analysis import fetch_pnl_timeseries

        assert fetch_pnl_timeseries(db_session) == []

    def test_builds_timeseries(self, db_session):
        from bet_agent.tools.master_analysis import fetch_pnl_timeseries

        match = _add_match(db_session)
        now = datetime.now(timezone.utc)

        _add_bet(db_session, match, status=BetStatus.WON,
                 pnl=Decimal("10.00"), resolved_at=now)
        _add_bet(db_session, match, status=BetStatus.LOST,
                 pnl=Decimal("-5.00"), resolved_at=now, selection="away")

        ts = fetch_pnl_timeseries(db_session, days=7)
        assert len(ts) == 1
        assert ts[0]["pnl"] == 5.0  # 10 - 5
        assert ts[0]["cumulative_pnl"] == 5.0
        assert ts[0]["bets_count"] == 2


class TestFetchPending:
    """Tests for fetch_pending_for_human."""

    def test_returns_pending_bets(self, db_session):
        from bet_agent.tools.master_analysis import fetch_pending_for_human

        match = _add_match(db_session)
        _add_bet(db_session, match, status=BetStatus.PENDING)
        _add_bet(db_session, match, status=BetStatus.WON, selection="away",
                 pnl=Decimal("10.00"))  # Should not appear

        pending = fetch_pending_for_human(db_session)
        assert len(pending) == 1
        assert pending[0]["selection"] == "home"
        assert "bet_id" in pending[0]


class TestMarkBetPlaced:
    """Tests for mark_bet_placed_by_user (syndicate /placed command).

    The updated /placed flow captures actual odds/stake from the sportsbook,
    recalculates EV, deducts stake from bankroll, and warns on -EV.
    """

    def test_marks_pending_bet_with_actual_values(self, db_session):
        from bet_agent.tools.master_analysis import mark_bet_placed_by_user

        match = _add_match(db_session)
        bet = _add_bet(db_session, match, status=BetStatus.PENDING,
                       odds=Decimal("2.10"), stake=Decimal("10.00"))
        _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        result = mark_bet_placed_by_user(
            db_session, str(bet.id), "@olli",
            actual_odds=2.05, actual_stake=12.00,
        )

        assert result.get("success") is True
        assert result["placed_by"] == "@olli"
        assert result["odds"] == 2.05  # Actual odds stored
        assert result["stake_eur"] == 12.00  # Actual stake stored
        assert result["original_odds"] == 2.10  # Original for comparison
        assert bet.status == BetStatus.PLACED
        assert float(bet.odds_at_placement) == 2.05
        assert float(bet.stake_eur) == 12.00

    def test_bankroll_deducted_on_placement(self, db_session):
        from bet_agent.tools.master_analysis import mark_bet_placed_by_user

        match = _add_match(db_session)
        bet = _add_bet(db_session, match, status=BetStatus.PENDING, stake=Decimal("50.00"))
        ledger = _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        mark_bet_placed_by_user(
            db_session, str(bet.id), "@olli",
            actual_odds=2.10, actual_stake=50.00,
        )
        db_session.flush()

        assert ledger.balance == Decimal("950.00")  # 1000 - 50

    def test_warns_on_negative_ev(self, db_session):
        from bet_agent.tools.master_analysis import mark_bet_placed_by_user

        match = _add_match(db_session)
        # model_prob=0.55, original odds=2.10 → EV = 0.55*(2.10-1) - 0.45 = +0.155
        bet = _add_bet(db_session, match, status=BetStatus.PENDING,
                       odds=Decimal("2.10"), model_prob=Decimal("0.55"))
        _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        # Odds dropped to 1.50 → EV = 0.55*(1.50-1) - 0.45 = -0.175 → NEGATIVE
        result = mark_bet_placed_by_user(
            db_session, str(bet.id), "@olli",
            actual_odds=1.50, actual_stake=10.00,
        )

        assert result.get("success") is True
        assert result["ev"] < 0  # Negative EV!
        assert len(result["warnings"]) > 0
        assert "NEGATIVE EV" in result["warnings"][0]

    def test_no_warnings_on_positive_ev(self, db_session):
        from bet_agent.tools.master_analysis import mark_bet_placed_by_user

        match = _add_match(db_session)
        bet = _add_bet(db_session, match, status=BetStatus.PENDING,
                       odds=Decimal("2.10"), model_prob=Decimal("0.55"))
        _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        result = mark_bet_placed_by_user(
            db_session, str(bet.id), "@olli",
            actual_odds=2.10, actual_stake=10.00,
        )

        assert result.get("success") is True
        assert result["ev"] > 0
        assert len(result["warnings"]) == 0

    def test_rejects_already_settled(self, db_session):
        from bet_agent.tools.master_analysis import mark_bet_placed_by_user

        match = _add_match(db_session)
        bet = _add_bet(db_session, match, status=BetStatus.WON, pnl=Decimal("10.00"))

        result = mark_bet_placed_by_user(db_session, str(bet.id), "@olli")
        assert "error" in result

    def test_invalid_bet_id(self, db_session):
        from bet_agent.tools.master_analysis import mark_bet_placed_by_user

        result = mark_bet_placed_by_user(db_session, "not-a-uuid", "@olli")
        assert "error" in result

    def test_nonexistent_bet(self, db_session):
        from bet_agent.tools.master_analysis import mark_bet_placed_by_user

        result = mark_bet_placed_by_user(db_session, str(uuid.uuid4()), "@olli")
        assert "error" in result


class TestExpiryStaleBets:
    """Tests for expire_stale_bets (auto-void old PENDING tickets)."""

    def test_expires_old_pending_bet(self, db_session):
        from bet_agent.tools.settlement_engine import expire_stale_bets

        match = _add_match(db_session)
        bet = _add_bet(db_session, match, status=BetStatus.PENDING, stake=Decimal("25.00"))
        # PENDING bets never had stake deducted (deduct-at-placement model)
        ledger = _add_bankroll(db_session, LedgerType.REAL, Decimal("1000.00"))

        # Fake the placed_at to 2 hours ago
        bet.placed_at = datetime.now(timezone.utc) - timedelta(hours=2)
        db_session.flush()

        expired = expire_stale_bets(db_session, max_age_minutes=60)

        assert len(expired) == 1
        assert expired[0]["bet_id"] == str(bet.id)
        assert bet.status == BetStatus.VOID
        assert bet.pnl_eur == Decimal("0.00")
        # No refund — stake was never deducted for PENDING bets
        assert ledger.balance == Decimal("1000.00")

    def test_does_not_expire_fresh_bets(self, db_session):
        from bet_agent.tools.settlement_engine import expire_stale_bets

        match = _add_match(db_session)
        _add_bet(db_session, match, status=BetStatus.PENDING)
        _add_bankroll(db_session)

        expired = expire_stale_bets(db_session, max_age_minutes=60)
        assert len(expired) == 0

    def test_does_not_expire_settled_bets(self, db_session):
        from bet_agent.tools.settlement_engine import expire_stale_bets

        match = _add_match(db_session)
        bet = _add_bet(db_session, match, status=BetStatus.WON, pnl=Decimal("10.00"))
        bet.placed_at = datetime.now(timezone.utc) - timedelta(hours=2)
        db_session.flush()

        expired = expire_stale_bets(db_session, max_age_minutes=60)
        assert len(expired) == 0


class TestFormatting:
    """Tests for text formatting helpers."""

    def test_format_portfolio_text(self, db_session):
        from bet_agent.tools.master_analysis import (
            fetch_portfolio_summary,
            format_portfolio_text,
        )

        _add_bankroll(db_session, LedgerType.REAL, Decimal("1234.56"))

        summary = fetch_portfolio_summary(db_session)
        text = format_portfolio_text(summary)

        assert "1234.56" in text
        assert "REAL" in text

    def test_format_pending_text_empty(self):
        from bet_agent.tools.master_analysis import format_pending_text

        text = format_pending_text([])
        assert "No pending" in text

    def test_format_pending_text_with_bets(self, db_session):
        from bet_agent.tools.master_analysis import (
            fetch_pending_for_human,
            format_pending_text,
        )

        match = _add_match(db_session)
        _add_bet(db_session, match)

        pending = fetch_pending_for_human(db_session)
        text = format_pending_text(pending)

        assert "PENDING" in text
        assert "FC Bayern" in text


# ══════════════════════════════════════════════════════════════════════
# TELEGRAM BOT — Whitelist authentication
# ══════════════════════════════════════════════════════════════════════


class TestWhitelist:
    """Tests for Telegram bot whitelist enforcement."""

    def test_parse_allowed_ids(self):
        from bet_agent.interfaces.telegram_bot import parse_allowed_ids

        ids = parse_allowed_ids("123456,789012,345678")
        assert ids == {123456, 789012, 345678}

    def test_parse_with_spaces(self):
        from bet_agent.interfaces.telegram_bot import parse_allowed_ids

        ids = parse_allowed_ids(" 111 , 222 , 333 ")
        assert ids == {111, 222, 333}

    def test_parse_empty(self):
        from bet_agent.interfaces.telegram_bot import parse_allowed_ids

        assert parse_allowed_ids("") == set()
        assert parse_allowed_ids("   ") == set()

    def test_parse_with_negative_group_id(self):
        from bet_agent.interfaces.telegram_bot import parse_allowed_ids

        ids = parse_allowed_ids("123456,-100987654")
        assert 123456 in ids
        assert -100987654 in ids

    def test_is_authorized_with_valid_id(self):
        """Simulate authorized check with injected IDs."""
        import bet_agent.interfaces.telegram_bot as bot_mod

        original = bot_mod.ALLOWED_IDS
        try:
            bot_mod.ALLOWED_IDS = {12345, 67890}
            # Private DM: chat_id == user_id
            assert bot_mod.is_authorized(12345, chat_id=12345) is True
            assert bot_mod.is_authorized(99999, chat_id=99999) is False
        finally:
            bot_mod.ALLOWED_IDS = original

    def test_empty_whitelist_denies_all(self):
        """If ALLOWED_TELEGRAM_IDS is empty, nobody gets in."""
        import bet_agent.interfaces.telegram_bot as bot_mod

        original = bot_mod.ALLOWED_IDS
        try:
            bot_mod.ALLOWED_IDS = set()
            assert bot_mod.is_authorized(12345) is False
        finally:
            bot_mod.ALLOWED_IDS = original

    def test_private_dm_allowed(self):
        """Whitelisted user in a private DM (chat_id == user_id) is authorized."""
        import bet_agent.interfaces.telegram_bot as bot_mod

        original = bot_mod.ALLOWED_IDS
        try:
            bot_mod.ALLOWED_IDS = {12345}
            assert bot_mod.is_authorized(12345, chat_id=12345) is True
        finally:
            bot_mod.ALLOWED_IDS = original

    def test_official_group_allowed(self):
        """Whitelisted user in the official syndicate group is authorized."""
        import bet_agent.interfaces.telegram_bot as bot_mod

        original_ids = bot_mod.ALLOWED_IDS
        original_group = bot_mod.TELEGRAM_GROUP_ID
        try:
            bot_mod.ALLOWED_IDS = {12345}
            bot_mod.TELEGRAM_GROUP_ID = "-100999888"
            assert bot_mod.is_authorized(12345, chat_id=-100999888) is True
        finally:
            bot_mod.ALLOWED_IDS = original_ids
            bot_mod.TELEGRAM_GROUP_ID = original_group

    def test_random_group_blocked(self):
        """Whitelisted user in a random public group is BLOCKED (syndicate leak fix)."""
        import bet_agent.interfaces.telegram_bot as bot_mod

        original_ids = bot_mod.ALLOWED_IDS
        original_group = bot_mod.TELEGRAM_GROUP_ID
        try:
            bot_mod.ALLOWED_IDS = {12345}
            bot_mod.TELEGRAM_GROUP_ID = "-100999888"
            # User types /pnl in a random group — must be blocked!
            assert bot_mod.is_authorized(12345, chat_id=-100777666) is False
        finally:
            bot_mod.ALLOWED_IDS = original_ids
            bot_mod.TELEGRAM_GROUP_ID = original_group

    def test_no_group_configured_blocks_all_groups(self):
        """If no TELEGRAM_GROUP_ID is set, only private DMs work."""
        import bet_agent.interfaces.telegram_bot as bot_mod

        original_ids = bot_mod.ALLOWED_IDS
        original_group = bot_mod.TELEGRAM_GROUP_ID
        try:
            bot_mod.ALLOWED_IDS = {12345}
            bot_mod.TELEGRAM_GROUP_ID = ""
            # Any group → blocked
            assert bot_mod.is_authorized(12345, chat_id=-100777666) is False
            # Private DM → allowed
            assert bot_mod.is_authorized(12345, chat_id=12345) is True
        finally:
            bot_mod.ALLOWED_IDS = original_ids
            bot_mod.TELEGRAM_GROUP_ID = original_group

    def test_backwards_compat_no_chat_id(self):
        """If chat_id is not provided, fall back to user-only check."""
        import bet_agent.interfaces.telegram_bot as bot_mod

        original = bot_mod.ALLOWED_IDS
        try:
            bot_mod.ALLOWED_IDS = {12345}
            assert bot_mod.is_authorized(12345) is True
            assert bot_mod.is_authorized(99999) is False
        finally:
            bot_mod.ALLOWED_IDS = original

    def test_get_user_display_name(self):
        from bet_agent.interfaces.telegram_bot import get_user_display_name

        # None user
        assert get_user_display_name(None) == "Unknown"

        # Mock user with username
        class MockUser:
            def __init__(self, uid, username=None, first_name=None):
                self.id = uid
                self.username = username
                self.first_name = first_name

        assert get_user_display_name(MockUser(1, "olli")) == "@olli"
        assert get_user_display_name(MockUser(2, None, "Oliver")) == "Oliver"
        assert get_user_display_name(MockUser(3, None, None)) == "3"


# ══════════════════════════════════════════════════════════════════════
# NOTIFIER — Syndicate broadcasting
# ══════════════════════════════════════════════════════════════════════


class TestSyndicateBroadcaster:
    """Tests for TelegramSyndicateBroadcaster."""

    def test_get_target_ids(self):
        from bet_agent.tools.notifier import TelegramSyndicateBroadcaster

        bc = TelegramSyndicateBroadcaster(
            bot_token="fake",
            allowed_ids="111,222,333",
            group_id="-100999",
        )
        targets = bc._get_target_ids()
        assert "111" in targets
        assert "222" in targets
        assert "333" in targets
        assert "-100999" in targets
        assert len(targets) == 4

    def test_deduplicates_group_id(self):
        from bet_agent.tools.notifier import TelegramSyndicateBroadcaster

        bc = TelegramSyndicateBroadcaster(
            bot_token="fake",
            allowed_ids="111,-100999",
            group_id="-100999",
        )
        targets = bc._get_target_ids()
        assert targets.count("-100999") == 1

    def test_no_targets_returns_false(self):
        from bet_agent.tools.notifier import TelegramSyndicateBroadcaster

        bc = TelegramSyndicateBroadcaster(
            bot_token="fake",
            allowed_ids="",
            group_id="",
        )
        assert bc.send("test") is False

    def test_no_token_returns_false(self):
        from bet_agent.tools.notifier import TelegramSyndicateBroadcaster

        bc = TelegramSyndicateBroadcaster(
            bot_token="",
            allowed_ids="111",
        )
        assert bc.send("test") is False


class TestNotifierFactory:
    """Tests for get_notifiers() factory."""

    def test_console_always_present(self):
        """ConsoleNotifier is always in the list."""
        import os
        from bet_agent.tools.notifier import ConsoleNotifier, get_notifiers

        # Clear telegram env vars to ensure no telegram notifier
        old_token = os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        old_ids = os.environ.pop("ALLOWED_TELEGRAM_IDS", None)
        old_chat = os.environ.pop("TELEGRAM_CHAT_ID", None)
        try:
            notifiers = get_notifiers()
            assert any(isinstance(n, ConsoleNotifier) for n in notifiers)
        finally:
            if old_token:
                os.environ["TELEGRAM_BOT_TOKEN"] = old_token
            if old_ids:
                os.environ["ALLOWED_TELEGRAM_IDS"] = old_ids
            if old_chat:
                os.environ["TELEGRAM_CHAT_ID"] = old_chat

    def test_syndicate_mode_when_allowed_ids_set(self):
        """TelegramSyndicateBroadcaster is used when ALLOWED_TELEGRAM_IDS is set."""
        import os
        from bet_agent.tools.notifier import TelegramSyndicateBroadcaster, get_notifiers

        old_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        old_ids = os.environ.get("ALLOWED_TELEGRAM_IDS")
        old_chat = os.environ.pop("TELEGRAM_CHAT_ID", None)
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = "fake_token"
            os.environ["ALLOWED_TELEGRAM_IDS"] = "111,222"
            notifiers = get_notifiers()
            assert any(isinstance(n, TelegramSyndicateBroadcaster) for n in notifiers)
        finally:
            if old_token:
                os.environ["TELEGRAM_BOT_TOKEN"] = old_token
            else:
                os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            if old_ids:
                os.environ["ALLOWED_TELEGRAM_IDS"] = old_ids
            else:
                os.environ.pop("ALLOWED_TELEGRAM_IDS", None)
            if old_chat:
                os.environ["TELEGRAM_CHAT_ID"] = old_chat


class TestMasterAgentBridge:
    """Tests for the NL bridge to Master Agent."""

    def test_default_bridge_returns_acknowledgment(self):
        from unittest.mock import patch

        from bet_agent.interfaces.telegram_bot import MasterAgentBridge

        bridge = MasterAgentBridge()
        # Without LLM credentials the bridge returns a graceful fallback
        response = bridge.query("How is our NHL model?", "@olli")
        assert "Master Agent" in response

    def test_bridge_routes_through_llm(self):
        from unittest.mock import MagicMock, patch

        from bet_agent.interfaces.telegram_bot import MasterAgentBridge
        from bet_agent.llm.client import LLMClient, ProviderConfig, TierConfig

        bridge = MasterAgentBridge()
        fake_client = MagicMock(spec=LLMClient)
        fake_client.chat.return_value = "NHL xgboost Brier 0.19, ROI +2.1%"

        with patch.object(bridge, "_get_llm", return_value=fake_client):
            response = bridge.query("How is our NHL model?", "@olli")

        assert "NHL" in response
        assert "Brier" in response
        fake_client.chat.assert_called_once()

    def test_custom_bridge_injectable(self):
        from bet_agent.interfaces.telegram_bot import MasterAgentBridge, set_master_bridge

        class MockBridge(MasterAgentBridge):
            def query(self, message, user_name):
                return f"Mock response for {user_name}: {message}"

        import bet_agent.interfaces.telegram_bot as bot_mod
        original = bot_mod._master_bridge
        try:
            set_master_bridge(MockBridge())
            assert bot_mod._master_bridge.query("test", "@user") == "Mock response for @user: test"
        finally:
            bot_mod._master_bridge = original


class TestInlineKeyboardConstants:
    """Tests for InlineKeyboard callback data prefixes and conversation states."""

    def test_callback_prefixes_defined(self):
        from bet_agent.interfaces.telegram_bot import (
            CALLBACK_PLACE_CUSTOM,
            CALLBACK_PLACE_STD,
        )

        assert CALLBACK_PLACE_STD.startswith("place_std:")
        assert CALLBACK_PLACE_CUSTOM.startswith("place_cst:")

    def test_conversation_states_defined(self):
        from bet_agent.interfaces.telegram_bot import (
            CONV_AWAITING_ODDS,
            CONV_AWAITING_STAKE,
            CONV_TIMEOUT_SECONDS,
        )

        assert CONV_AWAITING_ODDS == 0
        assert CONV_AWAITING_STAKE == 1
        assert CONV_TIMEOUT_SECONDS == 300  # 5 minutes

    def test_callback_data_format_with_bet_id(self):
        """Callback data should safely encode bet_id after the prefix."""
        from bet_agent.interfaces.telegram_bot import CALLBACK_PLACE_STD

        bet_id = "abc123de-f456-7890-abcd-ef1234567890"
        data = f"{CALLBACK_PLACE_STD}{bet_id}"
        assert data.startswith("place_std:")
        extracted = data[len(CALLBACK_PLACE_STD):]
        assert extracted == bet_id


class TestAlertDigest:
    """Tests for the alert digest/batching system."""

    def test_queue_and_clear(self):
        import bet_agent.interfaces.telegram_bot as bot_mod

        # Drain any existing items
        while not bot_mod._alert_queue.empty():
            bot_mod._alert_queue.get_nowait()
        try:
            bot_mod.queue_alert("Test alert 1")
            bot_mod.queue_alert("Test alert 2")
            assert bot_mod._alert_queue.qsize() == 2
            assert bot_mod._alert_queue.get_nowait() == "Test alert 1"
            assert bot_mod._alert_queue.get_nowait() == "Test alert 2"
        finally:
            # Clean up
            while not bot_mod._alert_queue.empty():
                bot_mod._alert_queue.get_nowait()

    def test_digest_interval_configured(self):
        from bet_agent.interfaces.telegram_bot import _DIGEST_INTERVAL_SECONDS

        assert _DIGEST_INTERVAL_SECONDS == 600  # 10 minutes

    def test_sync_fetch_pending_data_returns_list(self, db_session):
        """_sync_fetch_pending_data returns structured dicts (for InlineKeyboard)."""
        from bet_agent.tools.master_analysis import fetch_pending_for_human

        match = _add_match(db_session)
        _add_bet(db_session, match, status=BetStatus.PENDING)

        pending = fetch_pending_for_human(db_session)
        assert isinstance(pending, list)
        assert len(pending) == 1
        # Keys needed for InlineKeyboard rendering
        assert "bet_id" in pending[0]
        assert "match" in pending[0]
        assert "selection" in pending[0]
        assert "odds" in pending[0]
        assert "stake_eur" in pending[0]
