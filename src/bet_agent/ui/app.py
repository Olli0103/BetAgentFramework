"""OpenClaw Watchtower — Streamlit Control Tower.

Institutional-grade monitoring dashboard for the BetAgent MAS.
Accessible on the local network via http://<mac-mini-ip>:8501

Launch:
    streamlit run src/bet_agent/ui/app.py --server.address 0.0.0.0

Features:
    - Global sport filter (sidebar) that flows into every tab
    - Live pipeline Kanban with sport-aware KPIs
    - Portfolio PnL with sport breakdown
    - MLOps model health with killswitch indicators
    - Agent status grid
    - Bet execution with readiness gate status
    - Live log viewer with level filter and search

Charts: Plotly with dark theme, hover tooltips, fill-to-zero.
Auto-refresh: streamlit-autorefresh (proper component, no meta-refresh hack).

Golden Rule: This file contains ZERO betting logic. Read-only DB queries only.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

from sqlalchemy import func, select

# ── Page config (must be first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="OpenClaw Watchtower",
    page_icon="\U0001f3af",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for tighter, more responsive layout ───────────────────
st.markdown("""
<style>
/* Tighter metrics */
[data-testid="stMetric"] {
    padding: 8px 12px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
}
/* Compact Kanban cards */
.kanban-card {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    border-left: 3px solid #42a5f5;
    font-size: 0.85rem;
}
.kanban-card.vetoed { border-left-color: #ff1744; }
.kanban-card.approved { border-left-color: #00c853; }
.kanban-card.settled-won { border-left-color: #00c853; }
.kanban-card.settled-lost { border-left-color: #ff1744; }
.kanban-card.pending { border-left-color: #ffc107; }
/* Sidebar branding */
[data-testid="stSidebar"] [data-testid="stMarkdown"] h1 {
    font-size: 1.3rem !important;
}
/* Log viewer */
.log-line { font-family: monospace; font-size: 0.78rem; line-height: 1.5; }
.log-WARNING { color: #ffc107; }
.log-ERROR { color: #ff1744; }
.log-INFO { color: #90caf9; }
.log-DEBUG { color: #666; }
/* Readiness badge */
.readiness-pass { color: #00c853; font-weight: 600; }
.readiness-fail { color: #ff1744; font-weight: 600; }
/* Tab indicator fix */
button[data-baseweb="tab"] { font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh (every 30s) ────────────────────────────────────────
if HAS_AUTOREFRESH:
    st_autorefresh(interval=30_000, limit=None, key="global_autorefresh")

# ── Database connection ──────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    st.error(
        "DATABASE_URL not set. Export it before launching:\n\n"
        "```\nexport DATABASE_URL=postgresql://...\nstreamlit run src/bet_agent/ui/app.py\n```"
    )
    st.stop()

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
    TeamAlias,
)
from bet_agent.db.session import get_session
from bet_agent.ui.helpers import SPORT_EMOJI, get_match_display, resolve_display_name


# ── Working window: 07:00 UTC → 07:00 UTC next day ──────────────────

def _get_working_window() -> tuple[datetime, datetime]:
    """Return the current working window (07:00 UTC to 07:00 UTC next day)."""
    now = datetime.now(timezone.utc)
    today_7am = datetime.combine(now.date(), time(7, 0), tzinfo=timezone.utc)

    if now >= today_7am:
        return today_7am, today_7am + timedelta(days=1)
    else:
        yesterday_7am = today_7am - timedelta(days=1)
        return yesterday_7am, today_7am


# ── In-memory log handler for the Logs tab ───────────────────────────

class _RingBufferHandler(logging.Handler):
    """Captures log records into a fixed-size ring buffer for the UI."""

    _MAX = 500  # Keep last 500 entries

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)
        if len(self.records) > self._MAX:
            self.records = self.records[-self._MAX:]


# Singleton: attach once to the root 'bet_agent' logger
_log_handler: _RingBufferHandler | None = None


def _ensure_log_handler() -> _RingBufferHandler:
    global _log_handler
    if _log_handler is None:
        _log_handler = _RingBufferHandler()
        _log_handler.setLevel(logging.DEBUG)
        _log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger("bet_agent").addHandler(_log_handler)
    return _log_handler


_ensure_log_handler()


# ── Cached data loaders ──────────────────────────────────────────────

@st.cache_data(ttl=60)
def _cached_pnl_timeseries(days: int, ledger_type_value: str | None) -> list[dict]:
    from bet_agent.tools.master_analysis import fetch_pnl_timeseries
    lt = LedgerType(ledger_type_value) if ledger_type_value else None
    with get_session() as sess:
        return fetch_pnl_timeseries(sess, days=days, ledger_type=lt)


@st.cache_data(ttl=60)
def _cached_sport_exposure() -> list[dict]:
    from bet_agent.tools.master_analysis import fetch_sport_exposure
    with get_session() as sess:
        results = fetch_sport_exposure(sess)
        return [
            {"sport": e.sport, "pending_count": e.pending_count,
             "total_stake": float(e.total_stake), "avg_odds": float(e.avg_odds),
             "avg_ev": float(e.avg_ev)}
            for e in results
        ]


@st.cache_data(ttl=60)
def _cached_model_health(sport: str | None) -> list[dict]:
    from bet_agent.tools.master_analysis import fetch_model_health
    with get_session() as sess:
        results = fetch_model_health(sess, sport=sport)
        return [
            {"model_name": r.model_name, "latest_brier": r.latest_brier,
             "latest_roi": r.latest_roi, "record_win": r.record_win,
             "record_loss": r.record_loss, "total_bets": r.total_bets,
             "trend": r.trend, "is_degraded": r.is_degraded}
            for r in results
        ]


@st.cache_data(ttl=30)
def _cached_recent_activity() -> dict:
    from bet_agent.tools.master_analysis import fetch_recent_activity
    with get_session() as sess:
        a = fetch_recent_activity(sess)
        return {
            "predictions_today": a.predictions_today,
            "settled_today": a.settled_today,
            "last_settlement_pnl": float(a.last_settlement_pnl),
            "approved_today": a.approved_today,
            "vetoed_today": a.vetoed_today,
            "placed_today": a.placed_today,
        }


@st.cache_data(ttl=60)
def _cached_brier_history(days: int) -> list[dict]:
    with get_session() as sess:
        metrics = list(
            sess.execute(
                select(ModelMetrics)
                .where(ModelMetrics.date >= date.today() - timedelta(days=days))
                .order_by(ModelMetrics.date)
            ).scalars().all()
        )
        return [
            {"model_name": m.model_name, "date": m.date.isoformat(),
             "brier_score": float(m.brier_score)}
            for m in metrics
        ]


# ── Plotly dark theme helper ─────────────────────────────────────────

_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    font=dict(size=12),
    hovermode="x unified",
)


def _plotly_layout(**overrides):
    """Return a dark-theme Plotly layout with overrides."""
    layout = dict(_PLOTLY_LAYOUT)
    layout.update(overrides)
    return layout


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("\U0001f3af OpenClaw Watchtower")
st.sidebar.caption("Multi-Agent Sports Betting System")
st.sidebar.divider()

# ── Global Sport Filter ──────────────────────────────────────────────
ALL_SPORTS = [s.value for s in Sport]
SPORT_OPTIONS = {s: f"{SPORT_EMOJI.get(s, '')} {s.replace('_', ' ').title()}" for s in ALL_SPORTS}

selected_sports = st.sidebar.multiselect(
    "Filter Sports",
    options=ALL_SPORTS,
    default=ALL_SPORTS,
    format_func=lambda s: SPORT_OPTIONS[s],
    help="Filter all tabs by sport. Deselect to hide.",
)

# If nothing selected, show all (UX safety net)
if not selected_sports:
    selected_sports = ALL_SPORTS

selected_sport_enums = [Sport(s) for s in selected_sports]

st.sidebar.divider()

# Quick bankroll display
try:
    with get_session() as _sess:
        _ledgers = list(_sess.execute(select(BankrollLedger)).scalars().all())
        for _l in _ledgers:
            _icon = "\U0001f4b5" if _l.ledger_type == LedgerType.REAL else "\U0001f4dd"
            st.sidebar.metric(
                f"{_icon} {_l.ledger_type.value.upper()}",
                f"{_l.balance:.2f} EUR",
            )
except Exception:
    st.sidebar.warning("DB not reachable")

st.sidebar.divider()

# Working window info
w_start, w_end = _get_working_window()
st.sidebar.caption(
    f"Window: {w_start.strftime('%H:%M')} \u2013 {w_end.strftime('%H:%M')} UTC "
    f"({w_start.strftime('%Y-%m-%d')})"
)

if HAS_AUTOREFRESH:
    st.sidebar.caption("\u26a1 Auto-refresh 30s")
else:
    st.sidebar.caption("Install streamlit-autorefresh for auto-refresh")


# ── Helper: sport-filtered match query ───────────────────────────────

def _get_filtered_matches(sess, w_start, w_end):
    """Get matches in working window filtered by selected sports."""
    return list(
        sess.execute(
            select(Match).where(
                Match.scheduled_at >= w_start,
                Match.scheduled_at <= w_end,
                Match.sport.in_(selected_sport_enums),
            ).order_by(Match.scheduled_at)
        ).scalars().all()
    )


# ── Tab layout ───────────────────────────────────────────────────────

tab_cmd, tab_portfolio, tab_mlops, tab_agents, tab_execution, tab_logs = st.tabs([
    "\U0001f3af Command Center",
    "\U0001f4c8 Portfolio & PnL",
    "\U0001f9e0 MLOps & Health",
    "\U0001f916 Agent Status",
    "\U0001f4b0 Bet Execution",
    "\U0001f4dc Logs",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: COMMAND CENTER — KPI bar + Pipeline Kanban
# ══════════════════════════════════════════════════════════════════════

with tab_cmd:
    st.header("Command Center")

    # Sport filter summary
    if len(selected_sports) < len(ALL_SPORTS):
        active_emojis = " ".join(SPORT_EMOJI.get(s, "") for s in selected_sports)
        st.caption(f"Filtered: {active_emojis} ({len(selected_sports)}/{len(ALL_SPORTS)} sports)")

    w_start, w_end = _get_working_window()

    try:
        with get_session() as sess:
            matches = _get_filtered_matches(sess, w_start, w_end)
            match_ids = [m.id for m in matches]

            predictions = []
            if match_ids:
                predictions = list(
                    sess.execute(
                        select(Prediction).where(Prediction.match_id.in_(match_ids))
                    ).scalars().all()
                )

            # ── KPI bar ──────────────────────────────────────────
            n_matches = len(matches)
            n_predictions = len(predictions)
            n_pending = sum(1 for p in predictions if p.status == PredictionStatus.PENDING)
            n_approved = sum(1 for p in predictions if p.status == PredictionStatus.APPROVED)
            n_vetoed = sum(1 for p in predictions if p.status == PredictionStatus.VETOED)
            n_placed = sum(1 for p in predictions if p.status == PredictionStatus.PLACED)

            settled_bets = []
            if match_ids:
                settled_bets = list(
                    sess.execute(
                        select(PlacedBet).where(
                            PlacedBet.match_id.in_(match_ids),
                            PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST, BetStatus.VOID]),
                        )
                    ).scalars().all()
                )
            n_settled = len(settled_bets)
            settled_pnl = sum(float(b.pnl_eur or 0) for b in settled_bets)

            n_positive_ev = sum(
                1 for p in predictions
                if p.status == PredictionStatus.APPROVED and p.ev and p.ev > 0
            )
            n_has_odds = sum(
                1 for p in predictions
                if p.status == PredictionStatus.APPROVED and p.best_odds is not None
            )

            # Pending PlacedBets (from pipeline bridge)
            n_pending_bets = 0
            if match_ids:
                n_pending_bets = sess.execute(
                    select(func.count(PlacedBet.id)).where(
                        PlacedBet.match_id.in_(match_ids),
                        PlacedBet.status == BetStatus.PENDING,
                    )
                ).scalar() or 0

            k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
            k1.metric("Matches", n_matches)
            k2.metric("Predictions", n_predictions)
            k3.metric("Pending ML", n_pending)
            k4.metric("Approved", n_approved, delta=f"+{n_approved}" if n_approved else None)
            k5.metric("+EV/Odds", f"{n_positive_ev}/{n_has_odds}",
                      help="Positive EV / Has shopped odds")
            k6.metric("Vetoed", n_vetoed)
            k7.metric("Awaiting", n_pending_bets, help="PlacedBet(PENDING) via bridge")
            k8.metric("Settled", n_settled, delta=f"{settled_pnl:+.2f}\u20ac" if settled_bets else None)

            st.divider()

            # ── Kanban pipeline ──────────────────────────────────
            kanban: dict[str, list] = {
                "NOT STARTED": [],
                "PENDING ML": [],
                "VETOED": [],
                "APPROVED": [],
                "SETTLED": [],
            }

            pred_match_ids = {p.match_id for p in predictions}

            for m in matches:
                if m.id not in pred_match_ids:
                    kanban["NOT STARTED"].append(m)

            for p in predictions:
                match = sess.get(Match, p.match_id)
                entry = {"prediction": p, "match": match}

                if p.status == PredictionStatus.PENDING:
                    kanban["PENDING ML"].append(entry)
                elif p.status == PredictionStatus.VETOED:
                    kanban["VETOED"].append(entry)
                elif p.status in (PredictionStatus.APPROVED, PredictionStatus.PLACED):
                    bet = sess.execute(
                        select(PlacedBet).where(
                            PlacedBet.match_id == p.match_id,
                            PlacedBet.selection == p.selection,
                            PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST, BetStatus.VOID]),
                        )
                    ).scalar_one_or_none()

                    if bet:
                        entry["bet"] = bet
                        kanban["SETTLED"].append(entry)
                    else:
                        kanban["APPROVED"].append(entry)

            # Render Kanban columns
            cols = st.columns(5)
            headers = list(kanban.keys())
            colors = ["\U0001f7e4", "\U0001f7e1", "\U0001f534", "\U0001f7e2", "\u2705"]

            for col, header, color in zip(cols, headers, colors):
                with col:
                    st.subheader(f"{color} {header}")
                    st.caption(f"{len(kanban[header])} items")
                    st.divider()

                    items = kanban[header]
                    for item in items[:20]:
                        if header == "NOT STARTED":
                            m = item
                            disp = get_match_display(sess, m)
                            st.markdown(
                                f"<div class='kanban-card'>"
                                f"{disp['sport_emoji']} <b>{disp['home']}</b> vs <b>{disp['away']}</b><br>"
                                f"<small>{disp['sport']} | {disp['league']} | {disp['kickoff']}</small>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        elif header == "VETOED":
                            p = item["prediction"]
                            m = item["match"]
                            disp = get_match_display(sess, m)
                            st.markdown(
                                f"<div class='kanban-card vetoed'>"
                                f"{disp['sport_emoji']} <b>{disp['home']}</b> vs <b>{disp['away']}</b><br>"
                                f"<code>{p.selection}</code> | EV: {p.ev:.4f}<br>"
                                f"<small><i>{p.veto_reason or 'N/A'}</i></small>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        elif header == "SETTLED":
                            p = item["prediction"]
                            m = item["match"]
                            b = item["bet"]
                            disp = get_match_display(sess, m)
                            pnl_str = f"{b.pnl_eur:+.2f}" if b.pnl_eur else "0.00"
                            css_class = "settled-won" if b.status == BetStatus.WON else "settled-lost"
                            status_icon = {
                                BetStatus.WON: "\U0001f7e2",
                                BetStatus.LOST: "\U0001f534",
                                BetStatus.VOID: "\u26aa",
                            }.get(b.status, "\u2753")
                            st.markdown(
                                f"<div class='kanban-card {css_class}'>"
                                f"{status_icon} <b>{disp['home']}</b> vs <b>{disp['away']}</b><br>"
                                f"<code>{p.selection}</code> @ {b.odds_at_placement:.2f}<br>"
                                f"<b>{pnl_str} \u20ac</b> | {b.status.value.upper()}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # PENDING ML or APPROVED
                            p = item["prediction"]
                            m = item["match"]
                            disp = get_match_display(sess, m)
                            css = "approved" if header == "APPROVED" else "pending"
                            odds_str = f"@ {float(p.best_odds):.2f}" if p.best_odds else ""
                            ev_color = "#00c853" if p.ev > 0 else "#ff1744"
                            st.markdown(
                                f"<div class='kanban-card {css}'>"
                                f"{disp['sport_emoji']} <b>{disp['home']}</b> vs <b>{disp['away']}</b><br>"
                                f"<code>{p.selection}</code> {odds_str} | "
                                f"<span style='color:{ev_color}'>EV: {p.ev:.4f}</span><br>"
                                f"<small>Prob: {p.model_prob:.1%} | {p.model_source}</small>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    if len(items) > 20:
                        st.caption(f"... and {len(items) - 20} more")

    except Exception as e:
        st.error(f"Pipeline query failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 2: PORTFOLIO & PnL
# ══════════════════════════════════════════════════════════════════════

with tab_portfolio:
    st.header("Portfolio & PnL")

    try:
        fc1, fc2 = st.columns(2)
        with fc1:
            lookback = st.selectbox("Lookback", [7, 14, 30, 60, 90], index=2, key="pf_lookback")
        with fc2:
            ledger_filter = st.selectbox("Ledger", ["Both", "REAL", "PAPER"], key="pf_ledger")

        lt_val = None
        if ledger_filter == "REAL":
            lt_val = LedgerType.REAL.value
        elif ledger_filter == "PAPER":
            lt_val = LedgerType.PAPER.value

        ts_data = _cached_pnl_timeseries(lookback, lt_val)

        if ts_data:
            dates = [d["date"] for d in ts_data]
            daily_pnl = [d["pnl"] for d in ts_data]
            cumulative = [d["cumulative_pnl"] for d in ts_data]
            counts = [d["bets_count"] for d in ts_data]

            # Summary metrics (top of tab)
            total_pnl = sum(daily_pnl)
            total_bets = sum(counts)
            best_day = max(daily_pnl) if daily_pnl else 0
            worst_day = min(daily_pnl) if daily_pnl else 0
            win_days = sum(1 for d in daily_pnl if d > 0)
            total_days = len([d for d in daily_pnl if d != 0])
            win_rate = (win_days / total_days * 100) if total_days > 0 else 0

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Total PnL", f"{total_pnl:+.2f}\u20ac")
            mc2.metric("Bets", total_bets)
            mc3.metric("Win Rate", f"{win_rate:.0f}%", help="Days with positive PnL")
            mc4.metric("Best Day", f"{best_day:+.2f}\u20ac")
            mc5.metric("Worst Day", f"{worst_day:+.2f}\u20ac")

            # ── Cumulative PnL chart ────────────────────────────
            if HAS_PLOTLY:
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=dates, y=cumulative,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(0, 200, 83, 0.15)",
                    line=dict(color="#00c853", width=2),
                    name="Cumulative PnL",
                    hovertemplate="%{x}<br>PnL: %{y:+.2f}\u20ac<extra></extra>",
                ))
                fig_cum.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_cum.update_layout(**_plotly_layout(yaxis_title="EUR", height=300))
                st.plotly_chart(fig_cum, use_container_width=True)

                # ── Daily PnL bar chart ─────────────────────────
                bar_colors = ["#00c853" if v >= 0 else "#ff1744" for v in daily_pnl]
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(
                    x=dates, y=daily_pnl,
                    marker_color=bar_colors,
                    name="Daily PnL",
                    hovertemplate="%{x}<br>PnL: %{y:+.2f}\u20ac<extra></extra>",
                ))
                fig_daily.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_daily.update_layout(**_plotly_layout(yaxis_title="EUR", height=250))
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.line_chart({d["date"]: d["cumulative_pnl"] for d in ts_data})
                st.bar_chart({d["date"]: d["pnl"] for d in ts_data})
        else:
            st.info("No settled bets in the selected period.")

        # ── Sport exposure ───────────────────────────────────────
        st.subheader("Exposure by Sport")

        exposure = _cached_sport_exposure()
        # Filter by selected sports
        exposure = [e for e in exposure if e["sport"] in selected_sports]

        if exposure:
            if HAS_PLOTLY:
                sports = [f"{SPORT_EMOJI.get(e['sport'], '')} {e['sport']}" for e in exposure]
                stakes = [e["total_stake"] for e in exposure]
                fig_exp = go.Figure()
                fig_exp.add_trace(go.Bar(
                    x=sports, y=stakes,
                    marker_color="#42a5f5",
                    hovertemplate="%{x}<br>Stake: %{y:.2f}\u20ac<extra></extra>",
                ))
                fig_exp.update_layout(**_plotly_layout(yaxis_title="EUR Staked", height=280))
                st.plotly_chart(fig_exp, use_container_width=True)

            # Sport detail cards in compact columns
            exp_cols = st.columns(min(len(exposure), 4))
            for i, e in enumerate(exposure):
                with exp_cols[i % len(exp_cols)]:
                    emoji = SPORT_EMOJI.get(e["sport"], "\U0001f3c6")
                    st.markdown(
                        f"**{emoji} {e['sport'].replace('_', ' ').title()}**\n\n"
                        f"{e['pending_count']} bets | {e['total_stake']:.2f}\u20ac\n\n"
                        f"Avg odds: {e['avg_odds']:.2f} | EV: {e['avg_ev']:.4f}"
                    )
        else:
            st.info("No exposure for selected sports.")

    except Exception as e:
        st.error(f"Portfolio query failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 3: MLOps & MODEL HEALTH
# ══════════════════════════════════════════════════════════════════════

with tab_mlops:
    st.header("MLOps & Model Health")

    try:
        sport_filter = st.selectbox(
            "Filter by Sport",
            ["All"] + [s.value for s in Sport],
            key="mlops_sport",
        )
        sf = None if sport_filter == "All" else sport_filter

        health_reports = _cached_model_health(sf)

        if health_reports:
            # Killswitch alert
            degraded = [r for r in health_reports if r["is_degraded"]]
            if degraded:
                st.error(
                    f"\U0001f6a8 **KILLSWITCH ACTIVE** \u2014 "
                    f"{len(degraded)} model(s) degraded: "
                    + ", ".join(f"`{r['model_name']}`" for r in degraded)
                    + "\n\nBetting HALTED for these models until human retraining approval."
                )

            # Model cards
            for report in health_reports:
                status_icon = "\U0001f534" if report["is_degraded"] else "\U0001f7e2"
                trend_icon = {
                    "improving": "\u2197\ufe0f",
                    "stable": "\u27a1\ufe0f",
                    "declining": "\u2198\ufe0f",
                }.get(report["trend"], "\u2753")

                with st.expander(
                    f"{status_icon} {report['model_name']} | "
                    f"Brier: {report['latest_brier']:.4f} | "
                    f"ROI: {report['latest_roi']:+.1f}% | "
                    f"{trend_icon} {report['trend']}",
                    expanded=report["is_degraded"],
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Brier Score", f"{report['latest_brier']:.4f}",
                              delta="DEGRADED" if report["latest_brier"] > 0.22 else "OK",
                              delta_color="inverse" if report["latest_brier"] > 0.22 else "normal")
                    c2.metric("ROI", f"{report['latest_roi']:+.1f}%",
                              delta="DEGRADED" if report["latest_roi"] < -5 else "OK",
                              delta_color="inverse" if report["latest_roi"] < -5 else "normal")
                    c3.metric("Record", f"{report['record_win']}W-{report['record_loss']}L")
                    c4.metric("Total Bets", report["total_bets"])

            # Brier score history chart
            st.subheader("Brier Score History")
            metrics = _cached_brier_history(30)

            if metrics:
                if HAS_PLOTLY:
                    model_series: dict[str, tuple[list, list]] = {}
                    for m in metrics:
                        if m["model_name"] not in model_series:
                            model_series[m["model_name"]] = ([], [])
                        model_series[m["model_name"]][0].append(m["date"])
                        model_series[m["model_name"]][1].append(m["brier_score"])

                    fig_brier = go.Figure()
                    for model_name, (dates_b, scores) in model_series.items():
                        fig_brier.add_trace(go.Scatter(
                            x=dates_b, y=scores,
                            mode="lines+markers",
                            name=model_name,
                            hovertemplate="%{x}<br>Brier: %{y:.4f}<extra></extra>",
                        ))

                    fig_brier.add_hline(
                        y=0.22, line_dash="dash", line_color="#ff1744",
                        annotation_text="Degradation (0.22)",
                        annotation_position="top right",
                    )
                    fig_brier.update_layout(
                        **_plotly_layout(
                            yaxis_title="Brier Score",
                            height=350,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        )
                    )
                    st.plotly_chart(fig_brier, use_container_width=True)
                else:
                    chart: dict[str, dict[str, float]] = defaultdict(dict)
                    for m in metrics:
                        chart[m["date"]][m["model_name"]] = m["brier_score"]
                    st.line_chart(chart)

        else:
            st.warning("No model metrics yet.")
            st.markdown("**Prerequisites for model metrics:**")
            try:
                with get_session() as diag_sess:
                    pred_count = diag_sess.execute(
                        select(func.count(Prediction.id))
                    ).scalar() or 0
                    placed_count = diag_sess.execute(
                        select(func.count(PlacedBet.id)).where(
                            PlacedBet.status == BetStatus.PLACED
                        )
                    ).scalar() or 0
                    settled_count = diag_sess.execute(
                        select(func.count(PlacedBet.id)).where(
                            PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST, BetStatus.VOID])
                        )
                    ).scalar() or 0
                    metrics_count = diag_sess.execute(
                        select(func.count(ModelMetrics.id))
                    ).scalar() or 0

                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Predictions", pred_count,
                           delta="OK" if pred_count > 0 else "NEEDED",
                           delta_color="normal" if pred_count > 0 else "inverse")
                dc2.metric("Placed Bets", placed_count,
                           delta="OK" if placed_count > 0 else "NEEDED",
                           delta_color="normal" if placed_count > 0 else "inverse")
                dc3.metric("Settled Bets", settled_count,
                           delta="OK" if settled_count >= 10 else f"Need {max(0, 10 - settled_count)} more",
                           delta_color="normal" if settled_count >= 10 else "inverse")
                dc4.metric("Metric Records", metrics_count)

                st.markdown(
                    "**Pipeline:** Predictions \u2192 Place bets \u2192 Settle results \u2192 "
                    "Auditor morning audit (05:00 UTC) \u2192 Metrics\n\n"
                    "Metrics are calculated after the Auditor settles finished matches "
                    "and evaluates Brier scores + ROI across at least 10 settled bets."
                )
            except Exception:
                st.info("Run the Auditor morning audit to generate metrics.")

    except Exception as e:
        st.error(f"MLOps query failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 4: AGENT STATUS — Health cards for all 9 agents
# ══════════════════════════════════════════════════════════════════════

AGENT_DEFS = [
    {"id": "master", "name": "Master Agent", "role": "Portfolio Manager & Concierge",
     "emoji": "\U0001f451", "tier": "Tier 1"},
    {"id": "scout", "name": "Scout Agent", "role": "Data Gatherer & Live Monitor",
     "emoji": "\U0001f50d", "tier": "Tier 2"},
    {"id": "data_janitor", "name": "Data Janitor", "role": "ETL & Name Resolution",
     "emoji": "\U0001f9f9", "tier": "Tier 2"},
    {"id": "quant", "name": "Quant Agent", "role": "Probability & ML Engine",
     "emoji": "\U0001f4ca", "tier": "Tier 2"},
    {"id": "devils_advocate", "name": "Devil's Advocate", "role": "Veto Power",
     "emoji": "\U0001f608", "tier": "Tier 1"},
    {"id": "line_shopper", "name": "Line Shopper", "role": "Odds Maximizer",
     "emoji": "\U0001f6d2", "tier": "Tier 2"},
    {"id": "risk_manager", "name": "Risk Manager", "role": "Bankroll Sizing",
     "emoji": "\U0001f6e1\ufe0f", "tier": "Tier 2"},
    {"id": "moonshot", "name": "Moonshot Architect", "role": "Parlay Builder",
     "emoji": "\U0001f680", "tier": "Tier 1"},
    {"id": "auditor", "name": "Auditor", "role": "Performance Evaluator",
     "emoji": "\U0001f4d1", "tier": "Tier 1"},
]

with tab_agents:
    st.header("Agent Status")

    try:
        with get_session() as sess:
            activity = _cached_recent_activity()

            agent_activity = {}

            recent_matches_with_odds = sess.execute(
                select(func.count(Match.id)).where(
                    Match.scheduled_at >= w_start,
                    Match.scheduled_at <= w_end,
                    Match.sport.in_(selected_sport_enums),
                )
            ).scalar() or 0
            agent_activity["scout"] = {
                "status": "active" if recent_matches_with_odds > 0 else "idle",
                "detail": f"{recent_matches_with_odds} matches in window",
            }

            agent_activity["quant"] = {
                "status": "active" if activity["predictions_today"] > 0 else "idle",
                "detail": f"{activity['predictions_today']} predictions today",
            }

            agent_activity["devils_advocate"] = {
                "status": "active" if activity["vetoed_today"] > 0 else "idle",
                "detail": f"{activity['vetoed_today']} vetoed today",
            }

            agent_activity["risk_manager"] = {
                "status": "active" if activity["approved_today"] > 0 else "idle",
                "detail": f"{activity['approved_today']} sized today",
            }

            agent_activity["auditor"] = {
                "status": "active" if activity["settled_today"] > 0 else "idle",
                "detail": f"{activity['settled_today']} settled today",
            }

            agent_activity["master"] = {
                "status": "active",
                "detail": "Orchestrating pipeline",
            }

            approved_with_odds = sess.execute(
                select(func.count(Prediction.id)).where(
                    Prediction.status == PredictionStatus.APPROVED,
                    Prediction.best_odds.isnot(None),
                )
            ).scalar() or 0
            agent_activity["line_shopper"] = {
                "status": "active" if approved_with_odds > 0 else "idle",
                "detail": f"{approved_with_odds} lines shopped",
            }

            alias_count = sess.execute(
                select(func.count(TeamAlias.id))
            ).scalar() or 0
            canonical_count = sess.execute(
                select(func.count(func.distinct(TeamAlias.canonical_name)))
            ).scalar() or 0
            agent_activity["data_janitor"] = {
                "status": "active" if recent_matches_with_odds > 0 else "idle",
                "detail": f"{alias_count} aliases, {canonical_count} canonical",
            }

            agent_activity["moonshot"] = {
                "status": "idle",
                "detail": "Waiting for approved singles",
            }

        # Render agent cards in 3x3 grid
        for row_start in range(0, len(AGENT_DEFS), 3):
            row_agents = AGENT_DEFS[row_start:row_start + 3]
            cols = st.columns(len(row_agents))

            for col, agent_def in zip(cols, row_agents):
                with col:
                    aid = agent_def["id"]
                    info = agent_activity.get(aid, {"status": "idle", "detail": ""})
                    status_dot = "\U0001f7e2" if info["status"] == "active" else "\u26aa"

                    st.markdown(
                        f"### {agent_def['emoji']} {agent_def['name']}\n\n"
                        f"{status_dot} **{info['status'].upper()}** | {agent_def['tier']}\n\n"
                        f"_{agent_def['role']}_\n\n"
                        f"{info['detail']}"
                    )
                    st.divider()

    except Exception as e:
        st.error(f"Agent status query failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 5: BET EXECUTION — Pending bets with readiness gate status
# ══════════════════════════════════════════════════════════════════════

with tab_execution:
    st.header("Bet Execution")
    st.caption(
        "Approved bets awaiting manual placement. "
        "Place the bet on the listed sportsbook, then confirm via Telegram /pending."
    )

    try:
        with get_session() as sess:
            # Get PENDING PlacedBets (from pipeline bridge) — sport filtered
            pending_query = (
                select(PlacedBet)
                .join(Match, Match.id == PlacedBet.match_id)
                .where(
                    PlacedBet.status == BetStatus.PENDING,
                    Match.sport.in_(selected_sport_enums),
                )
                .order_by(Match.scheduled_at)
            )
            pending_bets = list(sess.execute(pending_query).scalars().all())

            if not pending_bets:
                st.info("No bets awaiting execution. All clear.")
            else:
                # Split REAL / PAPER
                real_bets = [b for b in pending_bets if b.ledger_type == LedgerType.REAL]
                paper_bets = [b for b in pending_bets if b.ledger_type == LedgerType.PAPER]

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Total Pending", len(pending_bets))
                rc2.metric("REAL", len(real_bets))
                rc3.metric("PAPER", len(paper_bets))

                if real_bets:
                    st.success(f"\U0001f4b0 **{len(real_bets)} REAL bet(s) ready to place**")

                for bet in pending_bets:
                    match = sess.get(Match, bet.match_id)
                    if not match:
                        continue

                    disp = get_match_display(sess, match)

                    # Readiness gate check
                    pred = sess.execute(
                        select(Prediction).where(
                            Prediction.match_id == bet.match_id,
                            Prediction.market_type == bet.market_type,
                            Prediction.selection == bet.selection,
                        )
                    ).scalar_one_or_none()

                    readiness_html = ""
                    if pred:
                        from bet_agent.tools.notifier import check_bet_readiness
                        readiness = check_bet_readiness(pred, match, float(bet.stake_eur))
                        if readiness.is_ready:
                            readiness_html = "<span class='readiness-pass'>READY</span>"
                        else:
                            failed = ", ".join(readiness.failed_checks)
                            readiness_html = f"<span class='readiness-fail'>FAIL: {failed}</span>"

                    ledger_color = "#00c853" if bet.ledger_type == LedgerType.REAL else "#ffc107"
                    ledger_label = bet.ledger_type.value.upper()
                    card_css = "approved" if bet.ledger_type == LedgerType.REAL else "pending"

                    st.markdown(
                        f"<div class='kanban-card {card_css}'>"
                        f"{disp['sport_emoji']} <b>{disp['home']}</b> vs <b>{disp['away']}</b> "
                        f"<small>({disp['kickoff']} | {disp['league']})</small><br>"
                        f"<code>{bet.selection}</code> @ <b>{bet.odds_at_placement:.2f}</b> | "
                        f"Stake: <b>{bet.stake_eur:.2f}\u20ac</b> | "
                        f"EV: {bet.ev_at_placement:.4f} | "
                        f"<span style='color:{ledger_color}'>[{ledger_label}]</span> | "
                        f"Gate: {readiness_html}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Recently settled
            st.divider()
            st.subheader("Recently Settled")

            recent_settled = list(
                sess.execute(
                    select(PlacedBet)
                    .join(Match, Match.id == PlacedBet.match_id)
                    .where(
                        PlacedBet.status.in_([BetStatus.WON, BetStatus.LOST, BetStatus.VOID]),
                        Match.sport.in_(selected_sport_enums),
                    )
                    .order_by(PlacedBet.resolved_at.desc())
                    .limit(10)
                ).scalars().all()
            )

            if recent_settled:
                for bet in recent_settled:
                    match = sess.get(Match, bet.match_id)
                    if not match:
                        continue
                    disp = get_match_display(sess, match)
                    pnl_str = f"{bet.pnl_eur:+.2f}\u20ac" if bet.pnl_eur else ""
                    icon = {BetStatus.WON: "\U0001f7e2", BetStatus.LOST: "\U0001f534", BetStatus.VOID: "\u26aa"}.get(bet.status, "")
                    st.markdown(
                        f"{icon} **{disp['vs']}** \u2014 `{bet.selection}` @ {bet.odds_at_placement:.2f} "
                        f"| {bet.stake_eur:.2f}\u20ac | **{pnl_str}**"
                    )
            else:
                st.caption("No recently settled bets.")

    except Exception as e:
        st.error(f"Execution query failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 6: LOGS — Live log viewer
# ══════════════════════════════════════════════════════════════════════

with tab_logs:
    st.header("System Logs")

    lc1, lc2, lc3 = st.columns([1, 1, 2])
    with lc1:
        log_level = st.selectbox(
            "Min Level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
            key="log_level",
        )
    with lc2:
        log_limit = st.selectbox("Show last", [50, 100, 200, 500], index=1, key="log_limit")
    with lc3:
        log_search = st.text_input("Search", placeholder="Filter logs...", key="log_search")

    handler = _ensure_log_handler()
    level_num = getattr(logging, log_level)

    # Filter records
    filtered = [
        r for r in handler.records
        if r.levelno >= level_num
        and (not log_search or log_search.lower() in handler.format(r).lower())
    ]

    # Show newest first
    filtered = filtered[-log_limit:]
    filtered.reverse()

    if filtered:
        st.caption(f"Showing {len(filtered)} log entries (newest first)")

        log_lines = []
        for record in filtered:
            formatted = handler.format(record)
            css_class = f"log-{record.levelname}"
            # Escape HTML in log message
            import html as _html
            safe = _html.escape(formatted)
            log_lines.append(f"<div class='log-line {css_class}'>{safe}</div>")

        st.markdown("\n".join(log_lines), unsafe_allow_html=True)
    else:
        st.info(
            "No log entries yet. Logs appear here as the system runs.\n\n"
            "Tip: The log handler captures all `bet_agent.*` logger output."
        )
