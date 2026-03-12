"""OpenClaw Quant Command Center — Streamlit WebUI.

Institutional-grade Control Tower dashboard for the BetAgent MAS.
Accessible on the local network via http://<mac-mini-ip>:8501

Launch:
    streamlit run src/bet_agent/ui/app.py --server.address 0.0.0.0

Tabs:
    1. Command Center — KPI bar + Kanban pipeline with full team names
    2. Portfolio       — Time-series PnL, bankroll growth, sport exposure
    3. MLOps           — Model health, Brier Scores, ROI, killswitch indicators
    4. Agent Status    — Live health cards for all 9 agents
    5. Bet Execution   — Pending bets to place, with clear instructions

Charts: Plotly with dark theme, hover tooltips, fill-to-zero.
Auto-refresh: streamlit-autorefresh (proper component, no meta-refresh hack).

Golden Rule: This file contains ZERO betting logic. Read-only DB queries only.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

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
    page_title="OpenClaw Command Center",
    page_icon="\U0001f3af",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

st.sidebar.title("\U0001f3af OpenClaw MAS")
st.sidebar.markdown("**Quant Trading Syndicate**")
st.sidebar.divider()

# Quick bankroll display
try:
    with get_session() as _sess:
        _ledgers = list(_sess.execute(select(BankrollLedger)).scalars().all())
        for _l in _ledgers:
            _icon = "\U0001f4b5" if _l.ledger_type == LedgerType.REAL else "\U0001f4dd"
            st.sidebar.metric(
                f"{_icon} {_l.ledger_type.value.upper()} Balance",
                f"{_l.balance:.2f} EUR",
            )
except Exception:
    st.sidebar.warning("DB not reachable")

st.sidebar.divider()

# Working window info
w_start, w_end = _get_working_window()
st.sidebar.caption(
    f"Working window: {w_start.strftime('%H:%M')} \u2013 {w_end.strftime('%H:%M')} UTC\n\n"
    f"({w_start.strftime('%Y-%m-%d')} \u2013 {w_end.strftime('%Y-%m-%d')})"
)

if HAS_AUTOREFRESH:
    st.sidebar.caption("Auto-refreshes every 30s")
else:
    st.sidebar.caption("Install streamlit-autorefresh for auto-refresh")

# ── Tab layout ───────────────────────────────────────────────────────

tab_cmd, tab_portfolio, tab_mlops, tab_agents, tab_execution = st.tabs([
    "\U0001f3af Command Center",
    "\U0001f4c8 Portfolio & PnL",
    "\U0001f9e0 MLOps & Health",
    "\U0001f916 Agent Status",
    "\U0001f4b0 Bet Execution",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: COMMAND CENTER — KPI bar + Pipeline Kanban
# ══════════════════════════════════════════════════════════════════════

with tab_cmd:
    st.header("Command Center")

    w_start, w_end = _get_working_window()

    try:
        with get_session() as sess:
            # Get matches in working window
            matches = list(
                sess.execute(
                    select(Match).where(
                        Match.scheduled_at >= w_start,
                        Match.scheduled_at <= w_end,
                    ).order_by(Match.scheduled_at)
                ).scalars().all()
            )

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

            # Count settled bets
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

            k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
            k1.metric("Matches", n_matches)
            k2.metric("Predictions", n_predictions)
            k3.metric("Pending", n_pending)
            k4.metric("Approved", n_approved, delta=f"{n_approved}" if n_approved else None)
            k5.metric("Vetoed", n_vetoed)
            k6.metric("Placed", n_placed)
            k7.metric("Settled", n_settled, delta=f"{settled_pnl:+.2f} EUR" if settled_bets else None)

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
                                f"{disp['sport_emoji']} **{disp['home']}** vs **{disp['away']}**\n\n"
                                f"`{disp['sport']}` | {disp['league']}\n\n"
                                f"Kickoff: {disp['kickoff']}"
                            )
                        elif header == "VETOED":
                            p = item["prediction"]
                            m = item["match"]
                            disp = get_match_display(sess, m)
                            st.markdown(
                                f"{disp['sport_emoji']} **{disp['home']}** vs **{disp['away']}**\n\n"
                                f"`{p.selection}` | EV: {p.ev:.4f}\n\n"
                                f"Reason: _{p.veto_reason or 'N/A'}_"
                            )
                        elif header == "SETTLED":
                            p = item["prediction"]
                            m = item["match"]
                            b = item["bet"]
                            disp = get_match_display(sess, m)
                            pnl_str = f"{b.pnl_eur:+.2f}" if b.pnl_eur else "0.00"
                            status_icon = {
                                BetStatus.WON: "\U0001f7e2",
                                BetStatus.LOST: "\U0001f534",
                                BetStatus.VOID: "\u26aa",
                            }.get(b.status, "\u2753")
                            st.markdown(
                                f"{status_icon} **{disp['home']}** vs **{disp['away']}**\n\n"
                                f"`{p.selection}` @ {b.odds_at_placement:.2f}\n\n"
                                f"PnL: **{pnl_str} EUR** | {b.status.value.upper()}"
                            )
                        else:
                            # PENDING ML or APPROVED
                            p = item["prediction"]
                            m = item["match"]
                            disp = get_match_display(sess, m)
                            ev_color = "green" if p.ev > 0 else "red"
                            st.markdown(
                                f"{disp['sport_emoji']} **{disp['home']}** vs **{disp['away']}**\n\n"
                                f"`{p.selection}` | EV: :{ev_color}[{p.ev:.4f}]\n\n"
                                f"Prob: {p.model_prob:.1%} | {p.model_source}"
                            )
                        st.divider()

    except Exception as e:
        st.error(f"Pipeline query failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 2: PORTFOLIO & PnL
# ══════════════════════════════════════════════════════════════════════

with tab_portfolio:
    st.header("Portfolio & PnL")

    try:
        col_days, col_ledger = st.columns(2)
        with col_days:
            lookback = st.selectbox("Lookback", [7, 14, 30, 60, 90], index=2)
        with col_ledger:
            ledger_filter = st.selectbox("Ledger", ["Both", "REAL", "PAPER"])

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

            # ── Cumulative PnL chart ────────────────────────────
            st.subheader("Cumulative PnL")

            if HAS_PLOTLY:
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=dates,
                    y=cumulative,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(0, 200, 83, 0.15)",
                    line=dict(color="#00c853", width=2),
                    name="Cumulative PnL",
                    hovertemplate="%{x}<br>PnL: %{y:+.2f} EUR<extra></extra>",
                ))
                fig_cum.add_hline(
                    y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                )
                fig_cum.update_layout(
                    **_plotly_layout(
                        yaxis_title="EUR",
                        xaxis_title="",
                        height=350,
                    )
                )
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                chart_data = {d["date"]: d["cumulative_pnl"] for d in ts_data}
                st.line_chart(chart_data)

            # ── Daily PnL bar chart ─────────────────────────────
            st.subheader("Daily PnL")

            if HAS_PLOTLY:
                colors = [
                    "#00c853" if v >= 0 else "#ff1744" for v in daily_pnl
                ]
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(
                    x=dates,
                    y=daily_pnl,
                    marker_color=colors,
                    name="Daily PnL",
                    hovertemplate="%{x}<br>PnL: %{y:+.2f} EUR<extra></extra>",
                ))
                fig_daily.add_hline(
                    y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                )
                fig_daily.update_layout(
                    **_plotly_layout(
                        yaxis_title="EUR",
                        xaxis_title="",
                        height=300,
                    )
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                bar_data = {d["date"]: d["pnl"] for d in ts_data}
                st.bar_chart(bar_data)

            # Summary metrics
            total_pnl = sum(daily_pnl)
            total_bets = sum(counts)
            best_day = max(daily_pnl) if daily_pnl else 0
            worst_day = min(daily_pnl) if daily_pnl else 0

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total PnL", f"{total_pnl:+.2f} EUR")
            mc2.metric("Total Bets", total_bets)
            mc3.metric("Best Day", f"{best_day:+.2f} EUR")
            mc4.metric("Worst Day", f"{worst_day:+.2f} EUR")
        else:
            st.info("No settled bets in the selected period.")

        # Sport exposure
        st.subheader("Exposure by Sport")

        exposure = _cached_sport_exposure()
        if exposure:
            if HAS_PLOTLY:
                sports = [e["sport"] for e in exposure]
                stakes = [e["total_stake"] for e in exposure]
                fig_exp = go.Figure()
                fig_exp.add_trace(go.Bar(
                    x=sports,
                    y=stakes,
                    marker_color="#42a5f5",
                    hovertemplate="%{x}<br>Stake: %{y:.2f} EUR<extra></extra>",
                ))
                fig_exp.update_layout(
                    **_plotly_layout(
                        yaxis_title="EUR Staked",
                        height=300,
                    )
                )
                st.plotly_chart(fig_exp, use_container_width=True)
            else:
                exp_data = {e["sport"]: e["total_stake"] for e in exposure}
                st.bar_chart(exp_data)

            for e in exposure:
                emoji = SPORT_EMOJI.get(e["sport"], "\U0001f3c6")
                st.markdown(
                    f"{emoji} **{e['sport']}**: {e['pending_count']} bets | "
                    f"{e['total_stake']:.2f} EUR staked | "
                    f"Avg odds: {e['avg_odds']:.2f} | Avg EV: {e['avg_ev']:.4f}"
                )
        else:
            st.info("No pending exposure.")

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
                        annotation_text="Degradation threshold (0.22)",
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

                st.caption("Degradation threshold: Brier > 0.22 (red zone)")
        else:
            st.info("No model metrics yet. Run the Auditor morning audit first.")

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

            # Build per-agent activity heuristics from DB
            agent_activity = {}

            # Scout: check recent matches with odds
            recent_matches_with_odds = sess.execute(
                select(func.count(Match.id)).where(
                    Match.scheduled_at >= w_start,
                    Match.scheduled_at <= w_end,
                )
            ).scalar() or 0
            agent_activity["scout"] = {
                "status": "active" if recent_matches_with_odds > 0 else "idle",
                "detail": f"{recent_matches_with_odds} matches in window",
            }

            # Quant: predictions today
            agent_activity["quant"] = {
                "status": "active" if activity["predictions_today"] > 0 else "idle",
                "detail": f"{activity['predictions_today']} predictions today",
            }

            # Devil's Advocate: vetoed today
            agent_activity["devils_advocate"] = {
                "status": "active" if activity["vetoed_today"] > 0 else "idle",
                "detail": f"{activity['vetoed_today']} vetoed today",
            }

            # Risk Manager: approved/sized today
            agent_activity["risk_manager"] = {
                "status": "active" if activity["approved_today"] > 0 else "idle",
                "detail": f"{activity['approved_today']} sized today",
            }

            # Auditor: settled today
            agent_activity["auditor"] = {
                "status": "active" if activity["settled_today"] > 0 else "idle",
                "detail": f"{activity['settled_today']} settled today",
            }

            # Master: always active
            agent_activity["master"] = {
                "status": "active",
                "detail": "Orchestrating pipeline",
            }

            # Line Shopper: check if any predictions have best_odds
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

            # Data Janitor: inferred from scout activity
            agent_activity["data_janitor"] = {
                "status": "active" if recent_matches_with_odds > 0 else "idle",
                "detail": "Processing crawl data" if recent_matches_with_odds > 0 else "Waiting for data",
            }

            # Moonshot: check parlays (if any placed bets are parlays)
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
# TAB 5: BET EXECUTION — Pending bets with clear placement instructions
# ══════════════════════════════════════════════════════════════════════

with tab_execution:
    st.header("Bet Execution")
    st.caption(
        "Approved bets awaiting manual placement. "
        "Place the bet on the listed sportsbook, then mark as placed."
    )

    try:
        with get_session() as sess:
            # Get APPROVED predictions not yet placed
            approved_preds = list(
                sess.execute(
                    select(Prediction).where(
                        Prediction.status == PredictionStatus.APPROVED,
                    ).order_by(Prediction.created_at.desc())
                ).scalars().all()
            )

            if not approved_preds:
                st.info("No bets awaiting execution. All clear.")
            else:
                st.success(f"\U0001f4b0 **{len(approved_preds)} bet(s) ready to place**")

                for pred in approved_preds:
                    match = sess.get(Match, pred.match_id)
                    if not match:
                        continue

                    disp = get_match_display(sess, match)
                    emoji = disp["sport_emoji"]

                    # Get best odds info
                    odds_display = f"{pred.best_odds:.2f}" if pred.best_odds else "N/A"
                    book_display = pred.best_bookmaker or "Check line shopper"
                    stake_display = f"{pred.stake_eur:.2f} EUR" if pred.stake_eur else "Not sized"
                    ev_display = f"{pred.ev:.4f}" if pred.ev else "N/A"
                    ledger_display = pred.ledger_type.value.upper() if pred.ledger_type else "TBD"

                    with st.container():
                        st.markdown(f"---")
                        st.markdown(
                            f"### {emoji} {disp['home']} vs {disp['away']}\n\n"
                            f"**Kickoff:** {disp['kickoff']} UTC | "
                            f"**League:** {disp['league']}"
                        )

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Selection", pred.selection)
                        c2.metric("Best Odds", odds_display)
                        c3.metric("Stake", stake_display)
                        c4.metric("EV", ev_display)

                        bc1, bc2 = st.columns(2)
                        bc1.markdown(f"**Sportsbook:** `{book_display}`")
                        bc2.markdown(f"**Ledger:** `{ledger_display}`")

                        st.markdown(
                            f"**Model:** {pred.model_source} | "
                            f"**Prob:** {pred.model_prob:.1%} | "
                            f"**Market:** {pred.market_type.value if pred.market_type else 'N/A'}"
                        )

            # Also show recently placed (for confirmation)
            st.divider()
            st.subheader("Recently Placed")

            placed_preds = list(
                sess.execute(
                    select(Prediction).where(
                        Prediction.status == PredictionStatus.PLACED,
                    ).order_by(Prediction.created_at.desc())
                    .limit(10)
                ).scalars().all()
            )

            if placed_preds:
                for pred in placed_preds:
                    match = sess.get(Match, pred.match_id)
                    if not match:
                        continue
                    disp = get_match_display(sess, match)
                    odds_str = f"@ {pred.best_odds:.2f}" if pred.best_odds else ""
                    st.markdown(
                        f"\u2705 **{disp['vs']}** \u2014 `{pred.selection}` {odds_str} "
                        f"| {pred.stake_eur:.2f} EUR" if pred.stake_eur else
                        f"\u2705 **{disp['vs']}** \u2014 `{pred.selection}` {odds_str}"
                    )
            else:
                st.caption("No recently placed bets.")

    except Exception as e:
        st.error(f"Execution query failed: {e}")
