"""OpenClaw Quant Command Center — Streamlit WebUI.

Institutional-grade read-only dashboard for the BetAgent MAS.
Accessible on the local network via http://<mac-mini-ip>:8501

Launch:
    streamlit run src/bet_agent/ui/app.py --server.address 0.0.0.0

Tabs:
    1. Pipeline     — Kanban view of today's prediction flow
    2. Portfolio     — Time-series PnL, bankroll growth, sport exposure
    3. MLOps         — Model health, Brier Scores, ROI, killswitch indicators
    4. Agent Logs    — Live tail of agent execution logs

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

from sqlalchemy import select

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


# ── Cached data loaders ──────────────────────────────────────────────
# Streamlit re-runs the entire script on every interaction (tab switch,
# button click, auto-refresh). @st.cache_data keeps expensive DB queries
# in RAM for `ttl` seconds so the UI stays responsive at scale.


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
        # Convert dataclasses to dicts for Streamlit serialization
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
if HAS_AUTOREFRESH:
    st.sidebar.caption("Auto-refreshes every 30s (streamlit-autorefresh)")
else:
    st.sidebar.caption("Install streamlit-autorefresh for auto-refresh")

# ── Tab layout ───────────────────────────────────────────────────────

tab_pipeline, tab_portfolio, tab_mlops, tab_logs = st.tabs([
    "\U0001f4cb Pipeline",
    "\U0001f4c8 Portfolio & PnL",
    "\U0001f9e0 MLOps & Health",
    "\U0001f4dc Agent Logs",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: PIPELINE — Kanban view of today's prediction flow
# ══════════════════════════════════════════════════════════════════════

with tab_pipeline:
    st.header("Today's Pipeline")

    today_start = datetime.combine(date.today(), time.min, tzinfo=timezone.utc)
    today_end = datetime.combine(date.today(), time.max, tzinfo=timezone.utc)

    try:
        with get_session() as sess:
            # Get today's matches
            matches = list(
                sess.execute(
                    select(Match).where(
                        Match.scheduled_at >= today_start,
                        Match.scheduled_at <= today_end,
                    ).order_by(Match.scheduled_at)
                ).scalars().all()
            )

            # Get predictions for today's matches
            match_ids = [m.id for m in matches]
            predictions = []
            if match_ids:
                predictions = list(
                    sess.execute(
                        select(Prediction).where(Prediction.match_id.in_(match_ids))
                    ).scalars().all()
                )

            # Group predictions by status
            kanban: dict[str, list] = {
                "NOT STARTED": [],
                "PENDING ML": [],
                "VETOED": [],
                "APPROVED & SIZED": [],
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
                    # Check if bet is settled
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
                        kanban["APPROVED & SIZED"].append(entry)

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
                    for item in items[:20]:  # Cap display
                        if header == "NOT STARTED":
                            m = item
                            st.markdown(
                                f"**{m.home_team}** vs **{m.away_team}**\n\n"
                                f"`{m.sport.value}` | {m.league}\n\n"
                                f"Kickoff: {m.scheduled_at.strftime('%H:%M')}"
                            )
                        elif header == "VETOED":
                            p = item["prediction"]
                            m = item["match"]
                            st.markdown(
                                f"**{m.home_team}** vs **{m.away_team}**\n\n"
                                f"`{p.selection}` | EV: {p.ev:.4f}\n\n"
                                f"Reason: _{p.veto_reason or 'N/A'}_"
                            )
                        elif header == "SETTLED":
                            p = item["prediction"]
                            m = item["match"]
                            b = item["bet"]
                            pnl_str = f"{b.pnl_eur:+.2f}" if b.pnl_eur else "0.00"
                            status_icon = {
                                BetStatus.WON: "\U0001f7e2",
                                BetStatus.LOST: "\U0001f534",
                                BetStatus.VOID: "\u26aa",
                            }.get(b.status, "\u2753")
                            st.markdown(
                                f"{status_icon} **{m.home_team}** vs **{m.away_team}**\n\n"
                                f"`{p.selection}` @ {b.odds_at_placement:.2f}\n\n"
                                f"PnL: **{pnl_str} EUR** | {b.status.value.upper()}"
                            )
                        else:
                            p = item["prediction"]
                            m = item["match"]
                            st.markdown(
                                f"**{m.home_team}** vs **{m.away_team}**\n\n"
                                f"`{p.selection}` | EV: {p.ev:.4f}\n\n"
                                f"Model: {p.model_source} | Prob: {p.model_prob:.1%}"
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
                st.markdown(
                    f"**{e['sport']}**: {e['pending_count']} bets | "
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
                    f"\U0001f6a8 **KILLSWITCH ACTIVE** — "
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
                    # Group by model_name
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

                    # Degradation threshold line
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
# TAB 4: AGENT LOGS
# ══════════════════════════════════════════════════════════════════════

with tab_logs:
    st.header("Agent Activity Logs")

    log_file = os.getenv("BETAGENT_LOG_FILE", "logs/betagent.log")

    if Path(log_file).exists():
        num_lines = st.slider("Tail lines", 50, 500, 100)

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                tail = lines[-num_lines:]

            # Color-code log levels
            for line in tail:
                if "ERROR" in line or "CRITICAL" in line:
                    st.markdown(f":red[{line.rstrip()}]")
                elif "WARNING" in line or "DEGRADED" in line:
                    st.markdown(f":orange[{line.rstrip()}]")
                elif "Settlement" in line or "Settled" in line:
                    st.markdown(f":green[{line.rstrip()}]")
                else:
                    st.text(line.rstrip())

        except Exception as e:
            st.error(f"Failed to read log file: {e}")
    else:
        st.info(
            f"Log file not found at `{log_file}`.\n\n"
            f"Set `BETAGENT_LOG_FILE` env var or ensure logging is configured.\n\n"
            f"Example: `export BETAGENT_LOG_FILE=logs/betagent.log`"
        )

        # Fallback: show recent DB activity
        st.subheader("Recent Database Activity (fallback)")
        try:
            activity = _cached_recent_activity()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predictions Today", activity["predictions_today"])
            c2.metric("Settled Today", activity["settled_today"])
            c3.metric("Settlement PnL", f"{activity['last_settlement_pnl']:+.2f} EUR")

            c4, c5, c6 = st.columns(3)
            c4.metric("Approved", activity["approved_today"])
            c5.metric("Vetoed", activity["vetoed_today"])
            c6.metric("Placed", activity["placed_today"])
        except Exception as e:
            st.error(f"Activity query failed: {e}")
