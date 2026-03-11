# BetAgent — Multi-Agent Sports Betting System

A 9-agent orchestrated framework for institutional sports betting with human-in-the-loop controls, deterministic financial math, and GlüStV compliance.

**No LLM ever touches money.** All calculations (EV, Kelly, Brier, PnL) run in pure Python. LLMs handle reasoning, research, and coordination — never arithmetic.

---

## Architecture

```
                         ┌──────────────┐
                         │    MASTER    │  Tier 1 LLM
                         │  (Portfolio  │  Hub of all comms
                         │   Manager)   │  Human-in-the-Loop
                         └──────┬───────┘
                                │
          ┌─────────┬───────────┼───────────┬──────────┬──────────┐
          ▼         ▼           ▼           ▼          ▼          ▼
     ┌─────────┐ ┌──────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ ┌────────┐
     │  SCOUT  │ │ DATA │ │  QUANT  │ │ DEVIL'S │ │ LINE  │ │  RISK  │
     │(Crawler)│ │JANITOR│ │ (Math)  │ │ADVOCATE│ │SHOPPER│ │MANAGER │
     └─────────┘ └──────┘ └─────────┘ └─────────┘ └───────┘ └────────┘
       Tier 2      Tier 2    Tier 2      Tier 1      Tier 2    Tier 2

     ┌──────────┐ ┌──────────┐
     │ MOONSHOT │ │ AUDITOR  │
     │(Parlays) │ │(Feedback)│
     └──────────┘ └──────────┘
       Tier 1       Tier 1
```

**LLM Tiers:**
- **Tier 1 (Heavy Reasoning):** OpenClaw primary + Gemini fallback — for research, veto analysis, NL queries
- **Tier 2 (Local Tools):** Ollama/gemma3:4b on Mac Mini — for crawling, parsing, calculations (privacy-first)

---

## Agent Roster

| Agent | Role | Key Responsibilities |
|-------|------|---------------------|
| **Master** | Portfolio Manager | Pipeline orchestration, Telegram concierge, alert broadcasts |
| **Scout** | Data Gatherer | Pre-match odds scraping, daily Cloudflare crawl (04:00 UTC) |
| **Data Janitor** | ETL Engine | Parse HTML/JSON, normalize team names, resolve aliases |
| **Quant** | Math & ML Engine | Daily predictions (analytical + XGBoost), parameter estimation |
| **Devil's Advocate** | Qualitative Veto | Research injuries/news/weather, VETO or APPROVE each pick |
| **Line Shopper** | Odds Maximizer | Compare odds across 5 legal German sportsbooks |
| **Risk Manager** | Bankroll Guardian | Quarter-Kelly sizing, loss limits, Paper/Real ledger routing |
| **Moonshot** | Parlay Builder | Smart +EV parlays, correlation analysis, hard-cap 1.00 EUR |
| **Auditor** | Feedback Loop | Nightly settlement (05:00 UTC), Brier Score, ROI, degradation alerts |

---

## Pipeline Flow (Golden Path)

```
04:00 UTC ─── Scout (crawl daily stats)
                │
                ▼
          Data Janitor (parse & clean)
                │
                ▼
          Quant (run_daily_predictions)
                │
                ▼
          Master (filter +EV predictions)
                │
                ▼
          Devil's Advocate (veto check: injuries, news, fatigue)
                │
                ▼
          Line Shopper (best odds across 5 books)
                │
                ▼
          Risk Manager (Quarter-Kelly stake + loss limit check)
                │
                ▼
          Master ──► Telegram Alert to Syndicate
                │
                ▼
          Human confirms /placed at sportsbook  ◄── HUMAN-IN-THE-LOOP
                │
                ▼
05:00 UTC ─── Auditor (fetch results → settle → audit)
                │
                ▼
          Dashboard updated (Streamlit + Telegram digest)
```

---

## Sports & Probability Models

| Sport | Model | Primary Input |
|-------|-------|---------------|
| Football | Poisson (xG-based) | home_xg, away_xg |
| Tennis | Hierarchical serve/break | p_serve_home, p_serve_away |
| Ice Hockey | Corsi/Fenwick Poisson | adjusted xG |
| Basketball | Pace-adjusted Normal | ORtg, DRtg, pace |
| American Football | EPA Power Ratings | home/away power_rtg |
| Darts | Leg Probability | average leg score, form |

Markets: Match Winner, Over/Under, BTTS, Spread

---

## Tool Inventory (17 Modules)

```
src/bet_agent/tools/
├── prediction_runner.py     # Daily ML/analytical predictions
├── ev_calculator.py         # EV = P(win) × Odds − 1
├── kelly_calculator.py      # Quarter-Kelly (0.25 fraction)
├── sizing_engine.py         # Risk-aware stake sizing
├── veto_engine.py           # Qualitative risk assessment
├── settlement_engine.py     # PnL calculation & bankroll updates
├── auditor_metrics.py       # Brier Score, ROI, degradation
├── line_shopper.py          # Odds comparison across books
├── results_fetcher.py       # Final score fetching
├── cloudflare_crawler.py    # Daily stats crawl orchestrator
├── data_parser.py           # HTML/JSON → structured stats
├── param_estimator.py       # Analytical model parameters
├── feature_factory.py       # ML feature engineering
├── historical_importer.py   # Bulk historical data loading
├── notifier.py              # Multi-channel alerts
├── master_analysis.py       # Read-only concierge tools
└── prob_models/             # 6 sport-specific models + registry
```

---

## Database Schema (PostgreSQL, 9 Tables)

| Table | Purpose |
|-------|---------|
| `matches` | Match metadata (sport, teams, scores, state) |
| `predictions` | Model predictions (prob, EV, status) |
| `placed_bets` | Placed wagers (stake, odds, PnL, ledger type) |
| `odds_markets` | Historical odds snapshots per market/selection |
| `bankroll_ledger` | REAL and PAPER balance tracking |
| `team_daily_stats` | Daily metrics as JSONB (xG, pace, EPA, etc.) |
| `model_metrics` | Brier Score, ROI, degradation status per model |
| `team_aliases` | Sportsbook name → canonical name mapping |
| `historical_matches` | Historical data for ML training |

**Bet Lifecycle:**
```
PENDING ──► PLACED ──► WON / LOST / VOID
   │                        ▲
   │ (never placed)         │ (settlement engine)
   └──► VOID (auto-expired) │
                             │
   PUSHED_TO_HUMAN ─────────┘ (edge case review)
```

---

## Interfaces

### Streamlit WebUI (Port 8501)

```bash
streamlit run src/bet_agent/ui/app.py
```

4 tabs:
- **Pipeline:** Kanban board (NOT_STARTED → PENDING → APPROVED → SETTLED)
- **Portfolio:** Time-series PnL, bankroll growth, sport exposure (Plotly charts)
- **MLOps:** Model health, Brier Scores, ROI, degradation indicators
- **Agent Logs:** Live tail of execution logs

### Telegram Syndicate Bot

```bash
python -m bet_agent.interfaces.telegram_bot
```

Commands:
- `/status` — Portfolio summary
- `/pending` — Bets awaiting execution (InlineKeyboard: Standard or Custom)
- `/pnl` — Balance & P&L
- `/health` — Model health overview
- `/placed <id> <odds> <stake>` — Manual placement
- `/cancel` — Abort custom odds entry
- Free text → Master Agent (Tier 1) for NL analysis

**EV Gate:** When entering custom odds, the system recalculates EV. If the bet becomes -EV at the new odds, placement is **automatically aborted** with an explanation.

---

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Ollama (for Tier 2 local models)

### Installation

```bash
# Clone
git clone <repo-url> && cd BetAgentFramework

# Install with all extras
pip install -e ".[ui,telegram,dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

```env
DATABASE_URL=postgresql://betagent:password@localhost:5432/betagent
TELEGRAM_BOT_TOKEN=           # From @BotFather
ALLOWED_TELEGRAM_IDS=123456   # Comma-separated whitelist
TELEGRAM_GROUP_ID=            # Optional syndicate group
GEMINI_API_KEY=               # Tier 1 fallback
OLLAMA_BASE_URL=http://localhost:11434
CLOUDFLARE_ACCOUNT_ID=        # For daily crawls
CLOUDFLARE_API_TOKEN=
```

### Database Setup

```bash
# Create database
createdb betagent

# Tables auto-create via SQLAlchemy on first run
# Or use Alembic for migrations:
alembic upgrade head
```

### Running Tests

```bash
pytest tests/ -q        # 310 tests, SQLite in-memory
pytest tests/ -x -v     # Stop on first failure, verbose
```

---

## Risk Controls

| Control | Limit |
|---------|-------|
| Max single bet | 5% of bankroll |
| Daily loss stop | 10% of bankroll |
| Weekly loss cap | 20% of bankroll |
| Parlay hard cap | 1.00 EUR (non-negotiable) |
| Degradation threshold | Brier > 0.22 or ROI < -5% |
| Stale bet expiry | 60 minutes auto-void |
| EV gate | Custom odds must be +EV or placement aborted |

**Dual Ledger System:**
- **PAPER** — Sandbox for unproven models (no real money)
- **REAL** — Promoted after Auditor validates performance

---

## Golden Rules

1. **NO LLM MATH** — Python tools handle all EV, Kelly, Brier, PnL calculations
2. **STATEFUL MEMORY** — PostgreSQL is the single source of truth
3. **HUMAN-IN-THE-LOOP** — Master never auto-places bets; always pushes to Telegram
4. **PAPER FIRST** — New models sandbox on Paper ledger until Auditor promotes
5. **MOONSHOT CAP** — All parlays capped at 1.00 EUR
6. **DEDUCT AT PLACEMENT** — Stake is only deducted when the human confirms via /placed

---

## Configuration

```
config/
├── agents.yaml      # 9-agent definitions (roles, tools, schedules, limits)
└── llm_tiers.yaml   # LLM routing (Tier 1: heavy reasoning, Tier 2: local tools)

agents/
├── master/SOUL.md        # Agent identity & behavioral instructions
├── scout/SOUL.md
├── quant/SOUL.md
├── devils_advocate/SOUL.md
├── line_shopper/SOUL.md
├── risk_manager/SOUL.md
├── moonshot/SOUL.md
├── data_janitor/SOUL.md
└── auditor/SOUL.md
```

---

## Project Structure

```
BetAgentFramework/
├── src/bet_agent/
│   ├── tools/               # 17 deterministic Python tools
│   │   └── prob_models/     # 6 sport-specific probability models
│   ├── db/                  # SQLAlchemy ORM (models.py, session.py)
│   ├── ml/                  # XGBoost training pipeline
│   ├── ingest/              # 6 sport-specific historical data importers
│   ├── ui/                  # Streamlit institutional dashboard
│   └── interfaces/          # Telegram syndicate bot
├── config/                  # Agent + LLM tier configuration (YAML)
├── agents/                  # 9 agent SOUL.md character sheets
├── tests/                   # 310 tests (pytest, SQLite in-memory)
├── pyproject.toml           # Dependencies & build config
└── .env.example             # Environment variable template
```

---

## License

Private. All rights reserved.
