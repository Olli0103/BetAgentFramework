.PHONY: install setup db test ui bot lint clean help

# Default target
help:
	@echo ""
	@echo "  BetAgent — Multi-Agent Sports Betting System"
	@echo "  ─────────────────────────────────────────────"
	@echo ""
	@echo "  Quick start:"
	@echo "    make setup        Full first-time setup (install + db + .env)"
	@echo ""
	@echo "  Individual commands:"
	@echo "    make install      Install Python dependencies (all extras)"
	@echo "    make db           Create database tables"
	@echo "    make test         Run test suite (320 tests, SQLite in-memory)"
	@echo "    make ui           Launch Streamlit dashboard on port 8501"
	@echo "    make bot          Start Telegram bot"
	@echo "    make clean        Remove build artifacts and caches"
	@echo ""

# ── Full first-time setup ────────────────────────────────────────────

setup: install _ensure-env db
	@echo ""
	@echo "  Setup complete."
	@echo "  Edit .env with your credentials, then run:"
	@echo "    make ui     → open http://localhost:8501"
	@echo "    make bot    → start the Telegram bot"
	@echo ""

# ── Install ──────────────────────────────────────────────────────────

install:
	pip install -e ".[ui,telegram,dev]"

# ── Database ─────────────────────────────────────────────────────────

db:
	python -c "from bet_agent.db.session import init_db; init_db(); print('Tables created.')"

# ── Tests ────────────────────────────────────────────────────────────

test:
	BETAGENT_ENV=test pytest tests/ -q

test-verbose:
	BETAGENT_ENV=test pytest tests/ -x -v

# ── Run services ─────────────────────────────────────────────────────

ui:
	streamlit run src/bet_agent/ui/app.py --server.address 0.0.0.0 --server.port 8501

bot:
	python -m bet_agent.interfaces.telegram_bot

# ── Housekeeping ─────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache build dist

_ensure-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "  Created .env from .env.example — edit it with your credentials."; \
	else \
		echo "  .env already exists, skipping."; \
	fi
