"""OpenClaw Syndicate Telegram Bot — Multiplayer Quant Concierge.

Secure, multi-user Telegram bot with strict whitelist authentication.
All members of the syndicate can interact with the system via:
  - /status   — daily portfolio summary
  - /pending  — bets waiting for human execution
  - /pnl      — current balance and P&L
  - /placed   — confirm manual bet placement (syncs to team)
  - /health   — model health overview
  - Free text — routed to the Master Agent (Tier 1 LLM) for NL answers

Security:
  - ALLOWED_TELEGRAM_IDS from .env enforces strict whitelist
  - Unauthorized users are blocked silently (logged for audit)
  - All interactions are logged with user identification

Launch:
    python -m bet_agent.interfaces.telegram_bot

Env vars required:
    TELEGRAM_BOT_TOKEN    — Bot API token from @BotFather
    ALLOWED_TELEGRAM_IDS  — Comma-separated whitelist: "123456,789012,..."
    DATABASE_URL          — PostgreSQL connection string
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from bet_agent.db.models import Base

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_TELEGRAM_IDS_RAW = os.getenv("ALLOWED_TELEGRAM_IDS", "")
TELEGRAM_GROUP_ID = os.getenv("TELEGRAM_GROUP_ID", "")


def parse_allowed_ids(raw: str) -> set[int]:
    """Parse comma-separated Telegram user IDs into a set."""
    if not raw.strip():
        return set()
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
        elif part.lstrip("-").isdigit():
            ids.add(int(part))  # Negative = group IDs
    return ids


ALLOWED_IDS: set[int] = parse_allowed_ids(ALLOWED_TELEGRAM_IDS_RAW)


# ── Whitelist enforcement ────────────────────────────────────────────


def is_authorized(user_id: int) -> bool:
    """Check if a Telegram user is whitelisted.

    The Door-man: strict whitelist check. If ALLOWED_TELEGRAM_IDS is
    empty, NO ONE is authorized (fail-closed, not fail-open).
    """
    if not ALLOWED_IDS:
        logger.warning("ALLOWED_TELEGRAM_IDS is empty — all access denied")
        return False
    return user_id in ALLOWED_IDS


def get_user_display_name(user) -> str:
    """Extract display name from a Telegram User object."""
    if user is None:
        return "Unknown"
    if user.username:
        return f"@{user.username}"
    if user.first_name:
        return user.first_name
    return str(user.id)


# ── Database session helper ──────────────────────────────────────────


def _get_session():
    """Create a new DB session for command handlers."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")

    engine = create_engine(db_url, echo=False, pool_pre_ping=True)
    return sessionmaker(bind=engine)()


# ── Master Agent bridge (NL routing) ────────────────────────────────


class MasterAgentBridge:
    """Bridge for routing natural language queries to the Master Agent.

    In production, this calls the actual LLM endpoint. The bridge is
    injectable for testing purposes.
    """

    def query(self, message: str, user_name: str) -> str:
        """Send a natural language query to the Master Agent.

        Override this method to integrate with your LLM orchestration layer.
        Default implementation returns a structured acknowledgment.
        """
        logger.info("NL query from %s: %s", user_name, message)
        return (
            f"[Master Agent] Received your query: \"{message}\"\n\n"
            f"This will be routed to the Tier 1 LLM for analysis. "
            f"In production, connect MasterAgentBridge.query() to your "
            f"LLM orchestration endpoint."
        )


# Default bridge (override in production)
_master_bridge = MasterAgentBridge()


def set_master_bridge(bridge: MasterAgentBridge) -> None:
    """Inject a custom Master Agent bridge (for production LLM routing)."""
    global _master_bridge
    _master_bridge = bridge


# ── Command handlers ─────────────────────────────────────────────────


async def cmd_start(update, context) -> None:
    """Handle /start — welcome message."""
    user = update.effective_user
    if not is_authorized(user.id):
        logger.warning("Unauthorized /start from user_id=%d", user.id)
        return  # Silent block

    await update.message.reply_text(
        "Welcome to the OpenClaw Syndicate.\n\n"
        "Commands:\n"
        "/status  — Portfolio summary\n"
        "/pending — Bets awaiting execution\n"
        "/pnl     — Balance & P&L\n"
        "/placed <bet_id> — Confirm bet placement\n"
        "/health  — Model health overview\n\n"
        "Or just type a question in natural language."
    )


async def cmd_status(update, context) -> None:
    """Handle /status — daily portfolio summary."""
    user = update.effective_user
    if not is_authorized(user.id):
        logger.warning("Unauthorized /status from user_id=%d", user.id)
        return

    try:
        sess = _get_session()
        from bet_agent.tools.master_analysis import (
            fetch_portfolio_summary,
            format_portfolio_text,
        )

        summary = fetch_portfolio_summary(sess)
        text = format_portfolio_text(summary)
        sess.close()

        await update.message.reply_text(f"```\n{text}\n```", parse_mode="Markdown")
    except Exception as e:
        logger.error("Error in /status: %s", e)
        await update.message.reply_text(f"Error fetching status: {e}")


async def cmd_pending(update, context) -> None:
    """Handle /pending — bets waiting for human execution."""
    user = update.effective_user
    if not is_authorized(user.id):
        return

    try:
        sess = _get_session()
        from bet_agent.tools.master_analysis import (
            fetch_pending_for_human,
            format_pending_text,
        )

        pending = fetch_pending_for_human(sess)
        text = format_pending_text(pending)
        sess.close()

        await update.message.reply_text(f"```\n{text}\n```", parse_mode="Markdown")
    except Exception as e:
        logger.error("Error in /pending: %s", e)
        await update.message.reply_text(f"Error: {e}")


async def cmd_pnl(update, context) -> None:
    """Handle /pnl — current balance and P&L."""
    user = update.effective_user
    if not is_authorized(user.id):
        return

    try:
        sess = _get_session()
        from bet_agent.tools.master_analysis import format_pnl_text

        text = format_pnl_text(sess)
        sess.close()

        await update.message.reply_text(f"```\n{text}\n```", parse_mode="Markdown")
    except Exception as e:
        logger.error("Error in /pnl: %s", e)
        await update.message.reply_text(f"Error: {e}")


async def cmd_health(update, context) -> None:
    """Handle /health — model health overview."""
    user = update.effective_user
    if not is_authorized(user.id):
        return

    try:
        sess = _get_session()
        from bet_agent.tools.master_analysis import fetch_model_health

        reports = fetch_model_health(sess)
        sess.close()

        if not reports:
            await update.message.reply_text("No model metrics available yet.")
            return

        lines = ["MODEL HEALTH REPORT", ""]
        for r in reports:
            status = "\U0001f534 DEGRADED" if r.is_degraded else "\U0001f7e2 OK"
            lines.append(
                f"{status} {r.model_name}\n"
                f"  Brier: {r.latest_brier:.4f} | ROI: {r.latest_roi:+.1f}%\n"
                f"  Record: {r.record_win}W-{r.record_loss}L | {r.total_bets} bets\n"
                f"  Trend: {r.trend}"
            )
            lines.append("")

        await update.message.reply_text(
            f"```\n{chr(10).join(lines)}\n```", parse_mode="Markdown"
        )
    except Exception as e:
        logger.error("Error in /health: %s", e)
        await update.message.reply_text(f"Error: {e}")


async def cmd_placed(update, context) -> None:
    """Handle /placed <bet_id> — confirm manual bet placement.

    When a syndicate member places a bet on a sportsbook, they trigger
    this command. The bot updates the DB and broadcasts to the team.
    """
    user = update.effective_user
    if not is_authorized(user.id):
        return

    args = context.args if context.args else []
    if not args:
        await update.message.reply_text(
            "Usage: /placed <bet_id>\n\n"
            "Use /pending to see available bet IDs."
        )
        return

    bet_id = args[0]
    user_name = get_user_display_name(user)

    try:
        sess = _get_session()
        from bet_agent.tools.master_analysis import mark_bet_placed_by_user

        result = mark_bet_placed_by_user(sess, bet_id, user_name)

        if "error" in result:
            sess.close()
            await update.message.reply_text(f"Error: {result['error']}")
            return

        sess.commit()
        sess.close()

        # Confirmation to the user who placed
        await update.message.reply_text(
            f"\u2705 Bet confirmed as placed!\n\n"
            f"Match: {result['match']}\n"
            f"Selection: {result['selection']}\n"
            f"Stake: {result['stake_eur']:.2f} EUR @ {result['odds']:.2f}\n"
            f"Placed by: {user_name}"
        )

        # Broadcast to the group (if configured)
        if TELEGRAM_GROUP_ID and context.bot:
            broadcast_msg = (
                f"\U0001f4e2 BET PLACED by {user_name}\n\n"
                f"Match: {result['match']}\n"
                f"Selection: {result['selection']}\n"
                f"Stake: {result['stake_eur']:.2f} EUR @ {result['odds']:.2f}\n"
                f"ID: {bet_id[:8]}..."
            )
            try:
                await context.bot.send_message(
                    chat_id=TELEGRAM_GROUP_ID,
                    text=broadcast_msg,
                )
            except Exception as exc:
                logger.error("Group broadcast failed: %s", exc)

        # Also broadcast to all whitelisted individual users
        for uid in ALLOWED_IDS:
            if uid != user.id and uid != int(TELEGRAM_GROUP_ID or 0):
                try:
                    await context.bot.send_message(
                        chat_id=uid,
                        text=(
                            f"\U0001f4e2 {user_name} placed: "
                            f"{result['match']} — {result['selection']} "
                            f"@ {result['odds']:.2f} ({result['stake_eur']:.2f} EUR)"
                        ),
                    )
                except Exception:
                    pass  # User may not have started the bot yet

    except Exception as e:
        logger.error("Error in /placed: %s", e)
        await update.message.reply_text(f"Error: {e}")


async def handle_message(update, context) -> None:
    """Handle free-text messages — route to Master Agent via NL bridge."""
    user = update.effective_user
    if not is_authorized(user.id):
        logger.warning(
            "Unauthorized message from user_id=%d: %s",
            user.id, (update.message.text or "")[:50],
        )
        return  # Silent block

    text = update.message.text
    if not text:
        return

    user_name = get_user_display_name(user)
    logger.info("NL query from %s (id=%d): %s", user_name, user.id, text[:100])

    # Route to Master Agent
    try:
        response = _master_bridge.query(text, user_name)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error("Master Agent bridge error: %s", e)
        await update.message.reply_text(
            "The Master Agent encountered an error processing your request. "
            "Please try again or use a slash command."
        )


# ── Bot builder ──────────────────────────────────────────────────────


def build_application():
    """Build the Telegram Application with all handlers registered.

    Returns:
        telegram.ext.Application instance (not yet running).
    """
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
    )

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. "
            "Create a bot via @BotFather and set the token in .env"
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pending", cmd_pending))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("placed", cmd_placed))

    # Free-text → Master Agent NL bridge (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(
        "Telegram bot configured with %d whitelisted users",
        len(ALLOWED_IDS),
    )

    return app


def run_bot() -> None:
    """Run the Telegram bot with long-polling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if not ALLOWED_IDS:
        logger.error(
            "ALLOWED_TELEGRAM_IDS is empty! No users will be able to interact. "
            "Set it in .env: ALLOWED_TELEGRAM_IDS=123456,789012"
        )

    app = build_application()
    logger.info("Starting OpenClaw Syndicate Telegram bot (long-polling)...")
    app.run_polling(drop_pending_updates=True)


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run_bot()
