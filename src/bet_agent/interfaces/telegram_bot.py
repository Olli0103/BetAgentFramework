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
  - Chat context validation: only private DMs or the official TELEGRAM_GROUP_ID
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

import asyncio
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


def is_authorized(user_id: int, chat_id: int | None = None) -> bool:
    """Check if a Telegram interaction is authorized.

    The Door-man enforces TWO layers:
      1. User whitelist — is the sender on ALLOWED_TELEGRAM_IDS?
      2. Chat context   — is this a private DM or the official syndicate group?

    If a whitelisted user types /pnl in a random public group, the bot
    stays silent to prevent leaking fund data to strangers.

    Args:
        user_id: The Telegram user ID of the sender.
        chat_id: The Telegram chat ID where the message was sent.
                 If None, only user whitelist is checked (backwards compat).
    """
    if not ALLOWED_IDS:
        logger.warning("ALLOWED_TELEGRAM_IDS is empty — all access denied")
        return False

    if user_id not in ALLOWED_IDS:
        return False

    # If no chat context provided, fall back to user-only check
    if chat_id is None:
        return True

    # Private DM: chat_id == user_id (always allowed for whitelisted users)
    if chat_id == user_id:
        return True

    # Group chat: only the official syndicate group is allowed
    group_id = _parse_group_id()
    if group_id is not None and chat_id == group_id:
        return True

    # Any other chat (random groups, channels) → BLOCK
    logger.warning(
        "Whitelisted user_id=%d attempted command in unauthorized chat_id=%d",
        user_id, chat_id,
    )
    return False


def _parse_group_id() -> int | None:
    """Parse TELEGRAM_GROUP_ID into an int, returning None if not set."""
    raw = TELEGRAM_GROUP_ID.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def get_user_display_name(user) -> str:
    """Extract display name from a Telegram User object."""
    if user is None:
        return "Unknown"
    if user.username:
        return f"@{user.username}"
    if user.first_name:
        return user.first_name
    return str(user.id)


# ── Sync DB wrappers (run in thread to avoid blocking event loop) ────


def _sync_fetch_status():
    """Synchronous: fetch portfolio summary and format text."""
    from bet_agent.db.session import get_session
    from bet_agent.tools.master_analysis import (
        fetch_portfolio_summary,
        format_portfolio_text,
    )
    with get_session() as sess:
        summary = fetch_portfolio_summary(sess)
        return format_portfolio_text(summary)


def _sync_fetch_pending():
    """Synchronous: fetch pending bets and format text."""
    from bet_agent.db.session import get_session
    from bet_agent.tools.master_analysis import (
        fetch_pending_for_human,
        format_pending_text,
    )
    with get_session() as sess:
        pending = fetch_pending_for_human(sess)
        return format_pending_text(pending)


def _sync_fetch_pnl():
    """Synchronous: fetch PnL and format text."""
    from bet_agent.db.session import get_session
    from bet_agent.tools.master_analysis import format_pnl_text
    with get_session() as sess:
        return format_pnl_text(sess)


def _sync_fetch_health():
    """Synchronous: fetch model health reports."""
    from bet_agent.db.session import get_session
    from bet_agent.tools.master_analysis import fetch_model_health
    with get_session() as sess:
        return fetch_model_health(sess)


def _sync_place_bet(bet_id: str, user_name: str):
    """Synchronous: mark bet as placed and commit."""
    from bet_agent.db.session import get_session
    from bet_agent.tools.master_analysis import mark_bet_placed_by_user
    with get_session() as sess:
        result = mark_bet_placed_by_user(sess, bet_id, user_name)
        # get_session() auto-commits on success
        return result


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


# ── Auth helper for handlers ─────────────────────────────────────────


def _check_auth(update) -> bool:
    """Check user + chat authorization from an Update object."""
    user = update.effective_user
    chat = update.effective_chat
    chat_id = chat.id if chat else None
    return is_authorized(user.id, chat_id)


# ── Command handlers ─────────────────────────────────────────────────


async def cmd_start(update, context) -> None:
    """Handle /start — welcome message."""
    if not _check_auth(update):
        logger.warning("Unauthorized /start from user_id=%d", update.effective_user.id)
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
    if not _check_auth(update):
        logger.warning("Unauthorized /status from user_id=%d", update.effective_user.id)
        return

    try:
        text = await asyncio.to_thread(_sync_fetch_status)
        await update.message.reply_text(f"```\n{text}\n```", parse_mode="Markdown")
    except Exception as e:
        logger.error("Error in /status: %s", e)
        await update.message.reply_text(f"Error fetching status: {e}")


async def cmd_pending(update, context) -> None:
    """Handle /pending — bets waiting for human execution."""
    if not _check_auth(update):
        return

    try:
        text = await asyncio.to_thread(_sync_fetch_pending)
        await update.message.reply_text(f"```\n{text}\n```", parse_mode="Markdown")
    except Exception as e:
        logger.error("Error in /pending: %s", e)
        await update.message.reply_text(f"Error: {e}")


async def cmd_pnl(update, context) -> None:
    """Handle /pnl — current balance and P&L."""
    if not _check_auth(update):
        return

    try:
        text = await asyncio.to_thread(_sync_fetch_pnl)
        await update.message.reply_text(f"```\n{text}\n```", parse_mode="Markdown")
    except Exception as e:
        logger.error("Error in /pnl: %s", e)
        await update.message.reply_text(f"Error: {e}")


async def cmd_health(update, context) -> None:
    """Handle /health — model health overview."""
    if not _check_auth(update):
        return

    try:
        reports = await asyncio.to_thread(_sync_fetch_health)

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
    if not _check_auth(update):
        return

    user = update.effective_user
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
        result = await asyncio.to_thread(_sync_place_bet, bet_id, user_name)

        if "error" in result:
            await update.message.reply_text(f"Error: {result['error']}")
            return

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
    if not _check_auth(update):
        user = update.effective_user
        logger.warning(
            "Unauthorized message from user_id=%d in chat_id=%d: %s",
            user.id, update.effective_chat.id if update.effective_chat else 0,
            (update.message.text or "")[:50],
        )
        return  # Silent block

    text = update.message.text
    if not text:
        return

    user = update.effective_user
    user_name = get_user_display_name(user)
    logger.info("NL query from %s (id=%d): %s", user_name, user.id, text[:100])

    # Route to Master Agent (bridge.query may be slow — run in thread)
    try:
        response = await asyncio.to_thread(_master_bridge.query, text, user_name)
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
