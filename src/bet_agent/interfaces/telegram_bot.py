"""OpenClaw Syndicate Telegram Bot — Multiplayer Quant Concierge.

Secure, multi-user Telegram bot with strict whitelist authentication.
All members of the syndicate can interact with the system via:
  - /status   — daily portfolio summary
  - /pending  — bets waiting for human execution (InlineKeyboard buttons)
  - /pnl      — current balance & P&L
  - /placed   — confirm manual bet placement (syncs to team)
  - /health   — model health overview
  - /cancel   — abort custom odds/stake conversation
  - Free text — routed to the Master Agent (Tier 1 LLM) for NL answers

UX:
  - /pending renders each bet as an InlineKeyboard with two buttons:
    "✅ Standard" (use model values) or "✏️ Custom" (enter real odds/stake)
  - Custom flow uses ConversationHandler with proper state machine + timeout
  - After placement, buttons are replaced with confirmation via edit_message_text
    to prevent double-placement in group chat
  - EV gate: if custom odds make the bet -EV, placement is ABORTED

Security:
  - ALLOWED_TELEGRAM_IDS from .env enforces strict whitelist
  - Chat context validation: only private DMs or the official TELEGRAM_GROUP_ID
  - Unauthorized users are blocked silently (logged for audit)

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


def _sync_fetch_pending_data():
    """Synchronous: fetch pending bets as structured dicts."""
    from bet_agent.db.session import get_session
    from bet_agent.tools.master_analysis import fetch_pending_for_human
    with get_session() as sess:
        return fetch_pending_for_human(sess)


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


def _sync_place_bet(
    bet_id: str,
    user_name: str,
    actual_odds: float | None = None,
    actual_stake: float | None = None,
):
    """Synchronous: mark bet as placed with actual sportsbook values."""
    from bet_agent.db.session import get_session
    from bet_agent.tools.master_analysis import mark_bet_placed_by_user
    with get_session() as sess:
        result = mark_bet_placed_by_user(
            sess, bet_id, user_name,
            actual_odds=actual_odds,
            actual_stake=actual_stake,
        )
        # get_session() auto-commits on success
        return result


def _sync_check_ev(bet_id: str, custom_odds: float) -> dict:
    """Synchronous: pre-check EV at custom odds WITHOUT placing the bet.

    Returns dict with 'ev', 'model_prob', 'is_positive' keys.
    """
    from bet_agent.db.session import get_session
    from bet_agent.db.models import PlacedBet
    import uuid as _uuid

    try:
        uid = _uuid.UUID(bet_id)
    except ValueError:
        return {"error": f"Invalid bet_id: {bet_id}"}

    with get_session() as sess:
        bet = sess.get(PlacedBet, uid)
        if bet is None:
            return {"error": f"Bet {bet_id} not found"}
        model_prob = float(bet.model_prob)
        ev = (model_prob * (custom_odds - 1.0)) - (1.0 - model_prob)
        return {
            "ev": round(ev, 4),
            "model_prob": model_prob,
            "is_positive": ev >= 0,
            "model_odds": float(bet.odds_at_placement),
            "model_stake": float(bet.stake_eur),
        }


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


# ── Callback data prefixes ──────────────────────────────────────────

CALLBACK_PLACE_STD = "place_std:"     # place_std:<bet_id>
CALLBACK_PLACE_CUSTOM = "place_cst:"  # place_cst:<bet_id>

# ConversationHandler states
CONV_AWAITING_ODDS = 0
CONV_AWAITING_STAKE = 1
CONV_TIMEOUT_SECONDS = 300  # 5 minutes


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
        "/pending — Bets awaiting execution (tap to place)\n"
        "/pnl     — Balance & P&L\n"
        "/placed <id> <odds> <stake> — Manual placement\n"
        "/health  — Model health overview\n"
        "/cancel  — Abort custom odds/stake entry\n\n"
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
    """Handle /pending — bets waiting for human execution with InlineKeyboard buttons.

    Each pending bet is rendered as a card with two action buttons:
      ✅ Standard — place at model odds/stake (one tap)
      ✏️ Custom   — enter your actual sportsbook odds/stake
    """
    if not _check_auth(update):
        return

    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        pending = await asyncio.to_thread(_sync_fetch_pending_data)

        if not pending:
            await update.message.reply_text("No pending bets. All clear.")
            return

        await update.message.reply_text(
            f"\U0001f4cb **{len(pending)} bet(s) waiting for placement:**",
            parse_mode="Markdown",
        )

        for bet in pending:
            short_id = bet["bet_id"][:8]
            text = (
                f"\u26bd **{bet['match']}**\n"
                f"Selection: `{bet['selection']}` ({bet['market']})\n"
                f"Odds: {bet['odds']:.2f} | Stake: {bet['stake_eur']:.2f} EUR\n"
                f"Ledger: {bet['ledger']} | ID: `{short_id}...`"
            )

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(
                        "\u2705 Place (standard)",
                        callback_data=f"{CALLBACK_PLACE_STD}{bet['bet_id']}",
                    ),
                    InlineKeyboardButton(
                        "\u270f\ufe0f Custom odds/stake",
                        callback_data=f"{CALLBACK_PLACE_CUSTOM}{bet['bet_id']}",
                    ),
                ]
            ])

            await update.message.reply_text(
                text, reply_markup=keyboard, parse_mode="Markdown",
            )

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
    """Handle /placed <bet_id> <actual_odds> <actual_stake> — confirm manual bet placement.

    The human MUST provide the actual odds and stake obtained at the
    sportsbook. The system recalculates EV on the real numbers, warns
    on -EV, and deducts the actual stake from the bankroll ledger.

    NOTE: The preferred UX is via /pending -> InlineKeyboard buttons.
    This command is kept as a fallback for power users.
    """
    if not _check_auth(update):
        return

    user = update.effective_user
    args = context.args if context.args else []
    if len(args) < 3:
        await update.message.reply_text(
            "Usage: /placed <bet_id> <actual_odds> <actual_stake>\n\n"
            "Example: /placed 1234abcd 1.85 45.00\n\n"
            "You MUST provide the real odds and stake from the sportsbook.\n"
            "Easier: Use /pending for clickable buttons."
        )
        return

    bet_id = args[0]

    try:
        actual_odds = float(args[1])
    except ValueError:
        await update.message.reply_text(f"Invalid odds: {args[1]} — must be a number (e.g. 1.85)")
        return

    try:
        actual_stake = float(args[2])
    except ValueError:
        await update.message.reply_text(f"Invalid stake: {args[2]} — must be a number (e.g. 45.00)")
        return

    if actual_odds <= 1.0:
        await update.message.reply_text("Odds must be > 1.00")
        return
    if actual_stake <= 0:
        await update.message.reply_text("Stake must be > 0")
        return

    user_name = get_user_display_name(user)

    try:
        result = await asyncio.to_thread(
            _sync_place_bet, bet_id, user_name, actual_odds, actual_stake,
        )
        await _send_placement_confirmation(update.message, context, result, bet_id, user_name, user)
    except Exception as e:
        logger.error("Error in /placed: %s", e)
        await update.message.reply_text(f"Error: {e}")


# ── InlineKeyboard callback handler ─────────────────────────────────


async def handle_callback_query(update, context) -> None:
    """Handle InlineKeyboard button presses for bet placement.

    Callback data formats:
      - place_std:<bet_id>   -> place at model odds/stake
      - place_cst:<bet_id>   -> initiate ConversationHandler for custom odds/stake
    """
    query = update.callback_query
    if query is None:
        return

    user = query.from_user
    if not is_authorized(user.id, query.message.chat_id if query.message else None):
        await query.answer("Unauthorized.", show_alert=True)
        return

    data = query.data or ""
    user_name = get_user_display_name(user)

    # ── Standard placement (model values) ────────────────────────────
    if data.startswith(CALLBACK_PLACE_STD):
        bet_id = data[len(CALLBACK_PLACE_STD):]
        await query.answer("Placing bet...")

        try:
            result = await asyncio.to_thread(
                _sync_place_bet, bet_id, user_name,
            )

            if "error" in result:
                await query.edit_message_text(
                    f"\u274c Placement failed: {result['error']}\n\n"
                    f"(Original bet ID: `{bet_id[:8]}...`)",
                    parse_mode="Markdown",
                )
                return

            # State-aware: replace buttons with confirmation
            ev_str = f"{result['ev']:+.4f}"
            ev_emoji = "\u2705" if result["ev"] >= 0 else "\u26a0\ufe0f"
            confirmation = (
                f"{ev_emoji} **PLACED** by {user_name}\n\n"
                f"Match: {result['match']}\n"
                f"Selection: `{result['selection']}`\n"
                f"Stake: {result['stake_eur']:.2f} EUR @ {result['odds']:.2f}\n"
                f"EV: {ev_str}"
            )

            for warning in result.get("warnings", []):
                confirmation += f"\n\u26a0\ufe0f {warning}"

            await query.edit_message_text(confirmation, parse_mode="Markdown")

            # Broadcast
            await _broadcast_placement(context, result, bet_id, user_name, user)

        except Exception as e:
            logger.error("Callback placement error: %s", e)
            await query.edit_message_text(f"\u274c Error: {e}")
        return

    # ── Custom placement (enter ConversationHandler) ─────────────────
    if data.startswith(CALLBACK_PLACE_CUSTOM):
        bet_id = data[len(CALLBACK_PLACE_CUSTOM):]
        await query.answer()

        # Store bet_id in user_data for the ConversationHandler
        context.user_data["custom_bet_id"] = bet_id

        await query.edit_message_text(
            f"\u270f\ufe0f **Custom placement** for `{bet_id[:8]}...`\n\n"
            f"Enter the actual odds from the sportsbook:\n"
            f"(e.g. `1.85`)\n\n"
            f"Type /cancel to abort.",
            parse_mode="Markdown",
        )
        return

    await query.answer("Unknown action.")


# ── ConversationHandler for custom odds/stake ───────────────────────
#
# State machine:
#   [✏️ Custom button] → CONV_AWAITING_ODDS → user enters odds
#     → EV check: if -EV → ABORT (no ledger change)
#     → if +EV → CONV_AWAITING_STAKE → user enters stake → place bet
#   /cancel at any point → abort cleanly
#


async def conv_receive_odds(update, context) -> int:
    """ConversationHandler: receive custom odds from user.

    Performs an EV gate check: if the odds make the bet -EV, the
    placement is ABORTED and the user is informed. No ledger change.
    """
    if not _check_auth(update):
        return -1  # ConversationHandler.END

    text = (update.message.text or "").strip()
    bet_id = context.user_data.get("custom_bet_id", "")

    try:
        odds = float(text)
    except ValueError:
        await update.message.reply_text(
            "Invalid number. Enter the odds as a decimal (e.g. `1.85`).\n"
            "Type /cancel to abort.",
            parse_mode="Markdown",
        )
        return CONV_AWAITING_ODDS

    if odds <= 1.0:
        await update.message.reply_text("Odds must be > 1.00. Try again or /cancel:")
        return CONV_AWAITING_ODDS

    # ── EV gate: check if the bet is still +EV at these odds ─────────
    ev_check = await asyncio.to_thread(_sync_check_ev, bet_id, odds)

    if "error" in ev_check:
        await update.message.reply_text(f"Error: {ev_check['error']}")
        _clear_conv_data(context)
        return -1  # ConversationHandler.END

    if not ev_check["is_positive"]:
        # ABORT: -EV at custom odds → do NOT place, do NOT touch ledger
        await update.message.reply_text(
            f"\U0001f6a8 **ABBRUCH: Negative EV!**\n\n"
            f"Bei Quote {odds:.2f} ist der EV negativ ({ev_check['ev']:+.4f}).\n"
            f"Model-Prob: {ev_check['model_prob']:.1%} | "
            f"Modell-Quote war: {ev_check['model_odds']:.2f}\n\n"
            f"Wette wurde **NICHT** im Ledger verbucht.\n"
            f"Nutze /pending um eine andere Aktion zu wählen.",
            parse_mode="Markdown",
        )
        _clear_conv_data(context)
        return -1  # ConversationHandler.END

    # +EV: proceed to stake entry
    context.user_data["custom_odds"] = odds
    await update.message.reply_text(
        f"Odds: **{odds:.2f}** \u2705 (EV: {ev_check['ev']:+.4f})\n\n"
        f"Now enter the actual stake in EUR (e.g. `45.00`).\n"
        f"Model suggested: {ev_check['model_stake']:.2f} EUR\n\n"
        f"Type /cancel to abort.",
        parse_mode="Markdown",
    )
    return CONV_AWAITING_STAKE


async def conv_receive_stake(update, context) -> int:
    """ConversationHandler: receive custom stake, place the bet."""
    if not _check_auth(update):
        return -1

    text = (update.message.text or "").strip()

    try:
        stake = float(text)
    except ValueError:
        await update.message.reply_text(
            "Invalid number. Enter the stake in EUR (e.g. `45.00`).\n"
            "Type /cancel to abort.",
            parse_mode="Markdown",
        )
        return CONV_AWAITING_STAKE

    if stake <= 0:
        await update.message.reply_text("Stake must be > 0. Try again or /cancel:")
        return CONV_AWAITING_STAKE

    bet_id = context.user_data.get("custom_bet_id", "")
    odds = context.user_data.get("custom_odds", 0.0)
    user = update.effective_user
    user_name = get_user_display_name(user)

    _clear_conv_data(context)

    try:
        result = await asyncio.to_thread(
            _sync_place_bet, bet_id, user_name, odds, stake,
        )
        await _send_placement_confirmation(
            update.message, context, result, bet_id, user_name, user,
        )
    except Exception as e:
        logger.error("Custom placement error: %s", e)
        await update.message.reply_text(f"Error: {e}")

    return -1  # ConversationHandler.END


async def conv_cancel(update, context) -> int:
    """ConversationHandler: handle /cancel — abort custom placement."""
    _clear_conv_data(context)
    await update.message.reply_text(
        "Custom placement cancelled. Use /pending to start over."
    )
    return -1  # ConversationHandler.END


async def conv_timeout(update, context) -> int:
    """ConversationHandler: timeout handler — auto-cancel after inactivity."""
    _clear_conv_data(context)
    # Timeout callbacks receive update=None in some versions
    if update and update.effective_user:
        logger.info(
            "Custom placement timed out for user %d",
            update.effective_user.id,
        )
    return -1  # ConversationHandler.END


def _clear_conv_data(context) -> None:
    """Remove custom placement data from user_data."""
    context.user_data.pop("custom_bet_id", None)
    context.user_data.pop("custom_odds", None)


# ── Free-text handler (Master Agent NL routing) ─────────────────────


async def handle_message(update, context) -> None:
    """Handle free-text messages — route to Master Agent via NL bridge.

    NOTE: Custom odds/stake input is handled by the ConversationHandler,
    NOT this function. This only processes messages outside any conversation.
    """
    if not _check_auth(update):
        user = update.effective_user
        logger.warning(
            "Unauthorized message from user_id=%d in chat_id=%d: %s",
            user.id, update.effective_chat.id if update.effective_chat else 0,
            (update.message.text or "")[:50],
        )
        return  # Silent block

    text = (update.message.text or "").strip()
    if not text:
        return

    user = update.effective_user
    user_name = get_user_display_name(user)
    logger.info("NL query from %s (id=%d): %s", user_name, user.id, text[:100])

    try:
        response = await asyncio.to_thread(_master_bridge.query, text, user_name)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error("Master Agent bridge error: %s", e)
        await update.message.reply_text(
            "The Master Agent encountered an error processing your request. "
            "Please try again or use a slash command."
        )


# ── Shared placement confirmation + broadcast ───────────────────────


async def _send_placement_confirmation(message, context, result, bet_id, user_name, user):
    """Send placement confirmation (reusable for /placed and button flows)."""
    if "error" in result:
        await message.reply_text(f"Error: {result['error']}")
        return

    ev_str = f"{result['ev']:+.4f}"
    ev_emoji = "\u2705" if result["ev"] >= 0 else "\u26a0\ufe0f"
    lines = [
        f"{ev_emoji} Bet confirmed as placed!",
        "",
        f"Match: {result['match']}",
        f"Selection: {result['selection']}",
        f"Stake: {result['stake_eur']:.2f} EUR @ {result['odds']:.2f}",
        f"EV at actual odds: {ev_str}",
        f"Placed by: {user_name}",
    ]

    if result.get("original_odds") and abs(result["odds"] - result["original_odds"]) > 0.001:
        lines.append(
            f"\nOriginal model odds: {result['original_odds']:.2f} "
            f"-> Actual: {result['odds']:.2f}"
        )

    for warning in result.get("warnings", []):
        lines.append(f"\n\u26a0\ufe0f {warning}")

    await message.reply_text("\n".join(lines))
    await _broadcast_placement(context, result, bet_id, user_name, user)


async def _broadcast_placement(context, result, bet_id, user_name, user):
    """Broadcast placement to group and individual syndicate members."""
    ev_str = f"{result['ev']:+.4f}"

    if TELEGRAM_GROUP_ID and context.bot:
        broadcast_msg = (
            f"\U0001f4e2 BET PLACED by {user_name}\n\n"
            f"Match: {result['match']}\n"
            f"Selection: {result['selection']}\n"
            f"Stake: {result['stake_eur']:.2f} EUR @ {result['odds']:.2f}\n"
            f"EV: {ev_str}\n"
            f"ID: {bet_id[:8]}..."
        )
        try:
            await context.bot.send_message(
                chat_id=TELEGRAM_GROUP_ID,
                text=broadcast_msg,
            )
        except Exception as exc:
            logger.error("Group broadcast failed: %s", exc)

    for uid in ALLOWED_IDS:
        if uid != user.id and uid != int(TELEGRAM_GROUP_ID or 0):
            try:
                await context.bot.send_message(
                    chat_id=uid,
                    text=(
                        f"\U0001f4e2 {user_name} placed: "
                        f"{result['match']} — {result['selection']} "
                        f"@ {result['odds']:.2f} ({result['stake_eur']:.2f} EUR) "
                        f"EV: {ev_str}"
                    ),
                )
            except Exception:
                pass  # User may not have started the bot yet


# ── Alert digest / batching (thread-safe asyncio.Queue) ─────────────

_alert_queue: asyncio.Queue[str] = asyncio.Queue()
_DIGEST_INTERVAL_SECONDS = 600  # 10 minutes


def queue_alert(text: str) -> None:
    """Add an alert to the digest queue (thread-safe, called from any thread)."""
    try:
        _alert_queue.put_nowait(text)
    except asyncio.QueueFull:
        logger.warning("Alert queue full, dropping alert: %s", text[:50])


async def flush_alert_digest(context) -> None:
    """Flush queued alerts as a single digest message.

    Called by the Application job_queue every DIGEST_INTERVAL_SECONDS.
    If only 1-2 alerts, send immediately. If 3+, combine into a digest.
    """
    alerts: list[str] = []
    while not _alert_queue.empty():
        try:
            alerts.append(_alert_queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    if not alerts:
        return

    if len(alerts) <= 2:
        for alert in alerts:
            await _send_to_all(context, alert)
    else:
        digest = (
            f"\U0001f4cb **Alert Digest** ({len(alerts)} items)\n"
            + "\n---\n".join(alerts)
        )
        await _send_to_all(context, digest)


async def _send_to_all(context, text: str) -> None:
    """Send a message to the group and all whitelisted users."""
    if TELEGRAM_GROUP_ID and context.bot:
        try:
            await context.bot.send_message(chat_id=TELEGRAM_GROUP_ID, text=text)
        except Exception as exc:
            logger.error("Digest broadcast failed: %s", exc)

    for uid in ALLOWED_IDS:
        if uid != int(TELEGRAM_GROUP_ID or 0):
            try:
                await context.bot.send_message(chat_id=uid, text=text)
            except Exception:
                pass


# ── Bot builder ──────────────────────────────────────────────────────


def build_application():
    """Build the Telegram Application with all handlers registered.

    Uses ConversationHandler for the custom odds/stake flow:
      - Entry: CallbackQuery with CALLBACK_PLACE_CUSTOM prefix
      - State 0 (AWAITING_ODDS): user enters odds → EV gate
      - State 1 (AWAITING_STAKE): user enters stake → place bet
      - /cancel or timeout (5min) → clean abort

    Returns:
        telegram.ext.Application instance (not yet running).
    """
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ConversationHandler,
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

    # ConversationHandler for custom odds/stake entry
    conv_handler = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(
                _conv_entry,
                pattern=f"^{CALLBACK_PLACE_CUSTOM.replace(':', ':')}",
            ),
        ],
        states={
            CONV_AWAITING_ODDS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, conv_receive_odds),
            ],
            CONV_AWAITING_STAKE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, conv_receive_stake),
            ],
            ConversationHandler.TIMEOUT: [
                MessageHandler(filters.ALL, conv_timeout),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", conv_cancel),
        ],
        conversation_timeout=CONV_TIMEOUT_SECONDS,
        per_user=True,
        per_chat=True,
    )
    app.add_handler(conv_handler)

    # Standard placement button (outside ConversationHandler)
    app.add_handler(CallbackQueryHandler(
        handle_callback_query,
        pattern=f"^{CALLBACK_PLACE_STD}",
    ))

    # Free-text → Master Agent NL bridge (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Schedule alert digest flush every 10 minutes
    if app.job_queue is not None:
        app.job_queue.run_repeating(
            flush_alert_digest, interval=_DIGEST_INTERVAL_SECONDS, first=60,
        )

    logger.info(
        "Telegram bot configured with %d whitelisted users, "
        "ConversationHandler + InlineKeyboard enabled",
        len(ALLOWED_IDS),
    )

    return app


async def _conv_entry(update, context) -> int:
    """ConversationHandler entry point: handle ✏️ Custom button click."""
    query = update.callback_query
    if query is None:
        return -1

    user = query.from_user
    if not is_authorized(user.id, query.message.chat_id if query.message else None):
        await query.answer("Unauthorized.", show_alert=True)
        return -1

    data = query.data or ""
    bet_id = data[len(CALLBACK_PLACE_CUSTOM):]
    await query.answer()

    context.user_data["custom_bet_id"] = bet_id

    await query.edit_message_text(
        f"\u270f\ufe0f **Custom placement** for `{bet_id[:8]}...`\n\n"
        f"Enter the actual odds from the sportsbook:\n"
        f"(e.g. `1.85`)\n\n"
        f"Type /cancel to abort. Auto-cancels after 5 min.",
        parse_mode="Markdown",
    )
    return CONV_AWAITING_ODDS


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
