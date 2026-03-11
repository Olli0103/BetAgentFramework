"""Master Alert System — push notifications for final betting tickets.

Supports multiple notification channels:
  - Telegram webhook (production)
  - macOS osascript desktop notification (development)
  - Console/logging fallback (always available)

The Final Ticket consolidates vetted, sized, and shopped bet data
into a human-readable alert for the operator.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from decimal import Decimal

from bet_agent.db.models import LedgerType, Match, Prediction

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────

_SPORT_EMOJI = {
    "football": "\u26bd",
    "tennis": "\U0001f3be",
    "ice_hockey": "\U0001f3d2",
    "basketball": "\U0001f3c0",
    "american_football": "\U0001f3c8",
    "darts": "\U0001f3af",
}


@dataclass
class BetTicket:
    """A final, human-readable bet ticket ready for notification."""

    prediction_id: object  # UUID
    sport: str
    match_description: str  # "Zverev vs Alcaraz"
    league: str
    market: str  # "Match Winner - Alcaraz"
    selection: str
    stake_eur: float
    best_odds: float
    best_sportsbook: str
    model_edge_pct: float  # e.g., 8.4
    ev: float
    veto_status: str  # "PASSED" or reason
    ledger_type: str  # "REAL" or "PAPER"
    model_source: str


def build_ticket(
    prediction: Prediction,
    match: Match,
    stake_eur: float,
    ledger_type: LedgerType,
) -> BetTicket:
    """Build a BetTicket from prediction, match, and sizing data."""
    best_odds = float(prediction.best_odds) if prediction.best_odds else 0.0
    best_book = prediction.best_sportsbook or "unknown"

    market_display = prediction.market_type.value.replace("_", " ").title()
    market_str = f"{market_display} - {prediction.selection}"

    edge_pct = float(prediction.prob_edge) * 100.0

    veto = "PASSED"
    if prediction.veto_reason:
        veto = prediction.veto_reason

    return BetTicket(
        prediction_id=prediction.id,
        sport=match.sport.value,
        match_description=f"{match.home_team} vs {match.away_team}",
        league=match.league,
        market=market_str,
        selection=prediction.selection,
        stake_eur=stake_eur,
        best_odds=best_odds,
        best_sportsbook=best_book,
        model_edge_pct=round(edge_pct, 1),
        ev=round(float(prediction.ev), 4),
        ledger_type=ledger_type.value.upper(),
        model_source=prediction.model_source or "analytical",
        veto_status=veto,
    )


# ── Message formatting ───────────────────────────────────────────────


def format_ticket_message(ticket: BetTicket) -> str:
    """Format a BetTicket as a human-readable notification message."""
    emoji = _SPORT_EMOJI.get(ticket.sport, "\U0001f3c6")
    ledger_tag = f"[{ticket.ledger_type}]" if ticket.ledger_type == "PAPER" else ""

    lines = [
        "\U0001f6a8 NEW +EV BET READY " + ledger_tag,
        f"{emoji} Match: {ticket.match_description}",
        f"\U0001f3c6 League: {ticket.league}",
        f"\U0001f4c8 Market: {ticket.market}",
        f"\U0001f4b0 Stake: {ticket.stake_eur:.2f}EUR (Quarter-Kelly)",
        f"\U0001f3c6 Best Odds: {ticket.best_odds:.2f} ({ticket.best_sportsbook})",
        f"\U0001f9e0 Model Edge: +{ticket.model_edge_pct}%",
        f"\U0001f6e1\ufe0f Veto Check: {ticket.veto_status}",
        f"\U0001f4ca Source: {ticket.model_source}",
    ]
    return "\n".join(lines)


def format_daily_summary(tickets: list[BetTicket]) -> str:
    """Format a daily summary of all bet tickets."""
    if not tickets:
        return "\u2705 Daily Summary: No +EV bets found today."

    total_stake = sum(t.stake_eur for t in tickets)
    real_count = sum(1 for t in tickets if t.ledger_type == "REAL")
    paper_count = sum(1 for t in tickets if t.ledger_type == "PAPER")

    header = (
        f"\U0001f4cb DAILY BET SUMMARY\n"
        f"Bets: {len(tickets)} ({real_count} real, {paper_count} paper)\n"
        f"Total Stake: {total_stake:.2f}EUR\n"
        f"{'=' * 40}"
    )

    body = "\n\n".join(format_ticket_message(t) for t in tickets)
    return f"{header}\n\n{body}"


# ── Notification backends ────────────────────────────────────────────


class NotificationBackend:
    """Base class for notification backends."""

    def send(self, message: str, title: str = "BetAgent Alert") -> bool:
        """Send a notification. Returns True on success."""
        raise NotImplementedError


class ConsoleNotifier(NotificationBackend):
    """Logs alerts to console/logger. Always available."""

    def send(self, message: str, title: str = "BetAgent Alert") -> bool:
        logger.info("[%s] %s", title, message)
        return True


class TelegramNotifier(NotificationBackend):
    """Sends alerts via Telegram Bot API webhook."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    def send(self, message: str, title: str = "BetAgent Alert") -> bool:
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured (missing bot_token or chat_id)")
            return False

        try:
            import urllib.request

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False


class MacOSNotifier(NotificationBackend):
    """macOS desktop notification via osascript."""

    def send(self, message: str, title: str = "BetAgent Alert") -> bool:
        if platform.system() != "Darwin":
            logger.debug("macOS notifier skipped (not on macOS)")
            return False

        try:
            # Truncate for osascript (max ~1000 chars)
            short_msg = message[:500].replace('"', '\\"')
            safe_title = title.replace('"', '\\"')
            script = (
                f'display notification "{short_msg}" '
                f'with title "{safe_title}" '
                f'sound name "Glass"'
            )
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
            return True
        except Exception as exc:
            logger.error("macOS notification failed: %s", exc)
            return False


# ── Main notification dispatcher ─────────────────────────────────────


def get_notifiers() -> list[NotificationBackend]:
    """Build list of available notification backends based on environment."""
    notifiers: list[NotificationBackend] = [ConsoleNotifier()]

    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        notifiers.append(TelegramNotifier())

    if platform.system() == "Darwin":
        notifiers.append(MacOSNotifier())

    return notifiers


def push_alert(
    ticket: BetTicket,
    notifiers: list[NotificationBackend] | None = None,
) -> bool:
    """Push a single bet ticket alert through all configured notifiers.

    Returns True if at least one notifier succeeded.
    """
    if notifiers is None:
        notifiers = get_notifiers()

    message = format_ticket_message(ticket)
    success = False

    for notifier in notifiers:
        try:
            if notifier.send(message):
                success = True
        except Exception as exc:
            logger.error("Notifier %s failed: %s", type(notifier).__name__, exc)

    return success


def push_daily_summary(
    tickets: list[BetTicket],
    notifiers: list[NotificationBackend] | None = None,
) -> bool:
    """Push a daily summary of all bet tickets.

    Returns True if at least one notifier succeeded.
    """
    if notifiers is None:
        notifiers = get_notifiers()

    message = format_daily_summary(tickets)
    success = False

    for notifier in notifiers:
        try:
            if notifier.send(message, title="BetAgent Daily Summary"):
                success = True
        except Exception as exc:
            logger.error("Notifier %s failed: %s", type(notifier).__name__, exc)

    return success
