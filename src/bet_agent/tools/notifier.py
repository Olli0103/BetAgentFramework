"""Master Alert System — push notifications for final betting tickets.

Supports multiple notification channels:
  - Telegram syndicate broadcast (production) — sends to ALL whitelisted IDs
  - Telegram group chat (if TELEGRAM_GROUP_ID is set)
  - macOS osascript desktop notification (development)
  - Console/logging fallback (always available)

The Final Ticket consolidates vetted, sized, and shopped bet data
into a human-readable alert for the operator and syndicate members.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from bet_agent.db.models import LedgerType, Match, OddsMarket, Prediction

logger = logging.getLogger(__name__)

# Maximum age (minutes) for odds to be considered fresh
_ODDS_FRESHNESS_MINUTES = 120

# Minimum team name length to pass quality check
_MIN_NAME_LEN = 4


# ── Readiness gate ──────────────────────────────────────────────────


@dataclass
class ReadinessCheck:
    """Result of a bettable readiness check."""

    is_ready: bool
    checks: dict[str, bool]  # check_name → passed
    reason: str  # Human-readable reason if not ready

    @property
    def failed_checks(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


def check_bet_readiness(
    prediction: Prediction,
    match: Match,
    stake_eur: float,
) -> ReadinessCheck:
    """Final quality gate before a bet is recommended for REAL placement.

    Checks:
      1. odds_available — prediction has best_odds populated
      2. odds_reasonable — best_odds > 1.0 and < 100.0
      3. team_names_ok — home/away names are not abbreviations (>= 4 chars)
      4. stake_positive — stake > 0
      5. ev_positive — EV > 0

    Returns a ReadinessCheck. If not ready, the bet should be routed to
    PAPER ledger or flagged as DO_NOT_BET in the ticket.
    """
    checks: dict[str, bool] = {}

    # 1. Odds available
    checks["odds_available"] = prediction.best_odds is not None and float(prediction.best_odds) > 0

    # 2. Odds reasonable
    if prediction.best_odds is not None:
        odds_val = float(prediction.best_odds)
        checks["odds_reasonable"] = 1.0 < odds_val < 100.0
    else:
        checks["odds_reasonable"] = False

    # 3. Team name quality
    checks["team_names_ok"] = (
        len(match.home_team.strip()) >= _MIN_NAME_LEN
        and len(match.away_team.strip()) >= _MIN_NAME_LEN
    )

    # 4. Stake positive
    checks["stake_positive"] = stake_eur > 0

    # 5. EV positive
    checks["ev_positive"] = float(prediction.ev) > 0

    is_ready = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    reason = ""
    if not is_ready:
        reason = f"Failed: {', '.join(failed)}"
        logger.warning(
            "Readiness gate FAILED for %s: %s",
            prediction.selection, reason,
        )

    return ReadinessCheck(is_ready=is_ready, checks=checks, reason=reason)


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

    prediction_id: uuid.UUID
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
    """Build a BetTicket from prediction, match, and sizing data.

    Runs the readiness gate — if the bet fails quality checks, it's
    downgraded to PAPER ledger and the veto_status reflects the failure.
    """
    # ── Readiness gate ────────────────────────────────────────
    readiness = check_bet_readiness(prediction, match, stake_eur)
    if not readiness.is_ready and ledger_type == LedgerType.REAL:
        logger.warning(
            "Downgrading %s to PAPER — readiness gate failed: %s",
            prediction.selection, readiness.reason,
        )
        ledger_type = LedgerType.PAPER

    best_odds = float(prediction.best_odds) if prediction.best_odds else 0.0
    best_book = prediction.best_sportsbook or "unknown"

    market_display = prediction.market_type.value.replace("_", " ").title()
    market_str = f"{market_display} - {prediction.selection}"

    edge_pct = float(prediction.prob_edge) * 100.0

    veto = "PASSED"
    if prediction.veto_reason:
        veto = prediction.veto_reason
    if not readiness.is_ready:
        veto = f"READINESS FAIL: {readiness.reason}"

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

    readiness_warn = ""
    if "READINESS FAIL" in ticket.veto_status:
        readiness_warn = "\n\u26a0\ufe0f DO NOT BET (REAL) \u2014 data quality issues detected"

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
    if readiness_warn:
        lines.append(readiness_warn)
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
    """Sends alerts to a single Telegram chat (legacy single-user mode)."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    def _send_to_chat(self, chat_id: str, message: str) -> bool:
        """Send a message to a specific Telegram chat_id."""
        import urllib.request

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
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

    def send(self, message: str, title: str = "BetAgent Alert") -> bool:
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured (missing bot_token or chat_id)")
            return False

        try:
            return self._send_to_chat(self.chat_id, message)
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False


class TelegramSyndicateBroadcaster(NotificationBackend):
    """Broadcasts alerts to ALL whitelisted syndicate members.

    Reads ALLOWED_TELEGRAM_IDS from env and sends to each member.
    Also sends to TELEGRAM_GROUP_ID if set.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        allowed_ids: str | None = None,
        group_id: str | None = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._allowed_raw = allowed_ids or os.environ.get("ALLOWED_TELEGRAM_IDS", "")
        self.group_id = group_id or os.environ.get("TELEGRAM_GROUP_ID", "")

    def _get_target_ids(self) -> list[str]:
        """Parse all target chat IDs for broadcasting."""
        targets: list[str] = []

        # Individual whitelisted users
        for part in self._allowed_raw.split(","):
            part = part.strip()
            if part:
                targets.append(part)

        # Group chat (if configured and not already in list)
        if self.group_id and self.group_id not in targets:
            targets.append(self.group_id)

        return targets

    def send(self, message: str, title: str = "BetAgent Alert") -> bool:
        if not self.bot_token:
            logger.warning("Telegram bot token not configured for broadcast")
            return False

        targets = self._get_target_ids()
        if not targets:
            logger.warning("No broadcast targets configured")
            return False

        import urllib.request

        success_count = 0
        for chat_id in targets:
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                payload = json.dumps({
                    "chat_id": chat_id,
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
                    if resp.status == 200:
                        success_count += 1
            except Exception as exc:
                logger.warning("Broadcast to %s failed: %s", chat_id, exc)

        logger.info(
            "Syndicate broadcast: %d/%d targets reached",
            success_count, len(targets),
        )
        return success_count > 0


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
    """Build list of available notification backends based on environment.

    Priority:
      1. Console (always)
      2. Syndicate Broadcaster (if ALLOWED_TELEGRAM_IDS is set — broadcasts to all)
      3. Single TelegramNotifier fallback (if only TELEGRAM_CHAT_ID is set)
      4. macOS desktop (if on Darwin)
    """
    notifiers: list[NotificationBackend] = [ConsoleNotifier()]

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    allowed_ids = os.environ.get("ALLOWED_TELEGRAM_IDS", "")
    single_chat = os.environ.get("TELEGRAM_CHAT_ID", "")

    if bot_token and allowed_ids:
        # Syndicate mode: broadcast to all whitelisted members
        notifiers.append(TelegramSyndicateBroadcaster())
    elif bot_token and single_chat:
        # Legacy single-user mode
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
