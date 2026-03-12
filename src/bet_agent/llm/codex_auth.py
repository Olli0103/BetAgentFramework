"""Auto-refresh for OpenAI Codex OAuth tokens.

Reads ``~/.codex/auth.json`` (written by ``codex login``), and transparently
refreshes the ``access_token`` using the ``refresh_token`` before it expires.

The module is intentionally self-contained — it only needs ``requests`` and
the stdlib.  No OpenAI SDK required.

Usage::

    from bet_agent.llm.codex_auth import get_access_token

    token = get_access_token()  # always fresh
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

_AUTH_FILE = Path.home() / ".codex" / "auth.json"

# Public Codex CLI OAuth client — not a secret.
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"

# Refresh 5 minutes before actual expiry to avoid mid-request failures.
_REFRESH_MARGIN_SECONDS = 300


# ── Token manager (singleton, thread-safe) ───────────────────────────────


class CodexTokenManager:
    """Thread-safe manager that caches and auto-refreshes the Codex token."""

    def __init__(self, auth_file: Path = _AUTH_FILE) -> None:
        self._auth_file = auth_file
        self._lock = threading.Lock()
        self._access_token: str | None = None
        self._expires_at: float = 0.0  # epoch
        self._refresh_token: str | None = None

    # ── Public API ───────────────────────────────────────────────────

    def get_token(self) -> str:
        """Return a valid access token, refreshing if needed.

        Raises ``RuntimeError`` if no token can be obtained.
        """
        with self._lock:
            # 1. If we have a cached token that isn't close to expiry, use it
            if self._access_token and time.time() < self._expires_at - _REFRESH_MARGIN_SECONDS:
                return self._access_token

            # 2. Try to refresh using the stored refresh_token
            if self._refresh_token:
                try:
                    self._do_refresh()
                    return self._access_token  # type: ignore[return-value]
                except Exception:
                    logger.warning("Token refresh failed — reloading from disk")

            # 3. (Re-)load from ~/.codex/auth.json
            self._load_from_disk()

            # 4. If the loaded token is already expired, try refresh
            if time.time() >= self._expires_at - _REFRESH_MARGIN_SECONDS:
                if self._refresh_token:
                    self._do_refresh()
                else:
                    raise RuntimeError(
                        f"Codex token expired and no refresh_token in {self._auth_file}. "
                        "Run `codex login` to re-authenticate."
                    )

            if not self._access_token:
                raise RuntimeError(
                    f"No Codex access token available. "
                    f"Run `codex login` or set OPENCLAW_OAUTH_TOKEN."
                )

            return self._access_token

    # ── Internals ────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        """Read ``~/.codex/auth.json`` and extract tokens + expiry."""
        if not self._auth_file.exists():
            raise RuntimeError(
                f"{self._auth_file} not found. Run `codex login` first."
            )

        data: dict[str, Any] = json.loads(self._auth_file.read_text())
        tokens = data.get("tokens", {})

        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")

        if not access_token:
            raise RuntimeError(
                f"No access_token in {self._auth_file}. Run `codex login`."
            )

        # Decode JWT expiry without a JWT library — the payload is the
        # second dot-separated segment, base64url-encoded.
        self._expires_at = _decode_jwt_exp(access_token)
        self._access_token = access_token
        self._refresh_token = refresh_token or None

        logger.debug(
            "Loaded Codex token from disk (expires in %.0f min)",
            (self._expires_at - time.time()) / 60,
        )

    def _do_refresh(self) -> None:
        """Exchange the ``refresh_token`` for a new ``access_token``."""
        if not self._refresh_token:
            raise RuntimeError("No refresh_token available")

        logger.info("Refreshing Codex OAuth access token …")

        resp = requests.post(
            _TOKEN_ENDPOINT,
            data={
                "grant_type": "refresh_token",
                "client_id": _CLIENT_ID,
                "refresh_token": self._refresh_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()

        new_access = body.get("access_token", "")
        new_refresh = body.get("refresh_token", self._refresh_token)

        if not new_access:
            raise RuntimeError("Token refresh response missing access_token")

        self._access_token = new_access
        self._refresh_token = new_refresh
        self._expires_at = _decode_jwt_exp(new_access)

        # Persist the refreshed tokens back to disk so Codex CLI and
        # future process restarts see the latest credentials.
        self._persist_to_disk(body)

        logger.info(
            "Codex token refreshed (expires in %.0f min)",
            (self._expires_at - time.time()) / 60,
        )

    def _persist_to_disk(self, token_response: dict[str, Any]) -> None:
        """Write refreshed tokens back to ``~/.codex/auth.json``."""
        try:
            if self._auth_file.exists():
                data = json.loads(self._auth_file.read_text())
            else:
                data = {}

            tokens = data.setdefault("tokens", {})
            tokens["access_token"] = token_response.get(
                "access_token", tokens.get("access_token")
            )
            if "refresh_token" in token_response:
                tokens["refresh_token"] = token_response["refresh_token"]
            if "id_token" in token_response:
                tokens["id_token"] = token_response["id_token"]

            self._auth_file.write_text(json.dumps(data, indent=2) + "\n")
            logger.debug("Persisted refreshed tokens to %s", self._auth_file)
        except Exception:
            # Non-fatal — we already have the token in memory
            logger.warning("Could not persist refreshed token to disk", exc_info=True)


# ── JWT helpers (no dependency) ──────────────────────────────────────────

import base64


def _decode_jwt_exp(token: str) -> float:
    """Extract ``exp`` (epoch) from a JWT without validating the signature."""
    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("Invalid JWT — expected at least 2 dot-separated parts")

    # Base64url decode the payload (2nd segment)
    payload_b64 = parts[1]
    # Add padding if needed
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding

    payload_bytes = base64.urlsafe_b64decode(payload_b64)
    payload = json.loads(payload_bytes)

    exp = payload.get("exp")
    if exp is None:
        raise ValueError("JWT payload has no 'exp' claim")

    return float(exp)


# ── Singleton + convenience ──────────────────────────────────────────────

_manager: CodexTokenManager | None = None


def get_access_token(auth_file: Path | None = None) -> str:
    """Return a fresh Codex access token (auto-refreshes if needed).

    This is the primary entry point for the rest of the codebase.
    """
    global _manager
    if _manager is None:
        _manager = CodexTokenManager(auth_file or _AUTH_FILE)
    return _manager.get_token()
