"""Tier-aware LLM client with ordered provider fallback chain.

Reads ``config/llm_tiers.yaml`` and corresponding env vars, then exposes a
simple ``chat()`` function that tries providers in order and transparently
falls back when one is unavailable (missing credentials, timeout, HTTP error).

Design decisions:
  * Uses raw ``requests`` — no vendor SDK required.  All providers expose
    OpenAI-compatible ``/chat/completions`` endpoints.
  * Provider chain: OpenClaw → OpenRouter → Gemini (configurable via YAML).
  * One ``LLMClient`` instance per tier; the module-level ``chat()``
    convenience function targets **tier1_heavy_reasoning** by default.
  * Stateless — each call is an independent HTTP request.
  * OpenClaw / Codex OAuth tokens are auto-refreshed via ``codex_auth``
    when ``auth: codex_cli`` is set in the tier config.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

logger = logging.getLogger(__name__)

# ── Configuration data classes ────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "llm_tiers.yaml"


@dataclass(frozen=True)
class ProviderConfig:
    """Connection details for a single LLM provider."""

    name: str
    base_url: str
    model: str
    api_key: str
    max_tokens: int = 8192
    temperature: float = 0.2
    timeout: int = 60
    # When set, called instead of using ``api_key`` directly.
    # Returns a fresh token string each time (for OAuth auto-refresh).
    _token_fn: Any = None  # Callable[[], str] | None


@dataclass
class TierConfig:
    """Ordered list of providers for one tier."""

    tier_name: str
    providers: list[ProviderConfig] = field(default_factory=list)

    # Backward-compat properties for code that reads .primary / .fallback
    @property
    def primary(self) -> ProviderConfig | None:
        return self.providers[0] if self.providers else None

    @property
    def fallback(self) -> ProviderConfig | None:
        return self.providers[1] if len(self.providers) > 1 else None


# ── Provider URL builders ─────────────────────────────────────────────

_PROVIDER_URLS: dict[str, str] = {
    "openclaw": "https://api.openclaw.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "",  # filled from env
}


def _resolve_provider(cfg: dict[str, Any]) -> ProviderConfig | None:
    """Build a ``ProviderConfig`` from a raw YAML dict, or *None* if
    credentials are missing."""
    provider = cfg.get("provider", "")
    auth_mode = cfg.get("auth", "")
    token_fn = None

    # ── Codex CLI auto-refresh (reads ~/.codex/auth.json) ────────
    if auth_mode == "codex_cli":
        try:
            from bet_agent.llm.codex_auth import get_access_token

            # Verify that auth.json exists and is readable
            test_token = get_access_token()
            api_key = test_token  # initial value
            token_fn = get_access_token
            logger.info("Codex CLI auth active — tokens auto-refresh from ~/.codex/auth.json")
        except Exception as exc:
            logger.debug("Codex CLI auth unavailable (%s) — trying env fallback", exc)
            # Fall through to env-based auth below
            token_fn = None
            api_key = ""

    # ── Standard env-based auth ──────────────────────────────────
    if token_fn is None:
        api_key = ""
        if "auth_env" in cfg:
            api_key = os.environ.get(cfg["auth_env"], "")
        elif "api_key_env" in cfg:
            api_key = os.environ.get(cfg["api_key_env"], "")

    if not api_key and token_fn is None:
        logger.debug("No credentials for provider %s — skipping", provider)
        return None

    # Base URL
    if provider == "ollama":
        base_url = os.environ.get(cfg.get("base_url_env", ""), "http://localhost:11434") + "/v1"
    else:
        base_url = _PROVIDER_URLS.get(provider, cfg.get("base_url", ""))

    if not base_url:
        return None

    # Model: env override takes precedence over YAML default
    model = cfg.get("model", "")
    if "model_env" in cfg:
        model = os.environ.get(cfg["model_env"], "") or model

    return ProviderConfig(
        name=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=cfg.get("max_tokens", 8192),
        temperature=cfg.get("temperature", 0.2),
        timeout=cfg.get("timeout_seconds", 60),
        _token_fn=token_fn,
    )


# ── Config loader ─────────────────────────────────────────────────────


def load_tier_configs(
    config_path: Path | str = _CONFIG_PATH,
) -> dict[str, TierConfig]:
    """Parse ``llm_tiers.yaml`` and return a mapping of tier name → config.

    Supports both the new ``providers`` list format and the legacy
    ``primary`` / ``fallback`` format for backward compatibility.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("LLM tier config not found at %s", path)
        return {}

    raw = yaml.safe_load(path.read_text())
    tiers: dict[str, TierConfig] = {}
    for tier_name, tier_data in (raw.get("tiers") or {}).items():
        resolved: list[ProviderConfig] = []

        # New format: ordered providers list
        if "providers" in tier_data:
            for prov_cfg in tier_data["providers"]:
                p = _resolve_provider(prov_cfg)
                if p:
                    resolved.append(p)
        else:
            # Legacy format: primary + fallback
            primary = _resolve_provider(tier_data.get("primary") or {})
            if primary:
                resolved.append(primary)
            fallback_raw = tier_data.get("fallback")
            if fallback_raw:
                fb = _resolve_provider(fallback_raw)
                if fb:
                    resolved.append(fb)

        tiers[tier_name] = TierConfig(
            tier_name=tier_name,
            providers=resolved,
        )
    return tiers


# ── Core LLM client ──────────────────────────────────────────────────


@dataclass
class LLMClient:
    """Tier-aware LLM client with automatic fallback.

    Usage::

        client = LLMClient.for_tier("tier1_heavy_reasoning")
        reply = client.chat("Summarise today's portfolio risk.")
    """

    tier: TierConfig
    _providers: list[ProviderConfig] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._providers = list(self.tier.providers)

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def for_tier(
        cls,
        tier_name: str = "tier1_heavy_reasoning",
        config_path: Path | str = _CONFIG_PATH,
    ) -> LLMClient:
        tiers = load_tier_configs(config_path)
        tier = tiers.get(tier_name)
        if tier is None:
            raise ValueError(
                f"Tier '{tier_name}' not found in {config_path}. "
                f"Available: {list(tiers)}"
            )
        return cls(tier=tier)

    # ── Chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request, falling back on failure.

        Returns the assistant's reply text, or raises if all providers fail.
        """
        if not self._providers:
            raise RuntimeError(
                f"No LLM providers available for tier '{self.tier.tier_name}'. "
                f"Set OPENCLAW_OAUTH_TOKEN, OPENROUTER_API_KEY, or GEMINI_API_KEY "
                f"in your environment."
            )

        last_err: Exception | None = None
        for prov in self._providers:
            try:
                return self._call(prov, user_message, system_prompt, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Provider %s (%s) failed: %s — trying next",
                    prov.name,
                    prov.model,
                    exc,
                )
                last_err = exc

        raise RuntimeError(
            f"All LLM providers exhausted for tier '{self.tier.tier_name}'"
        ) from last_err

    # ── Internal HTTP call ────────────────────────────────────────────

    @staticmethod
    def _call(
        prov: ProviderConfig,
        user_message: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload: dict[str, Any] = {
            "model": prov.model,
            "messages": messages,
            "max_tokens": max_tokens or prov.max_tokens,
            "temperature": temperature if temperature is not None else prov.temperature,
        }

        # Use dynamic token if available (Codex auto-refresh), else static key
        token = prov._token_fn() if prov._token_fn else prov.api_key

        url = f"{prov.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        logger.debug("LLM request → %s (%s)", prov.name, prov.model)
        resp = requests.post(url, json=payload, headers=headers, timeout=prov.timeout)
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def active_provider(self) -> str | None:
        """Name of the first available provider (for logging)."""
        return self._providers[0].name if self._providers else None


# ── Module-level convenience ──────────────────────────────────────────

_default_client: LLMClient | None = None


def _get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient.for_tier("tier1_heavy_reasoning")
    return _default_client


def chat(
    message: str,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Quick tier-1 chat — ``from bet_agent.llm import chat``."""
    return _get_default_client().chat(
        message,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
