"""Tests for the LLM client — tier loading, fallback routing, and MasterAgentBridge."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from bet_agent.llm.client import (
    LLMClient,
    ProviderConfig,
    TierConfig,
    load_tier_configs,
)


# ── Helpers ───────────────────────────────────────────────────────────

SAMPLE_TIER_YAML = textwrap.dedent("""\
    tiers:
      tier1_heavy_reasoning:
        description: "test tier"
        providers:
          - provider: openclaw
            model: openclaw/codex
            auth: oauth
            auth_env: OPENCLAW_OAUTH_TOKEN
            max_tokens: 8192
            temperature: 0.2
            timeout_seconds: 60
          - provider: openrouter
            model: anthropic/claude-sonnet-4
            api_key_env: OPENROUTER_API_KEY
            max_tokens: 8192
            temperature: 0.2
            timeout_seconds: 60
          - provider: google
            model: gemini-2.0-flash
            api_key_env: GEMINI_API_KEY
            max_tokens: 8192
            temperature: 0.3
            timeout_seconds: 45
""")

# Legacy format for backward-compat test
LEGACY_TIER_YAML = textwrap.dedent("""\
    tiers:
      tier1_heavy_reasoning:
        description: "legacy format"
        primary:
          provider: openclaw
          model: openclaw/codex
          auth: oauth
          auth_env: OPENCLAW_OAUTH_TOKEN
          max_tokens: 8192
          temperature: 0.2
          timeout_seconds: 60
        fallback:
          provider: google
          model: gemini-2.0-flash
          api_key_env: GEMINI_API_KEY
          max_tokens: 8192
          temperature: 0.3
          timeout_seconds: 45
""")

CHAT_RESPONSE = {
    "choices": [{"message": {"content": "Hello from LLM"}}],
}


@pytest.fixture
def tier_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "llm_tiers.yaml"
    p.write_text(SAMPLE_TIER_YAML)
    return p


@pytest.fixture
def legacy_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "llm_tiers.yaml"
    p.write_text(LEGACY_TIER_YAML)
    return p


# ── Config loading ────────────────────────────────────────────────────


def test_load_tier_configs_all_providers(tier_yaml: Path) -> None:
    """All three providers resolve when env vars are set."""
    env = {
        "OPENCLAW_OAUTH_TOKEN": "tok-123",
        "OPENROUTER_API_KEY": "or-456",
        "GEMINI_API_KEY": "gem-789",
    }
    with patch.dict("os.environ", env, clear=False):
        tiers = load_tier_configs(tier_yaml)

    t1 = tiers["tier1_heavy_reasoning"]
    assert len(t1.providers) == 3
    assert t1.providers[0].name == "openclaw"
    assert t1.providers[0].api_key == "tok-123"
    assert t1.providers[1].name == "openrouter"
    assert t1.providers[1].api_key == "or-456"
    assert t1.providers[2].name == "google"
    assert t1.providers[2].api_key == "gem-789"


def test_load_tier_configs_middle_missing(tier_yaml: Path) -> None:
    """OpenRouter skipped when its key is unset; others still resolve."""
    env = {"OPENCLAW_OAUTH_TOKEN": "tok-123", "GEMINI_API_KEY": "gem-789"}
    with patch.dict("os.environ", env, clear=False), \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
        tiers = load_tier_configs(tier_yaml)

    t1 = tiers["tier1_heavy_reasoning"]
    assert len(t1.providers) == 2
    assert t1.providers[0].name == "openclaw"
    assert t1.providers[1].name == "google"


def test_load_tier_configs_primary_missing(tier_yaml: Path) -> None:
    """Primary skipped when its env var is unset; rest still resolve."""
    env = {"OPENROUTER_API_KEY": "or-456", "GEMINI_API_KEY": "gem-789"}
    with patch.dict("os.environ", env, clear=False), \
         patch.dict("os.environ", {"OPENCLAW_OAUTH_TOKEN": ""}, clear=False):
        tiers = load_tier_configs(tier_yaml)

    t1 = tiers["tier1_heavy_reasoning"]
    assert len(t1.providers) == 2
    assert t1.providers[0].name == "openrouter"
    assert t1.providers[1].name == "google"


def test_load_tier_configs_legacy_format(legacy_yaml: Path) -> None:
    """Legacy primary/fallback format still works."""
    env = {"OPENCLAW_OAUTH_TOKEN": "tok-123", "GEMINI_API_KEY": "gem-456"}
    with patch.dict("os.environ", env, clear=False):
        tiers = load_tier_configs(legacy_yaml)

    t1 = tiers["tier1_heavy_reasoning"]
    assert len(t1.providers) == 2
    assert t1.primary is not None
    assert t1.primary.name == "openclaw"
    assert t1.fallback is not None
    assert t1.fallback.name == "google"


def test_load_tier_configs_missing_file(tmp_path: Path) -> None:
    """Gracefully returns empty dict when config file doesn't exist."""
    tiers = load_tier_configs(tmp_path / "missing.yaml")
    assert tiers == {}


def test_tier_config_backward_compat_properties() -> None:
    """TierConfig.primary and .fallback properties work correctly."""
    p1 = ProviderConfig(name="a", base_url="u", model="m", api_key="k")
    p2 = ProviderConfig(name="b", base_url="u", model="m", api_key="k")
    p3 = ProviderConfig(name="c", base_url="u", model="m", api_key="k")
    tier = TierConfig("test", providers=[p1, p2, p3])
    assert tier.primary == p1
    assert tier.fallback == p2

    empty = TierConfig("empty", providers=[])
    assert empty.primary is None
    assert empty.fallback is None

    single = TierConfig("single", providers=[p1])
    assert single.primary == p1
    assert single.fallback is None


# ── LLMClient ─────────────────────────────────────────────────────────


def _make_provider(name: str = "test", api_key: str = "key") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url="https://test.example.com/v1",
        model=f"{name}/model",
        api_key=api_key,
    )


def test_client_uses_primary_first() -> None:
    """Primary is called first; others are not touched on success."""
    primary = _make_provider("primary")
    secondary = _make_provider("secondary")
    fallback = _make_provider("fallback")
    tier = TierConfig("test", providers=[primary, secondary, fallback])
    client = LLMClient(tier=tier)

    with patch("bet_agent.llm.client.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = CHAT_RESPONSE
        mock_post.return_value.raise_for_status = lambda: None

        result = client.chat("hi")

    assert result == "Hello from LLM"
    mock_post.assert_called_once()
    assert "test.example.com" in mock_post.call_args[0][0]
    payload = mock_post.call_args[1]["json"]
    assert payload["model"] == "primary/model"


def test_client_falls_back_through_chain() -> None:
    """When primary and secondary fail, third provider succeeds."""
    primary = _make_provider("primary")
    secondary = _make_provider("secondary")
    fallback = _make_provider("fallback")
    tier = TierConfig("test", providers=[primary, secondary, fallback])
    client = LLMClient(tier=tier)

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        class FakeResp:
            status_code = 200 if call_count >= 3 else 500

            def json(self):
                return CHAT_RESPONSE

            def raise_for_status(self):
                if self.status_code != 200:
                    raise Exception(f"provider {call_count} down")

        return FakeResp()

    with patch("bet_agent.llm.client.requests.post", side_effect=side_effect):
        result = client.chat("test chain fallback")

    assert result == "Hello from LLM"
    assert call_count == 3  # primary + secondary failed, fallback succeeded


def test_client_falls_back_on_primary_failure() -> None:
    """When primary raises, second provider is tried."""
    primary = _make_provider("primary")
    fallback = _make_provider("fallback")
    tier = TierConfig("test", providers=[primary, fallback])
    client = LLMClient(tier=tier)

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        class FakeResp:
            status_code = 200 if call_count > 1 else 500

            def json(self):
                return CHAT_RESPONSE

            def raise_for_status(self):
                if self.status_code != 200:
                    raise Exception("primary down")

        return FakeResp()

    with patch("bet_agent.llm.client.requests.post", side_effect=side_effect):
        result = client.chat("test fallback")

    assert result == "Hello from LLM"
    assert call_count == 2  # primary failed, fallback succeeded


def test_client_raises_when_all_fail() -> None:
    """RuntimeError when all providers fail."""
    primary = _make_provider("primary")
    secondary = _make_provider("secondary")
    fallback = _make_provider("fallback")
    tier = TierConfig("test", providers=[primary, secondary, fallback])
    client = LLMClient(tier=tier)

    with patch("bet_agent.llm.client.requests.post", side_effect=Exception("down")):
        with pytest.raises(RuntimeError, match="All LLM providers exhausted"):
            client.chat("will fail")


def test_client_raises_when_no_providers() -> None:
    """RuntimeError when tier has no providers (no credentials)."""
    tier = TierConfig("empty", providers=[])
    client = LLMClient(tier=tier)

    with pytest.raises(RuntimeError, match="No LLM providers available"):
        client.chat("no providers")


def test_client_only_fallback_when_primary_missing() -> None:
    """When only the last provider has credentials, it's used directly."""
    fallback = _make_provider("gemini")
    tier = TierConfig("test", providers=[fallback])
    client = LLMClient(tier=tier)

    assert client.active_provider == "gemini"

    with patch("bet_agent.llm.client.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = CHAT_RESPONSE
        mock_post.return_value.raise_for_status = lambda: None

        result = client.chat("hello gemini")

    assert result == "Hello from LLM"
    payload = mock_post.call_args[1]["json"]
    assert payload["model"] == "gemini/model"


def test_client_system_prompt_included() -> None:
    """System prompt is sent as first message when provided."""
    prov = _make_provider("test")
    tier = TierConfig("test", providers=[prov])
    client = LLMClient(tier=tier)

    with patch("bet_agent.llm.client.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = CHAT_RESPONSE
        mock_post.return_value.raise_for_status = lambda: None

        client.chat("q", system_prompt="You are a test agent.")

    messages = mock_post.call_args[1]["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "You are a test agent."}
    assert messages[1] == {"role": "user", "content": "q"}


def test_openrouter_url_in_provider_urls() -> None:
    """OpenRouter URL is correctly configured."""
    from bet_agent.llm.client import _PROVIDER_URLS
    assert _PROVIDER_URLS["openrouter"] == "https://openrouter.ai/api/v1"


# ── MasterAgentBridge integration ─────────────────────────────────────


def test_master_bridge_calls_llm() -> None:
    """MasterAgentBridge.query() routes through Tier-1 LLM."""
    from bet_agent.interfaces.telegram_bot import MasterAgentBridge

    bridge = MasterAgentBridge()
    fake_client = LLMClient(
        tier=TierConfig("test", providers=[_make_provider("test")])
    )

    with patch.object(bridge, "_get_llm", return_value=fake_client):
        with patch("bet_agent.llm.client.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Portfolio up 3.2%"}}]
            }
            mock_post.return_value.raise_for_status = lambda: None

            result = bridge.query("How are we doing?", "alice")

    assert "Portfolio up 3.2%" in result


def test_master_bridge_graceful_failure() -> None:
    """MasterAgentBridge returns friendly error when LLM is unreachable."""
    from bet_agent.interfaces.telegram_bot import MasterAgentBridge

    bridge = MasterAgentBridge()

    with patch.object(bridge, "_get_llm", side_effect=RuntimeError("no providers")):
        result = bridge.query("test", "bob")

    assert "temporarily unable" in result
