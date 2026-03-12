"""Tests for Codex OAuth token auto-refresh."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bet_agent.llm.codex_auth import (
    CodexTokenManager,
    _decode_jwt_exp,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_jwt(exp: float, extra: dict | None = None) -> str:
    """Build a minimal JWT with the given expiry (no real signature)."""
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()

    payload_data = {"exp": exp, "sub": "test"}
    if extra:
        payload_data.update(extra)
    payload = base64.urlsafe_b64encode(
        json.dumps(payload_data).encode()
    ).rstrip(b"=").decode()

    return f"{header}.{payload}.fakesig"


def _make_auth_json(
    tmp_path: Path,
    *,
    access_exp: float | None = None,
    refresh_token: str = "rt_test_refresh",
) -> Path:
    """Write a fake ~/.codex/auth.json and return the path."""
    if access_exp is None:
        access_exp = time.time() + 3600  # 1 hour from now

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": _make_jwt(access_exp),
            "refresh_token": refresh_token,
        },
        "last_refresh": "2026-03-12T09:00:00Z",
    }))
    return auth_file


# ── JWT decoding ─────────────────────────────────────────────────────────


class TestJWTDecoding:
    def test_decode_exp(self):
        exp = time.time() + 7200
        token = _make_jwt(exp)
        assert _decode_jwt_exp(token) == exp

    def test_decode_invalid_jwt_raises(self):
        with pytest.raises(ValueError, match="Invalid JWT"):
            _decode_jwt_exp("not-a-jwt")

    def test_decode_no_exp_raises(self):
        header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(b'{"sub":"test"}').rstrip(b"=").decode()
        with pytest.raises(ValueError, match="no 'exp' claim"):
            _decode_jwt_exp(f"{header}.{payload}.sig")


# ── Token manager ────────────────────────────────────────────────────────


class TestCodexTokenManager:
    def test_loads_valid_token_from_disk(self, tmp_path: Path):
        auth_file = _make_auth_json(tmp_path, access_exp=time.time() + 3600)
        mgr = CodexTokenManager(auth_file)
        token = mgr.get_token()
        assert token.startswith("eyJ")

    def test_raises_when_file_missing(self, tmp_path: Path):
        mgr = CodexTokenManager(tmp_path / "nonexistent.json")
        with pytest.raises(RuntimeError, match="not found"):
            mgr.get_token()

    def test_raises_when_no_access_token(self, tmp_path: Path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"tokens": {}}))
        mgr = CodexTokenManager(auth_file)
        with pytest.raises(RuntimeError, match="No access_token"):
            mgr.get_token()

    def test_returns_cached_token_without_reloading(self, tmp_path: Path):
        auth_file = _make_auth_json(tmp_path, access_exp=time.time() + 3600)
        mgr = CodexTokenManager(auth_file)

        token1 = mgr.get_token()
        # Delete the file — should still return cached token
        auth_file.unlink()
        token2 = mgr.get_token()
        assert token1 == token2

    def test_refreshes_expired_token(self, tmp_path: Path):
        # Token that expires in 2 minutes (within the 5-min margin)
        auth_file = _make_auth_json(tmp_path, access_exp=time.time() + 120)
        mgr = CodexTokenManager(auth_file)

        new_exp = time.time() + 7200
        new_token = _make_jwt(new_exp)

        refresh_response = {
            "access_token": new_token,
            "refresh_token": "rt_new_refresh",
            "id_token": "id_new",
        }

        with patch("bet_agent.llm.codex_auth.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = refresh_response
            mock_post.return_value.raise_for_status = lambda: None

            token = mgr.get_token()

        assert token == new_token

        # Verify refresh request was correct
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "https://auth.openai.com/oauth/token"
        assert call_kwargs[1]["data"]["grant_type"] == "refresh_token"
        assert call_kwargs[1]["data"]["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"

    def test_persists_refreshed_token_to_disk(self, tmp_path: Path):
        auth_file = _make_auth_json(tmp_path, access_exp=time.time() + 120)
        mgr = CodexTokenManager(auth_file)

        new_exp = time.time() + 7200
        new_token = _make_jwt(new_exp)

        with patch("bet_agent.llm.codex_auth.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "access_token": new_token,
                "refresh_token": "rt_persisted",
            }
            mock_post.return_value.raise_for_status = lambda: None

            mgr.get_token()

        # Check disk was updated
        saved = json.loads(auth_file.read_text())
        assert saved["tokens"]["access_token"] == new_token
        assert saved["tokens"]["refresh_token"] == "rt_persisted"

    def test_raises_when_expired_and_no_refresh_token(self, tmp_path: Path):
        auth_file = _make_auth_json(
            tmp_path,
            access_exp=time.time() - 100,  # already expired
            refresh_token="",
        )
        mgr = CodexTokenManager(auth_file)
        with pytest.raises(RuntimeError, match="no refresh_token"):
            mgr.get_token()

    def test_refresh_failure_reloads_from_disk(self, tmp_path: Path):
        """If refresh HTTP call fails but disk has a valid token, use that."""
        # Start with near-expiry token and a refresh_token
        auth_file = _make_auth_json(tmp_path, access_exp=time.time() + 120)
        mgr = CodexTokenManager(auth_file)

        # Load once to populate refresh_token
        mgr._load_from_disk()
        # Manually set a stale cached token to force refresh path
        mgr._expires_at = 0

        # Now update the file with a fresh token (simulating codex CLI refresh)
        fresh_exp = time.time() + 7200
        _make_auth_json(tmp_path, access_exp=fresh_exp)

        with patch("bet_agent.llm.codex_auth.requests.post", side_effect=Exception("network")):
            token = mgr.get_token()

        # Should have reloaded from disk and gotten the fresh token
        assert token.startswith("eyJ")


# ── Integration with LLMClient ───────────────────────────────────────────


class TestCodexAuthLLMIntegration:
    def test_token_fn_called_on_each_request(self):
        """ProviderConfig._token_fn is called each time, not the static key."""
        from bet_agent.llm.client import LLMClient, ProviderConfig, TierConfig

        call_count = 0

        def fresh_token():
            nonlocal call_count
            call_count += 1
            return f"token-{call_count}"

        prov = ProviderConfig(
            name="openclaw",
            base_url="https://api.openclaw.ai/v1",
            model="openclaw/codex",
            api_key="initial-stale",
            _token_fn=fresh_token,
        )
        tier = TierConfig("test", primary=prov)
        client = LLMClient(tier=tier)

        with patch("bet_agent.llm.client.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
            }
            mock_post.return_value.raise_for_status = lambda: None

            client.chat("first")
            client.chat("second")

        assert call_count == 2
        # Second call should use "token-2"
        second_call_headers = mock_post.call_args_list[1][1]["headers"]
        assert second_call_headers["Authorization"] == "Bearer token-2"

    def test_static_key_used_when_no_token_fn(self):
        """Without _token_fn, static api_key is used as before."""
        from bet_agent.llm.client import LLMClient, ProviderConfig, TierConfig

        prov = ProviderConfig(
            name="google",
            base_url="https://example.com/v1",
            model="gemini",
            api_key="static-key-123",
        )
        tier = TierConfig("test", primary=prov)
        client = LLMClient(tier=tier)

        with patch("bet_agent.llm.client.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
            }
            mock_post.return_value.raise_for_status = lambda: None

            client.chat("hi")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer static-key-123"
