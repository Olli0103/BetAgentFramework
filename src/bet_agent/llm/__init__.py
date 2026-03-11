"""LLM client layer — tiered routing with automatic fallback."""

from bet_agent.llm.client import LLMClient, TierConfig, chat

__all__ = ["LLMClient", "TierConfig", "chat"]
