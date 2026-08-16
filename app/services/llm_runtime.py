"""Resolve LLM provider, model, and API key from persisted settings (not stale .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from app.tools.llm_client import resolve_api_key


@dataclass
class LlmRuntimeConfig:
    provider: str
    model: str
    api_key: str
    qwen_base_url: Optional[str] = None


def _is_local_qwen_url(url: str | None) -> bool:
    value = (url or "").lower()
    return "localhost" in value or "127.0.0.1" in value


def get_llm_config(memory: Any | None = None) -> LlmRuntimeConfig:
    """
    Read LLM settings from MemoryManager DB and sync to os.environ.
    Never reload .env with override here — that would wipe API keys saved via Settings.
    """
    if memory is None:
        from app.services.memory_service import get_memory_manager

        memory = get_memory_manager()

    memory._sync_llm_env()

    provider = (memory.get_var("llm_provider") or "qwen").lower().strip()
    stored_key = memory.get_var("api_key")
    api_key = resolve_api_key(provider, stored_key=stored_key)

    if provider == "gemini":
        model = memory.get_var("llm_model") or os.getenv("LLM_MODEL") or "gemini-2.0-flash-lite"
        qwen_base_url = None
    else:
        model = memory.get_var("llm_model") or os.getenv("LLM_MODEL") or "Qwen/Qwen2.5-72B-Instruct"
        qwen_base_url = (
            os.getenv("QWEN_BASE_URL")
            or memory.get_var("qwen_base_url")
            or "https://router.huggingface.co/v1"
        )

    return LlmRuntimeConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        qwen_base_url=qwen_base_url,
    )


def require_api_key(memory: Any | None = None) -> LlmRuntimeConfig:
    cfg = get_llm_config(memory)
    if cfg.provider == "qwen" and _is_local_qwen_url(cfg.qwen_base_url):
        return cfg
    if not cfg.api_key:
        label = "Google Gemini" if cfg.provider == "gemini" else "Hugging Face"
        raise ValueError(
            f"API key not configured for {label}. Save your key in Settings → General, then retry."
        )
    return cfg
