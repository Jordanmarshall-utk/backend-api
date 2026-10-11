"""Translate POLARIS runtime LLM settings into a CrewAI LLM."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

from app.tools.llm_client import DEFAULT_MODELS, resolve_api_key


@dataclass(frozen=True)
class PolarisCrewLLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None


def configure_crewai_environment() -> Path:
    """Keep CrewAI runtime files inside the portable POLARIS backend."""
    if os.name == "nt":
        local_app_data = Path(os.getenv("LOCALAPPDATA") or tempfile.gettempdir())
        try:
            (local_app_data / "crewai" / "credentials").mkdir(parents=True, exist_ok=True)
        except OSError:
            local_app_data = Path(tempfile.gettempdir()) / "polaris-localappdata"
            (local_app_data / "crewai" / "credentials").mkdir(parents=True, exist_ok=True)
            os.environ["LOCALAPPDATA"] = str(local_app_data)

    configured = os.getenv("CREWAI_STORAGE_DIR")
    storage_dir = Path(configured) if configured else Path(__file__).resolve().parents[3] / "data" / "crewai"
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        storage_dir = Path(tempfile.gettempdir()) / "polaris-crewai"
        storage_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("CREWAI_DISABLE_VERSION_CHECK", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    return storage_dir


def resolve_crewai_llm_config(memory: Any = None) -> PolarisCrewLLMConfig:
    def stored(key: str, default: Any = None) -> Any:
        if memory is None:
            return default
        try:
            return memory.get_var(key, default)
        except Exception:
            return default

    provider = str(stored("llm_provider") or os.getenv("LLM_PROVIDER") or "qwen").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"CrewAI does not support configured POLARIS provider: {provider}")
    model = str(stored("llm_model") or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider])
    key = resolve_api_key(provider, stored("api_key"))

    if provider == "qwen":
        base_url = str(
            stored("qwen_base_url")
            or os.getenv("QWEN_BASE_URL")
            or "https://router.huggingface.co/v1"
        )
        if not key and ("localhost" in base_url or "127.0.0.1" in base_url):
            key = "polaris-local"
        if not key:
            raise ValueError("No POLARIS Qwen/Hugging Face API key is configured")
        return PolarisCrewLLMConfig(provider, model, key, base_url)

    if not key:
        raise ValueError("No POLARIS Gemini API key is configured")
    return PolarisCrewLLMConfig(provider, model, key)


def build_crewai_llm(memory: Any = None):
    """Build lazily so normal POLARIS imports work when CrewAI is unavailable."""
    configure_crewai_environment()
    try:
        from crewai import LLM
    except ImportError as exc:
        raise RuntimeError("CrewAI is not installed; install crewai==1.15.17") from exc

    config = resolve_crewai_llm_config(memory)
    common = {
        "api_key": config.api_key,
        "temperature": 0.1,
        "timeout": 90.0,
        "max_retries": 1,
    }
    if config.provider == "qwen":
        return LLM(
            model=f"openai/{config.model}",
            custom_openai=True,
            base_url=config.base_url,
            **common,
        )
    return LLM(model=f"gemini/{config.model}", **common)
