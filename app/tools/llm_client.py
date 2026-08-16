"""
Unified LLM client for POLARIS.
Supports Qwen through either a hosted OpenAI-compatible router or local Ollama,
and Google Gemini.
"""

import logging
import os
import threading
import time
from typing import Optional

_logger = logging.getLogger(__name__)

_gemini_lock = threading.Lock()
_last_gemini_request_at: float = 0.0

SUPPORTED_PROVIDERS = ("qwen", "gemini")

DEFAULT_MODELS = {
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
    "gemini": "gemini-2.0-flash",
}


def _is_local_base_url(base_url: Optional[str]) -> bool:
    value = (base_url or "").lower()
    return "localhost" in value or "127.0.0.1" in value


def generate_text(
    prompt: str,
    api_key: str,
    provider: str = "qwen",
    model: Optional[str] = None,
    qwen_base_url: Optional[str] = None,
) -> str:
    """
    Generate text from the configured provider.

    Args:
        prompt: The text prompt to send.
        api_key: Provider API key (HF token or Gemini API key).
        provider: "qwen" or "gemini".
        model: Model ID override.
        qwen_base_url: Base URL for Qwen (HF router only).

    Returns:
        Generated text from the model.
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "qwen").lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Use one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    qwen_url = qwen_base_url or os.getenv("QWEN_BASE_URL") or "https://router.huggingface.co/v1"

    if not api_key or not api_key.strip():
        if provider == "qwen" and _is_local_base_url(qwen_url):
            api_key = "ollama-local"
        else:
            label = "Gemini" if provider == "gemini" else "Hugging Face"
            raise ValueError(f"API key is empty. Set your {label} API key in Settings.")

    if not api_key or not api_key.strip():
        label = "Gemini" if provider == "gemini" else "Hugging Face"
        raise ValueError(f"API key is empty. Set your {label} API key in Settings.")

    model_id = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    if provider == "gemini":
        return _generate_gemini(prompt, api_key.strip(), model_id)

    return _generate_qwen(
        prompt,
        api_key.strip(),
        model_id,
        qwen_url,
    )


def resolve_api_key(provider: str, stored_key: Optional[str] = None) -> str:
    """Resolve API key from stored value or environment for a provider."""
    if stored_key and str(stored_key).strip():
        return str(stored_key).strip()
    provider = (provider or "qwen").lower().strip()
    if provider == "gemini":
        return (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        ).strip()
    return (
        os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HF_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or ""
    ).strip()


def _generate_qwen(prompt: str, api_key: str, model_id: str, base_url: str) -> str:
    """Generate text using Qwen via Hugging Face OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for Qwen. Install with: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response or not response.choices:
        raise ValueError("No response from Qwen API")
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty content in Qwen response")
    return content


def _gemini_min_interval_sec() -> float:
    """Minimum seconds between Gemini calls (free tier ~5–15 RPM)."""
    return float(os.getenv("GEMINI_MIN_INTERVAL_SEC", "6"))


def _throttle_gemini() -> None:
    """Space out Gemini requests to avoid 429 on free-tier RPM limits."""
    global _last_gemini_request_at
    interval = _gemini_min_interval_sec()
    if interval <= 0:
        return
    with _gemini_lock:
        now = time.monotonic()
        wait = interval - (now - _last_gemini_request_at)
        if wait > 0:
            _logger.info("Gemini throttle: sleeping %.1fs before next request", wait)
            time.sleep(wait)
        _last_gemini_request_at = time.monotonic()


def _is_gemini_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).upper()
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "RATE_LIMIT" in text or "QUOTA" in text:
        return True
    name = type(exc).__name__.upper()
    return "RESOURCEEXHAUSTED" in name or "TOOMANYREQUESTS" in name


def _format_gemini_rate_limit_error(model_id: str) -> str:
    return (
        f"Gemini rate limit (429) for model {model_id}. "
        f"The Hypothesis agent runs several LLM calls per question (~4 on submit, +1 per choice). "
        f"Free tier allows roughly 5–15 requests/minute depending on model — check limits in "
        f"https://aistudio.google.com/ . "
        f"Try gemini-2.0-flash-lite in Settings, wait a minute and retry, enable billing for higher "
        f"limits, or set GEMINI_MIN_INTERVAL_SEC (default 6) higher. "
        f"Optional: HYPOTHESIS_FAST_MODE=1 (default), HYPOTHESIS_SKIP_READINESS_CHECK=1, "
        f"HYPOTHESIS_SKIP_ANALYSIS_ON_GENERATE=1, or use a paid Gemini tier / Qwen."
    )


def _generate_gemini(prompt: str, api_key: str, model_id: str) -> str:
    """Generate text using Google Gemini with throttling and 429 retries."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required for Gemini. Install with: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_id)
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    last_exc: Optional[BaseException] = None

    for attempt in range(max_retries):
        _throttle_gemini()
        try:
            response = gemini_model.generate_content(prompt)
            if not response:
                raise ValueError("No response from Gemini API")
            try:
                text = response.text
            except Exception as text_exc:
                feedback = getattr(response, "prompt_feedback", None)
                candidates = getattr(response, "candidates", None)
                detail = str(text_exc)
                if feedback:
                    detail = f"{detail}; feedback={feedback}"
                if candidates is not None:
                    detail = f"{detail}; candidates={candidates}"
                raise ValueError(f"Gemini blocked or empty response: {detail}") from text_exc
            if not text or not str(text).strip():
                raise ValueError("Empty content in Gemini response")
            return str(text).strip()
        except Exception as exc:
            last_exc = exc
            if _is_gemini_rate_limit_error(exc) and attempt < max_retries - 1:
                backoff = _gemini_min_interval_sec() * (2 ** (attempt + 1))
                _logger.warning(
                    "Gemini 429/rate limit (attempt %s/%s), retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                time.sleep(backoff)
                continue
            if _is_gemini_rate_limit_error(exc):
                raise ValueError(_format_gemini_rate_limit_error(model_id)) from exc
            raise

    if last_exc:
        if _is_gemini_rate_limit_error(last_exc):
            raise ValueError(_format_gemini_rate_limit_error(model_id)) from last_exc
        raise last_exc
    raise ValueError("Gemini request failed")
