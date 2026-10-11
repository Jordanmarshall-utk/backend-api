from typing import Annotated

from fastapi import APIRouter, Depends, Response

from app.core.deps import get_current_user
from app.core.auth import AuthUser
from app.schemas.api_models import AppSettings, AppSettingsPatch, AppSettingsResponse
from app.services.jupyter_upload import get_jupyter_config, merge_jupyter_config
from app.services.memory_service import get_memory_manager

router = APIRouter()

_SETTINGS_KEYS = {
    "llm_provider": "llm_provider",
    "llm_model": "llm_model",
    "qwen_base_url": "qwen_base_url",
    "routing_mode": "routing_mode",
    "reasoning_mode": "reasoning_mode",
    "max_hypothesis_rounds": "max_hypothesis_rounds",
    "watcher_directory": "watcher_directory",
    "watcher_results_dir": "watcher_results_dir",
    "watcher_enabled": "watcher_enabled",
    "experimental_mode": "experimental_mode",
}


def _read_settings(memory) -> AppSettingsResponse:
    api_key = memory.get_var("api_key")
    return AppSettingsResponse(
        llm_provider=memory.get_var("llm_provider"),
        llm_model=memory.get_var("llm_model"),
        qwen_base_url=memory.get_var("qwen_base_url"),
        routing_mode=memory.get_var("routing_mode"),
        reasoning_mode=memory.get_var("reasoning_mode", "Shadow"),
        max_hypothesis_rounds=memory.get_var("max_hypothesis_rounds"),
        watcher_directory=memory.get_var("watcher_directory"),
        watcher_results_dir=memory.get_var("watcher_results_dir"),
        watcher_enabled=memory.get_var("watcher_enabled"),
        experimental_mode=memory.get_var("experimental_mode"),
        jupyter_config=get_jupyter_config(memory),
        api_key_configured=bool(api_key),
        last_reasoning_plan=memory.get_var("last_reasoning_plan"),
    )


@router.get("/settings", response_model=AppSettingsResponse)
def get_settings_route(
    user: Annotated[AuthUser, Depends(get_current_user)],
    response: Response,
) -> AppSettingsResponse:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return _read_settings(get_memory_manager())


@router.patch("/settings", response_model=AppSettingsResponse)
def patch_settings(
    body: AppSettingsPatch,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> AppSettingsResponse:
    memory = get_memory_manager()
    for field, db_key in _SETTINGS_KEYS.items():
        value = getattr(body, field, None)
        if value is not None:
            memory.set_var(db_key, value)
    if body.api_key and body.api_key.strip():
        memory.set_var("api_key", body.api_key.strip())
        memory.set_var("api_key_source", "user")
    elif body.llm_provider is not None:
        # Provider changed without a new key — keep user-stored key, re-sync env for provider
        if memory.get_var("api_key") and memory.get_var("api_key_source") != "environment":
            memory.set_var("api_key_source", "user")
    if body.jupyter_config is not None:
        merge_jupyter_config(memory, body.jupyter_config.model_dump(exclude_unset=True))
    memory._sync_llm_env()
    return _read_settings(memory)
