from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.auth import AuthUser
from app.core.config import Settings, get_settings
from app.core.deps import get_current_user
from app.mcp.gateway import TOOL_NAMES
from app.schemas.api_models import McpOrchestrateRequest, McpOrchestrateResponse
from app.services.memory_service import get_memory_manager
from app.watcher.orchestrator_mcp import OrchestratorService, SearchPapersRequest

router = APIRouter()


@router.get("/mcp/status")
def mcp_status(
    user: Annotated[AuthUser, Depends(get_current_user)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    del user
    authentication_ready = not settings.auth_disabled and bool(settings.supabase_jwt_secret)
    host_allowlist_ready = bool(settings.mcp_allowed_host_list)
    available = settings.mcp_enabled and (
        settings.mcp_access_mode == "local"
        or (authentication_ready and host_allowlist_ready)
    )
    reason = None
    if not settings.mcp_enabled:
        reason = "disabled"
    elif settings.mcp_access_mode == "authenticated" and not authentication_ready:
        reason = "authentication_not_configured"
    elif settings.mcp_access_mode == "authenticated" and not host_allowlist_ready:
        reason = "allowed_hosts_not_configured"
    return {
        "enabled": settings.mcp_enabled,
        "available": available,
        "reason": reason,
        "endpoint": "/mcp",
        "transport": "streamable_http",
        "access_mode": settings.mcp_access_mode,
        "tools": list(TOOL_NAMES),
        "read_only": True,
        "hardware_execution_supported": False,
        "simulation_available": False,
        "protocol_status": "draft_validation_only",
    }


@router.post("/mcp/orchestrate", response_model=McpOrchestrateResponse)
def orchestrate_mcp(
    body: McpOrchestrateRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> McpOrchestrateResponse:
    memory = get_memory_manager()
    if body.experiment_id:
        memory.set_current_experiment(body.experiment_id, user.id)
    svc = OrchestratorService(memory=memory)
    gate = {}
    if body.require_hypothesis_gate:
        gate = svc._history_guard(body.query)
    literature = {}
    try:
        req = SearchPapersRequest(query=body.query, max_candidates=5)
        literature = svc.search_papers(req)
    except Exception as exc:
        literature = {"error": str(exc)}
    return McpOrchestrateResponse(
        status="blocked" if gate.get("blocked") else "ok",
        hypothesis_gate=gate,
        literature=literature if isinstance(literature, dict) else {"result": literature},
    )
