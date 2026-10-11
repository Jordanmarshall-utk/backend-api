import logging
from typing import Any, Dict, Optional

from app.core.session import SessionContext
from app.services.agent_headless import run_agent_headless
from app.services.agent_registry import build_agent_router, resolve_agent_by_name
from app.services.memory_service import get_memory_manager
from app.services.reasoning_integration import prepare_pipeline_reasoning


_logger = logging.getLogger(__name__)


def _apply_experiment(memory: Any, experiment_id: Optional[int]) -> None:
    if experiment_id and experiment_id > 0:
        memory.set_current_experiment(experiment_id)


def run_named_agent(
    agent_name: str,
    *,
    experiment_id: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    memory = get_memory_manager()
    _apply_experiment(memory, experiment_id)
    ctx = SessionContext.from_memory(memory, experiment_id)
    body = ctx.to_router_payload(payload or {})
    body.setdefault("source", "api")

    selected = resolve_agent_by_name(agent_name)
    if selected is None:
        return {
            "agent": agent_name,
            "status": "error",
            "message": f"Unknown agent: {agent_name}. Use names like 'Hypothesis Agent' or 'experiment'.",
            "data": {},
        }

    display_name = getattr(selected, "name", agent_name)
    reasoning = None
    if "hypothesis" not in str(display_name).lower():
        try:
            body, reasoning = prepare_pipeline_reasoning(
                memory,
                str(display_name),
                payload=body,
                requested_policy=str(body.get("reasoning_policy") or "auto"),
            )
        except Exception as exc:
            _logger.warning("Reasoning Council integration failed for %s: %s", display_name, exc)
    try:
        data = run_agent_headless(selected, memory, body)
        if not isinstance(data, dict):
            data = {"result": data}
        status = str(data.get("status") or ("success" if data.get("ready", True) else "skipped"))
        if status not in ("success", "error", "skipped"):
            status = "success"
        message = data.get("message") or (
            "Agent completed." if status == "success" else "Agent could not run."
        )
        response_payload = {k: v for k, v in data.items() if k not in ("status", "message")}
        if reasoning is not None:
            response_payload["reasoning"] = {
                "policy": reasoning.effective_policy,
                "plan": reasoning.plan.model_dump(mode="json"),
                "context_applied": bool(reasoning.context),
                "reasoning_required": reasoning.plan.reasoning_required,
                "selection_source": reasoning.plan.selection_source,
                "fallback_used": reasoning.plan.fallback_used,
            }
        return {
            "agent": display_name,
            "status": status,
            "message": message,
            "data": response_payload,
            "nextAgent": None,
        }
    except Exception as exc:
        return {
            "agent": display_name,
            "status": "error",
            "message": str(exc),
            "data": {},
        }


def get_agents_status(experiment_id: Optional[int] = None) -> Dict[str, Any]:
    """Readiness snapshot for all registered agents."""
    memory = get_memory_manager()
    _apply_experiment(memory, experiment_id)
    router = build_agent_router()
    agents_out = []
    for agent in router.agents:
        name = getattr(agent, "name", "Unknown")
        data = run_agent_headless(agent, memory, {})
        agents_out.append(
            {
                "name": name,
                "ready": data.get("ready", False),
                "message": data.get("message", ""),
                "hint_action": data.get("hint_action"),
            }
        )
    return {
        "stage": memory.get_var("stage"),
        "hypothesis_ready": bool(memory.get_var("hypothesis_ready")),
        "agents": agents_out,
    }
