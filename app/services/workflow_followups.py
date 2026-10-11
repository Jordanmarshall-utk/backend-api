"""Workflow step sync, agent name resolution, and post-step automation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.reasoning_integration import prepare_pipeline_reasoning

# Builder step labels → registered agent names
STEP_TO_AGENT_NAME: Dict[str, str] = {
    "Hypothesis Agent": "Hypothesis Agent",
    "Experiment Agent": "Experiment Agent",
    "Curve Fitting": "Curve Fitting Agent",
    "Curve Fitting Agent": "Curve Fitting Agent",
    "ML Models": "ML Models",
    "Analysis Agent": "Analysis Agent",
}

WORKFLOW_STEP_KEYS: Dict[str, str] = {
    "Hypothesis Agent": "hypothesis",
    "Experiment Agent": "experiment",
    "Curve Fitting": "curve_fitting",
    "Curve Fitting Agent": "curve_fitting",
    "ML Models": "ml_models",
    "Analysis Agent": "analysis",
}

WORKFLOW_STEP_PAGES: Dict[str, str] = {
    "hypothesis": "/agents/hypothesis",
    "experiment": "/agents/experiment",
    "curve_fitting": "/agents/curve-fitting",
    "ml_models": "/agents/ml-models",
    "analysis": "/agents/analysis",
}


def resolve_agent_name(step_name: str) -> str:
    return STEP_TO_AGENT_NAME.get(step_name, step_name)


def step_key(step_name: str) -> str:
    return WORKFLOW_STEP_KEYS.get(step_name, step_name.lower().replace(" ", "_"))


def sync_workflow_from_steps(memory: Any, steps: List[Dict[str, Any]]) -> None:
    """Keep manual_workflow order and automation flags aligned with builder steps."""
    step_names = [str(s["name"]) for s in steps if s.get("name")]
    memory.set_var("manual_workflow", step_names)
    auto_flags = {
        str(s["name"]): bool(s.get("automatic")) for s in steps if s.get("name")
    }
    memory.set_var("workflow_auto_flags", auto_flags)

    if "ML Models" in step_names:
        memory.set_var(
            "auto_ml_after_curve_fitting",
            bool(auto_flags.get("ML Models", False)),
        )
    if "Analysis Agent" in step_names:
        memory.set_var(
            "auto_route_to_analysis",
            bool(auto_flags.get("Analysis Agent", False)),
        )


def set_workflow_step(memory: Any, step_name: str) -> None:
    memory.set_var("workflow_step", step_key(step_name))


def next_step_after(memory: Any, completed_step: str) -> Optional[str]:
    manual = memory.get_var("manual_workflow") or []
    if not manual:
        return None
    try:
        idx = manual.index(completed_step)
    except ValueError:
        # Alias: Curve Fitting vs Curve Fitting Agent
        alt = (
            "Curve Fitting Agent"
            if completed_step == "Curve Fitting"
            else "Curve Fitting"
        )
        try:
            idx = manual.index(alt)
        except ValueError:
            return None
    if idx + 1 < len(manual):
        return manual[idx + 1]
    return None


def should_auto_run_ml_after_curve_fitting(memory: Any) -> bool:
    if memory.get_var("auto_ml_after_curve_fitting"):
        return True
    manual = memory.get_var("manual_workflow") or []
    flags = memory.get_var("workflow_auto_flags") or {}
    try:
        cf_idx = manual.index("Curve Fitting")
    except ValueError:
        try:
            cf_idx = manual.index("Curve Fitting Agent")
        except ValueError:
            return False
    next_idx = cf_idx + 1
    return (
        next_idx < len(manual)
        and manual[next_idx] == "ML Models"
        and bool(flags.get("ML Models", False))
    )


def on_curve_fitting_complete(memory: Any) -> Dict[str, Any]:
    """
    Update workflow progress after successful curve fitting.
    Optionally run ML when workflow flags request it.
    """
    memory.set_var("workflow_curve_fitting_completed", True)
    nxt = next_step_after(memory, "Curve Fitting") or next_step_after(
        memory, "Curve Fitting Agent"
    )
    out: Dict[str, Any] = {"next_step": nxt}

    if nxt:
        set_workflow_step(memory, nxt)
        out["workflow_step"] = step_key(nxt)
        out["next_page"] = WORKFLOW_STEP_PAGES.get(step_key(nxt))

    if not should_auto_run_ml_after_curve_fitting(memory):
        return out

    model = (
        memory.get_var("workflow_ml_model_choice")
        or memory.get_var("optimization_model_choice")
    )
    if model:
        memory.set_var("optimization_model_choice", model)

    from app.services.ml_session_service import run_ml_automation

    ml_payload, _ = prepare_pipeline_reasoning(
        memory,
        "ML Models",
        stage="ml_models",
        payload={"action": "run", "model_choice": model},
    )
    ml_result = run_ml_automation(memory, ml_payload)
    out["ml_automation"] = ml_result
    if ml_result.get("status") == "success" or (
        isinstance(ml_result.get("result"), dict)
        and ml_result["result"].get("success")
    ):
        set_workflow_step(memory, "ML Models")
        out["workflow_step"] = "ml_models"
        out["next_page"] = WORKFLOW_STEP_PAGES["ml_models"]

        if memory.get_var("auto_route_to_analysis") or (
            memory.get_var("workflow_auto_flags") or {}
        ).get("Analysis Agent"):
            from app.services.analysis_service import run_analysis_pipeline

            analysis_payload, _ = prepare_pipeline_reasoning(
                memory,
                "Analysis Agent",
                stage="analysis",
                payload={"action": "run"},
            )
            analysis_result = run_analysis_pipeline(memory, analysis_payload)
            out["analysis_automation"] = analysis_result
            if analysis_result.get("status") == "success":
                set_workflow_step(memory, "Analysis Agent")
                out["workflow_step"] = "analysis"
                out["next_page"] = WORKFLOW_STEP_PAGES["analysis"]

    return out


def build_auto_curve_fitting_payload(memory: Any) -> Optional[Dict[str, Any]]:
    """Build API payload for a pending workflow/demo auto curve fitting run."""
    if not memory.get_var("auto_run_curve_fitting"):
        return None
    data_file = memory.get_var("auto_run_data_file")
    if not data_file:
        return None
    params = memory.get_var("auto_run_params") or {}
    payload: Dict[str, Any] = {
        "data_file": str(data_file),
        "composition_file": memory.get_var("auto_run_comp_file"),
        "action": "run",
    }
    if isinstance(params, dict):
        payload.update(params)
    return payload


def consume_auto_curve_fitting_flag(memory: Any) -> None:
    memory.set_var("auto_run_curve_fitting", False)
