from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from uuid import uuid4

from app.core.deps import get_current_user
from app.core.auth import AuthUser
from app.schemas.api_models import WorkflowRunRequest, WorkflowRunResponse, WorkflowStatus
from app.services.memory_service import get_memory_manager
from app.services.workflow_service import (
    apply_named_workflow,
    delete_named_workflow,
    export_workflow_json,
    get_workflow_session,
    load_named_workflow,
    patch_workflow_session,
    run_demo_workflow,
    save_named_workflow,
    start_workflow,
    stop_workflow,
)

router = APIRouter()
_workflows: dict[str, WorkflowStatus] = {}


class WorkflowStepModel(BaseModel):
    name: str
    automatic: bool = False
    description: Optional[str] = None


class WorkflowSessionPatch(BaseModel):
    experiment_id: Optional[int] = None
    workflow_name: Optional[str] = None
    workflow_steps: Optional[List[WorkflowStepModel]] = None
    routing_mode: Optional[str] = None
    workflow_ml_model_choice: Optional[str] = None
    workflow_index: Optional[int] = None
    workflow_step: Optional[str] = None
    auto_ml_after_curve_fitting: Optional[bool] = None
    auto_route_to_analysis: Optional[bool] = None


class WorkflowSaveRequest(BaseModel):
    experiment_id: Optional[int] = None
    name: str
    steps: List[WorkflowStepModel]
    apply: bool = False


class WorkflowNameRequest(BaseModel):
    experiment_id: Optional[int] = None
    name: str


def _memory(experiment_id: Optional[int], user: AuthUser):
    memory = get_memory_manager()
    if experiment_id:
        memory.set_current_experiment(experiment_id, user.id)
    return memory


@router.get("/workflows/session")
def workflow_session_get(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    return get_workflow_session(_memory(experiment_id, user))


@router.patch("/workflows/session")
def workflow_session_patch(
    body: WorkflowSessionPatch,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    memory = _memory(body.experiment_id, user)
    steps = None
    if body.workflow_steps is not None:
        steps = [s.model_dump() for s in body.workflow_steps]
    return patch_workflow_session(
        memory,
        workflow_name=body.workflow_name,
        workflow_steps=steps,
        routing_mode=body.routing_mode,
        workflow_ml_model_choice=body.workflow_ml_model_choice,
        workflow_index=body.workflow_index,
        workflow_step=body.workflow_step,
        auto_ml_after_curve_fitting=body.auto_ml_after_curve_fitting,
        auto_route_to_analysis=body.auto_route_to_analysis,
    )


@router.post("/workflows/save")
def workflow_save(
    body: WorkflowSaveRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    memory = _memory(body.experiment_id, user)
    steps = [s.model_dump() for s in body.steps]
    out = save_named_workflow(memory, body.name, steps)
    if body.apply:
        out.update(apply_named_workflow(memory, body.name))
    return out


@router.post("/workflows/load")
def workflow_load(
    body: WorkflowNameRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    return load_named_workflow(_memory(body.experiment_id, user), body.name)


@router.post("/workflows/apply")
def workflow_apply(
    body: WorkflowNameRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    return apply_named_workflow(_memory(body.experiment_id, user), body.name)


@router.delete("/workflows/saved")
def workflow_delete(
    user: Annotated[AuthUser, Depends(get_current_user)],
    name: Annotated[str, Query()],
    experiment_id: Optional[int] = None,
) -> dict:
    return delete_named_workflow(_memory(experiment_id, user), name)


@router.post("/workflows/start")
def workflow_start(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    return start_workflow(_memory(experiment_id, user))


@router.post("/workflows/stop")
def workflow_stop(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    return stop_workflow(_memory(experiment_id, user))


def _prime_demo_tour_state(memory) -> None:
    steps = [
        {
            "name": "Hypothesis Agent",
            "automatic": False,
            "description": "Refine the research question with literature evidence and council guidance.",
        },
        {
            "name": "Experiment Agent",
            "automatic": False,
            "description": "Generate a constrained protocol, composition series, and plate layout.",
        },
        {
            "name": "Curve Fitting",
            "automatic": True,
            "description": "Resolve composition-dependent photoluminescence phase peaks.",
        },
        {
            "name": "ML Models",
            "automatic": True,
            "description": "Rank the next compositions using prediction uncertainty.",
        },
        {
            "name": "Analysis Agent",
            "automatic": True,
            "description": "Compare evidence, fitted results, and model suggestions.",
        },
    ]
    memory.set_var("current_workflow_name", "Phase-aware PL optimization")
    memory.set_var("workflow_name", "Phase-aware PL optimization")
    memory.set_var("workflow_steps", steps)
    memory.set_var("manual_workflow", [step["name"] for step in steps])
    memory.set_var("workflow_auto_flags", {step["name"]: step["automatic"] for step in steps})
    memory.set_var("workflow_active", True)
    memory.set_var("workflow_index", 0)
    memory.set_var("workflow_step", "hypothesis")
    memory.set_var("demo_workflow_running", True)
    memory.set_var("auto_run_curve_fitting", False)
    memory.set_var("auto_run_data_file", None)
    memory.set_var("auto_run_comp_file", None)
    memory.set_var("auto_run_params", {})
    memory.set_var("auto_ml_after_curve_fitting", True)
    memory.set_var("auto_route_to_analysis", True)
    memory.set_var(
        "agent_usage_counts",
        {
            "hypothesis": 1,
            "experiment": 1,
            "curve_fit": 1,
            "ml_models": 1,
            "analysis": 1,
            "router": 5,
        },
    )
    memory.set_var("reasoning_mode", "Active")
    memory.set_var(
        "last_reasoning_plan",
        {
            "primary_school": "humean",
            "supporting_schools": ["cartesian", "hegelian"],
            "assignments": {
                "humean": "Ground phase assignments and composition trends in the measured spectra.",
                "cartesian": "Separate composition, phase identity, peak fitting, and uncertainty checks.",
                "hegelian": "Reconcile competing phase emissions without forcing a monotonic peak shift.",
            },
            "rationale": (
                "Evidence grounding leads the analysis, decomposition keeps fitted quantities distinct, "
                "and conflict resolution handles coexistence between lower-dimensional and 3D phases."
            ),
            "confidence": 0.88,
            "reasoning_required": True,
            "task_summary": (
                "Assess how the PEA2PbI4 to FAPbI3 composition changes phase-resolved PL emission "
                "and identify informative next experiments."
            ),
            "reasoning_sequence": ["humean", "cartesian", "hegelian"],
            "unresolved_questions": [
                "Which emission bands represent coexisting phases rather than a continuous wavelength shift?",
                "Which compositions most reduce uncertainty around the dominant-phase transition?",
            ],
            "evidence_requirements": [
                "Composition-resolved PL spectra",
                "Multi-peak fit quality and uncertainty",
                "Replicate measurements near phase-transition compositions",
            ],
            "warnings": [],
            "mode": "Active",
            "selection_source": "crewai",
            "crew_success": True,
            "fallback_used": False,
            "error": None,
        },
    )


class WorkflowDemoRequest(BaseModel):
    experiment_id: Optional[int] = None
    auto_fit: bool = True
    precomputed: bool = False


@router.post("/workflows/demo")
def workflow_demo(
    user: Annotated[AuthUser, Depends(get_current_user)],
    body: WorkflowDemoRequest = WorkflowDemoRequest(),
) -> dict:
    memory = _memory(body.experiment_id, user)
    if body.precomputed:
        from app.services.presentation_demo_service import load_presentation_demo

        result = load_presentation_demo(memory)
    else:
        result = run_demo_workflow(memory, auto_fit=body.auto_fit)
    _prime_demo_tour_state(memory)
    if body.precomputed:
        # Bind the report after every presentation-state mutation so Analysis
        # cannot mistake it for output from an older hypothesis.
        from app.services.analysis_service import _hypothesis_fingerprint, _hypothesis_text

        memory.set_var(
            "research_goal",
            "Evaluate composition-dependent 2D, quasi-2D, and 3D emission features and recommend the next measurements.",
        )
        memory.set_var(
            "analysis_hypothesis_fingerprint",
            _hypothesis_fingerprint(_hypothesis_text(memory)),
        )
    return result


@router.get("/workflows/export")
def workflow_export(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    memory = _memory(experiment_id, user)
    return {"json": export_workflow_json(memory)}


@router.post("/workflows/run", response_model=WorkflowRunResponse, status_code=202)
def run_workflow(
    body: WorkflowRunRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> WorkflowRunResponse:
    memory = get_memory_manager()
    memory.set_current_experiment(body.experiment_id, user.id)
    memory.set_var("workflow_active", True)
    memory.set_var("routing_mode", "Autonomous (LLM)" if body.mode == "autonomous" else "Manual")
    if body.steps:
        step_dicts = [{"name": s, "automatic": False} for s in body.steps]
        memory.save_workflow("api_workflow", step_dicts)

    workflow_id = uuid4()
    status = WorkflowStatus(
        workflow_id=workflow_id,
        status="queued",
        current_step=body.steps[0] if body.steps else None,
    )
    _workflows[str(workflow_id)] = status
    return WorkflowRunResponse(workflow_id=workflow_id, status="queued")


@router.get("/workflows/{workflow_id}", response_model=WorkflowStatus)
def get_workflow(
    workflow_id: str,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> WorkflowStatus:
    from fastapi import HTTPException

    status = _workflows.get(workflow_id)
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return status
