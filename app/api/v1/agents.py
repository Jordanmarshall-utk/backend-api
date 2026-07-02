import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError

from app.core.deps import get_current_user
from app.core.auth import AuthUser
from app.schemas.api_models import (
    AgentRunRequest,
    AgentRunResponse,
    ExperimentSessionPatch,
    HypothesisChatRequest,
    HypothesisChatResponse,
)
from app.services.agent_runner import get_agents_status, run_named_agent
from app.services.experiment_service import get_experiment_session, patch_experiment_session
from app.services.curve_fitting_service import (
    find_plot_path,
    get_curve_fitting_results,
    get_curve_fitting_session,
    preview_table_file,
)
from app.services.curve_fitting_uploads import persist_curve_fitting_uploads
from app.services.hypothesis_chat import handle_chat
from app.services.hypothesis_stream import iter_sse_events
from app.services.memory_service import get_memory_manager

router = APIRouter()
_logger = logging.getLogger(__name__)

_AGENT_PATHS = {
    "hypothesis": "Hypothesis Agent",
    "experiment": "Experiment Agent",
    "curve-fitting": "Curve Fitting Agent",
    "ml": "ML Models",
    "analysis": "Analysis Agent",
}


@router.get("/agents/status")
def agents_status(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    return get_agents_status(experiment_id)


@router.get("/agents/reasoning-policies")
def list_reasoning_policies(
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    from app.tools.scientific_reasoning_policy_layer import list_available_policies, POLICY_ALIASES
    return {
        "policies": list_available_policies(),
        "aliases": POLICY_ALIASES,
    }


@router.post("/agents/hypothesis/chat", response_model=HypothesisChatResponse)
def hypothesis_chat(
    body: HypothesisChatRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> HypothesisChatResponse:
    memory = get_memory_manager()
    if body.experiment_id:
        memory.set_current_experiment(body.experiment_id, user.id)
    stage = str(memory.get_var("stage") or "initial")
    try:
        result = handle_chat(
            memory,
            action=body.action,
            question=body.question,
            choice=body.choice,
            reasoning_policy=body.reasoning_policy or "auto",
        )
        try:
            return HypothesisChatResponse(**result)
        except ValidationError as val_exc:
            _logger.exception("Hypothesis response validation failed")
            return HypothesisChatResponse(
                stage=stage,
                error=f"Invalid agent response shape: {val_exc}",
            )
    except ValueError as exc:
        return HypothesisChatResponse(stage=stage, error=str(exc))
    except Exception as exc:
        _logger.exception("Hypothesis chat failed")
        return HypothesisChatResponse(
            stage=stage,
            error=str(exc) or "Hypothesis agent failed",
        )


@router.post("/agents/hypothesis/chat/stream")
def hypothesis_chat_stream(
    body: HypothesisChatRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> StreamingResponse:
    memory = get_memory_manager()
    if body.experiment_id:
        memory.set_current_experiment(body.experiment_id, user.id)

    def generate():
        yield from iter_sse_events(
            memory,
            action=body.action,
            question=body.question,
            choice=body.choice,
            reasoning_policy=body.reasoning_policy or "auto",
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/agents/hypothesis", response_model=AgentRunResponse)
def run_hypothesis(
    body: AgentRunRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> AgentRunResponse:
    result = run_named_agent(_AGENT_PATHS["hypothesis"], experiment_id=body.experiment_id, payload=body.payload)
    return AgentRunResponse(**result)


@router.get("/agents/experiment/session")
def experiment_session_get(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    memory = get_memory_manager()
    if experiment_id:
        memory.set_current_experiment(experiment_id, user.id)
    return get_experiment_session(memory)


@router.patch("/agents/experiment/session")
def experiment_session_patch(
    body: ExperimentSessionPatch,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    memory = get_memory_manager()
    if body.experiment_id:
        memory.set_current_experiment(body.experiment_id, user.id)
    return patch_experiment_session(
        memory,
        experimental_constraints=body.experimental_constraints,
        manual_inputs=body.manual_inputs,
    )


@router.post("/agents/experiment", response_model=AgentRunResponse)
def run_experiment(
    body: AgentRunRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> AgentRunResponse:
    result = run_named_agent(_AGENT_PATHS["experiment"], experiment_id=body.experiment_id, payload=body.payload)
    return AgentRunResponse(**result)


@router.post("/agents/curve-fitting/preview")
async def curve_fitting_preview(
    user: Annotated[AuthUser, Depends(get_current_user)],
    data_file: Annotated[Optional[UploadFile], File()] = None,
    composition_file: Annotated[Optional[UploadFile], File()] = None,
    data_file_path: Annotated[Optional[str], Form()] = None,
    composition_file_path: Annotated[Optional[str], Form()] = None,
) -> dict:
    try:
        data_path, comp_path = await persist_curve_fitting_uploads(
            data_file,
            composition_file,
            data_file_path=data_file_path,
            composition_file_path=composition_file_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    memory = get_memory_manager()
    if data_path:
        memory.set_var("curve_fitting_last_data_file", data_path)
    if comp_path:
        memory.set_var("curve_fitting_last_composition_file", comp_path)

    return {
        "data_preview": preview_table_file(data_path) if data_path else None,
        "composition_preview": preview_table_file(comp_path) if comp_path else None,
        "data_file": data_path,
        "composition_file": comp_path,
    }


@router.get("/agents/curve-fitting/session")
def curve_fitting_session_get(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    memory = get_memory_manager()
    if experiment_id:
        memory.set_current_experiment(experiment_id, user.id)
    return get_curve_fitting_session(memory)


@router.get("/agents/curve-fitting/results")
def curve_fitting_results_get(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    memory = get_memory_manager()
    if experiment_id:
        memory.set_current_experiment(experiment_id, user.id)
    return get_curve_fitting_results(memory)


@router.get("/agents/curve-fitting/plot")
def curve_fitting_plot(
    user: Annotated[AuthUser, Depends(get_current_user)],
    well: Annotated[str, Query()],
    read: Annotated[Optional[str], Query()] = None,
) -> FileResponse:
    memory = get_memory_manager()
    plot_path = find_plot_path(memory, well, read)
    if not plot_path:
        raise HTTPException(status_code=404, detail="Plot not found for this well/read")
    return FileResponse(
        path=str(plot_path),
        media_type="image/png",
        filename=plot_path.name,
    )


@router.post("/agents/curve-fitting", response_model=AgentRunResponse)
async def run_curve_fitting(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Annotated[Optional[int], Form()] = None,
    data_file: Annotated[Optional[UploadFile], File()] = None,
    composition_file: Annotated[Optional[UploadFile], File()] = None,
    data_file_path: Annotated[Optional[str], Form()] = None,
    composition_file_path: Annotated[Optional[str], Form()] = None,
) -> AgentRunResponse:
    try:
        data_path, comp_path = await persist_curve_fitting_uploads(
            data_file,
            composition_file,
            data_file_path=data_file_path,
            composition_file_path=composition_file_path,
        )
    except ValueError as exc:
        return AgentRunResponse(
            agent=_AGENT_PATHS["curve-fitting"],
            status="error",
            message=str(exc),
            data={},
        )

    payload: dict = {
        "action": "run",
        "auto_trigger": True,
        "data_file": data_path,
        "source": "api_upload",
    }
    if comp_path:
        payload["composition_file"] = comp_path
    if data_file and data_file.filename:
        payload["data_file_name"] = data_file.filename
    if composition_file and composition_file.filename:
        payload["composition_file_name"] = composition_file.filename

    result = run_named_agent(
        _AGENT_PATHS["curve-fitting"],
        experiment_id=experiment_id,
        payload=payload,
    )
    if isinstance(result.get("data"), dict):
        result["data"]["data_file"] = data_path
        if comp_path:
            result["data"]["composition_file"] = comp_path
    return AgentRunResponse(**result)


@router.post("/agents/ml", response_model=AgentRunResponse)
def run_ml(
    body: AgentRunRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> AgentRunResponse:
    result = run_named_agent(_AGENT_PATHS["ml"], experiment_id=body.experiment_id, payload=body.payload)
    return AgentRunResponse(**result)


@router.post("/agents/analysis", response_model=AgentRunResponse)
def run_analysis(
    body: AgentRunRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> AgentRunResponse:
    result = run_named_agent(_AGENT_PATHS["analysis"], experiment_id=body.experiment_id, payload=body.payload)
    return AgentRunResponse(**result)
