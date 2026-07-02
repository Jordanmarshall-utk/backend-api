from __future__ import annotations

from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field

from app.core.deps import get_current_user
from app.core.auth import AuthUser
from app.services.curve_fitting_uploads import save_upload_file
from app.services.memory_service import get_memory_manager
from app.services.ml_session_service import get_ml_session, patch_ml_session, run_ml_automation
from app.services.curve_fitting_exports import resolve_results_path

router = APIRouter()


class MlSessionPatch(BaseModel):
    experiment_id: Optional[int] = None
    model_choice: Optional[str] = None
    ml_model_config: Optional[Dict[str, Any]] = None
    auto_ml_after_curve_fitting: Optional[bool] = None
    json_path: Optional[str] = None
    csv_path: Optional[str] = None
    composition_path: Optional[str] = None


class MlRunRequest(BaseModel):
    experiment_id: Optional[int] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


@router.get("/ml/session")
def ml_session_get(
    user: Annotated[AuthUser, Depends(get_current_user)],
    experiment_id: Optional[int] = None,
) -> dict:
    memory = get_memory_manager()
    if experiment_id:
        memory.set_current_experiment(experiment_id, user.id)
    return get_ml_session(memory)


@router.patch("/ml/session")
def ml_session_patch(
    body: MlSessionPatch,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    memory = get_memory_manager()
    if body.experiment_id:
        memory.set_current_experiment(body.experiment_id, user.id)
    return patch_ml_session(
        memory,
        model_choice=body.model_choice,
        ml_model_config=body.ml_model_config,
        auto_ml_after_curve_fitting=body.auto_ml_after_curve_fitting,
        json_path=body.json_path,
        csv_path=body.csv_path,
        composition_path=body.composition_path,
    )


@router.post("/ml/composition")
async def ml_composition_upload(
    user: Annotated[AuthUser, Depends(get_current_user)],
    composition_file: Annotated[UploadFile | None, File()] = None,
    composition_file_path: Annotated[str | None, Form()] = None,
    experiment_id: Annotated[int | None, Form()] = None,
) -> dict:
    """Upload or point to a composition CSV for single-objective GP."""
    memory = get_memory_manager()
    if experiment_id:
        memory.set_current_experiment(experiment_id, user.id)

    comp_path: str | None = None
    if composition_file and composition_file.filename:
        comp_path = await save_upload_file(composition_file, prefix="composition")
    elif composition_file_path and str(composition_file_path).strip():
        resolved = resolve_results_path(str(composition_file_path).strip())
        if resolved and resolved.is_file():
            comp_path = str(resolved)
        else:
            from pathlib import Path

            p = Path(str(composition_file_path).strip()).expanduser()
            if not p.is_file():
                return {"status": "error", "message": f"Composition file not found: {p}"}
            comp_path = str(p.resolve())

    if not comp_path:
        return {
            "status": "error",
            "message": "Provide composition_file (upload) or composition_file_path (server path).",
        }

    memory.set_var("ml_auto_composition_path", comp_path)
    memory.set_var("curve_fitting_last_composition_file", comp_path)
    return {"status": "success", "message": "Composition file linked.", **get_ml_session(memory)}


@router.post("/ml/run")
def ml_run(
    body: MlRunRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    memory = get_memory_manager()
    if body.experiment_id:
        memory.set_current_experiment(body.experiment_id, user.id)
    return run_ml_automation(memory, body.payload)
