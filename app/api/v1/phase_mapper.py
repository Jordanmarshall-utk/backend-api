"""FastAPI router — Phase Mapper library CRUD and uploads."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.services.phase_mapper_service import PhaseMapperService

router = APIRouter()


def get_service() -> PhaseMapperService:
    return PhaseMapperService()


class CreateLibraryRequest(BaseModel):
    name: str
    composition_profile: str = ""


class LibraryResponse(BaseModel):
    id: int
    user_id: str
    name: str
    slug: str
    composition_profile: str
    created_at: str
    updated_at: str


class UploadResponse(BaseModel):
    path: str
    warnings: list[str]


UploadKind = Literal["composition", "plate_reader", "raw_xrd", "xrd_labels_reviewed"]


@router.post("/phase-mapper/libraries", response_model=LibraryResponse)
def create_library(
    body: CreateLibraryRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    return svc.create_library(user.id, body.name, body.composition_profile)


@router.get("/phase-mapper/libraries", response_model=list[LibraryResponse])
def list_libraries(
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[dict]:
    svc = get_service()
    return svc.list_libraries(user.id)


@router.get("/phase-mapper/libraries/{library_id}", response_model=LibraryResponse)
def get_library(
    library_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        return svc.get_library_owned(library_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")


@router.delete("/phase-mapper/libraries/{library_id}")
def delete_library(
    library_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        svc.delete_library(library_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")
    return {"status": "deleted"}


@router.post("/phase-mapper/libraries/{library_id}/uploads/{kind}", response_model=UploadResponse)
async def upload_file(
    library_id: int,
    kind: UploadKind,
    user: Annotated[AuthUser, Depends(get_current_user)],
    file: UploadFile = File(...),
) -> dict:
    svc = get_service()
    try:
        return await svc.save_upload(library_id, user.id, kind, file)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class RunResponse(BaseModel):
    id: int
    library_id: int
    status: str
    dir_path: str
    job_pid: int | None
    return_code: int | None
    error_message: str
    frozen: bool
    created_at: str
    completed_at: str | None


@router.post("/phase-mapper/libraries/{library_id}/runs", response_model=RunResponse)
def start_run(
    library_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        return svc.start_run(library_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/phase-mapper/runs/{run_id}", response_model=RunResponse)
def get_run(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        return svc.get_run(run_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
