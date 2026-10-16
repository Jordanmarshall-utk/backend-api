"""FastAPI router — Literature Agent endpoints."""
from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.tools.literature_agent_service import (
    DOMAIN_PROFILES,
    EXTRACTION_PROFILES,
    HALIDE_PV_DOMAIN_PROFILE,
    NON_PV_FAMILY_GATING,
    PCE_EXTRACTION_PROFILE,
    LiteratureAgentConfig,
    LiteratureAgentService,
)

router = APIRouter()

_JOB_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,128}$')


def _validate_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id format")

VALID_STAGES = {"extract_batch", "vision_pass", "sanitize_summaries", "integrate_and_model", "knowledge_graph", "evidence_packet"}
VALID_EXTRACTION_PROFILES = EXTRACTION_PROFILES
VALID_DOMAIN_PROFILES = DOMAIN_PROFILES


def get_service() -> LiteratureAgentService:
    return LiteratureAgentService(LiteratureAgentConfig.load())


# ── Request / Response models ───────────────────────────────────────────────

class LiteratureSearchRequest(BaseModel):
    query: str
    limit: int = 5


class PaperHit(BaseModel):
    paper_slug: str
    title: str
    doi: str | None = None
    score: int
    summary_excerpt: str


class JobSummary(BaseModel):
    job_id: str
    stage: str
    status: str
    created_at: float


class JobDetail(JobSummary):
    log_tail: str
    return_code: int | None = None


class StartStageRequest(BaseModel):
    stage: str
    search_query: str = "perovskite solar cell stability T80 retention"
    max_papers: int = 100
    extraction_profile: str | None = None
    family_gating: str | None = None
    domain_profile: str | None = None
    ontology_path: str | None = None


class EvidencePacketRequest(BaseModel):
    query: str
    limit: int = 5
    max_chars: int = 16000


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/literature/health")
def get_literature_health(
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    try:
        svc = get_service()
        health = svc.health()
        return {
            "ok": health.get("ok", False),
            "domain_profile": health.get("domain_profile"),
            "domain_profiles": health.get("domain_profiles", sorted(VALID_DOMAIN_PROFILES)),
            "active_jobs": health.get("active_jobs", []),
            "paths": health.get("paths", {}),
            "path_checks": health.get("path_checks", {}),
        }
    except Exception:
        return {
            "ok": False,
            "domain_profile": None,
            "domain_profiles": sorted(VALID_DOMAIN_PROFILES),
            "active_jobs": [],
            "paths": {},
            "path_checks": {},
        }


@router.post("/literature/search", response_model=list[PaperHit])
def search_literature(
    body: LiteratureSearchRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[PaperHit]:
    svc = get_service()
    hits = svc.search(body.query, limit=body.limit)
    return [PaperHit(**h) for h in hits]


@router.post("/literature/evidence_packet")
def get_literature_evidence_packet(
    body: EvidencePacketRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    svc = get_service()
    return svc.write_evidence_packet(
        body.query.strip(),
        limit=max(1, min(body.limit, 20)),
        max_chars=max(1000, min(body.max_chars, 50000)),
    )


@router.get("/literature/jobs", response_model=list[JobSummary])
def list_literature_jobs(
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[JobSummary]:
    svc = get_service()
    jobs = svc.list_jobs()
    return [
        JobSummary(
            job_id=j.job_id,
            stage=j.stage,
            status=j.status,
            created_at=j.created_at,
        )
        for j in jobs
    ]


@router.get("/literature/jobs/{job_id}", response_model=JobDetail)
def get_literature_job(
    job_id: str,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> JobDetail:
    _validate_job_id(job_id)
    svc = get_service()
    status = svc.job_status(job_id)
    return JobDetail(
        job_id=status["job_id"],
        stage=status["stage"],
        status=status["status"],
        created_at=status["created_at"],
        log_tail=status.get("log_tail", ""),
        return_code=status.get("return_code"),
    )


@router.post("/literature/start_stage")
def start_literature_stage(
    body: StartStageRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    if body.stage not in VALID_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage '{body.stage}'. Must be one of: {sorted(VALID_STAGES)}",
        )
    if body.domain_profile is not None and body.domain_profile not in VALID_DOMAIN_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain_profile '{body.domain_profile}'. Must be one of: {sorted(VALID_DOMAIN_PROFILES)}",
        )
    if body.extraction_profile is not None and body.extraction_profile not in VALID_EXTRACTION_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid extraction_profile '{body.extraction_profile}'. Must be one of: {sorted(VALID_EXTRACTION_PROFILES)}",
        )
    svc = get_service()
    domain_profile = body.domain_profile or svc.config.domain_profile
    extraction_profile = body.extraction_profile or (
        PCE_EXTRACTION_PROFILE
        if domain_profile == HALIDE_PV_DOMAIN_PROFILE
        else "general_literature"
    )
    family_gating = (
        body.family_gating or "strict"
        if domain_profile == HALIDE_PV_DOMAIN_PROFILE
        else NON_PV_FAMILY_GATING
    )
    try:
        job = svc.start_stage(
            body.stage,
            request={
                "search_query": body.search_query,
                "max_papers": body.max_papers,
                "extraction_profile": extraction_profile,
                "family_gating": family_gating,
                "domain_profile": domain_profile,
                "ontology_path": body.ontology_path,
                "disable_google_drive": True,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "domain_profile": domain_profile,
    }


@router.delete("/literature/jobs/{job_id}")
def cancel_literature_job(
    job_id: str,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    _validate_job_id(job_id)
    svc = get_service()
    result = svc.cancel_job(job_id)
    return {"job_id": result["job_id"], "status": result["status"]}
