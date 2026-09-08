from fastapi import APIRouter

from app.api.v1 import (
    agents,
    analysis_session,
    dashboard,
    documents,
    experiments,
    feedback,
    health,
    history,
    literature,
    llm,
    mcp,
    ml_session,
    phase_mapper,
    pipeline,
    session,
    settings,
    watcher,
    workflows,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(llm.router, tags=["llm"])
api_router.include_router(experiments.router, tags=["experiments"])
api_router.include_router(dashboard.router, tags=["dashboard"])
api_router.include_router(workflows.router, tags=["workflows"])
api_router.include_router(ml_session.router, tags=["ml"])
api_router.include_router(analysis_session.router, tags=["analysis"])
api_router.include_router(agents.router, tags=["agents"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(watcher.router, tags=["watcher"])
api_router.include_router(mcp.router, tags=["mcp"])
api_router.include_router(settings.router, tags=["settings"])
api_router.include_router(session.router, tags=["session"])
api_router.include_router(history.router, tags=["history"])
api_router.include_router(pipeline.router, tags=["pipeline"])
api_router.include_router(literature.router, tags=["literature"])
api_router.include_router(phase_mapper.router, tags=["phase-mapper"])
api_router.include_router(feedback.router, tags=["feedback"])
