"""Unit tests for the literature API router — no real corpus required."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.literature import router
from app.core.auth import AuthUser


def _mock_user():
    return AuthUser(id="test-user", email="test@example.com")



@pytest.fixture
def svc():
    m = MagicMock()
    m.config.domain_profile = "halide_pv"
    m.health.return_value = {
        "ok": True,
        "domain_profile": "halide_pv",
        "domain_profiles": ["general_science", "halide_pv", "oxide_perovskite"],
        "active_jobs": [],
        "paths": {"knowledge_graph_builder": "literature_agent/scripts/build_literature_knowledge_graph.py"},
        "path_checks": {"knowledge_graph_builder": True},
    }
    m.search.return_value = [
        {
            "paper_slug": "slug1",
            "title": "Test Paper",
            "doi": "10.1234/test",
            "score": 5,
            "summary_excerpt": "A summary excerpt.",
        }
    ]
    m.list_jobs.return_value = []
    m.job_status.return_value = {
        "job_id": "job_abc",
        "stage": "extract_batch",
        "status": "completed",
        "created_at": 1700000000.0,
        "log_tail": "Done.",
        "return_code": 0,
    }
    m.start_stage.return_value = {"job_id": "job_new", "status": "running"}
    m.cancel_job.return_value = {"job_id": "job_abc", "status": "cancelled"}
    return m


def test_health_ok(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.get("/literature/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["domain_profile"] == "halide_pv"
    assert "general_science" in data["domain_profiles"]
    assert "active_jobs" in data
    assert data["path_checks"]["knowledge_graph_builder"] is True
    assert "knowledge_graph_builder" in data["paths"]


def test_search_returns_hits(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.post("/literature/search", json={"query": "perovskite", "limit": 5})
    assert resp.status_code == 200
    hits = resp.json()
    assert isinstance(hits, list)
    assert hits[0]["paper_slug"] == "slug1"


def test_start_stage_validates_stage(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.post(
            "/literature/start_stage",
            json={"stage": "bad_stage", "search_query": "test", "max_papers": 10},
        )
    assert resp.status_code == 400


def test_start_stage_valid(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.post(
            "/literature/start_stage",
            json={"stage": "extract_batch", "search_query": "perovskite", "max_papers": 50},
        )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "job_new"
    assert resp.json()["domain_profile"] == "halide_pv"
    request = svc.start_stage.call_args.kwargs["request"]
    assert request["domain_profile"] == "halide_pv"
    assert request["extraction_profile"] == "pce_stability_modeling"
    assert request["family_gating"] == "strict"
    assert request["ontology_path"] is None


def test_start_stage_non_halide_uses_general_profile_and_non_pv_gating(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.post(
            "/literature/start_stage",
            json={
                "stage": "extract_batch",
                "domain_profile": "oxide_perovskite",
                "ontology_path": "literature_agent/config/domain_packs/oxide_perovskite.json",
                "search_query": "oxygen vacancy conductivity",
                "max_papers": 25,
                "family_gating": "strict",
            },
        )

    assert resp.status_code == 200
    assert resp.json()["domain_profile"] == "oxide_perovskite"
    request = svc.start_stage.call_args.kwargs["request"]
    assert request["domain_profile"] == "oxide_perovskite"
    assert request["extraction_profile"] == "general_literature"
    assert request["family_gating"] == "off"
    assert request["ontology_path"] == "literature_agent/config/domain_packs/oxide_perovskite.json"


def test_start_stage_rejects_unknown_domain_profile(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.post(
            "/literature/start_stage",
            json={"stage": "extract_batch", "domain_profile": "unknown"},
        )

    assert resp.status_code == 400
    assert "domain_profile" in resp.json()["detail"]
    svc.start_stage.assert_not_called()


def test_start_stage_maps_domain_safety_error_to_bad_request(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    svc.start_stage.side_effect = ValueError(
        "integrate_and_model is only available for domain_profile=halide_pv"
    )

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.post(
            "/literature/start_stage",
            json={
                "stage": "integrate_and_model",
                "domain_profile": "general_science",
            },
        )

    assert resp.status_code == 400
    assert "halide_pv" in resp.json()["detail"]


def test_delete_job_cancels(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.delete("/literature/jobs/job_abc")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_health_returns_ok_false_on_unconfigured_service():
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", side_effect=OSError("path not found")):
        client = TestClient(app)
        resp = client.get("/literature/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_get_job_rejects_path_traversal(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.get("/literature/jobs/../../../etc/passwd")
    # FastAPI normalises %2F in path params, so the router receives the raw string;
    # the job_id regex blocks anything containing slashes or dots
    assert resp.status_code in (400, 404)


def test_cancel_job_rejects_path_traversal(svc):
    import app.api.v1.literature as lit_module
    import app.core.deps as deps_module

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    with patch.object(lit_module, "get_service", return_value=svc):
        client = TestClient(app)
        resp = client.delete("/literature/jobs/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404, 422)
