"""API tests for phase-mapper run orchestration."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.phase_mapper import router
from app.core.auth import AuthUser


def _mock_user():
    return AuthUser(id="user-1", email="test@example.com")


@pytest.fixture
def svc():
    m = MagicMock()
    m.get_library_owned.return_value = {"id": 1, "user_id": "user-1", "name": "DMA-Rb", "slug": "DMA-Rb", "composition_profile": "", "created_at": "t", "updated_at": "t"}
    m.start_run.return_value = {"id": 10, "library_id": 1, "status": "running", "dir_path": "/data/user-1/lib-a/runs/x", "job_pid": 123, "return_code": None, "error_message": "", "frozen": False, "created_at": "t", "completed_at": None}
    m.get_run.return_value = m.start_run.return_value
    return m


def _client(svc):
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    return app, TestClient(app)


def test_start_run(svc):
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.post("/phase-mapper/libraries/1/runs")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_start_run_conflict_when_already_running(svc):
    svc.start_run.side_effect = RuntimeError("A run is already in progress for this library")
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.post("/phase-mapper/libraries/1/runs")
    assert resp.status_code == 409


def test_get_run(svc):
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10")
    assert resp.status_code == 200
    assert resp.json()["id"] == 10


def test_get_run_not_found(svc):
    svc.get_run.side_effect = LookupError("not found")
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/999")
    assert resp.status_code == 404


def test_get_run_response_omits_dir_path(svc):
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10")
    assert resp.status_code == 200
    assert "dir_path" not in resp.json()


def test_delete_run(svc):
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.delete("/phase-mapper/runs/10")
    assert resp.status_code == 200
    svc.delete_run.assert_called_once_with(10, "user-1")


def test_delete_run_not_found_for_non_owned_run(svc):
    svc.delete_run.side_effect = LookupError("not found")
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.delete("/phase-mapper/runs/999")
    assert resp.status_code == 404
