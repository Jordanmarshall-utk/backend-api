"""API test for the phase-mapper Library Runs history endpoint."""
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
    m.list_runs.return_value = [
        {"id": 2, "library_id": 1, "status": "succeeded", "job_pid": None, "return_code": 0,
         "error_message": "", "frozen": True, "created_at": "t2", "completed_at": "t2"},
        {"id": 1, "library_id": 1, "status": "failed", "job_pid": None, "return_code": 1,
         "error_message": "boom", "frozen": False, "created_at": "t1", "completed_at": "t1"},
    ]
    return m


def test_list_runs(svc):
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/libraries/1/runs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2
    assert resp.json()[0]["id"] == 2


def test_list_runs_not_found(svc):
    svc.list_runs.side_effect = LookupError("not found")
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/libraries/999/runs")
    assert resp.status_code == 404
