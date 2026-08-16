"""API test for the phase-mapper Remeasurement tab endpoint."""
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
    m.get_remeasurement.return_value = [{"well": "B3", "optical_remeasurement_priority_score": 0.55}]
    return m


def test_get_remeasurement(svc):
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10/remeasurement")
    assert resp.status_code == 200
    assert resp.json()[0]["well"] == "B3"


def test_get_remeasurement_not_found(svc):
    svc.get_remeasurement.side_effect = LookupError("not found")
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/999/remeasurement")
    assert resp.status_code == 404
