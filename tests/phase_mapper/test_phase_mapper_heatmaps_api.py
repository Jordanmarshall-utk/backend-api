"""API test for the phase-mapper Heatmaps tab endpoint."""
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
    m.get_heatmaps.return_value = {
        "columns": ["pl_peak_nm", "pred_xrd_clean_3d_perovskite_prob"],
        "wells": [{"well": "A1", "row": "A", "col": 1, "pl_peak_nm": 512.3, "pred_xrd_clean_3d_perovskite_prob": 0.84}],
    }
    return m


def test_get_heatmaps(svc):
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10/heatmaps")
    assert resp.status_code == 200
    body = resp.json()
    assert "pl_peak_nm" in body["columns"]
    assert body["wells"][0]["well"] == "A1"


def test_get_heatmaps_not_found(svc):
    svc.get_heatmaps.side_effect = LookupError("not found")
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/999/heatmaps")
    assert resp.status_code == 404
