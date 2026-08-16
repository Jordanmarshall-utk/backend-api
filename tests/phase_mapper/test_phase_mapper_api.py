"""API tests for phase-mapper library CRUD and uploads."""
import io
from unittest.mock import AsyncMock, MagicMock, patch

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
    m.create_library.return_value = {"id": 1, "user_id": "user-1", "name": "DMA-Rb", "slug": "DMA-Rb", "composition_profile": "", "created_at": "t", "updated_at": "t"}
    m.list_libraries.return_value = [m.create_library.return_value]
    m.get_library_owned.return_value = m.create_library.return_value
    return m


def _client(svc):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides.clear()
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    return app, TestClient(app)


def test_create_library(svc):
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.post("/phase-mapper/libraries", json={"name": "DMA-Rb", "composition_profile": ""})
    assert resp.status_code == 200
    assert resp.json()["name"] == "DMA-Rb"
    svc.create_library.assert_called_once_with("user-1", "DMA-Rb", "")


def test_list_libraries(svc):
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/libraries")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_library_not_found_returns_404(svc):
    svc.get_library_owned.side_effect = LookupError("not found")
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/libraries/999")
    assert resp.status_code == 404


def test_upload_composition_csv(svc):
    svc.save_upload = AsyncMock(return_value={"path": "/data/user-1/lib-a/inputs/composition.csv", "warnings": []})
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.post(
            "/phase-mapper/libraries/1/uploads/composition",
            files={"file": ("composition.csv", io.BytesIO(b"well\nA1\n"), "text/csv")},
        )
    assert resp.status_code == 200
    assert resp.json()["warnings"] == []
    assert "path" not in resp.json()
