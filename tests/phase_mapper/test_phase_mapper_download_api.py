"""API test for the phase-mapper run download (ZIP export) endpoint."""
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
    m.get_run_download_bytes.return_value = b"PK\x03\x04fake-zip-bytes"
    return m


def test_download_run(svc):
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert resp.content == b"PK\x03\x04fake-zip-bytes"


def test_download_run_not_found(svc):
    svc.get_run_download_bytes.side_effect = LookupError("not found")
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/999/download")
    assert resp.status_code == 404
