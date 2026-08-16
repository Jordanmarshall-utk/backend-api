"""API test for the phase-mapper Optical Spectra tab endpoint."""
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
    m.get_optical_spectra.return_value = [{"well": "A1", "wavelength_nm": 500, "signal": 1.0, "read_name": "Abs_initial"}]
    return m


def _client(svc):
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    return TestClient(app)


def test_get_optical_spectra(svc):
    client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10/optical-spectra")
    assert resp.status_code == 200
    assert resp.json()[0]["well"] == "A1"
    svc.get_optical_spectra.assert_called_once_with(10, "user-1", None)


def test_get_optical_spectra_with_well_filter(svc):
    client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10/optical-spectra?well=A1")
    assert resp.status_code == 200
    svc.get_optical_spectra.assert_called_once_with(10, "user-1", "A1")


def test_get_optical_spectra_not_found(svc):
    svc.get_optical_spectra.side_effect = LookupError("not found")
    client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/999/optical-spectra")
    assert resp.status_code == 404
