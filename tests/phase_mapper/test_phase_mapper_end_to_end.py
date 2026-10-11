"""End-to-end test: create a library, upload fixtures, run the pipeline as a
real subprocess, poll to completion, and fetch the Reads tab — all driven
through FastAPI's TestClient (real HTTP requests), not by calling
PhaseMapperService directly.

Every other route-layer test in this suite (test_phase_mapper_api.py,
test_phase_mapper_runs_api.py, test_phase_mapper_reads_api.py) mocks the
service out with a MagicMock, so this is the only test that exercises the
route <-> service seam — request parsing, get_current_user -> user.id
propagation, HTTPException mapping, and response_model serialization against
a real DB dict — with real objects on both sides. The pipeline subprocess
still actually executes; only the entry point changes from direct Python
calls to HTTP requests against the app.
"""
import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.phase_mapper import router
from app.core.auth import AuthUser
from app.tools.database import DatabaseManager

_FIXTURES = Path(__file__).parent / "fixtures"


def _mock_user():
    return AuthUser(id="user-1", email="test@example.com")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from app.db import engine as engine_module
    engine_module._engine = None
    monkeypatch.setattr("app.tools.phase_mapper_paths.get_user_data_dir", lambda: str(tmp_path))

    DatabaseManager().init_schema()

    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user

    yield TestClient(app)
    engine_module._engine = None


@pytest.mark.slow
def test_full_library_pipeline_round_trip_over_http(client):
    resp = client.post("/phase-mapper/libraries", json={"name": "Fixture Library"})
    assert resp.status_code == 200
    library = resp.json()

    resp = client.post(
        f"/phase-mapper/libraries/{library['id']}/uploads/composition",
        files={"file": ("composition_map_fixture.csv", (_FIXTURES / "composition_map_fixture.csv").read_bytes(), "text/csv")},
    )
    assert resp.status_code == 200
    assert "path" not in resp.json()  # finding 4: no server filesystem path leaked

    resp = client.post(
        f"/phase-mapper/libraries/{library['id']}/uploads/plate_reader",
        files={"file": ("plate_reader_fixture.csv", (_FIXTURES / "plate_reader_fixture.csv").read_bytes(), "text/csv")},
    )
    assert resp.status_code == 200

    resp = client.post(f"/phase-mapper/libraries/{library['id']}/runs")
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "running"
    assert "dir_path" not in run  # finding 4: no server filesystem path leaked

    deadline = time.time() + 60
    while time.time() < deadline:
        resp = client.get(f"/phase-mapper/runs/{run['id']}")
        assert resp.status_code == 200
        run = resp.json()
        if run["status"] != "running":
            break
        time.sleep(1)

    assert run["status"] == "succeeded", run["error_message"]

    resp = client.get(f"/phase-mapper/runs/{run['id']}/reads")
    assert resp.status_code == 200
    reads = resp.json()
    assert len(reads) >= 1
    assert reads[0]["read_type"] == "absorbance"

    resp = client.get(f"/phase-mapper/runs/{run['id']}/predictions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) > 0

    resp = client.get(f"/phase-mapper/runs/{run['id']}/xrd-recommendations")
    assert resp.status_code == 200
    xrd_recs = resp.json()
    assert isinstance(xrd_recs, list)
    if xrd_recs:
        assert "well" in xrd_recs[0]

    resp = client.get(f"/phase-mapper/runs/{run['id']}/remeasurement")
    assert resp.status_code == 200
    remeasurement = resp.json()
    assert isinstance(remeasurement, list)
    if remeasurement:
        assert "well" in remeasurement[0]

    resp = client.get(f"/phase-mapper/runs/{run['id']}/optical-spectra")
    assert resp.status_code == 200
    assert len(resp.json()) > 0
    resp = client.get(f"/phase-mapper/runs/{run['id']}/optical-spectra?well=A1")
    assert resp.status_code == 200
    assert all(row["well"] == "A1" for row in resp.json())

    resp = client.get(f"/phase-mapper/runs/{run['id']}/heatmaps")
    assert resp.status_code == 200
    heatmaps_body = resp.json()
    assert "Cs_fraction" in heatmaps_body["columns"]
    assert any(w["well"] == "A1" for w in heatmaps_body["wells"])

    resp = client.get(f"/phase-mapper/libraries/{library['id']}/runs")
    assert resp.status_code == 200
    assert any(r["id"] == run["id"] for r in resp.json())

    resp = client.get(f"/phase-mapper/runs/{run['id']}/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        assert "parsed_reads_summary.csv" in names
        # Finding 3 (final review): job bookkeeping files leak server
        # filesystem paths and job internals and must never be in the zip.
        assert "job_payload.json" not in names
        assert "run.log" not in names
        assert not any(n.endswith(".returncode") for n in names)

    # Finding 5: frozen sample_key must be keyed off the real library name
    # ("Fixture Library::A1"), not the literal placeholder string "library".
    db = DatabaseManager()
    run_row = db.get_phase_mapper_run(run["id"])
    descriptors = pd.read_csv(Path(run_row["dir_path"]) / "composition_descriptors_per_well.csv")
    assert descriptors["sample_key"].str.startswith("Fixture Library::").all()
    assert not descriptors["sample_key"].str.startswith("library::").any()

    # Finding 3: a completed run can be deleted, e.g. to let a user manually
    # clear a stuck run and unblock starting a new one.
    resp = client.delete(f"/phase-mapper/runs/{run['id']}")
    assert resp.status_code == 200
    resp = client.get(f"/phase-mapper/runs/{run['id']}")
    assert resp.status_code == 404
