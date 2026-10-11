# Phase-Mapper Remaining Result Tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up the seven remaining phase-mapper result tabs whose data comes entirely from `run_pipeline()`'s existing output (Predictions, XRD Recommendations, Remeasurement, Optical Spectra, Heatmaps, Library Runs history, Downloads), completing the backend for everything except the raw-XRD-dependent tabs.

**Architecture:** Extend `PhaseMapperService` and `app/api/v1/phase_mapper.py` (both already established in the walking-skeleton plan) with one method + one route per tab, following the exact pattern `get_reads` set: fetch the owned run, read one or more CSVs out of `run["dir_path"]`, serialize to JSON. Add a small shared helper first (`_get_owned_run`, `_read_csv_records`) since that preamble would otherwise repeat seven more times.

**Tech Stack:** Same as the walking-skeleton plan — FastAPI, pandas, the existing `DatabaseManager`/`PhaseMapperService` stack, pytest.

## Global Constraints

- No changes to the ported pipeline algorithm code (`app/services/phase_mapper/*.py`) — this plan is pure serialization of already-correct output files.
- Every new route enforces per-user ownership via the existing `get_library_owned`/run-ownership pattern — no route may return data for a run belonging to another user.
- Every new service method gets BOTH a mocked API-layer test (proving the route wiring) AND a real, unmocked service-level test against a real `DatabaseManager` and real files on disk (proving the actual logic) — added together, not as a follow-up fix. (The walking-skeleton plan's task-by-task reviews found this gap in three separate tasks — Task 5, Task 6, and Task 7 — each requiring a fix round to add the unmocked test after the fact. Do it right the first time here.)
- Out of scope, deliberately: XRD Spectra, XRD Peak Fits, and XRD label review. Their data depends on raw-XRD processing (`_prepare_raw_xrd_tables` and related orchestration in the reference app's `app.py`) that was never ported — only the reference algorithm modules (`xrd_peak_fitting.py`, `xrd_auto_assignment.py`, `xrd_processing.py`) were ported in the walking-skeleton plan, not the Streamlit-level orchestration that wires raw XRD uploads into them. That orchestration is a separate, sizable piece of work for a later plan.
- Python 3.12, existing `backend-api` dependency set — no new third-party packages needed (pandas, `zipfile`, `io` are already available).

## Working directory note

Same as the walking-skeleton plan: if this executes inside an SDD worktree (`backend-api/.claude/worktrees/<name>/`), that worktree's root **is** backend-api's root — any `cd backend-api` in a snippet is a no-op there. Task 9 touches the separate `shared-types` repository, not this one.

---

### Task 1: Shared ownership/CSV-read helpers

**Files:**
- Modify: `app/services/phase_mapper_service.py` (add two helpers, refactor `get_run`, `delete_run`, `get_reads` to use them)
- Test: `tests/phase_mapper/test_phase_mapper_service.py` (existing tests for `get_run`/`delete_run`/`get_reads` must still pass unchanged — this task is a pure refactor, not a behavior change)

**Interfaces:**
- Produces (used by every task in this plan):
  - `PhaseMapperService._get_owned_run(self, run_id: int, user_id: str) -> dict` — raises `LookupError` if the run doesn't exist or its library isn't owned by `user_id`; otherwise returns the run row dict
  - `PhaseMapperService._read_csv_records(self, run_dir: str, filename: str) -> list[dict[str, Any]]` — returns `[]` if the file doesn't exist, otherwise the CSV as a list of JSON-safe records

- [ ] **Step 1: Add the helpers**

In `app/services/phase_mapper_service.py`, add these two methods to `PhaseMapperService` (near the top of the class, before `get_run`):

```python
    def _get_owned_run(self, run_id: int, user_id: str) -> dict[str, Any]:
        run = self.db.get_phase_mapper_run(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        self.get_library_owned(run["library_id"], user_id)
        return run

    @staticmethod
    def _read_csv_records(run_dir: str, filename: str) -> list[dict[str, Any]]:
        csv_path = Path(run_dir) / filename
        if not csv_path.exists():
            return []
        df = pd.read_csv(csv_path)
        return json.loads(df.to_json(orient="records"))
```

- [ ] **Step 2: Refactor the three existing methods to use them**

Replace `get_run`'s current existence/ownership preamble:

```python
    def get_run(self, run_id: int, user_id: str) -> dict[str, Any]:
        run = self.db.get_phase_mapper_run(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        self.get_library_owned(run["library_id"], user_id)  # raises LookupError if not owned
```

with:

```python
    def get_run(self, run_id: int, user_id: str) -> dict[str, Any]:
        run = self._get_owned_run(run_id, user_id)
```

(the rest of `get_run`'s body — the returncode-file polling logic — is unchanged, just re-indented if needed to reference the new `run` variable, which holds the same shape as before).

Replace `delete_run`'s body:

```python
    def delete_run(self, run_id: int, user_id: str) -> None:
        run = self._get_owned_run(run_id, user_id)
        self.db.delete_phase_mapper_run(run_id)
```

Replace `get_reads` entirely:

```python
    def get_reads(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        return self._read_csv_records(run["dir_path"], "parsed_reads_summary.csv")
```

- [ ] **Step 3: Run the existing tests to confirm the refactor is behavior-preserving**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/ -v`
Expected: PASS, same counts as before this change (no test file touched in this task) — this refactor must not change what any existing test asserts.

- [ ] **Step 4: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py
git commit -m "refactor: extract shared ownership/CSV-read helpers in PhaseMapperService"
```

---

### Task 2: Predictions endpoint

**Files:**
- Modify: `app/services/phase_mapper_service.py` (add `get_predictions`)
- Modify: `app/api/v1/phase_mapper.py` (add route)
- Test: `tests/phase_mapper/test_phase_mapper_predictions_api.py` (mocked route test), `tests/phase_mapper/test_phase_mapper_service.py` (real service test)

**Interfaces:**
- Consumes: `PhaseMapperService._get_owned_run`, `_read_csv_records` (Task 1)
- Produces: `PhaseMapperService.get_predictions(run_id: int, user_id: str) -> list[dict]`; route `GET /phase-mapper/runs/{run_id}/predictions`

- [ ] **Step 1: Write the failing tests**

`tests/phase_mapper/test_phase_mapper_predictions_api.py`:

```python
"""API test for the phase-mapper Predictions tab endpoint."""
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
    m.get_predictions.return_value = [{"well": "A1", "pred_xrd_clean_3d_perovskite_prob": 0.84}]
    return m


def test_get_predictions(svc):
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10/predictions")
    assert resp.status_code == 200
    assert resp.json()[0]["well"] == "A1"


def test_get_predictions_not_found(svc):
    svc.get_predictions.side_effect = LookupError("not found")
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/999/predictions")
    assert resp.status_code == 404
```

Add to `tests/phase_mapper/test_phase_mapper_service.py` (append; follow the file's existing fixture conventions — a real `DatabaseManager` + real `PhaseMapperService`, a library/run created via `db.create_phase_mapper_library`/`db.create_phase_mapper_run`, and a real CSV written into the run's `dir_path`):

```python
def test_get_predictions_returns_csv_records_for_owning_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "model_predictions_all_wells.csv").write_text(
        "well,pred_xrd_clean_3d_perovskite_prob\nA1,0.84\nA2,0.12\n", encoding="utf-8"
    )
    records = service.get_predictions(run_id, "user-A")
    assert records == [
        {"well": "A1", "pred_xrd_clean_3d_perovskite_prob": 0.84},
        {"well": "A2", "pred_xrd_clean_3d_perovskite_prob": 0.12},
    ]


def test_get_predictions_raises_for_different_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    with pytest.raises(LookupError):
        service.get_predictions(run_id, "user-B")


def test_get_predictions_returns_empty_list_when_csv_absent(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    assert service.get_predictions(run_id, "user-A") == []
```

`_make_run_with_dir(db, tmp_path, *, user_id: str = "user-A") -> tuple[int, os.PathLike]` already exists in `test_phase_mapper_service.py` (added alongside the walking-skeleton plan's `get_reads` tests, at the file's current line 105) — it creates a library, a run row, and a real `run-dir` under `tmp_path`, returning `(run_id, run_dir)`. Use it exactly as shown above; do not redefine it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_predictions_api.py -v`
Expected: FAIL — route doesn't exist yet.

- [ ] **Step 3: Add the service method**

```python
    def get_predictions(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        return self._read_csv_records(run["dir_path"], "model_predictions_all_wells.csv")
```

- [ ] **Step 4: Add the route**

```python
@router.get("/phase-mapper/runs/{run_id}/predictions")
def get_predictions(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[dict]:
    svc = get_service()
    try:
        return svc.get_predictions(run_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_predictions_api.py tests/phase_mapper/test_phase_mapper_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_predictions_api.py tests/phase_mapper/test_phase_mapper_service.py
git commit -m "feat: add phase-mapper Predictions tab endpoint"
```

---

### Task 3: XRD Recommendations endpoint

**Files:** same shape as Task 2, targeting `xrd_recommendations_next_wells.csv`.

**Interfaces:**
- Produces: `PhaseMapperService.get_xrd_recommendations(run_id: int, user_id: str) -> list[dict]`; route `GET /phase-mapper/runs/{run_id}/xrd-recommendations`

- [ ] **Step 1: Write the failing tests**

Create `tests/phase_mapper/test_phase_mapper_xrd_recommendations_api.py`, identical in structure to Task 2's `test_phase_mapper_predictions_api.py` (same fixture, same two test functions), except:
- mock/assert on `svc.get_xrd_recommendations` instead of `svc.get_predictions`
- request path `/phase-mapper/runs/10/xrd-recommendations` / `/phase-mapper/runs/999/xrd-recommendations`
- sample response row: `{"well": "A1", "S_XRD": 0.71, "reason_for_recommendation": "high model uncertainty"}`

Add the matching three real-service tests to `tests/phase_mapper/test_phase_mapper_service.py` (owning-user success, different-user `LookupError`, empty-when-absent), targeting `xrd_recommendations_next_wells.csv`, using `_make_run_with_dir(db, tmp_path, user_id="user-A")` exactly as Task 2's tests do (see that task for the literal pattern: unpack `run_id, run_dir`, write the fixture CSV via `(run_dir / "...").write_text(...)`, assert on `service.get_xrd_recommendations(...)`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_xrd_recommendations_api.py -v`
Expected: FAIL

- [ ] **Step 3: Add the service method**

```python
    def get_xrd_recommendations(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        return self._read_csv_records(run["dir_path"], "xrd_recommendations_next_wells.csv")
```

- [ ] **Step 4: Add the route**

```python
@router.get("/phase-mapper/runs/{run_id}/xrd-recommendations")
def get_xrd_recommendations(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[dict]:
    svc = get_service()
    try:
        return svc.get_xrd_recommendations(run_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_xrd_recommendations_api.py tests/phase_mapper/test_phase_mapper_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_xrd_recommendations_api.py tests/phase_mapper/test_phase_mapper_service.py
git commit -m "feat: add phase-mapper XRD Recommendations tab endpoint"
```

---

### Task 4: Remeasurement endpoint

**Files:** same shape as Task 2, targeting `optical_remeasurement_recommendations.csv`.

**Interfaces:**
- Produces: `PhaseMapperService.get_remeasurement(run_id: int, user_id: str) -> list[dict]`; route `GET /phase-mapper/runs/{run_id}/remeasurement`

- [ ] **Step 1: Write the failing tests**

Create `tests/phase_mapper/test_phase_mapper_remeasurement_api.py`, same structure as Task 2/3, targeting `svc.get_remeasurement`, path `/phase-mapper/runs/{run_id}/remeasurement`, sample row `{"well": "B3", "optical_remeasurement_priority_score": 0.55}`.

Add the three matching real-service tests to `test_phase_mapper_service.py`, targeting `optical_remeasurement_recommendations.csv`, using `_make_run_with_dir` exactly as in Task 2.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_remeasurement_api.py -v`
Expected: FAIL

- [ ] **Step 3: Add the service method**

```python
    def get_remeasurement(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        return self._read_csv_records(run["dir_path"], "optical_remeasurement_recommendations.csv")
```

- [ ] **Step 4: Add the route**

```python
@router.get("/phase-mapper/runs/{run_id}/remeasurement")
def get_remeasurement(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[dict]:
    svc = get_service()
    try:
        return svc.get_remeasurement(run_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_remeasurement_api.py tests/phase_mapper/test_phase_mapper_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_remeasurement_api.py tests/phase_mapper/test_phase_mapper_service.py
git commit -m "feat: add phase-mapper Remeasurement tab endpoint"
```

---

### Task 5: Optical Spectra endpoint (with optional well filter)

**Files:** same shape as Task 2, targeting `spectral_data_long_format.csv`, plus an optional `well` query parameter since this file is long-format (one row per well × wavelength × read) and can be large across a full 96-well plate.

**Interfaces:**
- Produces: `PhaseMapperService.get_optical_spectra(run_id: int, user_id: str, well: str | None = None) -> list[dict]`; route `GET /phase-mapper/runs/{run_id}/optical-spectra?well=A1`

- [ ] **Step 1: Write the failing tests**

`tests/phase_mapper/test_phase_mapper_optical_spectra_api.py`:

```python
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
```

Add to `test_phase_mapper_service.py`, using `_make_run_with_dir` exactly as in Task 2:

```python
def test_get_optical_spectra_returns_all_rows_by_default(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "spectral_data_long_format.csv").write_text(
        "well,wavelength_nm,signal\nA1,500,1.0\nA2,500,1.1\n", encoding="utf-8"
    )
    records = service.get_optical_spectra(run_id, "user-A")
    assert len(records) == 2


def test_get_optical_spectra_filters_by_well(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "spectral_data_long_format.csv").write_text(
        "well,wavelength_nm,signal\nA1,500,1.0\nA2,500,1.1\n", encoding="utf-8"
    )
    records = service.get_optical_spectra(run_id, "user-A", well="A1")
    assert records == [{"well": "A1", "wavelength_nm": 500, "signal": 1.0}]


def test_get_optical_spectra_raises_for_different_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    with pytest.raises(LookupError):
        service.get_optical_spectra(run_id, "user-B")


def test_get_optical_spectra_returns_empty_list_when_csv_absent(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    assert service.get_optical_spectra(run_id, "user-A") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_optical_spectra_api.py -v`
Expected: FAIL

- [ ] **Step 3: Add the service method**

```python
    def get_optical_spectra(self, run_id: int, user_id: str, well: str | None = None) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        records = self._read_csv_records(run["dir_path"], "spectral_data_long_format.csv")
        if well:
            records = [r for r in records if r.get("well") == well]
        return records
```

- [ ] **Step 4: Add the route**

```python
@router.get("/phase-mapper/runs/{run_id}/optical-spectra")
def get_optical_spectra(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
    well: str | None = None,
) -> list[dict]:
    svc = get_service()
    try:
        return svc.get_optical_spectra(run_id, user.id, well)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_optical_spectra_api.py tests/phase_mapper/test_phase_mapper_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_optical_spectra_api.py tests/phase_mapper/test_phase_mapper_service.py
git commit -m "feat: add phase-mapper Optical Spectra tab endpoint"
```

---

### Task 6: Heatmaps endpoint (multi-file merge)

**Files:**
- Modify: `app/services/phase_mapper_service.py` (add `get_heatmaps`)
- Modify: `app/api/v1/phase_mapper.py` (add route)
- Test: `tests/phase_mapper/test_phase_mapper_heatmaps_api.py`, `tests/phase_mapper/test_phase_mapper_service.py`

**Interfaces:**
- Produces: `PhaseMapperService.get_heatmaps(run_id: int, user_id: str) -> dict` — shape `{"columns": list[str], "wells": list[dict]}`, where each well dict has `well`, `row`, `col`, and one key per numeric column found across the merged source files; route `GET /phase-mapper/runs/{run_id}/heatmaps`

Unlike Tasks 2-5, this reads from multiple files and merges them on `well`, since the reference app's Heatmaps tab draws from several output files at once (composition descriptors for row/col position, optical features, model predictions, and both recommendation tables). Rather than hardcoding which specific columns to expose (fragile — column names live in the ported pipeline code and could drift), this merges every available file and returns every numeric column found, letting the frontend choose what to render. `composition_descriptors_per_well.csv` is the base table because it's the one file guaranteed to have `row`/`col` position columns (from `well_row_col()` in the ported `composition_features.py`).

- [ ] **Step 1: Write the failing tests**

`tests/phase_mapper/test_phase_mapper_heatmaps_api.py`:

```python
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
```

Add to `test_phase_mapper_service.py`:

```python
def test_get_heatmaps_merges_available_files(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "composition_descriptors_per_well.csv").write_text(
        "well,row,col,Cs_fraction\nA1,A,1,0.5\nA2,A,2,0.7\n", encoding="utf-8"
    )
    (run_dir / "optical_features_per_well.csv").write_text(
        "well,pl_peak_nm\nA1,512.3\nA2,498.1\n", encoding="utf-8"
    )
    result = service.get_heatmaps(run_id, "user-A")
    assert "pl_peak_nm" in result["columns"]
    assert "Cs_fraction" in result["columns"]
    a1 = next(w for w in result["wells"] if w["well"] == "A1")
    assert a1["row"] == "A"
    assert a1["col"] == 1
    assert a1["pl_peak_nm"] == 512.3


def test_get_heatmaps_raises_for_different_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    with pytest.raises(LookupError):
        service.get_heatmaps(run_id, "user-B")


def test_get_heatmaps_returns_empty_when_no_files_present(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    result = service.get_heatmaps(run_id, "user-A")
    assert result == {"columns": [], "wells": []}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_heatmaps_api.py -v`
Expected: FAIL

- [ ] **Step 3: Add the service method**

```python
    _HEATMAP_SOURCE_FILES = [
        "composition_descriptors_per_well.csv",
        "optical_features_per_well.csv",
        "model_predictions_all_wells.csv",
        "xrd_recommendations_next_wells.csv",
        "optical_remeasurement_recommendations.csv",
    ]

    def get_heatmaps(self, run_id: int, user_id: str) -> dict[str, Any]:
        run = self._get_owned_run(run_id, user_id)
        run_dir = run["dir_path"]

        base_path = Path(run_dir) / self._HEATMAP_SOURCE_FILES[0]
        if not base_path.exists():
            return {"columns": [], "wells": []}
        merged = pd.read_csv(base_path)

        for filename in self._HEATMAP_SOURCE_FILES[1:]:
            path = Path(run_dir) / filename
            if not path.exists():
                continue
            other = pd.read_csv(path)
            overlap = [c for c in other.columns if c in merged.columns and c != "well"]
            other = other.drop(columns=overlap)
            merged = merged.merge(other, on="well", how="left")

        non_metric_cols = {"well", "row", "col", "library", "sample_key", "is_control", "replicate_group", "notes"}
        numeric_cols = [
            c for c in merged.columns
            if c not in non_metric_cols and pd.api.types.is_numeric_dtype(merged[c])
        ]
        records = json.loads(merged.to_json(orient="records"))
        return {"columns": numeric_cols, "wells": records}
```

- [ ] **Step 4: Add the route**

```python
@router.get("/phase-mapper/runs/{run_id}/heatmaps")
def get_heatmaps(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        return svc.get_heatmaps(run_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_heatmaps_api.py tests/phase_mapper/test_phase_mapper_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_heatmaps_api.py tests/phase_mapper/test_phase_mapper_service.py
git commit -m "feat: add phase-mapper Heatmaps tab endpoint"
```

---

### Task 7: Library Runs history endpoint

**Files:**
- Modify: `app/services/phase_mapper_service.py` (add `list_runs`)
- Modify: `app/api/v1/phase_mapper.py` (add route + `RunResponse`-shaped list)
- Test: `tests/phase_mapper/test_phase_mapper_library_runs_api.py`, `tests/phase_mapper/test_phase_mapper_service.py`

This is the cheapest task in the plan: `DatabaseManager.list_phase_mapper_runs(library_id)` already exists (from the walking-skeleton plan's Task 1) and has never had a route. This task only adds the ownership-checked wrapper and the route.

**Interfaces:**
- Consumes: `DatabaseManager.list_phase_mapper_runs(library_id: int) -> list[dict]` (existing), `get_library_owned` (existing)
- Produces: `PhaseMapperService.list_runs(library_id: int, user_id: str) -> list[dict]`; route `GET /phase-mapper/libraries/{library_id}/runs`

- [ ] **Step 1: Write the failing tests**

`tests/phase_mapper/test_phase_mapper_library_runs_api.py`:

```python
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
```

Add to `test_phase_mapper_service.py`:

```python
def test_list_runs_returns_all_runs_for_owning_user(db, service):
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")
    first = db.create_phase_mapper_run(library_id, "/data/lib-a/runs/1")
    db.update_phase_mapper_run(first, status="succeeded")
    second = db.create_phase_mapper_run(library_id, "/data/lib-a/runs/2")
    runs = service.list_runs(library_id, "user-A")
    assert [r["id"] for r in runs] == [second, first]


def test_list_runs_raises_for_different_user(db, service):
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")
    with pytest.raises(LookupError):
        service.list_runs(library_id, "user-B")
```

Note this test's use of `db.create_phase_mapper_library`/`db.create_phase_mapper_run` directly matches the exact pattern already used elsewhere in `test_phase_mapper_service.py` (e.g. the `get_library_owned` tests from the walking-skeleton plan) — no new fixture helper is needed for this task, unlike Tasks 2-6.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_library_runs_api.py -v`
Expected: FAIL

- [ ] **Step 3: Add the service method**

```python
    def list_runs(self, library_id: int, user_id: str) -> list[dict[str, Any]]:
        self.get_library_owned(library_id, user_id)
        return self.db.list_phase_mapper_runs(library_id)
```

- [ ] **Step 4: Add the route**

Reuse the existing `RunResponse` model (already defined in this file) as the list item type:

```python
@router.get("/phase-mapper/libraries/{library_id}/runs", response_model=list[RunResponse])
def list_runs(
    library_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[dict]:
    svc = get_service()
    try:
        return svc.list_runs(library_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_library_runs_api.py tests/phase_mapper/test_phase_mapper_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_library_runs_api.py tests/phase_mapper/test_phase_mapper_service.py
git commit -m "feat: add phase-mapper Library Runs history endpoint"
```

---

### Task 8: Downloads endpoint (ZIP export)

**Files:**
- Modify: `app/services/phase_mapper_service.py` (add `get_run_download_bytes`)
- Modify: `app/api/v1/phase_mapper.py` (add route returning a binary ZIP response)
- Test: `tests/phase_mapper/test_phase_mapper_download_api.py`, `tests/phase_mapper/test_phase_mapper_service.py`

**Interfaces:**
- Produces: `PhaseMapperService.get_run_download_bytes(run_id: int, user_id: str) -> bytes` — a ZIP archive of every file directly under the run's `dir_path` (non-recursive, matching the reference app's `_zip_outputs`); route `GET /phase-mapper/runs/{run_id}/download`

- [ ] **Step 1: Write the failing tests**

`tests/phase_mapper/test_phase_mapper_download_api.py`:

```python
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
```

Add to `test_phase_mapper_service.py`:

```python
def test_get_run_download_bytes_zips_run_directory(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "parsed_reads_summary.csv").write_text("read_num\n1\n", encoding="utf-8")
    zip_bytes = service.get_run_download_bytes(run_id, "user-A")

    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert "parsed_reads_summary.csv" in zf.namelist()
        assert zf.read("parsed_reads_summary.csv") == b"read_num\n1\n"


def test_get_run_download_bytes_raises_for_different_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    with pytest.raises(LookupError):
        service.get_run_download_bytes(run_id, "user-B")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_download_api.py -v`
Expected: FAIL

- [ ] **Step 3: Add the service method**

Add `import io` and `import zipfile` to the top of `app/services/phase_mapper_service.py` if not already present, then:

```python
    def get_run_download_bytes(self, run_id: int, user_id: str) -> bytes:
        run = self._get_owned_run(run_id, user_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in Path(run["dir_path"]).glob("*"):
                if path.is_file():
                    zf.write(path, arcname=path.name)
        return buffer.getvalue()
```

- [ ] **Step 4: Add the route**

Add `from fastapi import Response` to the existing `fastapi` import in `app/api/v1/phase_mapper.py` if not already imported, then:

```python
@router.get("/phase-mapper/runs/{run_id}/download")
def download_run(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> Response:
    svc = get_service()
    try:
        content = svc.get_run_download_bytes(run_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}_outputs.zip"'},
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_download_api.py tests/phase_mapper/test_phase_mapper_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_download_api.py tests/phase_mapper/test_phase_mapper_service.py
git commit -m "feat: add phase-mapper run download (ZIP export) endpoint"
```

---

### Task 9: OpenAPI contract

**Files:**
- Modify: `shared-types/openapi/polaris-v1.yaml`
- Modify (regenerated): `shared-types/src/generated/api.ts`

**Interfaces:**
- Consumes: the exact response shapes from Tasks 2-8
- Produces: typed client methods for the follow-up web/mobile plans

- [ ] **Step 1: Confirm you're working in the right repository and branch**

This task's changes belong in the separate `shared-types` repository (`/Users/arielthompson/Documents/POLARIS/shared-types`), NOT in `backend-api`. Create a new branch there off `main` (which already contains the walking-skeleton plan's phase-mapper contract additions, merged): e.g. `git checkout -b phase-mapper-remaining-tabs-contract` from `/Users/arielthompson/Documents/POLARIS/shared-types`.

- [ ] **Step 2: Add the new paths and schemas**

Add under `paths:` (matching the existing `/phase-mapper/*` paths' style exactly — read a couple of them first, e.g. `/phase-mapper/runs/{run_id}/reads`, to copy the established formatting):

```yaml
  /phase-mapper/runs/{run_id}/predictions:
    get:
      operationId: getPhaseMapperPredictions
      tags: [phase-mapper]
      parameters:
        - { name: run_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: Model predictions for every well in this run
          content:
            application/json:
              schema: { type: array, items: { type: object } }
        '404': { description: Run not found }
  /phase-mapper/runs/{run_id}/xrd-recommendations:
    get:
      operationId: getPhaseMapperXrdRecommendations
      tags: [phase-mapper]
      parameters:
        - { name: run_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: Recommended next wells for XRD measurement
          content:
            application/json:
              schema: { type: array, items: { type: object } }
        '404': { description: Run not found }
  /phase-mapper/runs/{run_id}/remeasurement:
    get:
      operationId: getPhaseMapperRemeasurement
      tags: [phase-mapper]
      parameters:
        - { name: run_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: Optical remeasurement priority recommendations
          content:
            application/json:
              schema: { type: array, items: { type: object } }
        '404': { description: Run not found }
  /phase-mapper/runs/{run_id}/optical-spectra:
    get:
      operationId: getPhaseMapperOpticalSpectra
      tags: [phase-mapper]
      parameters:
        - { name: run_id, in: path, required: true, schema: { type: integer } }
        - { name: well, in: query, required: false, schema: { type: string } }
      responses:
        '200':
          description: Long-format optical spectra, optionally filtered to one well
          content:
            application/json:
              schema: { type: array, items: { type: object } }
        '404': { description: Run not found }
  /phase-mapper/runs/{run_id}/heatmaps:
    get:
      operationId: getPhaseMapperHeatmaps
      tags: [phase-mapper]
      parameters:
        - { name: run_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: Per-well metrics merged from this run's output files, for heatmap rendering
          content:
            application/json:
              schema: { $ref: '#/components/schemas/PhaseMapperHeatmaps' }
        '404': { description: Run not found }
  /phase-mapper/libraries/{library_id}/runs:
    get:
      operationId: listPhaseMapperLibraryRuns
      tags: [phase-mapper]
      parameters:
        - { name: library_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: Run history for a library, newest first
          content:
            application/json:
              schema:
                type: array
                items: { $ref: '#/components/schemas/PhaseMapperRun' }
        '404': { description: Library not found }
  /phase-mapper/runs/{run_id}/download:
    get:
      operationId: downloadPhaseMapperRun
      tags: [phase-mapper]
      parameters:
        - { name: run_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: ZIP archive of this run's output files
          content:
            application/zip:
              schema: { type: string, format: binary }
        '404': { description: Run not found }
```

Add under `components.schemas:`:

```yaml
    PhaseMapperHeatmaps:
      type: object
      required: [columns, wells]
      properties:
        columns:
          type: array
          items: { type: string }
        wells:
          type: array
          items: { type: object }
```

Note: `PhaseMapperRun` already exists in this file (from the walking-skeleton plan) and does not need to be redefined — the new `listPhaseMapperLibraryRuns` operation just references it.

- [ ] **Step 3: Validate and regenerate**

```bash
cd shared-types
npm run lint:openapi
npm run codegen
```

Expected: `lint:openapi` reports 0 errors (the existing `operation-description`/`operation-tag-defined` warnings are pre-existing and not something this task needs to fix). `codegen` regenerates `src/generated/api.ts` with the 7 new operations and the `PhaseMapperHeatmaps` type.

- [ ] **Step 4: Commit**

```bash
cd shared-types
git add openapi/polaris-v1.yaml src/generated/api.ts
git commit -m "feat: add remaining phase-mapper result-tab endpoints to API contract"
```

---

### Task 10: End-to-end smoke test covering all seven new endpoints

**Files:**
- Modify: `tests/phase_mapper/test_phase_mapper_end_to_end.py` (extend the existing HTTP-driven e2e test)

**Interfaces:**
- Consumes: everything from Tasks 2-8, plus the existing e2e test's fixture/flow from the walking-skeleton plan

- [ ] **Step 1: Read the existing e2e test**

Read `tests/phase_mapper/test_phase_mapper_end_to_end.py` in full first — it already drives create → upload → start_run → poll → get_reads → delete_run through a real `TestClient` (fixture `client`) against a real subprocess-executed pipeline run, inside `test_full_library_pipeline_round_trip_over_http`. Library and run responses are held in local variables `library` and `run` (dicts from `resp.json()`), so existing calls look like `client.get(f"/phase-mapper/runs/{run['id']}")`. This task extends that same test function; it does not create a second e2e test.

- [ ] **Step 2: Write the failing assertions**

Insert the following after the existing `reads[0]["read_type"] == "absorbance"` assertion and before the "Finding 5" sample_key comment/check, so the new endpoints are exercised against the same completed run before it gets deleted at the end of the test:

```python
    resp = client.get(f"/phase-mapper/runs/{run['id']}/predictions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = client.get(f"/phase-mapper/runs/{run['id']}/xrd-recommendations")
    assert resp.status_code == 200

    resp = client.get(f"/phase-mapper/runs/{run['id']}/remeasurement")
    assert resp.status_code == 200

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
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        assert "parsed_reads_summary.csv" in zf.namelist()
```

This reuses the existing test's `resp` variable name (already used throughout the function for exactly this "call then check" pattern) rather than introducing eight new per-endpoint variable names.

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/test_phase_mapper_end_to_end.py -v -m slow`
Expected: FAIL — the new endpoint calls 404 or the new assertions don't hold, since Tasks 2-8 haven't run yet if this task is executed out of order. (In practice this task runs last, after Tasks 2-8 are already merged, so this step should mostly just confirm the extended test compiles and the assertions are meaningful — if everything from Tasks 2-8 is already in place, this step may in fact pass immediately. If so, treat that as confirmation rather than a process violation: the point of RED here is "prove the assertions can fail," which you can satisfy by temporarily reverting one endpoint's route registration and confirming that specific assertion fails, then restoring it.)

- [ ] **Step 4: Run the full suite to verify everything passes**

Run: `cd backend-api && DATABASE_URL="" .venv/bin/python -m pytest tests/phase_mapper/ -v`
Expected: PASS (all tests from this plan and the walking-skeleton plan; the pre-existing 14 skips unchanged)

- [ ] **Step 5: Commit**

```bash
cd backend-api
git add tests/phase_mapper/test_phase_mapper_end_to_end.py
git commit -m "test: extend phase-mapper e2e test to cover remaining result-tab endpoints"
```

---

## After this plan

Backend coverage for the phase-mapper library workflow is now complete except for the raw-XRD-dependent tabs (XRD Spectra, XRD Peak Fits) and XRD label review — those need the reference app's raw-XRD orchestration logic ported (not just the algorithm modules, which are already in `app/services/phase_mapper/`), and belong in their own plan given the write-workflow complexity (manual peak-assignment review, reference library management). The web frontend (Plan 2) and mobile (Plan 3) can now be built against a materially complete backend for every tab except those two.
