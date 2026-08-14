# Phase-Mapper Backend Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the backend spine for the phase-mapper library workflow — library CRUD, per-user upload storage, the ported reference-app pipeline running as a background job, and one fully-wired result endpoint (Reads) — proving the whole architecture end to end before the remaining nine result tabs are added.

**Architecture:** Port `POLARIS_Integration_Handoff_20260810/src/*.py` unmodified into `app/services/phase_mapper/`. Wrap it with new per-user path helpers, a `DatabaseManager` extension for library/run bookkeeping, an upload-persistence service, and a background-subprocess job runner modeled on `app/tools/literature_agent_job_runner.py`. Expose it all through a new `app/api/v1/phase_mapper.py` router and the `@polaris/shared-types` OpenAPI contract.

**Tech Stack:** FastAPI, SQLAlchemy async (`sqlite+aiosqlite` in dev, `asyncpg` in prod) via the existing `DatabaseManager`, pandas/numpy/scikit-learn (ported pipeline), pytest + `fastapi.testclient.TestClient`.

## Global Constraints

- No scientific-logic changes: `src/*.py` algorithmic code is copied verbatim; only I/O (paths, config source) is adapted. (Design doc, "Non-goals".)
- Data is scoped per authenticated user: every library and run is namespaced under `<user_data_dir>/phase_mapper/<user_id>/<library_slug>/`. (Design doc, "Architecture".)
- Long-running pipeline execution happens via the existing background-subprocess job pattern, not inline in the request. (Design doc, "Architecture".)
- Upload validation keeps the reference app's semantics: fail only on a missing `well` column; warn (don't block) on incomplete well coverage or composition fractions not summing to ~1. (Design doc, "Error handling".)
- Same-user concurrent "Run Pipeline" calls on one library return `409` instead of racing. (Design doc, "Error handling".)
- `library::well` remains the sample identity; manual XRD labels stay authoritative over automatic proposals (carried forward from the handoff doc; enforced by later tasks in the follow-up "remaining tabs" plan, not this skeleton).
- Python 3.12, existing `backend-api` dependency set (`pyproject.toml`) — no new third-party packages needed for this plan (pandas/numpy/scikit-learn/openpyxl/matplotlib/scipy are already dependencies).

## Scope note

This plan covers the backend only, and only enough of it to prove the pattern:
library CRUD, uploads, pipeline-as-a-background-job, and the **Reads** tab endpoint.
The other nine result tabs (Heatmaps, Predictions, XRD Recommendations,
Remeasurement, Optical Spectra, XRD Spectra, XRD Peak Fits, Library Runs,
Downloads) and XRD label review are one mechanical serializer + route + test
each, following the exact pattern Task 7 establishes — they are intentionally
left to a follow-up plan so this one stays reviewable. Web (Plan 2) and mobile
(Plan 3) are separate plans per the design doc.

---

### Task 1: Library and run persistence in `DatabaseManager`

**Files:**
- Modify: `app/tools/database.py:243` (insert two new `CREATE TABLE` statements into `ddl_statements`, just before the list's closing `]`)
- Modify: `app/tools/database.py:806` (append new methods at end of file, still inside `DatabaseManager`)
- Create: `migrations/versions/002_phase_mapper.py`
- Test: `tests/phase_mapper/test_phase_mapper_database.py`

**Interfaces:**
- Produces (used by Tasks 4, 5, 6):
  - `DatabaseManager.create_phase_mapper_library(user_id: str, name: str, slug: str, composition_profile: str = "") -> int`
  - `DatabaseManager.list_phase_mapper_libraries(user_id: str) -> List[Dict]` — each dict has `id, user_id, name, slug, composition_profile, created_at, updated_at`
  - `DatabaseManager.get_phase_mapper_library(library_id: int) -> Optional[Dict]`
  - `DatabaseManager.delete_phase_mapper_library(library_id: int) -> None`
  - `DatabaseManager.create_phase_mapper_run(library_id: int, dir_path: str) -> int` — inserts with `status="running"`
  - `DatabaseManager.update_phase_mapper_run(run_id: int, *, status: Optional[str] = None, job_pid: Optional[int] = None, return_code: Optional[int] = None, error_message: Optional[str] = None, frozen: Optional[bool] = None, completed_at: Optional[str] = None) -> None`
  - `DatabaseManager.get_phase_mapper_run(run_id: int) -> Optional[Dict]`
  - `DatabaseManager.list_phase_mapper_runs(library_id: int) -> List[Dict]` — newest first
  - `DatabaseManager.get_latest_phase_mapper_run(library_id: int) -> Optional[Dict]` — most recent run with `status="succeeded"`
  - `DatabaseManager.has_active_phase_mapper_run(library_id: int) -> bool` — `True` if any run has `status="running"`
  - `DatabaseManager.delete_phase_mapper_run(run_id: int) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/phase_mapper/__init__.py` (empty) and `tests/phase_mapper/test_phase_mapper_database.py`:

```python
"""Unit tests for phase-mapper library/run persistence in DatabaseManager."""
import os

import pytest

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from app.tools.database import DatabaseManager  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    # Fresh in-memory engine per test so schema/state don't leak across tests.
    from app.db import engine as engine_module
    engine_module._engine = None
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    manager = DatabaseManager()
    manager.init_schema()
    yield manager
    engine_module._engine = None


def test_create_and_get_library(db):
    library_id = db.create_phase_mapper_library("user-1", "DMA-Rb", "DMA-Rb", "custom")
    library = db.get_phase_mapper_library(library_id)
    assert library["user_id"] == "user-1"
    assert library["name"] == "DMA-Rb"
    assert library["slug"] == "DMA-Rb"
    assert library["composition_profile"] == "custom"


def test_list_libraries_scoped_to_user(db):
    db.create_phase_mapper_library("user-1", "Lib A", "lib-a")
    db.create_phase_mapper_library("user-2", "Lib B", "lib-b")
    libraries = db.list_phase_mapper_libraries("user-1")
    assert [lib["name"] for lib in libraries] == ["Lib A"]


def test_delete_library(db):
    library_id = db.create_phase_mapper_library("user-1", "Lib A", "lib-a")
    db.delete_phase_mapper_library(library_id)
    assert db.get_phase_mapper_library(library_id) is None


def test_run_lifecycle_and_active_check(db):
    library_id = db.create_phase_mapper_library("user-1", "Lib A", "lib-a")
    run_id = db.create_phase_mapper_run(library_id, "/data/lib-a/runs/20260814_000000")
    assert db.has_active_phase_mapper_run(library_id) is True

    db.update_phase_mapper_run(run_id, status="succeeded", return_code=0, frozen=True, completed_at="2026-08-14 00:05:00")
    run = db.get_phase_mapper_run(run_id)
    assert run["status"] == "succeeded"
    assert run["frozen"] is True
    assert db.has_active_phase_mapper_run(library_id) is False

    latest = db.get_latest_phase_mapper_run(library_id)
    assert latest["id"] == run_id


def test_list_runs_newest_first(db):
    library_id = db.create_phase_mapper_library("user-1", "Lib A", "lib-a")
    first = db.create_phase_mapper_run(library_id, "/data/lib-a/runs/1")
    db.update_phase_mapper_run(first, status="succeeded")
    second = db.create_phase_mapper_run(library_id, "/data/lib-a/runs/2")
    runs = db.list_phase_mapper_runs(library_id)
    assert [r["id"] for r in runs] == [second, first]


def test_delete_run(db):
    library_id = db.create_phase_mapper_library("user-1", "Lib A", "lib-a")
    run_id = db.create_phase_mapper_run(library_id, "/data/lib-a/runs/1")
    db.delete_phase_mapper_run(run_id)
    assert db.get_phase_mapper_run(run_id) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_database.py -v`
Expected: FAIL — `AttributeError: 'DatabaseManager' object has no attribute 'create_phase_mapper_library'`

- [ ] **Step 3: Add the schema**

In `app/tools/database.py`, insert into the `ddl_statements` list (right before its closing `]` at line 244, i.e. after the `hypothesis_outcomes` table and before the list ends):

```python
            f"""CREATE TABLE IF NOT EXISTS phase_mapper_libraries (
                id {serial} PRIMARY KEY {autoincrement},
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                composition_profile TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""",
            f"""CREATE TABLE IF NOT EXISTS phase_mapper_runs (
                id {serial} PRIMARY KEY {autoincrement},
                library_id BIGINT NOT NULL,
                status TEXT NOT NULL,
                dir_path TEXT NOT NULL,
                job_pid BIGINT,
                return_code BIGINT,
                error_message TEXT,
                frozen INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (library_id) REFERENCES phase_mapper_libraries(id)
            )""",
```

- [ ] **Step 4: Add the methods**

Append at the end of `app/tools/database.py` (after the existing `_get_hypothesis_outcomes_async` method, still indented inside `class DatabaseManager`):

```python
    # ── phase mapper libraries ───────────────────────────────────────────────

    def create_phase_mapper_library(self, user_id: str, name: str, slug: str, composition_profile: str = "") -> int:
        return _run(self._create_phase_mapper_library_async(user_id, name, slug, composition_profile))

    async def _create_phase_mapper_library_async(self, user_id: str, name: str, slug: str, composition_profile: str) -> int:
        pg = _is_postgres(self._engine)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with self._engine.begin() as conn:
            if pg:
                result = await conn.execute(text(
                    "INSERT INTO phase_mapper_libraries (user_id, name, slug, composition_profile, created_at, updated_at) "
                    "VALUES (:uid, :n, :s, :cp, :ca, :ua) RETURNING id"
                ), {"uid": user_id, "n": name, "s": slug, "cp": composition_profile, "ca": now, "ua": now})
                return result.fetchone()[0]
            await conn.execute(text(
                "INSERT INTO phase_mapper_libraries (user_id, name, slug, composition_profile, created_at, updated_at) "
                "VALUES (:uid, :n, :s, :cp, :ca, :ua)"
            ), {"uid": user_id, "n": name, "s": slug, "cp": composition_profile, "ca": now, "ua": now})
            return (await conn.execute(text("SELECT last_insert_rowid()"))).fetchone()[0]

    def list_phase_mapper_libraries(self, user_id: str) -> List[Dict]:
        return _run(self._list_phase_mapper_libraries_async(user_id))

    async def _list_phase_mapper_libraries_async(self, user_id: str) -> List[Dict]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(
                "SELECT id, user_id, name, slug, composition_profile, created_at, updated_at "
                "FROM phase_mapper_libraries WHERE user_id = :uid ORDER BY created_at DESC"
            ), {"uid": user_id})).fetchall()
            return [self._library_row_to_dict(r) for r in rows]

    def get_phase_mapper_library(self, library_id: int) -> Optional[Dict]:
        return _run(self._get_phase_mapper_library_async(library_id))

    async def _get_phase_mapper_library_async(self, library_id: int) -> Optional[Dict]:
        async with self._engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT id, user_id, name, slug, composition_profile, created_at, updated_at "
                "FROM phase_mapper_libraries WHERE id = :id"
            ), {"id": library_id})).fetchone()
            return self._library_row_to_dict(row) if row else None

    @staticmethod
    def _library_row_to_dict(row) -> Dict:
        return {
            "id": row[0], "user_id": row[1], "name": row[2], "slug": row[3],
            "composition_profile": row[4] or "", "created_at": row[5], "updated_at": row[6],
        }

    def delete_phase_mapper_library(self, library_id: int) -> None:
        _run(self._delete_phase_mapper_library_async(library_id))

    async def _delete_phase_mapper_library_async(self, library_id: int) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text("DELETE FROM phase_mapper_runs WHERE library_id = :id"), {"id": library_id})
            await conn.execute(text("DELETE FROM phase_mapper_libraries WHERE id = :id"), {"id": library_id})

    # ── phase mapper runs ─────────────────────────────────────────────────────

    def create_phase_mapper_run(self, library_id: int, dir_path: str) -> int:
        return _run(self._create_phase_mapper_run_async(library_id, dir_path))

    async def _create_phase_mapper_run_async(self, library_id: int, dir_path: str) -> int:
        pg = _is_postgres(self._engine)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with self._engine.begin() as conn:
            if pg:
                result = await conn.execute(text(
                    "INSERT INTO phase_mapper_runs (library_id, status, dir_path, created_at) "
                    "VALUES (:lid, 'running', :dp, :ca) RETURNING id"
                ), {"lid": library_id, "dp": dir_path, "ca": now})
                return result.fetchone()[0]
            await conn.execute(text(
                "INSERT INTO phase_mapper_runs (library_id, status, dir_path, created_at) "
                "VALUES (:lid, 'running', :dp, :ca)"
            ), {"lid": library_id, "dp": dir_path, "ca": now})
            return (await conn.execute(text("SELECT last_insert_rowid()"))).fetchone()[0]

    def update_phase_mapper_run(
        self, run_id: int, *, status: Optional[str] = None, job_pid: Optional[int] = None,
        return_code: Optional[int] = None, error_message: Optional[str] = None,
        frozen: Optional[bool] = None, completed_at: Optional[str] = None,
    ) -> None:
        _run(self._update_phase_mapper_run_async(run_id, status, job_pid, return_code, error_message, frozen, completed_at))

    async def _update_phase_mapper_run_async(
        self, run_id: int, status, job_pid, return_code, error_message, frozen, completed_at,
    ) -> None:
        fields = {
            "status": status, "job_pid": job_pid, "return_code": return_code,
            "error_message": error_message, "completed_at": completed_at,
        }
        if frozen is not None:
            fields["frozen"] = int(frozen)
        fields = {k: v for k, v in fields.items() if v is not None}
        if not fields:
            return
        set_clause = ", ".join(f"{k} = :{k}" for k in fields)
        params = dict(fields)
        params["id"] = run_id
        async with self._engine.begin() as conn:
            await conn.execute(text(f"UPDATE phase_mapper_runs SET {set_clause} WHERE id = :id"), params)

    def get_phase_mapper_run(self, run_id: int) -> Optional[Dict]:
        return _run(self._get_phase_mapper_run_async(run_id))

    async def _get_phase_mapper_run_async(self, run_id: int) -> Optional[Dict]:
        async with self._engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT id, library_id, status, dir_path, job_pid, return_code, error_message, frozen, created_at, completed_at "
                "FROM phase_mapper_runs WHERE id = :id"
            ), {"id": run_id})).fetchone()
            return self._run_row_to_dict(row) if row else None

    def list_phase_mapper_runs(self, library_id: int) -> List[Dict]:
        return _run(self._list_phase_mapper_runs_async(library_id))

    async def _list_phase_mapper_runs_async(self, library_id: int) -> List[Dict]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(
                "SELECT id, library_id, status, dir_path, job_pid, return_code, error_message, frozen, created_at, completed_at "
                "FROM phase_mapper_runs WHERE library_id = :lid ORDER BY id DESC"
            ), {"lid": library_id})).fetchall()
            return [self._run_row_to_dict(r) for r in rows]

    def get_latest_phase_mapper_run(self, library_id: int) -> Optional[Dict]:
        return _run(self._get_latest_phase_mapper_run_async(library_id))

    async def _get_latest_phase_mapper_run_async(self, library_id: int) -> Optional[Dict]:
        async with self._engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT id, library_id, status, dir_path, job_pid, return_code, error_message, frozen, created_at, completed_at "
                "FROM phase_mapper_runs WHERE library_id = :lid AND status = 'succeeded' ORDER BY id DESC LIMIT 1"
            ), {"lid": library_id})).fetchone()
            return self._run_row_to_dict(row) if row else None

    def has_active_phase_mapper_run(self, library_id: int) -> bool:
        return _run(self._has_active_phase_mapper_run_async(library_id))

    async def _has_active_phase_mapper_run_async(self, library_id: int) -> bool:
        async with self._engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT 1 FROM phase_mapper_runs WHERE library_id = :lid AND status = 'running' LIMIT 1"
            ), {"lid": library_id})).fetchone()
            return row is not None

    def delete_phase_mapper_run(self, run_id: int) -> None:
        _run(self._delete_phase_mapper_run_async(run_id))

    async def _delete_phase_mapper_run_async(self, run_id: int) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text("DELETE FROM phase_mapper_runs WHERE id = :id"), {"id": run_id})

    @staticmethod
    def _run_row_to_dict(row) -> Dict:
        return {
            "id": row[0], "library_id": row[1], "status": row[2], "dir_path": row[3],
            "job_pid": row[4], "return_code": row[5], "error_message": row[6] or "",
            "frozen": bool(row[7]), "created_at": row[8], "completed_at": row[9],
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_database.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Add the matching Alembic migration**

Create `migrations/versions/002_phase_mapper.py`, mirroring the style of `001_initial_postgres.py`:

```python
"""Phase-mapper libraries and runs."""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "phase_mapper_libraries",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("composition_profile", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "phase_mapper_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("library_id", sa.BigInteger(), sa.ForeignKey("phase_mapper_libraries.id"), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("dir_path", sa.Text(), nullable=False),
        sa.Column("job_pid", sa.BigInteger(), nullable=True),
        sa.Column("return_code", sa.BigInteger(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("frozen", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("phase_mapper_runs")
    op.drop_table("phase_mapper_libraries")
```

- [ ] **Step 7: Commit**

```bash
cd backend-api
git add app/tools/database.py migrations/versions/002_phase_mapper.py tests/phase_mapper/
git commit -m "feat: add phase-mapper library/run persistence to DatabaseManager"
```

---

### Task 2: Per-user path helpers

**Files:**
- Create: `app/tools/phase_mapper_paths.py`
- Test: `tests/phase_mapper/test_phase_mapper_paths.py`

**Interfaces:**
- Consumes: `app.tools.paths.get_user_data_dir() -> str` (existing)
- Produces (used by Tasks 3, 4, 5, 6, 7):
  - `slugify(text: str) -> str`
  - `library_root(user_id: str, slug: str) -> Path`
  - `library_uploads_dir(user_id: str, slug: str, kind: str) -> Path` — `kind` is one of `"plate_reader"`, `"raw_xrd"`, `"xrd_labels_reviewed"` (composition maps are singletons and live under `library_inputs_dir` instead, not here)
  - `library_runs_dir(user_id: str, slug: str) -> Path`
  - `new_run_dir(user_id: str, slug: str) -> Path` — timestamped, not yet created on disk
  - `library_inputs_dir(user_id: str, slug: str) -> Path`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for per-user phase-mapper path helpers."""
from app.tools import phase_mapper_paths as pmp


def test_slugify_matches_reference_app_behavior():
    assert pmp.slugify("DMA-Rb") == "DMA-Rb"
    assert pmp.slugify("DMA / Rb library!") == "DMA_Rb_library"
    assert pmp.slugify("   ") == "library"


def test_library_root_is_namespaced_by_user(monkeypatch, tmp_path):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    root_a = pmp.library_root("user-1", "lib-a")
    root_b = pmp.library_root("user-2", "lib-a")
    assert root_a != root_b
    assert str(tmp_path) in str(root_a)
    assert "user-1" in str(root_a)


def test_uploads_dir_variants(monkeypatch, tmp_path):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    plate_dir = pmp.library_uploads_dir("user-1", "lib-a", "plate_reader")
    xrd_dir = pmp.library_uploads_dir("user-1", "lib-a", "raw_xrd")
    assert plate_dir != xrd_dir
    assert plate_dir.exists()  # created eagerly
    assert xrd_dir.exists()


def test_new_run_dir_is_timestamped_and_not_created(monkeypatch, tmp_path):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    run_dir = pmp.new_run_dir("user-1", "lib-a")
    assert run_dir.parent == pmp.library_runs_dir("user-1", "lib-a")
    assert not run_dir.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.tools.phase_mapper_paths'`

- [ ] **Step 3: Write the implementation**

```python
"""Per-user filesystem layout for the phase-mapper domain.

Mirrors the reference app's outputs/dashboard_run_history structure,
namespaced under a user id so libraries and runs never cross accounts.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from app.tools.paths import get_user_data_dir

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_UPLOAD_KINDS = {"plate_reader", "raw_xrd", "xrd_labels_reviewed"}


def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("_", text.strip()).strip("_")
    return slug or "library"


def _phase_mapper_root() -> Path:
    root = Path(get_user_data_dir()) / "phase_mapper"
    root.mkdir(parents=True, exist_ok=True)
    return root


def library_root(user_id: str, slug: str) -> Path:
    root = _phase_mapper_root() / user_id / slug
    root.mkdir(parents=True, exist_ok=True)
    return root


def library_inputs_dir(user_id: str, slug: str) -> Path:
    path = library_root(user_id, slug) / "inputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_uploads_dir(user_id: str, slug: str, kind: str) -> Path:
    if kind not in _UPLOAD_KINDS:
        raise ValueError(f"Unknown upload kind: {kind}")
    path = library_root(user_id, slug) / "uploads" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_runs_dir(user_id: str, slug: str) -> Path:
    path = library_root(user_id, slug) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_run_dir(user_id: str, slug: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return library_runs_dir(user_id, slug) / timestamp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_paths.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd backend-api
git add app/tools/phase_mapper_paths.py tests/phase_mapper/test_phase_mapper_paths.py
git commit -m "feat: add per-user path helpers for phase-mapper libraries"
```

---

### Task 3: Port the reference app's pipeline core, unmodified

**Files:**
- Create: `app/services/phase_mapper/__init__.py` (empty)
- Create: `app/services/phase_mapper/config.py`, `utils.py`, `composition_features.py`, `spectral_features.py`, `plate_reader.py`, `xrd_processing.py`, `xrd_labels.py`, `xrd_auto_assignment.py`, `xrd_peak_fitting.py`, `active_learning.py`, `active_learning_review.py`, `experiment_profiles.py`, `modeling.py`, `pipeline.py`, `plotting.py` — copied from `POLARIS_Integration_Handoff_20260810/src/*.py`
- Create: `app/tools/phase_mapper_pipeline_runner.py`
- Copy tests: `tests/phase_mapper/test_composition_features.py`, `test_plate_reader.py`, `test_spectral_features.py`, `test_active_learning.py`, `test_active_learning_review.py`, `test_experiment_profiles.py`, `test_modeling_feature_families.py`, `test_multitask_mlp.py`, `test_pipeline_library_merge.py`, `test_spin_coating_optimization.py`, `test_xrd_auto_assignment.py`, `test_xrd_labels.py`, `test_xrd_processing.py` — copied from `POLARIS_Integration_Handoff_20260810/tests/*.py`

**Interfaces:**
- Produces (used by Tasks 5, 6, 7):
  - `app.services.phase_mapper.config.load_config(path: str | Path) -> dict[str, Any]`
  - `app.services.phase_mapper.config.DEFAULT_CONFIG: dict[str, Any]`
  - `app.services.phase_mapper.pipeline.run_pipeline(cfg: dict[str, Any], config_path: Path | None = None) -> dict[str, Path]`

- [ ] **Step 1: Copy the algorithmic modules verbatim**

```bash
cd backend-api
mkdir -p app/services/phase_mapper
touch app/services/phase_mapper/__init__.py
SRC="../../Downloads/POLARIS_Integration_Handoff_20260810/src"
for f in config utils composition_features spectral_features plate_reader \
         xrd_processing xrd_labels xrd_auto_assignment xrd_peak_fitting \
         active_learning active_learning_review experiment_profiles \
         modeling pipeline plotting; do
  cp "$SRC/$f.py" "app/services/phase_mapper/$f.py"
done
```

Note: this plan assumes the handoff package sits at
`/Users/arielthompson/Downloads/POLARIS_Integration_Handoff_20260810`, two
directories up from `backend-api`, as it does in the current checkout —
adjust the `$SRC` path if the checkout differs.

- [ ] **Step 2: Fix internal imports**

The copied files import each other as `from src.xxx import yyy`. Rewrite those
to `from app.services.phase_mapper.xxx import yyy` throughout the copied
files:

```bash
cd backend-api
grep -rl "^from src\." app/services/phase_mapper/ | xargs sed -i '' 's/^from src\./from app.services.phase_mapper./'
grep -rl "^import src\." app/services/phase_mapper/ | xargs sed -i '' 's/^import src\./import app.services.phase_mapper./'
```

Then open each file and confirm no remaining `src.` reference:

```bash
grep -rn "\bsrc\." app/services/phase_mapper/*.py
```

Expected: no output.

- [ ] **Step 3: Copy the existing unit tests verbatim, fixing their imports the same way**

```bash
cd backend-api
TESTS="../../Downloads/POLARIS_Integration_Handoff_20260810/tests"
for f in test_composition_features test_plate_reader test_spectral_features \
         test_active_learning test_active_learning_review test_experiment_profiles \
         test_modeling_feature_families test_multitask_mlp test_pipeline_library_merge \
         test_spin_coating_optimization test_xrd_auto_assignment test_xrd_labels \
         test_xrd_processing; do
  cp "$TESTS/$f.py" "tests/phase_mapper/$f.py"
done
sed -i '' 's/^from src\./from app.services.phase_mapper./' tests/phase_mapper/test_*.py
sed -i '' 's/^import src\./import app.services.phase_mapper./' tests/phase_mapper/test_*.py
```

`test_spin_coating_optimization.py` exercises the spin-coating optimizer,
which is out of scope for this plan (design doc "Non-goals") — it's still
copied here because `active_learning_review.py` and `modeling.py` share
helpers with it and the file won't import cleanly if it's left behind with a
stale `src.` reference. It stays skipped, not run, until the spin-coating
sub-project plan lands:

```bash
cd backend-api
sed -i '' '1i\
import pytest\
pytestmark = pytest.mark.skip(reason="spin-coating optimizer is a later sub-project")
' tests/phase_mapper/test_spin_coating_optimization.py
```

- [ ] **Step 4: Run the copied test suite to verify the port is faithful**

Run: `cd backend-api && pytest tests/phase_mapper/ -v --ignore=tests/phase_mapper/test_phase_mapper_database.py --ignore=tests/phase_mapper/test_phase_mapper_paths.py`
Expected: PASS for every copied test file except the skipped spin-coating one. If any test fails, the failure is almost certainly an import-path issue from Step 2 (fix the import, don't touch the assertion — the algorithmic behavior must not change).

- [ ] **Step 5: Write the background-job runner entry point**

This is new code (not a copy): a thin subprocess entry point, modeled
directly on `app/tools/literature_agent_job_runner.py`, that loads a job
payload and calls the ported `run_pipeline`.

```python
"""Subprocess entry point for a phase-mapper pipeline run.

Mirrors app/tools/literature_agent_job_runner.py: reads a JSON payload
describing the run, executes it, and writes a returncode file the
polling service can observe without keeping a process handle open.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from app.services.phase_mapper.pipeline import run_pipeline


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    run_dir = Path(payload["run_dir"])
    return_code = 0
    try:
        run_pipeline(payload["config"], config_path=None)
    except Exception:
        return_code = 1
        (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    Path(payload["job_dir"], f"{payload['job_id']}.returncode").write_text(str(return_code), encoding="utf-8")
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
```

Note the one deliberate deviation from `run_pipeline`'s CLI wrapper
(`run_pipeline.py` in the reference app): `cfg["paths"]["output_dir"]` is set
directly to the run's own directory by the caller (Task 6), so this runner
calls `run_pipeline(cfg, config_path=None)` with an in-memory config dict
instead of writing and re-reading a YAML file — `config_path` is only used
by `run_pipeline` to resolve *relative* paths, and Task 6 always writes
absolute paths into `cfg["paths"]`, so `None` is safe (see
`src/config.py::resolve_path`, which falls back to `"."` when
`_config_dir` is absent — never exercised here because every path this
plan writes is already absolute).

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper/ app/tools/phase_mapper_pipeline_runner.py tests/phase_mapper/
git commit -m "feat: port phase-mapper pipeline core from reference app, unmodified"
```

---

### Task 4: Upload persistence service

**Files:**
- Create: `app/services/phase_mapper_uploads.py`
- Test: `tests/phase_mapper/test_phase_mapper_uploads.py`

**Interfaces:**
- Consumes: `app.tools.phase_mapper_paths.library_uploads_dir`, `library_inputs_dir` (Task 2)
- Produces (used by Task 5):
  - `async def save_composition_csv(user_id: str, slug: str, upload: UploadFile) -> str` — returns absolute saved path; raises `ValueError` if the CSV has no `well` column
  - `async def save_plate_reader_file(user_id: str, slug: str, upload: UploadFile) -> str`
  - `async def save_raw_xrd_file(user_id: str, slug: str, upload: UploadFile) -> str`
  - `async def save_reviewed_xrd_labels(user_id: str, slug: str, upload: UploadFile) -> str`
  - `validate_composition_csv(path: str) -> list[str]` — returns a list of warning strings (never raises; matches the reference app's warn-vs-fail semantics for coverage/fraction-sum issues)

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for phase-mapper upload persistence and validation."""
import io

import pytest
from fastapi import UploadFile

from app.services import phase_mapper_uploads as uploads


def _upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content))


@pytest.fixture(autouse=True)
def _isolate_user_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("app.tools.phase_mapper_paths.get_user_data_dir", lambda: str(tmp_path))


@pytest.mark.asyncio
async def test_save_composition_csv_round_trips(tmp_path):
    csv_bytes = b"well,row,col,Cs_fraction,Rb_fraction,FA_fraction,DMA_fraction,I_fraction,Br_fraction,Cl_fraction\nA1,A,1,0,0,1,0,1,0,0\n"
    path = await uploads.save_composition_csv("user-1", "lib-a", _upload("composition.csv", csv_bytes))
    assert path.endswith("composition.csv")
    assert "user-1" in path and "lib-a" in path


@pytest.mark.asyncio
async def test_save_composition_csv_rejects_missing_well_column(tmp_path):
    csv_bytes = b"row,col,Cs_fraction\nA,1,0\n"
    with pytest.raises(ValueError, match="well"):
        await uploads.save_composition_csv("user-1", "lib-a", _upload("bad.csv", csv_bytes))


def test_validate_composition_csv_warns_without_failing(tmp_path):
    path = tmp_path / "composition.csv"
    path.write_text("well,row,col,Cs_fraction,Rb_fraction,FA_fraction,DMA_fraction,I_fraction,Br_fraction,Cl_fraction\nA1,A,1,0.5,0,0.2,0,1,0,0\n")
    warnings = uploads.validate_composition_csv(str(path))
    assert any("Cs+Rb+FA+DMA" in w for w in warnings)


@pytest.mark.asyncio
async def test_save_plate_reader_file(tmp_path):
    path = await uploads.save_plate_reader_file("user-1", "lib-a", _upload("plate.xlsx", b"binary-stub"))
    assert path.endswith("plate.xlsx")
    assert "plate_reader" in path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_uploads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.phase_mapper_uploads'`

- [ ] **Step 3: Write the implementation**

```python
"""Persist and validate phase-mapper library uploads.

Mirrors app/services/curve_fitting_uploads.py's save-to-disk pattern, plus
the reference app's warn-vs-fail validation semantics: fail only on a
missing `well` column, warn (don't block) on incomplete well coverage or
composition fractions that don't sum to ~1.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from fastapi import UploadFile

from app.tools.phase_mapper_paths import library_inputs_dir, library_uploads_dir

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
_A_SITE_COLS = ["Cs_fraction", "Rb_fraction", "FA_fraction", "DMA_fraction"]
_X_SITE_COLS = ["I_fraction", "Br_fraction", "Cl_fraction"]
_EXPECTED_WELLS = 96


def _safe_filename(name: str | None, fallback: str) -> str:
    raw = (name or fallback).strip() or fallback
    base = Path(raw).name
    cleaned = _SAFE_NAME.sub("_", base)
    return cleaned[:200] or fallback


async def _save(user_id: str, slug: str, upload: UploadFile, dest_dir: Path, fallback: str) -> str:
    filename = _safe_filename(upload.filename, fallback)
    dest = dest_dir / filename
    content = await upload.read()
    dest.write_bytes(content)
    await upload.close()
    return str(dest.resolve())


async def save_composition_csv(user_id: str, slug: str, upload: UploadFile) -> str:
    # Composition map is a singleton per library (unlike plate-reader/XRD
    # uploads, which accumulate) — always written to a fixed filename so
    # start_run (Task 6) can find it without discovering the upload's
    # original name.
    dest_dir = library_inputs_dir(user_id, slug)
    dest = dest_dir / "composition.csv"
    content = await upload.read()
    dest.write_bytes(content)
    await upload.close()
    path = str(dest.resolve())
    df = pd.read_csv(path, nrows=1)
    if "well" not in df.columns:
        Path(path).unlink(missing_ok=True)
        raise ValueError("Composition CSV is missing the required 'well' column.")
    return path


def validate_composition_csv(path: str) -> list[str]:
    warnings: list[str] = []
    df = pd.read_csv(path)
    if len(df) != _EXPECTED_WELLS:
        warnings.append(f"Expected {_EXPECTED_WELLS} wells, found {len(df)}.")
    a_site_present = [c for c in _A_SITE_COLS if c in df.columns]
    if a_site_present:
        totals = df[a_site_present].sum(axis=1)
        if not totals.between(0.98, 1.02).all():
            warnings.append("Cs+Rb+FA+DMA does not sum to ~1 for one or more wells.")
    x_site_present = [c for c in _X_SITE_COLS if c in df.columns]
    if x_site_present:
        totals = df[x_site_present].sum(axis=1)
        if not totals.between(0.98, 1.02).all():
            warnings.append("I+Br+Cl does not sum to ~1 for one or more wells.")
    return warnings


async def save_plate_reader_file(user_id: str, slug: str, upload: UploadFile) -> str:
    dest_dir = library_uploads_dir(user_id, slug, "plate_reader")
    return await _save(user_id, slug, upload, dest_dir, "plate_reader.xlsx")


async def save_raw_xrd_file(user_id: str, slug: str, upload: UploadFile) -> str:
    dest_dir = library_uploads_dir(user_id, slug, "raw_xrd")
    return await _save(user_id, slug, upload, dest_dir, "raw_xrd.csv")


async def save_reviewed_xrd_labels(user_id: str, slug: str, upload: UploadFile) -> str:
    dest_dir = library_uploads_dir(user_id, slug, "xrd_labels_reviewed")
    return await _save(user_id, slug, upload, dest_dir, "xrd_labels_reviewed.csv")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_uploads.py -v`
Expected: PASS (4 tests). If `pytest-asyncio` complains about missing
`asyncio_mode`, check `pyproject.toml`/`pytest.ini` for the project's existing
setting (`dev` extras already include `pytest-asyncio>=0.23.0`) — do not add
a new config block if one already exists; match it.

- [ ] **Step 5: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_uploads.py tests/phase_mapper/test_phase_mapper_uploads.py
git commit -m "feat: add phase-mapper upload persistence and composition validation"
```

---

### Task 5: Library CRUD service and API routes

**Files:**
- Create: `app/services/phase_mapper_service.py`
- Create: `app/api/v1/phase_mapper.py`
- Modify: `app/api/v1/router.py` (add `phase_mapper` import and `include_router` call)
- Test: `tests/phase_mapper/test_phase_mapper_api.py`

**Interfaces:**
- Consumes: `DatabaseManager` (Task 1), `phase_mapper_paths` (Task 2), `phase_mapper_uploads` (Task 4)
- Produces (used by Task 6):
  - `PhaseMapperService.create_library(user_id: str, name: str, composition_profile: str = "") -> dict`
  - `PhaseMapperService.list_libraries(user_id: str) -> list[dict]`
  - `PhaseMapperService.get_library_owned(library_id: int, user_id: str) -> dict` — raises `LookupError` if missing or not owned by `user_id`
  - `PhaseMapperService.delete_library(library_id: int, user_id: str) -> None`
  - Routes: `POST /phase-mapper/libraries`, `GET /phase-mapper/libraries`, `GET /phase-mapper/libraries/{library_id}`, `DELETE /phase-mapper/libraries/{library_id}`, `POST /phase-mapper/libraries/{library_id}/uploads/{kind}`

- [ ] **Step 1: Write the failing test**

```python
"""API tests for phase-mapper library CRUD and uploads."""
import io
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
    svc.save_upload = MagicMock(return_value={"path": "/data/user-1/lib-a/inputs/composition.csv", "warnings": []})
    app, client = _client(svc)
    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.post(
            "/phase-mapper/libraries/1/uploads/composition",
            files={"file": ("composition.csv", io.BytesIO(b"well\nA1\n"), "text/csv")},
        )
    assert resp.status_code == 200
    assert resp.json()["warnings"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api.v1.phase_mapper'`

- [ ] **Step 3: Write the service**

```python
"""Library CRUD and upload orchestration for the phase-mapper domain."""
from __future__ import annotations

from typing import Any

from fastapi import UploadFile

from app.services import phase_mapper_uploads as uploads
from app.tools.database import DatabaseManager
from app.tools.phase_mapper_paths import slugify

_UPLOAD_KIND_MAP = {
    "composition": uploads.save_composition_csv,
    "plate_reader": uploads.save_plate_reader_file,
    "raw_xrd": uploads.save_raw_xrd_file,
    "xrd_labels_reviewed": uploads.save_reviewed_xrd_labels,
}


class PhaseMapperService:
    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or DatabaseManager()

    def create_library(self, user_id: str, name: str, composition_profile: str = "") -> dict[str, Any]:
        slug = slugify(name)
        library_id = self.db.create_phase_mapper_library(user_id, name, slug, composition_profile)
        return self.db.get_phase_mapper_library(library_id)

    def list_libraries(self, user_id: str) -> list[dict[str, Any]]:
        return self.db.list_phase_mapper_libraries(user_id)

    def get_library_owned(self, library_id: int, user_id: str) -> dict[str, Any]:
        library = self.db.get_phase_mapper_library(library_id)
        if library is None or library["user_id"] != user_id:
            raise LookupError(f"Library {library_id} not found for this user")
        return library

    def delete_library(self, library_id: int, user_id: str) -> None:
        self.get_library_owned(library_id, user_id)
        self.db.delete_phase_mapper_library(library_id)

    async def save_upload(self, library_id: int, user_id: str, kind: str, upload: UploadFile) -> dict[str, Any]:
        library = self.get_library_owned(library_id, user_id)
        if kind not in _UPLOAD_KIND_MAP:
            raise ValueError(f"Unknown upload kind: {kind}")
        path = await _UPLOAD_KIND_MAP[kind](user_id, library["slug"], upload)
        warnings = uploads.validate_composition_csv(path) if kind == "composition" else []
        return {"path": path, "warnings": warnings}
```

- [ ] **Step 4: Write the routes**

```python
"""FastAPI router — Phase Mapper library CRUD and uploads."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.services.phase_mapper_service import PhaseMapperService

router = APIRouter()


def get_service() -> PhaseMapperService:
    return PhaseMapperService()


class CreateLibraryRequest(BaseModel):
    name: str
    composition_profile: str = ""


class LibraryResponse(BaseModel):
    id: int
    user_id: str
    name: str
    slug: str
    composition_profile: str
    created_at: str
    updated_at: str


class UploadResponse(BaseModel):
    path: str
    warnings: list[str]


UploadKind = Literal["composition", "plate_reader", "raw_xrd", "xrd_labels_reviewed"]


@router.post("/phase-mapper/libraries", response_model=LibraryResponse)
def create_library(
    body: CreateLibraryRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    return svc.create_library(user.id, body.name, body.composition_profile)


@router.get("/phase-mapper/libraries", response_model=list[LibraryResponse])
def list_libraries(
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[dict]:
    svc = get_service()
    return svc.list_libraries(user.id)


@router.get("/phase-mapper/libraries/{library_id}", response_model=LibraryResponse)
def get_library(
    library_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        return svc.get_library_owned(library_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")


@router.delete("/phase-mapper/libraries/{library_id}")
def delete_library(
    library_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        svc.delete_library(library_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")
    return {"status": "deleted"}


@router.post("/phase-mapper/libraries/{library_id}/uploads/{kind}", response_model=UploadResponse)
async def upload_file(
    library_id: int,
    kind: UploadKind,
    user: Annotated[AuthUser, Depends(get_current_user)],
    file: UploadFile = File(...),
) -> dict:
    svc = get_service()
    try:
        return await svc.save_upload(library_id, user.id, kind, file)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
```

- [ ] **Step 5: Register the router**

In `app/api/v1/router.py`, add `phase_mapper` to the import block and register it:

```python
from app.api.v1 import (
    agents,
    analysis_session,
    dashboard,
    documents,
    experiments,
    health,
    history,
    literature,
    llm,
    mcp,
    ml_session,
    phase_mapper,
    pipeline,
    session,
    settings,
    watcher,
    workflows,
)
```

And after the `literature` line at the end of the file:

```python
api_router.include_router(literature.router, tags=["literature"])
api_router.include_router(phase_mapper.router, tags=["phase-mapper"])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py app/api/v1/router.py tests/phase_mapper/test_phase_mapper_api.py
git commit -m "feat: add phase-mapper library CRUD and upload API routes"
```

---

### Task 6: Pipeline run orchestration (background job)

**Files:**
- Modify: `app/services/phase_mapper_service.py` (add run orchestration methods)
- Modify: `app/api/v1/phase_mapper.py` (add run routes)
- Test: `tests/phase_mapper/test_phase_mapper_runs_api.py`

**Interfaces:**
- Consumes: `DatabaseManager.create_phase_mapper_run`, `.update_phase_mapper_run`, `.has_active_phase_mapper_run`, `.get_phase_mapper_run` (Task 1); `phase_mapper_paths.new_run_dir` (Task 2); `app.services.phase_mapper.config.DEFAULT_CONFIG` (Task 3); `app/tools/phase_mapper_pipeline_runner.py` (Task 3)
- Produces (used by Task 7):
  - `PhaseMapperService.start_run(library_id: int, user_id: str) -> dict` — raises `PermissionError`-style `LookupError` if not owned, raises `RuntimeError` (mapped to 409) if a run is already active
  - `PhaseMapperService.get_run(run_id: int, user_id: str) -> dict` — polls PID/returncode and updates status as a side effect, mirroring `LiteratureAgentService.job_status`
  - Routes: `POST /phase-mapper/libraries/{library_id}/runs`, `GET /phase-mapper/runs/{run_id}`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_runs_api.py -v`
Expected: FAIL — routes don't exist yet (404 on all three, `svc.start_run`/`svc.get_run` don't exist as real methods for the non-mocked path)

- [ ] **Step 3: Add run orchestration to the service**

Add to `app/services/phase_mapper_service.py` (new imports at top, new methods on `PhaseMapperService`):

```python
import copy
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from app.services.phase_mapper.config import DEFAULT_CONFIG
from app.tools.phase_mapper_paths import library_inputs_dir, library_uploads_dir, new_run_dir
```

```python
    def start_run(self, library_id: int, user_id: str) -> dict[str, Any]:
        library = self.get_library_owned(library_id, user_id)
        if self.db.has_active_phase_mapper_run(library_id):
            raise RuntimeError("A run is already in progress for this library")

        run_dir = new_run_dir(user_id, library["slug"])
        run_dir.mkdir(parents=True, exist_ok=True)

        composition_path = library_inputs_dir(user_id, library["slug"]) / "composition.csv"
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["paths"]["output_dir"] = str(run_dir)
        cfg["paths"]["composition_map"] = str(composition_path)
        plate_dir = library_uploads_dir(user_id, library["slug"], "plate_reader")
        cfg["paths"]["plate_reader_files"] = [str(p) for p in sorted(plate_dir.glob("*")) if p.is_file()]

        run_id = self.db.create_phase_mapper_run(library_id, str(run_dir))
        job_dir = run_dir  # job bookkeeping lives alongside this run's outputs
        payload_path = job_dir / "job_payload.json"
        payload_path.write_text(json.dumps({
            "job_id": str(run_id), "job_dir": str(job_dir), "run_dir": str(run_dir), "config": cfg,
        }), encoding="utf-8")

        runner = Path(__file__).resolve().parents[1] / "tools" / "phase_mapper_pipeline_runner.py"
        log_path = job_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(runner), str(payload_path)],
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        self.db.update_phase_mapper_run(run_id, job_pid=process.pid)
        return self.db.get_phase_mapper_run(run_id)

    def get_run(self, run_id: int, user_id: str) -> dict[str, Any]:
        run = self.db.get_phase_mapper_run(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        self.get_library_owned(run["library_id"], user_id)  # raises LookupError if not owned

        if run["status"] == "running" and run["job_pid"] and not _pid_exists(run["job_pid"]):
            returncode_path = Path(run["dir_path"]) / f"{run_id}.returncode"
            return_code = int(returncode_path.read_text()) if returncode_path.exists() else None
            status = "succeeded" if return_code == 0 else "failed"
            self.db.update_phase_mapper_run(
                run_id, status=status, return_code=return_code,
                frozen=(status == "succeeded"), completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            run = self.db.get_phase_mapper_run(run_id)
        return run
```

And a module-level helper near the bottom of the file (outside the class), matching `literature_agent_service.py`'s `_pid_exists`:

```python
def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
```

- [ ] **Step 4: Add the run routes**

Add to `app/api/v1/phase_mapper.py`:

```python
class RunResponse(BaseModel):
    id: int
    library_id: int
    status: str
    dir_path: str
    job_pid: int | None
    return_code: int | None
    error_message: str
    frozen: bool
    created_at: str
    completed_at: str | None


@router.post("/phase-mapper/libraries/{library_id}/runs", response_model=RunResponse)
def start_run(
    library_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        return svc.start_run(library_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Library not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/phase-mapper/runs/{run_id}", response_model=RunResponse)
def get_run(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    svc = get_service()
    try:
        return svc.get_run(run_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_runs_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_runs_api.py
git commit -m "feat: add phase-mapper background pipeline run orchestration"
```

---

### Task 7: Reads tab endpoint

**Files:**
- Modify: `app/services/phase_mapper_service.py` (add `get_reads`)
- Modify: `app/api/v1/phase_mapper.py` (add reads route)
- Test: `tests/phase_mapper/test_phase_mapper_reads_api.py`

**Interfaces:**
- Consumes: `PhaseMapperService.get_run` pattern (Task 6) for run lookup/ownership
- Produces (pattern to be repeated by the follow-up plan for the other nine tabs):
  - `PhaseMapperService.get_reads(run_id: int, user_id: str) -> list[dict]` — reads `parsed_reads_summary.csv` from the run's `dir_path` and returns records
  - Route: `GET /phase-mapper/runs/{run_id}/reads`

- [ ] **Step 1: Write the failing test**

```python
"""API test for the phase-mapper Reads tab endpoint."""
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
    m.get_reads.return_value = [
        {"read_num": 1, "read_label": "EM Spectrum", "read_name": "Abs_initial", "read_type": "absorbance", "n_wells_detected": 96},
    ]
    return m


def test_get_reads(svc):
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/10/reads")
    assert resp.status_code == 200
    assert resp.json()[0]["read_name"] == "Abs_initial"


def test_get_reads_not_found(svc):
    svc.get_reads.side_effect = LookupError("not found")
    app = FastAPI()
    app.include_router(router)
    import app.core.deps as deps_module
    app.dependency_overrides[deps_module.get_current_user] = _mock_user
    client = TestClient(app)

    import app.api.v1.phase_mapper as pm_module
    with patch.object(pm_module, "get_service", return_value=svc):
        resp = client.get("/phase-mapper/runs/999/reads")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_reads_api.py -v`
Expected: FAIL — route doesn't exist (404 with default FastAPI body, not the mocked payload)

- [ ] **Step 3: Add the service method**

Add to `app/services/phase_mapper_service.py`, plus `import pandas as pd` at the top:

```python
    def get_reads(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self.db.get_phase_mapper_run(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        self.get_library_owned(run["library_id"], user_id)

        csv_path = Path(run["dir_path"]) / "parsed_reads_summary.csv"
        if not csv_path.exists():
            return []
        df = pd.read_csv(csv_path)
        return json.loads(df.to_json(orient="records"))
```

- [ ] **Step 4: Add the route**

Add to `app/api/v1/phase_mapper.py`:

```python
@router.get("/phase-mapper/runs/{run_id}/reads")
def get_reads(
    run_id: int,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[dict]:
    svc = get_service()
    try:
        return svc.get_reads(run_id, user.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_reads_api.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
cd backend-api
git add app/services/phase_mapper_service.py app/api/v1/phase_mapper.py tests/phase_mapper/test_phase_mapper_reads_api.py
git commit -m "feat: add phase-mapper Reads tab endpoint"
```

---

### Task 8: OpenAPI contract

**Files:**
- Modify: `shared-types/openapi/polaris-v1.yaml`
- Modify (regenerated, not hand-edited): `shared-types/src/generated/api.ts`

**Interfaces:**
- Consumes: the exact request/response shapes from Tasks 5, 6, 7 (`LibraryResponse`, `UploadResponse`, `RunResponse`, reads record shape)
- Produces: typed client methods used by Plan 2 (web) and Plan 3 (mobile)

- [ ] **Step 1: Inspect the existing contract's structure**

Run: `head -n 60 shared-types/openapi/polaris-v1.yaml` from the `shared-types` repo root, and find one existing path entry (e.g. the literature endpoints) to match its exact `paths:`/`components.schemas:` formatting before adding new entries — this plan does not reproduce the whole file because the exact surrounding YAML (indentation, existing schema names) must be read first to avoid breaking it.

- [ ] **Step 2: Add the new paths and schemas**

Add under `paths:`, matching whatever indentation/style Step 1 found:

```yaml
  /phase-mapper/libraries:
    post:
      operationId: createPhaseMapperLibrary
      tags: [phase-mapper]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [name]
              properties:
                name: { type: string }
                composition_profile: { type: string }
      responses:
        '200':
          description: Created library
          content:
            application/json:
              schema: { $ref: '#/components/schemas/PhaseMapperLibrary' }
    get:
      operationId: listPhaseMapperLibraries
      tags: [phase-mapper]
      responses:
        '200':
          description: Libraries for the current user
          content:
            application/json:
              schema:
                type: array
                items: { $ref: '#/components/schemas/PhaseMapperLibrary' }
  /phase-mapper/libraries/{library_id}:
    get:
      operationId: getPhaseMapperLibrary
      tags: [phase-mapper]
      parameters:
        - { name: library_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: A library
          content:
            application/json:
              schema: { $ref: '#/components/schemas/PhaseMapperLibrary' }
        '404': { description: Library not found }
    delete:
      operationId: deletePhaseMapperLibrary
      tags: [phase-mapper]
      parameters:
        - { name: library_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200': { description: Deleted }
        '404': { description: Library not found }
  /phase-mapper/libraries/{library_id}/uploads/{kind}:
    post:
      operationId: uploadPhaseMapperFile
      tags: [phase-mapper]
      parameters:
        - { name: library_id, in: path, required: true, schema: { type: integer } }
        - name: kind
          in: path
          required: true
          schema: { type: string, enum: [composition, plate_reader, raw_xrd, xrd_labels_reviewed] }
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file]
              properties:
                file: { type: string, format: binary }
      responses:
        '200':
          description: Upload accepted
          content:
            application/json:
              schema: { $ref: '#/components/schemas/PhaseMapperUploadResult' }
        '404': { description: Library not found }
        '400': { description: Invalid file }
  /phase-mapper/libraries/{library_id}/runs:
    post:
      operationId: startPhaseMapperRun
      tags: [phase-mapper]
      parameters:
        - { name: library_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: Run started
          content:
            application/json:
              schema: { $ref: '#/components/schemas/PhaseMapperRun' }
        '404': { description: Library not found }
        '409': { description: A run is already in progress for this library }
  /phase-mapper/runs/{run_id}:
    get:
      operationId: getPhaseMapperRun
      tags: [phase-mapper]
      parameters:
        - { name: run_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: Run status
          content:
            application/json:
              schema: { $ref: '#/components/schemas/PhaseMapperRun' }
        '404': { description: Run not found }
  /phase-mapper/runs/{run_id}/reads:
    get:
      operationId: getPhaseMapperReads
      tags: [phase-mapper]
      parameters:
        - { name: run_id, in: path, required: true, schema: { type: integer } }
      responses:
        '200':
          description: Parsed plate-reader read assignments for this run
          content:
            application/json:
              schema:
                type: array
                items:
                  type: object
        '404': { description: Run not found }
```

Add under `components.schemas:`:

```yaml
    PhaseMapperLibrary:
      type: object
      required: [id, user_id, name, slug, composition_profile, created_at, updated_at]
      properties:
        id: { type: integer }
        user_id: { type: string }
        name: { type: string }
        slug: { type: string }
        composition_profile: { type: string }
        created_at: { type: string }
        updated_at: { type: string }
    PhaseMapperUploadResult:
      type: object
      required: [path, warnings]
      properties:
        path: { type: string }
        warnings: { type: array, items: { type: string } }
    PhaseMapperRun:
      type: object
      required: [id, library_id, status, dir_path, frozen, created_at]
      properties:
        id: { type: integer }
        library_id: { type: integer }
        status: { type: string, enum: [running, succeeded, failed] }
        dir_path: { type: string }
        job_pid: { type: integer, nullable: true }
        return_code: { type: integer, nullable: true }
        error_message: { type: string }
        frozen: { type: boolean }
        created_at: { type: string }
        completed_at: { type: string, nullable: true }
```

- [ ] **Step 3: Validate the YAML**

Run (from `shared-types`): `npx swagger-cli validate openapi/polaris-v1.yaml` (or whatever validator script `package.json` already defines — check its `scripts` block first; use the existing one rather than introducing a new tool).

Expected: no errors.

- [ ] **Step 4: Regenerate the TypeScript client**

Run (from `shared-types`): whatever `npm run` script currently produces
`src/generated/api.ts` — check `package.json`'s `scripts` block for its exact
name (do not hand-edit the generated file).

Expected: `src/generated/api.ts` now contains `PhaseMapperLibrary`,
`PhaseMapperRun`, `PhaseMapperUploadResult` types and the six new operations.

- [ ] **Step 5: Commit**

```bash
cd shared-types
git add openapi/polaris-v1.yaml src/generated/api.ts
git commit -m "feat: add phase-mapper library/run/reads endpoints to API contract"
```

---

### Task 9: End-to-end integration test

**Files:**
- Test: `tests/phase_mapper/test_phase_mapper_end_to_end.py`
- Create fixture: `tests/phase_mapper/fixtures/composition_map_fixture.csv`, `tests/phase_mapper/fixtures/plate_reader_fixture.csv`

**Interfaces:**
- Consumes: the full stack from Tasks 1–7, exercised through the real (non-mocked) `PhaseMapperService` and a temp SQLite DB — the one place in this plan where the pieces are proven to fit together, not just unit-tested in isolation.

- [ ] **Step 1: Write the fixture files**

`tests/phase_mapper/fixtures/composition_map_fixture.csv` — a 2-well minimal composition map (real pipeline code tolerates fewer than 96 wells; it only warns):

```csv
well,row,col,Cs_fraction,Rb_fraction,FA_fraction,DMA_fraction,I_fraction,Br_fraction,Cl_fraction,is_control,replicate_group,notes
A1,A,1,0.0,0.0,1.0,0.0,1.0,0.0,0.0,,,
A2,A,2,0.0,1.0,0.0,0.0,1.0,0.0,0.0,,,
```

`tests/phase_mapper/fixtures/plate_reader_fixture.csv` — matches the block
format `plate_reader.py` parses (`Read 1:` header, `Wavelength` column, well
columns):

```csv
Read 1:Spectrum
Wavelength,A1,A2
500,1.0,1.1
501,1.05,1.12
502,1.02,1.08
```

- [ ] **Step 2: Write the failing test**

```python
"""End-to-end test: create a library, upload fixtures, run the pipeline as a
real subprocess, poll to completion, and fetch the Reads tab.
"""
import time
from pathlib import Path

import pytest

from app.services.phase_mapper_service import PhaseMapperService
from app.tools.database import DatabaseManager

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from app.db import engine as engine_module
    engine_module._engine = None
    monkeypatch.setattr("app.tools.phase_mapper_paths.get_user_data_dir", lambda: str(tmp_path))
    db = DatabaseManager()
    db.init_schema()
    yield PhaseMapperService(db)
    engine_module._engine = None


@pytest.mark.slow
def test_full_library_pipeline_round_trip(service):
    library = service.create_library("user-1", "Fixture Library")

    import io
    from fastapi import UploadFile

    async def _upload(path: Path, save_fn):
        upload = UploadFile(filename=path.name, file=io.BytesIO(path.read_bytes()))
        await save_fn(library["id"], "user-1", "composition" if "composition" in path.name else "plate_reader", upload)

    import asyncio
    asyncio.run(_upload(_FIXTURES / "composition_map_fixture.csv", service.save_upload))
    asyncio.run(_upload(_FIXTURES / "plate_reader_fixture.csv", service.save_upload))

    run = service.start_run(library["id"], "user-1")
    assert run["status"] == "running"

    deadline = time.time() + 60
    while time.time() < deadline:
        run = service.get_run(run["id"], "user-1")
        if run["status"] != "running":
            break
        time.sleep(1)

    assert run["status"] == "succeeded", run["error_message"]

    reads = service.get_reads(run["id"], "user-1")
    assert len(reads) >= 1
    assert reads[0]["read_type"] == "absorbance"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_end_to_end.py -v -m slow`
Expected: FAIL at whatever the first missing piece is — by this point in the
plan it should fail only if a wiring detail (e.g. an actual column-name
mismatch between the fixture and what `plate_reader.py` expects) surfaces,
since all the individual pieces already have passing unit tests.

- [ ] **Step 4: Fix any wiring issues surfaced by the real subprocess run**

If the pipeline fails, read `run["error_message"]` — Task 6's job runner
writes the full traceback to `error.log` in the run directory, and Task 6's
`get_run` doesn't currently load that file into `error_message` (only
`return_code`/`status` are updated from the returncode file). Add that read
now, in `PhaseMapperService.get_run` (Task 6), immediately after computing
`status`:

```python
        if run["status"] == "running" and run["job_pid"] and not _pid_exists(run["job_pid"]):
            returncode_path = Path(run["dir_path"]) / f"{run_id}.returncode"
            return_code = int(returncode_path.read_text()) if returncode_path.exists() else None
            status = "succeeded" if return_code == 0 else "failed"
            error_log = Path(run["dir_path"]) / "error.log"
            error_message = error_log.read_text(encoding="utf-8") if error_log.exists() else ""
            self.db.update_phase_mapper_run(
                run_id, status=status, return_code=return_code, error_message=error_message,
                frozen=(status == "succeeded"), completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            run = self.db.get_phase_mapper_run(run_id)
```

This step exists because it's the one place in the plan where the
walking-skeleton nature of this increment shows: `error_message` was defined
in Task 1's schema and Task 6's `RunResponse`, but nothing populated it until
a real failure surfaced the gap.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend-api && pytest tests/phase_mapper/test_phase_mapper_end_to_end.py -v -m slow`
Expected: PASS (may take up to ~60s — it runs the real pipeline as a subprocess)

- [ ] **Step 6: Run the full phase-mapper test suite**

Run: `cd backend-api && pytest tests/phase_mapper/ -v`
Expected: PASS (all files except the still-skipped `test_spin_coating_optimization.py`)

- [ ] **Step 7: Commit**

```bash
cd backend-api
git add tests/phase_mapper/test_phase_mapper_end_to_end.py tests/phase_mapper/fixtures/ app/services/phase_mapper_service.py
git commit -m "test: add phase-mapper end-to-end pipeline integration test"
```

---

## After this plan

The remaining nine result tabs (Heatmaps, Predictions, XRD Recommendations,
Remeasurement, Optical Spectra, XRD Spectra, XRD Peak Fits, Library Runs,
Downloads) plus XRD label review each follow exactly the pattern in Tasks 7
and 8: one `PhaseMapperService` method reading a specific CSV/artifact out
of `run["dir_path"]`, one route, one contract entry, one test. That becomes
its own plan once this skeleton is reviewed and merged, so review effort here
stays focused on the architecture rather than nine repetitions of the same
shape.
