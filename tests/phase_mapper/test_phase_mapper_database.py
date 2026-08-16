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
