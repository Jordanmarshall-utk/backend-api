"""Unit tests for PhaseMapperService, exercised against a real DatabaseManager."""
import os

import pytest


os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from app.services.phase_mapper_service import PhaseMapperService  # noqa: E402
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


@pytest.fixture
def service(db):
    return PhaseMapperService(db)


def test_get_library_owned_returns_library_for_owner(service, db):
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")

    library = service.get_library_owned(library_id, "user-A")

    assert library["id"] == library_id
    assert library["user_id"] == "user-A"


def test_get_library_owned_raises_for_different_user(service, db):
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")

    with pytest.raises(LookupError):
        service.get_library_owned(library_id, "user-B")


def test_get_library_owned_raises_for_nonexistent_library(service):
    with pytest.raises(LookupError):
        service.get_library_owned(999999, "user-A")


def _make_running_run(db, tmp_path, *, job_pid: int) -> tuple[int, "os.PathLike"]:
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")
    run_dir = tmp_path / "run-dir"
    run_dir.mkdir()
    run_id = db.create_phase_mapper_run(library_id, str(run_dir))
    db.update_phase_mapper_run(run_id, job_pid=job_pid)
    return run_id, run_dir


def test_get_run_transitions_to_succeeded_when_returncode_file_exists(service, db, tmp_path):
    # Use our own PID: it is guaranteed to be alive (not a "not _pid_exists" case),
    # proving completion detection no longer depends on the job PID having exited.
    run_id, run_dir = _make_running_run(db, tmp_path, job_pid=os.getpid())
    (run_dir / f"{run_id}.returncode").write_text("0")

    run = service.get_run(run_id, "user-A")

    assert run["status"] == "succeeded"
    assert run["return_code"] == 0
    assert run["frozen"]
    assert run["completed_at"] is not None


def test_get_run_transitions_to_failed_when_returncode_file_is_nonzero(service, db, tmp_path):
    run_id, run_dir = _make_running_run(db, tmp_path, job_pid=os.getpid())
    (run_dir / f"{run_id}.returncode").write_text("1")

    run = service.get_run(run_id, "user-A")

    assert run["status"] == "failed"
    assert run["return_code"] == 1
    assert not run["frozen"]


def test_get_run_stays_running_when_returncode_file_absent(service, db, tmp_path):
    # Even with a still-alive PID and no returncode file yet, status must remain "running".
    run_id, run_dir = _make_running_run(db, tmp_path, job_pid=os.getpid())

    run = service.get_run(run_id, "user-A")

    assert run["status"] == "running"


def _make_run_with_dir(db, tmp_path, *, user_id: str = "user-A") -> tuple[int, "os.PathLike"]:
    library_id = db.create_phase_mapper_library(user_id, "Lib A", "lib-a")
    run_dir = tmp_path / "run-dir"
    run_dir.mkdir()
    run_id = db.create_phase_mapper_run(library_id, str(run_dir))
    return run_id, run_dir


def test_get_reads_returns_csv_records_for_owning_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "parsed_reads_summary.csv").write_text(
        "well,intensity\nA1,123.4\nA2,567.8\n", encoding="utf-8"
    )

    reads = service.get_reads(run_id, "user-A")

    assert reads == [
        {"well": "A1", "intensity": 123.4},
        {"well": "A2", "intensity": 567.8},
    ]


def test_get_reads_raises_for_different_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "parsed_reads_summary.csv").write_text(
        "well,intensity\nA1,123.4\n", encoding="utf-8"
    )

    with pytest.raises(LookupError):
        service.get_reads(run_id, "user-B")


def test_get_reads_returns_empty_list_when_csv_absent(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")

    reads = service.get_reads(run_id, "user-A")

    assert reads == []
