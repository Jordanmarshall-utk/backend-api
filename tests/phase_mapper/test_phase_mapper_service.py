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


def test_get_run_populates_error_message_from_error_log_on_failure(service, db, tmp_path):
    run_id, run_dir = _make_running_run(db, tmp_path, job_pid=os.getpid())
    (run_dir / f"{run_id}.returncode").write_text("1")
    (run_dir / "error.log").write_text("Traceback (most recent call last):\nValueError: boom\n")

    run = service.get_run(run_id, "user-A")

    assert run["status"] == "failed"
    assert "ValueError: boom" in run["error_message"]


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


def test_delete_run_removes_row_and_clears_active_run(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    library_id = db.get_phase_mapper_run(run_id)["library_id"]
    assert db.has_active_phase_mapper_run(library_id) is True  # newly-created runs start "running"

    service.delete_run(run_id, "user-A")

    assert db.get_phase_mapper_run(run_id) is None
    assert db.has_active_phase_mapper_run(library_id) is False


def test_delete_run_raises_for_different_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")

    with pytest.raises(LookupError):
        service.delete_run(run_id, "user-B")


def test_delete_run_raises_for_nonexistent_run(service):
    with pytest.raises(LookupError):
        service.delete_run(999999, "user-A")


def test_two_libraries_with_same_name_get_distinct_storage_dirs(service, monkeypatch, tmp_path):
    monkeypatch.setattr("app.tools.phase_mapper_paths.get_user_data_dir", lambda: str(tmp_path))

    lib_1 = service.create_library("user-A", "DMA-Rb")
    lib_2 = service.create_library("user-A", "DMA-Rb")

    assert lib_1["id"] != lib_2["id"]
    assert lib_1["slug"] == lib_2["slug"]  # same name -> same slug, by construction

    key_1 = service._storage_key(lib_1)
    key_2 = service._storage_key(lib_2)
    assert key_1 != key_2

    from app.tools.phase_mapper_paths import library_root
    root_1 = library_root("user-A", key_1)
    root_2 = library_root("user-A", key_2)
    assert root_1 != root_2


def test_delete_library_removes_on_disk_tree(service, monkeypatch, tmp_path):
    monkeypatch.setattr("app.tools.phase_mapper_paths.get_user_data_dir", lambda: str(tmp_path))

    library = service.create_library("user-A", "Lib A")
    from app.tools.phase_mapper_paths import library_inputs_dir
    inputs_dir = library_inputs_dir("user-A", service._storage_key(library))
    marker = inputs_dir / "composition.csv"
    marker.write_text("well\nA1\n")
    assert marker.exists()

    service.delete_library(library["id"], "user-A")

    assert not inputs_dir.exists()
    assert not marker.exists()


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


def test_get_xrd_recommendations_returns_csv_records_for_owning_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "xrd_recommendations_next_wells.csv").write_text(
        "well,S_XRD,reason_for_recommendation\nA1,0.71,high model uncertainty\nA2,0.45,low confidence\n", encoding="utf-8"
    )
    records = service.get_xrd_recommendations(run_id, "user-A")
    assert records == [
        {"well": "A1", "S_XRD": 0.71, "reason_for_recommendation": "high model uncertainty"},
        {"well": "A2", "S_XRD": 0.45, "reason_for_recommendation": "low confidence"},
    ]


def test_get_xrd_recommendations_raises_for_different_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    with pytest.raises(LookupError):
        service.get_xrd_recommendations(run_id, "user-B")


def test_get_xrd_recommendations_returns_empty_list_when_csv_absent(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    assert service.get_xrd_recommendations(run_id, "user-A") == []


def test_get_remeasurement_returns_csv_records_for_owning_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "optical_remeasurement_recommendations.csv").write_text(
        "well,optical_remeasurement_priority_score\nB3,0.55\nB4,0.42\n", encoding="utf-8"
    )
    records = service.get_remeasurement(run_id, "user-A")
    assert records == [
        {"well": "B3", "optical_remeasurement_priority_score": 0.55},
        {"well": "B4", "optical_remeasurement_priority_score": 0.42},
    ]


def test_get_remeasurement_raises_for_different_user(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    with pytest.raises(LookupError):
        service.get_remeasurement(run_id, "user-B")


def test_get_remeasurement_returns_empty_list_when_csv_absent(service, db, tmp_path):
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    assert service.get_remeasurement(run_id, "user-A") == []


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
