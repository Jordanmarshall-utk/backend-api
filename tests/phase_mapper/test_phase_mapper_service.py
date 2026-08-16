"""Unit tests for PhaseMapperService, exercised against a real DatabaseManager."""
import io
import json
import os
import zipfile

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
    # row/col/well are identifiers, not metrics: present on each well record
    # but must never appear in the `columns` list the frontend iterates over.
    assert "well" not in result["columns"]
    assert "row" not in result["columns"]
    assert "col" not in result["columns"]
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


def test_get_heatmaps_excludes_non_numeric_column_from_columns_but_keeps_value(service, db, tmp_path):
    # A text column (not one of the hardcoded identifier names) must survive
    # the merge on each well record, but must be excluded from `columns`
    # because it fails the numeric-dtype check, not because it's on the
    # hardcoded non_metric_cols list.
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "composition_descriptors_per_well.csv").write_text(
        "well,row,col,Cs_fraction\nA1,A,1,0.5\n", encoding="utf-8"
    )
    (run_dir / "model_predictions_all_wells.csv").write_text(
        "well,phase_label,pred_prob\nA1,perovskite,0.84\n", encoding="utf-8"
    )

    result = service.get_heatmaps(run_id, "user-A")

    assert "phase_label" not in result["columns"]
    assert "pred_prob" in result["columns"]
    a1 = next(w for w in result["wells"] if w["well"] == "A1")
    assert a1["phase_label"] == "perovskite"


def test_get_heatmaps_renames_colliding_columns_instead_of_dropping_data(service, db, tmp_path):
    # xrd_recommendations_next_wells.csv and optical_remeasurement_recommendations.csv
    # both define a "priority" column with unrelated meanings/values. The old
    # merge silently dropped the second file's column entirely; the fix must
    # keep both under distinct, non-colliding keys.
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "composition_descriptors_per_well.csv").write_text(
        "well,row,col,Cs_fraction\nA1,A,1,0.5\nA2,A,2,0.7\n", encoding="utf-8"
    )
    (run_dir / "xrd_recommendations_next_wells.csv").write_text(
        "well,priority\nA1,0.9\nA2,0.3\n", encoding="utf-8"
    )
    (run_dir / "optical_remeasurement_recommendations.csv").write_text(
        "well,priority\nA1,0.1\nA2,0.8\n", encoding="utf-8"
    )

    result = service.get_heatmaps(run_id, "user-A")

    assert "priority" in result["columns"]
    assert "priority_optical_remeasurement" in result["columns"]

    a1 = next(w for w in result["wells"] if w["well"] == "A1")
    a2 = next(w for w in result["wells"] if w["well"] == "A2")
    # Values from the earlier-merged file (xrd) keep the clean, unprefixed name.
    assert a1["priority"] == 0.9
    assert a2["priority"] == 0.3
    # The later, colliding file's values survive under the disambiguated name
    # rather than being silently overwritten or dropped.
    assert a1["priority_optical_remeasurement"] == 0.1
    assert a2["priority_optical_remeasurement"] == 0.8


def test_get_heatmaps_uses_predictions_normalized_value_not_raw_optical_value(service, db, tmp_path):
    # Regression test for the final-review finding: model_predictions_all_wells.csv
    # is written by the real pipeline as a *superset* of
    # composition_descriptors_per_well.csv and optical_features_per_well.csv,
    # but it also overwrites optical_heterogeneity_score_total,
    # pl_abs_disagreement_score, and optical_phase_complexity_score with
    # compute_xrd_priority-normalized (0-1 minmax) values -- different from
    # the raw values under the same column names in
    # optical_features_per_well.csv. GET /predictions reads
    # model_predictions_all_wells.csv directly, so /heatmaps must agree with
    # it on these keys rather than serving the stale raw optical values.
    #
    # This fixture is intentionally wider than a toy 2-4 column CSV: it
    # mirrors the real pipeline's column shape (composition descriptors, row/
    # col, optical features, prediction/priority columns) so the merge logic
    # is exercised the way it actually behaves against real pipeline output.
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "composition_descriptors_per_well.csv").write_text(
        "well,row,col,library,Cs_fraction,Rb_fraction,FA_fraction,A_site_entropy\n"
        "A1,A,1,Lib A,0.0,0.0,1.0,0.41\n"
        "A2,A,2,Lib A,0.0,1.0,0.0,0.55\n",
        encoding="utf-8",
    )
    (run_dir / "optical_features_per_well.csv").write_text(
        "well,library,pl_peak_nm,optical_heterogeneity_score_total,pl_abs_disagreement_score,"
        "optical_phase_complexity_score,measurement_QC_flag_count\n"
        "A1,Lib A,512.3,14.7,9.2,3.6,2\n"
        "A2,Lib A,498.1,22.1,4.4,1.1,0\n",
        encoding="utf-8",
    )
    (run_dir / "model_predictions_all_wells.csv").write_text(
        "well,row,col,library,Cs_fraction,Rb_fraction,FA_fraction,A_site_entropy,pl_peak_nm,"
        "optical_heterogeneity_score_total,pl_abs_disagreement_score,optical_phase_complexity_score,"
        "measurement_QC_flag_count,xrd_priority_score,model_uncertainty_score\n"
        "A1,A,1,Lib A,0.0,0.0,1.0,0.41,512.3,0.0,1.0,1.0,2,0.72,0.5\n"
        "A2,A,2,Lib A,0.0,1.0,0.0,0.55,498.1,1.0,0.0,0.0,0,0.31,0.2\n",
        encoding="utf-8",
    )
    (run_dir / "xrd_recommendations_next_wells.csv").write_text(
        "well,row,col,Cs_fraction,xrd_priority_score,reason_for_recommendation\n"
        "A1,A,1,0.0,0.72,high model uncertainty\n",
        encoding="utf-8",
    )
    (run_dir / "optical_remeasurement_recommendations.csv").write_text(
        "well,row,col,optical_remeasurement_priority_score,missing_spectra_score,"
        "measurement_QC_flag_count,reason_for_remeasurement\n"
        "A1,A,1,0.8,0.5,2,missing spectral features\n"
        "A2,A,2,0.1,0.0,0,no major optical QC issue\n",
        encoding="utf-8",
    )

    result = service.get_heatmaps(run_id, "user-A")

    a1 = next(w for w in result["wells"] if w["well"] == "A1")
    a2 = next(w for w in result["wells"] if w["well"] == "A2")

    # The clean, unprefixed key must carry the NORMALIZED (predictions) value,
    # not the raw optical value, for all three colliding metrics.
    assert a1["optical_heterogeneity_score_total"] == 0.0
    assert a2["optical_heterogeneity_score_total"] == 1.0
    assert a1["pl_abs_disagreement_score"] == 1.0
    assert a2["pl_abs_disagreement_score"] == 0.0
    assert a1["optical_phase_complexity_score"] == 1.0
    assert a2["optical_phase_complexity_score"] == 0.0

    # composition_descriptors_per_well.csv / optical_features_per_well.csv
    # must not be merged at all when model_predictions_all_wells.csv (a
    # confirmed superset) is present -- so no renamed duplicate of these
    # metrics should exist under any tag.
    for col in result["columns"]:
        assert not col.startswith("optical_heterogeneity_score_total_")
        assert not col.startswith("pl_abs_disagreement_score_")
        assert not col.startswith("optical_phase_complexity_score_")
        assert not col.startswith("measurement_QC_flag_count_")
        assert not col.startswith("Cs_fraction_")

    # Genuinely additional columns from the supplemental files still show up.
    assert "reason_for_recommendation" in result["wells"][0] or "reason_for_recommendation" in a1
    assert "optical_remeasurement_priority_score" in result["columns"]
    assert "missing_spectra_score" in result["columns"]

    # No positional/identity pseudo-metric ("col_<tag>", "row_<tag>", ...)
    # leaked into the metric-picker `columns` list.
    for col in result["columns"]:
        assert col not in ("col", "row", "well", "library")
        assert not col.startswith("col_")
        assert not col.startswith("row_")
        assert not col.startswith("well_")

    # Sanity: real metrics from the base and from the supplemental files are
    # still present as selectable columns.
    assert "xrd_priority_score" in result["columns"]
    assert "A_site_entropy" in result["columns"]


def test_get_heatmaps_falls_back_to_composition_and_optical_when_predictions_missing(service, db, tmp_path):
    # If model_predictions_all_wells.csv is absent (e.g. an older or partial
    # run), /heatmaps should still degrade to the older, narrower merge
    # instead of returning nothing.
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "composition_descriptors_per_well.csv").write_text(
        "well,row,col,Cs_fraction\nA1,A,1,0.5\nA2,A,2,0.7\n", encoding="utf-8"
    )
    (run_dir / "optical_features_per_well.csv").write_text(
        "well,optical_heterogeneity_score_total\nA1,14.7\nA2,22.1\n", encoding="utf-8"
    )

    result = service.get_heatmaps(run_id, "user-A")

    assert "Cs_fraction" in result["columns"]
    assert "optical_heterogeneity_score_total" in result["columns"]
    a1 = next(w for w in result["wells"] if w["well"] == "A1")
    assert a1["optical_heterogeneity_score_total"] == 14.7


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


def test_list_runs_reconciles_status_from_returncode_file_without_calling_get_run(service, db, tmp_path):
    # get_run reconciles a "running" DB row against a *.returncode file
    # dropped by the subprocess, but list_runs previously just returned raw
    # DB rows -- so a completed run could show "running" forever in a
    # library's run history until someone individually polled GET /runs/{id}.
    # This proves list_runs performs the same reconciliation itself.
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")
    run_dir = tmp_path / "run-dir"
    run_dir.mkdir()
    run_id = db.create_phase_mapper_run(library_id, str(run_dir))
    db.update_phase_mapper_run(run_id, job_pid=os.getpid())
    assert db.get_phase_mapper_run(run_id)["status"] == "running"

    # Write the returncode file directly -- not via get_run -- to prove
    # list_runs itself does the reconciliation.
    (run_dir / f"{run_id}.returncode").write_text("0", encoding="utf-8")

    runs = service.list_runs(library_id, "user-A")

    assert len(runs) == 1
    assert runs[0]["status"] == "succeeded"
    # And the DB row itself was updated, not just the returned dict.
    assert db.get_phase_mapper_run(run_id)["status"] == "succeeded"


def test_list_runs_does_not_reconcile_already_terminal_rows(service, db, tmp_path, monkeypatch):
    # A run already marked succeeded/failed shouldn't be re-checked -- no
    # need to touch the filesystem for rows that are already terminal.
    library_id = db.create_phase_mapper_library("user-A", "Lib A", "lib-a")
    run_dir = tmp_path / "run-dir"
    run_dir.mkdir()
    run_id = db.create_phase_mapper_run(library_id, str(run_dir))
    db.update_phase_mapper_run(run_id, status="succeeded")

    def _boom(self, run):
        raise AssertionError("_reconcile_run_status must not be called for an already-terminal row")

    monkeypatch.setattr(PhaseMapperService, "_reconcile_run_status", _boom)

    runs = service.list_runs(library_id, "user-A")

    assert runs[0]["status"] == "succeeded"


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


def test_get_run_download_bytes_excludes_job_internals(service, db, tmp_path):
    # job_payload.json contains absolute server paths and the full pipeline
    # config; run.log/error.log are raw subprocess output that can include
    # tracebacks with absolute paths; *.returncode is pure job plumbing.
    # None of these are pipeline output and none should leave the server.
    run_id, run_dir = _make_run_with_dir(db, tmp_path, user_id="user-A")
    (run_dir / "parsed_reads_summary.csv").write_text("read_num\n1\n", encoding="utf-8")
    (run_dir / "job_payload.json").write_text(
        json.dumps({"job_dir": "/Users/someone/POLARIS/data/lib-a/runs/1", "config": {"paths": {"composition_map": "/Users/someone/uploads/composition.csv"}}}),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text("Traceback (most recent call last):\n  File \"/Users/someone/app/pipeline.py\"\n", encoding="utf-8")
    (run_dir / "error.log").write_text("ValueError at /Users/someone/app/pipeline.py:42", encoding="utf-8")
    (run_dir / f"{run_id}.returncode").write_text("0", encoding="utf-8")

    zip_bytes = service.get_run_download_bytes(run_id, "user-A")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "parsed_reads_summary.csv" in names
        assert "job_payload.json" not in names
        assert "run.log" not in names
        assert "error.log" not in names
        assert f"{run_id}.returncode" not in names
        assert not any(n.endswith(".returncode") for n in names)
