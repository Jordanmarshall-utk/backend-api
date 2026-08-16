"""Library CRUD and upload orchestration for the phase-mapper domain."""
from __future__ import annotations

import copy
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fastapi import UploadFile

from app.services import phase_mapper_uploads as uploads
from app.services.phase_mapper.config import DEFAULT_CONFIG
from app.tools.database import DatabaseManager
from app.tools.phase_mapper_paths import (
    library_inputs_dir,
    library_root,
    library_uploads_dir,
    new_run_dir,
    slugify,
)

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

    def list_runs(self, library_id: int, user_id: str) -> list[dict[str, Any]]:
        self.get_library_owned(library_id, user_id)
        runs = self.db.list_phase_mapper_runs(library_id)
        return [self._reconcile_run_status(run) if run["status"] == "running" else run for run in runs]

    def get_library_owned(self, library_id: int, user_id: str) -> dict[str, Any]:
        library = self.db.get_phase_mapper_library(library_id)
        if library is None or library["user_id"] != user_id:
            raise LookupError(f"Library {library_id} not found for this user")
        return library

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

    @staticmethod
    def _storage_key(library: dict[str, Any]) -> str:
        """On-disk storage key for a library.

        Incorporates the numeric library id (not just the name-derived slug)
        so that two libraries with the same human-readable name — which slug
        to the same string — never resolve to the same directory tree on
        disk. Every call site that derives a library's on-disk location must
        go through this helper.
        """
        return f"{library['id']}-{library['slug']}"

    def delete_library(self, library_id: int, user_id: str) -> None:
        library = self.get_library_owned(library_id, user_id)
        self.db.delete_phase_mapper_library(library_id)
        root = library_root(library["user_id"], self._storage_key(library))
        shutil.rmtree(root, ignore_errors=True)

    async def save_upload(self, library_id: int, user_id: str, kind: str, upload: UploadFile) -> dict[str, Any]:
        library = self.get_library_owned(library_id, user_id)
        if kind not in _UPLOAD_KIND_MAP:
            raise ValueError(f"Unknown upload kind: {kind}")
        path = await _UPLOAD_KIND_MAP[kind](user_id, self._storage_key(library), upload)
        warnings = uploads.validate_composition_csv(path) if kind == "composition" else []
        return {"path": path, "warnings": warnings}

    def start_run(self, library_id: int, user_id: str) -> dict[str, Any]:
        library = self.get_library_owned(library_id, user_id)
        if self.db.has_active_phase_mapper_run(library_id):
            raise RuntimeError("A run is already in progress for this library")

        storage_key = self._storage_key(library)
        run_dir = new_run_dir(user_id, storage_key)
        run_dir.mkdir(parents=True, exist_ok=True)

        composition_path = library_inputs_dir(user_id, storage_key) / "composition.csv"
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["paths"]["output_dir"] = str(run_dir)
        cfg["paths"]["composition_map"] = str(composition_path)
        # NOTE: composition_profile (stored at library-creation time) and the
        # raw_xrd / xrd_labels_reviewed upload kinds are accepted through the
        # API today but not yet wired into the pipeline's config here — full
        # wiring is deferred to the follow-up plan that adds the remaining
        # result tabs. This is an intentional, documented gap.
        cfg["experiment"] = {"library_name": library["name"]}
        plate_dir = library_uploads_dir(user_id, storage_key, "plate_reader")
        cfg["paths"]["plate_reader_files"] = [str(p) for p in sorted(plate_dir.glob("*")) if p.is_file()]

        run_id = self.db.create_phase_mapper_run(library_id, str(run_dir))
        job_dir = run_dir  # job bookkeeping lives alongside this run's outputs
        payload_path = job_dir / "job_payload.json"
        payload_path.write_text(json.dumps({
            "job_id": str(run_id), "job_dir": str(job_dir), "run_dir": str(run_dir), "config": cfg,
        }), encoding="utf-8")

        # cwd/env are set explicitly rather than relying on the venv's editable-install
        # metadata to resolve `import app...` in the subprocess: that metadata records
        # an absolute source path baked in at `pip install -e .` time, which goes stale
        # (and the subprocess can no longer import app at all) whenever the checkout is
        # moved or an old worktree it was installed from is removed.
        project_root = Path(__file__).resolve().parents[2]
        runner = project_root / "app" / "tools" / "phase_mapper_pipeline_runner.py"
        log_path = job_dir / "run.log"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(project_root), env.get("PYTHONPATH", "")) if p)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(runner), str(payload_path)],
                stdout=log, stderr=subprocess.STDOUT, text=True,
                cwd=str(project_root), env=env,
            )
        self.db.update_phase_mapper_run(run_id, job_pid=process.pid)
        return self.db.get_phase_mapper_run(run_id)

    def _reconcile_run_status(self, run: dict[str, Any]) -> dict[str, Any]:
        """If `run` is still marked "running" but its subprocess has actually
        finished (a `<run_id>.returncode` file exists in its directory),
        update the DB row to the terminal status and return the refreshed row.

        Callers are expected to only invoke this for rows already known to
        have status == "running" -- already-terminal rows don't need (and
        this doesn't re-check) reconciliation.
        """
        run_id = run["id"]
        returncode_path = Path(run["dir_path"]) / f"{run_id}.returncode"
        if returncode_path.exists():
            return_code = int(returncode_path.read_text())
            status = "succeeded" if return_code == 0 else "failed"
            error_log = Path(run["dir_path"]) / "error.log"
            error_message = error_log.read_text(encoding="utf-8") if error_log.exists() else ""
            self.db.update_phase_mapper_run(
                run_id, status=status, return_code=return_code, error_message=error_message,
                frozen=(status == "succeeded"), completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return self.db.get_phase_mapper_run(run_id)
        return run

    def get_run(self, run_id: int, user_id: str) -> dict[str, Any]:
        run = self._get_owned_run(run_id, user_id)
        if run["status"] == "running":
            run = self._reconcile_run_status(run)
        return run

    def delete_run(self, run_id: int, user_id: str) -> None:
        run = self._get_owned_run(run_id, user_id)
        self.db.delete_phase_mapper_run(run_id)

    def get_reads(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        return self._read_csv_records(run["dir_path"], "parsed_reads_summary.csv")

    def get_predictions(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        return self._read_csv_records(run["dir_path"], "model_predictions_all_wells.csv")

    def get_xrd_recommendations(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        return self._read_csv_records(run["dir_path"], "xrd_recommendations_next_wells.csv")

    def get_remeasurement(self, run_id: int, user_id: str) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        return self._read_csv_records(run["dir_path"], "optical_remeasurement_recommendations.csv")

    def get_optical_spectra(self, run_id: int, user_id: str, well: str | None = None) -> list[dict[str, Any]]:
        run = self._get_owned_run(run_id, user_id)
        records = self._read_csv_records(run["dir_path"], "spectral_data_long_format.csv")
        if well:
            records = [r for r in records if r.get("well") == well]
        return records

    # model_predictions_all_wells.csv is written by the pipeline as `priority`
    # (see phase_mapper/pipeline.py), itself built from
    # `all_table = training.merge(predictions)` where
    # `training = comp_desc.merge(optical)`. `priority` is a `.copy()` of
    # `all_table`, so it retains every column of `training` (hence of both
    # comp_desc and optical) plus the prediction columns. It is therefore a
    # strict column superset of composition_descriptors_per_well.csv and
    # optical_features_per_well.csv -- confirmed empirically against a real
    # pipeline run (see the final-review-fix-report for the column diff).
    #
    # Critically, `compute_xrd_priority` (active_learning.py) also *mutates*
    # three of those carried-over columns in place --
    # optical_heterogeneity_score_total, pl_abs_disagreement_score, and
    # optical_phase_complexity_score -- replacing their raw values with
    # minmax-normalized ones. So model_predictions_all_wells.csv is not just
    # a superset, it's the file with the *correct* (normalized, matching
    # GET /predictions) values for those three metrics; the versions in
    # optical_features_per_well.csv are stale/raw duplicates under the same
    # column names.
    #
    # Because of this, when model_predictions_all_wells.csv is present it is
    # used as the sole base and composition_descriptors_per_well.csv /
    # optical_features_per_well.csv are skipped entirely (not merged) --
    # merging them would just re-collide on ~1400+ duplicate columns and, for
    # those three metrics, bury the correct normalized value under a renamed
    # key. If it's absent, we fall back to the older, narrower base so the
    # endpoint still returns something.
    _HEATMAP_BASE_FILE = "model_predictions_all_wells.csv"
    _HEATMAP_FALLBACK_BASE_FILES = [
        "composition_descriptors_per_well.csv",
        "optical_features_per_well.csv",
    ]
    # These carry genuinely additional columns not present in the base
    # (e.g. reason_for_recommendation, optical_remeasurement_priority_score)
    # alongside a lot of columns that just repeat the base's data for a
    # subset of wells -- the merge logic below skips the latter rather than
    # re-merging duplicates.
    _HEATMAP_SUPPLEMENTAL_FILES = [
        "xrd_recommendations_next_wells.csv",
        "optical_remeasurement_recommendations.csv",
    ]

    _HEATMAP_NON_METRIC_COLS = {"well", "row", "col", "library", "sample_key", "is_control", "replicate_group", "notes"}

    # Trailing tokens stripped from a source filename's stem to produce a
    # short, human-readable disambiguation tag (see _heatmap_source_tag).
    # Only the first matching suffix is stripped, so e.g.
    # "xrd_recommendations_next_wells" -> "xrd_recommendations" (keeps
    # "recommendations" so it stays distinct from
    # "optical_remeasurement_recommendations" -> "optical_remeasurement").
    _HEATMAP_TAG_SUFFIXES = ("_per_well", "_all_wells", "_next_wells", "_recommendations")

    @classmethod
    def _heatmap_source_tag(cls, filename: str) -> str:
        """Deterministic, human-readable short tag derived from a source filename.

        Used to disambiguate columns that collide across two heatmap source
        files (same column name, different meaning) without dropping either
        one's data.
        """
        stem = filename.removesuffix(".csv")
        for suffix in cls._HEATMAP_TAG_SUFFIXES:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _heatmap_merge_source(self, merged: pd.DataFrame, base_columns: set[str], path: Path, tag: str) -> pd.DataFrame:
        """Left-merge one supplemental/fallback heatmap source file onto `merged`.

        Columns already present in the base table (`base_columns`), or
        positional/identity columns (well/row/col/...), are skipped entirely
        rather than re-merged under a renamed key -- they're either
        duplicate data we already have from the base, or not metrics at all.
        A column that collides with something added by an *earlier*
        non-base source file (not the base itself) is renamed with a
        source-file tag so neither side's data is dropped.
        """
        other = pd.read_csv(path)
        if "well" not in other.columns:
            return merged
        keep_cols = ["well"]
        rename_map: dict[str, str] = {}
        for c in other.columns:
            if c == "well" or c in base_columns or c in self._HEATMAP_NON_METRIC_COLS:
                continue
            if c in merged.columns:
                new_name = f"{c}_{tag}"
                # Guard against the (unlikely) case where the disambiguated
                # name itself already exists, so no column is ever dropped.
                while new_name in merged.columns or new_name in rename_map.values():
                    new_name = f"{new_name}_2"
                rename_map[c] = new_name
            keep_cols.append(c)
        if len(keep_cols) == 1:
            return merged  # nothing new to add from this file
        other = other[keep_cols].rename(columns=rename_map)
        return merged.merge(other, on="well", how="left")

    def get_heatmaps(self, run_id: int, user_id: str) -> dict[str, Any]:
        run = self._get_owned_run(run_id, user_id)
        run_dir = Path(run["dir_path"])

        base_path = run_dir / self._HEATMAP_BASE_FILE
        if base_path.exists():
            merged = pd.read_csv(base_path)
            remaining_sources = list(self._HEATMAP_SUPPLEMENTAL_FILES)
        else:
            fallback_base_path = run_dir / self._HEATMAP_FALLBACK_BASE_FILES[0]
            if not fallback_base_path.exists():
                return {"columns": [], "wells": []}
            merged = pd.read_csv(fallback_base_path)
            remaining_sources = self._HEATMAP_FALLBACK_BASE_FILES[1:] + list(self._HEATMAP_SUPPLEMENTAL_FILES)

        base_columns = set(merged.columns)
        for filename in remaining_sources:
            path = run_dir / filename
            if not path.exists():
                continue
            tag = self._heatmap_source_tag(filename)
            merged = self._heatmap_merge_source(merged, base_columns, path, tag)

        non_metric_cols = self._HEATMAP_NON_METRIC_COLS
        numeric_cols = [
            c for c in merged.columns
            if c not in non_metric_cols and pd.api.types.is_numeric_dtype(merged[c])
        ]
        records = json.loads(merged.to_json(orient="records"))
        return {"columns": numeric_cols, "wells": records}

    # Job-bookkeeping files written alongside a run's actual pipeline output
    # artifacts. These carry server-internal details -- job_payload.json
    # contains absolute server paths and the full pipeline config (including
    # absolute composition-map paths); run.log/error.log are raw subprocess
    # stdout/stderr that can include tracebacks with absolute paths on
    # failure; *.returncode is pure job plumbing. None of them are pipeline
    # output and none should ever leave the server in a downloaded zip.
    _DOWNLOAD_EXCLUDED_FILENAMES = {"job_payload.json", "run.log", "error.log"}

    @classmethod
    def _is_download_excluded(cls, filename: str) -> bool:
        return filename in cls._DOWNLOAD_EXCLUDED_FILENAMES or filename.endswith(".returncode")

    def get_run_download_bytes(self, run_id: int, user_id: str) -> bytes:
        run = self._get_owned_run(run_id, user_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in Path(run["dir_path"]).glob("*"):
                if path.is_file() and not self._is_download_excluded(path.name):
                    zf.write(path, arcname=path.name)
        return buffer.getvalue()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
