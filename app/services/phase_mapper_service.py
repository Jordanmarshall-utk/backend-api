"""Library CRUD and upload orchestration for the phase-mapper domain."""
from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import sys
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

    def get_library_owned(self, library_id: int, user_id: str) -> dict[str, Any]:
        library = self.db.get_phase_mapper_library(library_id)
        if library is None or library["user_id"] != user_id:
            raise LookupError(f"Library {library_id} not found for this user")
        return library

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

        if run["status"] == "running":
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
                run = self.db.get_phase_mapper_run(run_id)
        return run

    def delete_run(self, run_id: int, user_id: str) -> None:
        run = self.db.get_phase_mapper_run(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        self.get_library_owned(run["library_id"], user_id)  # raises LookupError if not owned
        self.db.delete_phase_mapper_run(run_id)

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


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
