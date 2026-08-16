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
