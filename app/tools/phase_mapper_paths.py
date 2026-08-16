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
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_UPLOAD_KINDS = {"plate_reader", "raw_xrd", "xrd_labels_reviewed"}


def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("_", text.strip()).strip("_")
    return slug or "library"


def _validate_user_id(user_id: str) -> str:
    """Validate user_id against a strict safe charset.

    user_id is an opaque auth-provided identifier (see AuthUser.id in
    app/core/auth.py). It is rejected rather than sanitized/coerced so that
    two distinct real user ids can never be mangled into colliding paths.
    """
    if not isinstance(user_id, str) or not _USER_ID_RE.match(user_id):
        raise ValueError(f"Invalid user_id: {user_id!r}")
    return user_id


def _phase_mapper_root() -> Path:
    root = Path(get_user_data_dir()) / "phase_mapper"
    root.mkdir(parents=True, exist_ok=True)
    return root


def library_root(user_id: str, slug: str) -> Path:
    user_id = _validate_user_id(user_id)
    slug = slugify(slug)
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
