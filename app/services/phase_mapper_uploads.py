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
