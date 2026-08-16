"""Headless adapter for the bundled XRD peak fitting program."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .xrd_processing import infer_well_from_filename


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XRD_FITTING_ROOT = PROJECT_ROOT / "external" / "XRD fitting"


def _ensure_xrd_fitting_importable() -> None:
    if not XRD_FITTING_ROOT.exists():
        raise ImportError(f"Bundled XRD fitting program was not found at {XRD_FITTING_ROOT}")
    root = str(XRD_FITTING_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _fit_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = cfg or {}
    return dict(cfg.get("xrd_peak_fitting", {}))


def fit_xrd_file(path: str | Path, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Detect and fit peaks for one XRD scan using the bundled fitting engine."""
    _ensure_xrd_fitting_importable()

    from xrd_app.core.export import export_fitted_curve, peak_table
    from xrd_app.core.fitting import fit_sample
    from xrd_app.core.io import load_xrd_file
    from xrd_app.core.peak_detection import detect_peaks

    settings = _fit_cfg(cfg)
    path = Path(path)
    sample = load_xrd_file(path)
    sample.peaks = detect_peaks(
        sample,
        prominence=settings.get("prominence"),
        min_distance_deg=float(settings.get("min_distance_deg", 0.18)),
        smooth_window=int(settings.get("smooth_window", 11)),
        width_min_deg=float(settings.get("width_min_deg", 0.03)),
        width_max_deg=float(settings.get("width_max_deg", 2.5)),
    )

    summary: dict[str, Any] = {
        "source_file": str(path),
        "sample_id": path.stem,
        "well": infer_well_from_filename(path),
        "peak_fit_status": "not_run",
        "peak_fit_peak_count": int(len(sample.peaks)),
        "peak_fit_r2": pd.NA,
        "peak_fit_rmse": pd.NA,
        "peak_fit_warning": "",
    }

    if not sample.peaks:
        summary["peak_fit_status"] = "no_peaks_detected"
        return pd.DataFrame(), pd.DataFrame(), summary

    try:
        fit_sample(
            sample,
            background=str(settings.get("background", "adaptive")),
            max_nfev=int(settings.get("max_nfev", 120)),
            snapshot_label="dashboard_fit",
        )
        summary["peak_fit_status"] = "fitted"
        if sample.fit_result is not None:
            summary["peak_fit_r2"] = sample.fit_result.quality.get("r2", pd.NA)
            summary["peak_fit_rmse"] = sample.fit_result.quality.get("rmse", pd.NA)
    except Exception as exc:  # Keep the dashboard useful even if one scan is hard to fit.
        summary["peak_fit_status"] = "fit_failed"
        summary["peak_fit_warning"] = str(exc)

    peaks = peak_table(sample)
    if not peaks.empty:
        peaks.insert(0, "source_file", str(path))
        peaks.insert(0, "well", summary["well"])
        peaks.insert(0, "sample_id", path.stem)

    curve = pd.DataFrame()
    if sample.fit_result is not None:
        temp_path = path.with_suffix(path.suffix + ".fitted_curve.tmp.csv")
        try:
            export_fitted_curve(sample, temp_path)
            curve = pd.read_csv(temp_path)
            curve.insert(0, "source_file", str(path))
            curve.insert(0, "well", summary["well"])
            curve.insert(0, "sample_id", path.stem)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return peaks, curve, summary


def refit_xrd_file_with_peak_table(
    path: str | Path,
    reviewed_peaks: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit one XRD scan using a manually edited peak table."""
    _ensure_xrd_fitting_importable()

    from xrd_app.core.export import export_fitted_curve, peak_table
    from xrd_app.core.fitting import fit_sample
    from xrd_app.core.io import load_xrd_file
    from xrd_app.core.models import PeakComponent

    settings = _fit_cfg(cfg)
    path = Path(path)
    sample = load_xrd_file(path)
    sample.peaks = _components_from_reviewed_table(reviewed_peaks)

    summary: dict[str, Any] = {
        "source_file": str(path),
        "sample_id": path.stem,
        "well": infer_well_from_filename(path),
        "peak_fit_status": "not_run",
        "peak_fit_peak_count": int(len(sample.peaks)),
        "peak_fit_r2": pd.NA,
        "peak_fit_rmse": pd.NA,
        "peak_fit_warning": "",
        "reviewed": True,
    }

    if not sample.active_peaks():
        summary["peak_fit_status"] = "no_active_peaks"
        peaks = peak_table(sample)
        if not peaks.empty:
            peaks.insert(0, "source_file", str(path))
            peaks.insert(0, "well", summary["well"])
            peaks.insert(0, "sample_id", path.stem)
        return peaks, pd.DataFrame(), summary

    try:
        fit_sample(
            sample,
            background=str(settings.get("background", "adaptive")),
            max_nfev=int(settings.get("max_nfev", 120)),
            snapshot_label="reviewed_fit",
        )
        summary["peak_fit_status"] = "fitted"
        if sample.fit_result is not None:
            summary["peak_fit_r2"] = sample.fit_result.quality.get("r2", pd.NA)
            summary["peak_fit_rmse"] = sample.fit_result.quality.get("rmse", pd.NA)
    except Exception as exc:
        summary["peak_fit_status"] = "fit_failed"
        summary["peak_fit_warning"] = str(exc)

    peaks = peak_table(sample)
    if not peaks.empty:
        peaks.insert(0, "source_file", str(path))
        peaks.insert(0, "well", summary["well"])
        peaks.insert(0, "sample_id", path.stem)

    curve = pd.DataFrame()
    if sample.fit_result is not None:
        temp_path = path.with_suffix(path.suffix + ".reviewed_fitted_curve.tmp.csv")
        try:
            export_fitted_curve(sample, temp_path)
            curve = pd.read_csv(temp_path)
            curve.insert(0, "source_file", str(path))
            curve.insert(0, "well", summary["well"])
            curve.insert(0, "sample_id", path.stem)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return peaks, curve, summary


def _components_from_reviewed_table(reviewed_peaks: pd.DataFrame) -> list[Any]:
    _ensure_xrd_fitting_importable()
    from xrd_app.core.models import PeakComponent

    components: list[PeakComponent] = []
    if reviewed_peaks is None or reviewed_peaks.empty:
        return components

    for idx, row in reviewed_peaks.reset_index(drop=True).iterrows():
        position = pd.to_numeric(pd.Series([row.get("two_theta")]), errors="coerce").iloc[0]
        height = pd.to_numeric(pd.Series([row.get("height")]), errors="coerce").iloc[0]
        fwhm = pd.to_numeric(pd.Series([row.get("fwhm")]), errors="coerce").iloc[0]
        if pd.isna(position):
            continue
        status = str(row.get("status") or "manual").strip() or "manual"
        shape = str(row.get("shape") or "pseudo_voigt").strip()
        if shape not in {"pseudo_voigt", "gaussian", "lorentzian"}:
            shape = "pseudo_voigt"
        peak_id = pd.to_numeric(pd.Series([row.get("peak_id")]), errors="coerce").iloc[0]
        peak_id = int(peak_id) if not pd.isna(peak_id) else idx + 1
        eta = pd.to_numeric(pd.Series([row.get("eta")]), errors="coerce").iloc[0]
        component = PeakComponent(
            id=peak_id,
            position=float(position),
            height=float(height) if not pd.isna(height) else 100.0,
            fwhm=max(float(fwhm), 0.01) if not pd.isna(fwhm) else 0.20,
            area=float(pd.to_numeric(pd.Series([row.get("area")]), errors="coerce").iloc[0] or 0.0),
            shape=shape,
            eta=float(eta) if not pd.isna(eta) else 0.5,
            status=status,
            phase=str(row.get("phase") or ""),
            hkl=str(row.get("hkl") or ""),
            note=str(row.get("note") or ""),
            lock_position=bool(row.get("locked_position", False)),
            lock_height=bool(row.get("locked_height", False)),
            lock_fwhm=bool(row.get("locked_fwhm", False)),
            lock_eta=True,
            lower_position=float(position) - 0.5,
            upper_position=float(position) + 0.5,
            lower_height=0.0,
            lower_fwhm=0.01,
        )
        components.append(component)
    return components


def replace_rows_for_source(existing: pd.DataFrame, replacement: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Replace one scan's rows in a saved output table."""
    if existing.empty:
        return replacement.copy()
    kept = existing.loc[existing.get("source_file", "") != source_file].copy()
    if replacement.empty:
        return kept.reset_index(drop=True)
    return pd.concat([kept, replacement], ignore_index=True)


def load_reference_library(path: str | Path) -> pd.DataFrame:
    """Load confirmed reference phases from JSON or CSV into one peak table."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "phase",
                "reference_two_theta",
                "hkl",
                "relative_intensity",
                "tolerance",
                "selected",
                "visible",
                "priority",
                "color",
                "source_type",
            ]
        )
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for ref in payload.get("references", []):
            if not isinstance(ref, dict):
                continue
            phase = str(ref.get("name") or "").strip()
            if not phase:
                continue
            for peak in ref.get("peaks", []):
                if not isinstance(peak, dict):
                    continue
                two_theta = pd.to_numeric(pd.Series([peak.get("two_theta")]), errors="coerce").iloc[0]
                if pd.isna(two_theta):
                    continue
                rows.append(
                    {
                        "phase": phase,
                        "reference_two_theta": float(two_theta),
                        "hkl": str(peak.get("hkl") or ""),
                        "relative_intensity": peak.get("relative_intensity", pd.NA),
                        "tolerance": peak.get("tolerance", pd.NA),
                        "selected": peak.get("selected", True) is not False,
                        "visible": ref.get("visible", True) is not False,
                        "priority": int(ref.get("priority", 50) or 50),
                        "color": str(ref.get("color") or "#007aff"),
                        "source_type": str(ref.get("source_type") or ""),
                    }
                )
        return pd.DataFrame(rows)

    rules = pd.read_csv(path)
    rows = pd.DataFrame(
        {
            "phase": rules.get("phase_label", pd.Series(dtype=object)).astype(str),
            "reference_two_theta": pd.to_numeric(rules.get("peak_2theta_deg"), errors="coerce"),
            "hkl": "",
            "relative_intensity": pd.NA,
            "tolerance": pd.to_numeric(rules.get("tolerance_deg"), errors="coerce"),
            "selected": rules.get("enabled", True),
            "visible": True,
            "priority": 50,
            "color": "#007aff",
            "source_type": "rules_csv",
        }
    ).dropna(subset=["reference_two_theta"])
    if not isinstance(rows["selected"], bool):
        rows["selected"] = rows["selected"].fillna(True).astype(str).str.lower().isin({"1", "true", "yes", "y"})
    return rows


def save_reference_library_json(references: pd.DataFrame, path: str | Path) -> None:
    """Save an editable reference peak table as the fitting app JSON format."""
    path = Path(path)
    refs: list[dict[str, Any]] = []
    if references.empty:
        payload = {"version": 1, "references": refs}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    work = references.copy()
    for phase, group in work.groupby("phase", dropna=True):
        first = group.iloc[0]
        peaks = []
        for _, row in group.iterrows():
            two_theta = pd.to_numeric(pd.Series([row.get("reference_two_theta")]), errors="coerce").iloc[0]
            if pd.isna(two_theta):
                continue
            tol = pd.to_numeric(pd.Series([row.get("tolerance")]), errors="coerce").iloc[0]
            rel = pd.to_numeric(pd.Series([row.get("relative_intensity")]), errors="coerce").iloc[0]
            peaks.append(
                {
                    "two_theta": float(two_theta),
                    "hkl": str(row.get("hkl") or ""),
                    "relative_intensity": None if pd.isna(rel) else float(rel),
                    "tolerance": None if pd.isna(tol) else float(tol),
                    "selected": bool(row.get("selected", True)),
                    "note": str(row.get("note") or ""),
                }
            )
        refs.append(
            {
                "name": str(phase),
                "source_type": str(first.get("source_type") or "manual"),
                "source_path": str(first.get("source_path") or ""),
                "wavelength": float(first.get("wavelength", 1.540598) or 1.540598),
                "color": str(first.get("color") or "#007aff"),
                "visible": bool(first.get("visible", True)),
                "priority": int(first.get("priority", 50) or 50),
                "library_saved": True,
                "peaks": peaks,
            }
        )
    path.write_text(json.dumps({"version": 1, "references": refs}, indent=2), encoding="utf-8")


def reference_from_cif_file(path: str | Path, wavelength: float = 1.540598) -> pd.DataFrame:
    """Import one CIF as a reference peak table. Requires optional pymatgen."""
    _ensure_xrd_fitting_importable()
    from xrd_app.core.reference import reference_from_cif

    ref = reference_from_cif(path, wavelength=wavelength)
    rows = []
    for peak in ref.peaks:
        rows.append(
            {
                "phase": ref.name,
                "reference_two_theta": peak.two_theta,
                "hkl": peak.hkl,
                "relative_intensity": peak.relative_intensity,
                "tolerance": peak.tolerance,
                "selected": peak.selected,
                "visible": ref.visible,
                "priority": ref.priority,
                "color": ref.color,
                "source_type": ref.source_type,
                "source_path": str(ref.source_path or path),
                "wavelength": ref.wavelength,
            }
        )
    return pd.DataFrame(rows)


def collect_assignment_candidates(
    peaks: pd.DataFrame,
    references: pd.DataFrame,
    active_phases: list[str] | None = None,
    tolerance_deg: float = 0.28,
) -> pd.DataFrame:
    """Find candidate reference assignments for fitted peaks."""
    if peaks.empty or references.empty:
        return pd.DataFrame()
    refs = references.copy()
    refs = refs[refs["selected"].fillna(True).astype(bool) & refs["visible"].fillna(True).astype(bool)]
    if active_phases is not None:
        refs = refs[refs["phase"].isin(active_phases)]
    refs["reference_two_theta"] = pd.to_numeric(refs["reference_two_theta"], errors="coerce")
    refs["tolerance"] = pd.to_numeric(refs.get("tolerance"), errors="coerce")
    refs["priority"] = pd.to_numeric(refs.get("priority"), errors="coerce").fillna(50).astype(int)
    refs = refs.dropna(subset=["reference_two_theta"])
    if refs.empty:
        return pd.DataFrame()

    peak_rows = peaks.copy()
    peak_rows["two_theta"] = pd.to_numeric(peak_rows.get("two_theta"), errors="coerce")
    peak_rows = peak_rows.dropna(subset=["two_theta"])
    if "status" in peak_rows.columns:
        peak_rows = peak_rows[peak_rows["status"].fillna("").astype(str) != "ignored_noise"]
    rows: list[dict[str, Any]] = []
    for _, peak in peak_rows.iterrows():
        for _, ref in refs.iterrows():
            peak_tol = max(float(tolerance_deg), float(ref["tolerance"]) if pd.notna(ref["tolerance"]) else 0.0)
            if peak_tol <= 0:
                continue
            delta = float(peak["two_theta"]) - float(ref["reference_two_theta"])
            if abs(delta) > peak_tol:
                continue
            score = max(0.0, 1.0 - abs(delta) / peak_tol)
            rows.append(
                {
                    "library": peak.get("library", ""),
                    "sample_id": peak.get("sample_id", ""),
                    "well": peak.get("well", ""),
                    "source_file": peak.get("source_file", ""),
                    "peak_id": int(peak.get("peak_id", 0)),
                    "sample_two_theta": float(peak["two_theta"]),
                    "current_phase": peak.get("phase", ""),
                    "candidate_phase": ref["phase"],
                    "reference_two_theta": float(ref["reference_two_theta"]),
                    "delta_two_theta": delta,
                    "abs_delta_two_theta": abs(delta),
                    "tolerance_deg": peak_tol,
                    "hkl": ref.get("hkl", ""),
                    "priority": int(ref["priority"]),
                    "score": score,
                    "is_best": False,
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    best_idx = (
        candidates.sort_values(["peak_id", "priority", "abs_delta_two_theta", "score"], ascending=[True, True, True, False])
        .groupby("peak_id", sort=False)
        .head(1)
        .index
    )
    candidates.loc[best_idx, "is_best"] = True
    return candidates.sort_values(["sample_two_theta", "is_best", "priority", "abs_delta_two_theta"], ascending=[True, False, True, True]).reset_index(drop=True)


def apply_best_assignments(peaks: pd.DataFrame, candidates: pd.DataFrame, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign best candidate phase to each peak and return updated peaks plus assignment table."""
    best = candidates[candidates["is_best"].astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    return apply_assignment_candidates(peaks, best, overwrite=overwrite)


def apply_assignment_candidates(peaks: pd.DataFrame, candidates: pd.DataFrame, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign provided candidate rows to peaks and return updated peaks plus assignment table."""
    updated = peaks.copy()
    for col in ["phase", "hkl", "status"]:
        if col not in updated.columns:
            updated[col] = ""
        updated[col] = updated[col].astype("object")
    if candidates.empty or updated.empty:
        return updated, pd.DataFrame()
    assignment_rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        peak_id = int(candidate["peak_id"])
        mask = pd.to_numeric(updated.get("peak_id"), errors="coerce").fillna(-1).astype(int) == peak_id
        if not mask.any():
            continue
        row_index = updated.index[mask][0]
        current_phase = str(updated.at[row_index, "phase"]) if "phase" in updated.columns and pd.notna(updated.at[row_index, "phase"]) else ""
        if current_phase and not overwrite:
            continue
        updated.at[row_index, "phase"] = candidate["candidate_phase"]
        updated.at[row_index, "hkl"] = candidate.get("hkl", "")
        updated.at[row_index, "status"] = "assigned"
        assignment_rows.append(
            {
                "library": candidate.get("library", ""),
                "sample_id": candidate.get("sample_id", ""),
                "well": candidate.get("well", ""),
                "source_file": candidate.get("source_file", ""),
                "peak_id": peak_id,
                "sample_two_theta": candidate["sample_two_theta"],
                "phase": candidate["candidate_phase"],
                "reference_two_theta": candidate["reference_two_theta"],
                "delta_two_theta": candidate["delta_two_theta"],
                "hkl": candidate.get("hkl", ""),
                "assignment_status": "assigned",
                "priority": candidate["priority"],
                "score": candidate["score"],
            }
        )
    return updated, pd.DataFrame(assignment_rows)


def summarize_assignments(peaks: pd.DataFrame, references: pd.DataFrame, tolerance_deg: float = 0.28) -> pd.DataFrame:
    """Summarize assigned phase area and reference-peak coverage for one scan."""
    if peaks.empty or "phase" not in peaks.columns:
        return pd.DataFrame()
    assigned = peaks[peaks["phase"].fillna("").astype(str).str.len() > 0].copy()
    if assigned.empty:
        return pd.DataFrame()
    assigned["area"] = pd.to_numeric(assigned.get("area"), errors="coerce")
    assigned["height"] = pd.to_numeric(assigned.get("height"), errors="coerce")
    assigned["fwhm"] = pd.to_numeric(assigned.get("fwhm"), errors="coerce")
    assigned["area_fallback"] = assigned["area"].isna() | (assigned["area"] <= 0)
    assigned.loc[assigned["area_fallback"], "area"] = assigned["height"].fillna(0) * assigned["fwhm"].fillna(0.01)
    total_area = float(assigned["area"].sum()) or 1.0
    first = assigned.iloc[0]
    rows: list[dict[str, Any]] = []
    for phase, group in assigned.groupby("phase", dropna=True):
        ref = references[references["phase"] == phase].copy() if not references.empty else pd.DataFrame()
        if not ref.empty and "selected" in ref.columns:
            ref = ref[ref["selected"].fillna(True).astype(bool)]
        selected_count = len(ref)
        matched_refs: set[int] = set()
        for ref_idx, ref_row in ref.iterrows():
            ref_two_theta = pd.to_numeric(pd.Series([ref_row.get("reference_two_theta")]), errors="coerce").iloc[0]
            if pd.isna(ref_two_theta):
                continue
            ref_tol = pd.to_numeric(pd.Series([ref_row.get("tolerance")]), errors="coerce").iloc[0]
            peak_tol = max(float(tolerance_deg), float(ref_tol) if not pd.isna(ref_tol) else 0.0)
            if (pd.to_numeric(group["two_theta"], errors="coerce") - float(ref_two_theta)).abs().le(peak_tol).any():
                matched_refs.add(int(ref_idx))
        max_peak_row = group.loc[group["area"].idxmax()]
        area_sum = float(group["area"].sum())
        rows.append(
            {
                "library": first.get("library", ""),
                "sample_id": first.get("sample_id", ""),
                "well": first.get("well", ""),
                "source_file": first.get("source_file", ""),
                "phase": phase,
                "assigned_area_pct": area_sum / total_area * 100.0,
                "max_peak_pct": float(max_peak_row["area"]) / total_area * 100.0,
                "area_sum": area_sum,
                "peaks_used": int(len(group)),
                "coverage": f"{len(matched_refs)}/{selected_count}" if selected_count else "-",
                "max_peak_two_theta": float(max_peak_row["two_theta"]),
                "note": "area estimate" if group["area_fallback"].any() else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["assigned_area_pct", "phase"], ascending=[False, True]).reset_index(drop=True)


def williamson_hall_rows(peaks: pd.DataFrame, wavelength: float = 1.540598) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Williamson-Hall point rows and per-phase linear-fit summary."""
    if peaks.empty or "phase" not in peaks.columns:
        return pd.DataFrame(), pd.DataFrame()
    work = peaks.copy()
    work = work[work["phase"].fillna("").astype(str).str.len() > 0].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["two_theta"] = pd.to_numeric(work.get("two_theta"), errors="coerce")
    work["fwhm"] = pd.to_numeric(work.get("fwhm"), errors="coerce")
    work = work.dropna(subset=["two_theta", "fwhm"])
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    theta = np.deg2rad(work["two_theta"].to_numpy(dtype=float) / 2.0)
    beta = np.deg2rad(np.clip(work["fwhm"].to_numpy(dtype=float), 1e-9, None))
    work["beta_cos_theta"] = beta * np.cos(theta)
    work["four_sin_theta"] = 4.0 * np.sin(theta)
    work["d_spacing"] = wavelength / (2.0 * np.sin(theta))
    work["q"] = 4.0 * np.pi * np.sin(theta) / wavelength

    summary_rows: list[dict[str, Any]] = []
    for phase, group in work.groupby("phase", dropna=True):
        x = group["four_sin_theta"].to_numpy(dtype=float)
        y = group["beta_cos_theta"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "phase": phase,
            "num_points": int(len(group)),
            "strain": np.nan,
            "microstrain": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "crystallite_size_nm": np.nan,
        }
        if len(group) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
            row["strain"] = float(slope)
            row["microstrain"] = float(slope * 1e6)
            row["intercept"] = float(intercept)
            row["r2"] = float(1.0 - ss_res / ss_tot)
            if intercept > 0:
                row["crystallite_size_nm"] = float(0.9 * wavelength / intercept / 10.0)
        summary_rows.append(row)
    return work.reset_index(drop=True), pd.DataFrame(summary_rows)


def fit_xrd_files(
    files: list[str | Path],
    library: str = "uploaded_raw",
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a batch of XRD files and return peak table, fitted curves, and summaries."""
    peak_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for path in sorted(Path(file) for file in files):
        try:
            peaks, curve, summary = fit_xrd_file(path, cfg=cfg)
        except Exception as exc:
            summary = {
                "source_file": str(path),
                "sample_id": path.stem,
                "well": infer_well_from_filename(path),
                "peak_fit_status": "load_failed",
                "peak_fit_peak_count": 0,
                "peak_fit_r2": pd.NA,
                "peak_fit_rmse": pd.NA,
                "peak_fit_warning": str(exc),
            }
            peaks = pd.DataFrame()
            curve = pd.DataFrame()

        for frame in (peaks, curve):
            if not frame.empty:
                frame.insert(0, "library", library)
        summary["library"] = library
        summary_rows.append(summary)
        if not peaks.empty:
            peak_frames.append(peaks)
        if not curve.empty:
            curve_frames.append(curve)

    peak_table = pd.concat(peak_frames, ignore_index=True) if peak_frames else pd.DataFrame()
    fitted_curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    fit_summary = pd.DataFrame(summary_rows)

    if not peak_table.empty:
        grouped = peak_table.groupby("source_file", dropna=False)
        fit_summary = fit_summary.merge(
            grouped.agg(
                peak_fit_positions_deg=("two_theta", lambda values: ";".join(f"{float(v):.3f}" for v in values)),
                peak_fit_fwhm_deg=("fwhm", lambda values: ";".join(f"{float(v):.3f}" for v in values)),
                peak_fit_areas=("area", lambda values: ";".join(f"{float(v):.3g}" for v in values)),
            ).reset_index(),
            on="source_file",
            how="left",
        )

    return peak_table, fitted_curves, fit_summary


def attach_peak_fit_summary(features: pd.DataFrame, fit_summary: pd.DataFrame) -> pd.DataFrame:
    """Add compact peak-fit summary columns to raw XRD feature rows."""
    if features.empty or fit_summary.empty or "source_file" not in features.columns:
        return features
    columns = [
        "source_file",
        "peak_fit_status",
        "peak_fit_peak_count",
        "peak_fit_r2",
        "peak_fit_rmse",
        "peak_fit_positions_deg",
        "peak_fit_fwhm_deg",
        "peak_fit_areas",
        "peak_fit_warning",
    ]
    available = [col for col in columns if col in fit_summary.columns]
    return features.merge(fit_summary[available], on="source_file", how="left")
