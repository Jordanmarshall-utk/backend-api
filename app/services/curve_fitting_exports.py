"""Discover and materialize curve-fitting exports for ML and analysis agents."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.tools.paths import get_results_dir


def resolve_results_path(path_str: str) -> Optional[Path]:
    """Resolve a results file path (absolute, relative, or results/... prefix)."""
    if not path_str or not str(path_str).strip():
        return None
    raw = str(path_str).strip()
    p = Path(raw)
    if p.is_file():
        return p.resolve()

    results_root = Path(get_results_dir())
    # Strip redundant results/ prefix before joining
    rel = raw.replace("\\", "/")
    if rel.startswith("results/"):
        rel = rel[len("results/") :]
    for candidate in (
        results_root / rel,
        results_root / raw,
        Path.cwd() / raw,
        Path.cwd() / "results" / rel,
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def discover_curve_fitting_files(results_dir: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Recursively find peak-fit JSON and CSV exports under the results directory.
    Returns lists of absolute paths, newest first.
    """
    root = Path(results_dir or get_results_dir())
    if not root.is_dir():
        return [], []

    json_hits: List[Path] = []
    csv_hits: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            if name.endswith("_peak_fit_results.json"):
                json_hits.append(full)
            elif name.endswith("_peak_fit_export.csv"):
                csv_hits.append(full)
            elif name.endswith("all_wells_analysis.json"):
                json_hits.append(full)

    json_hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    csv_hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in json_hits], [str(p) for p in csv_hits]


def find_latest_curve_fitting_pair(results_dir: Optional[str] = None) -> Optional[Tuple[str, Optional[str]]]:
    """Newest JSON export and best-matching CSV (by basename), searching recursively."""
    json_files, csv_files = discover_curve_fitting_files(results_dir)
    if not json_files:
        return None
    json_path = json_files[0]
    base = Path(json_path).name.replace("_peak_fit_results.json", "").replace(
        "all_wells_analysis.json", ""
    )
    csv_path: Optional[str] = None
    for c in csv_files:
        if base and base in Path(c).name:
            csv_path = c
            break
    if not csv_path and csv_files:
        csv_path = csv_files[0]
    return json_path, csv_path


def _serialized_to_ml_json(stored: Dict[str, Any]) -> Dict[str, Any]:
    """Convert API-serialized curve_fitting_results to save_all_wells-style JSON for ML."""
    wells_out: Dict[str, Any] = {}
    wells = stored.get("wells")
    if isinstance(wells, list):
        for w in wells:
            if not isinstance(w, dict):
                continue
            name = str(w.get("well_name") or "")
            if not name:
                continue
            fit = w.get("fit") or {}
            peaks = []
            for i, p in enumerate(fit.get("peaks") or []):
                if not isinstance(p, dict):
                    continue
                peaks.append(
                    {
                        "id": f"p{i + 1}",
                        "center_nm": p.get("center"),
                        "FWHM_nm": p.get("fwhm"),
                        "height": p.get("height"),
                        "amplitude": 0,
                        "area": 0,
                        "frac": 0,
                    }
                )
            wells_out[name] = {
                "read": w.get("read"),
                "fitting_results": {
                    "success": bool(fit.get("success")),
                    "quality_metrics": {
                        "r_squared": fit.get("r2"),
                        "rmse": fit.get("rmse"),
                    },
                    "quality_peaks": peaks,
                    "all_peaks": peaks,
                },
                "quality_assessment": w.get("quality_assessment") or {},
            }
    elif isinstance(wells, dict):
        return stored

    summary = stored.get("summary") or {}
    return {
        "analysis_summary": {
            "total_wells": summary.get("total_wells") or len(wells_out),
            "successful_fits": summary.get("successful_fits") or 0,
        },
        "wells": wells_out,
    }


def materialize_curve_fitting_json(memory: Any) -> Optional[str]:
    """
    Write ML-compatible JSON to disk from session when exports are missing.
    Returns absolute path to JSON file or None.
    """
    stored = memory.get_var("curve_fitting_results")
    if not isinstance(stored, dict):
        return None

    payload = _serialized_to_ml_json(stored)
    if not payload.get("wells"):
        return None

    data_file = memory.get_var("curve_fitting_last_data_file") or "curve_fitting"
    base = re.sub(r'[<>:"/\\|?*]', "_", Path(str(data_file)).stem).strip() or "curve_fitting"
    out_dir = Path(get_results_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}_peak_fit_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(out_path.resolve())


def sync_ml_paths_from_curve_fitting(memory: Any) -> Dict[str, Optional[str]]:
    """
    Point ml_auto_* session vars at the latest on-disk exports (or materialize from session).
    """
    json_path: Optional[str] = None
    csv_path: Optional[str] = None

    files = memory.get_var("curve_fitting_results")
    if isinstance(files, dict):
        raw_files = files.get("files") or {}
        for key in ("json_results", "json_path", "peak_fit_json"):
            val = raw_files.get(key)
            if val:
                resolved = resolve_results_path(str(val))
                if resolved:
                    json_path = str(resolved)
                    break
        for key in ("csv_export", "csv_path", "peak_fit_csv"):
            val = raw_files.get(key)
            if val:
                resolved = resolve_results_path(str(val))
                if resolved:
                    csv_path = str(resolved)
                    break

    # Last run may store paths on the raw agent return via memory — check dedicated keys
    for key, target in (
        ("curve_fitting_last_json", "json"),
        ("curve_fitting_last_csv", "csv"),
    ):
        val = memory.get_var(key)
        if val:
            resolved = resolve_results_path(str(val))
            if resolved:
                if target == "json":
                    json_path = str(resolved)
                else:
                    csv_path = str(resolved)

    latest = find_latest_curve_fitting_pair()
    if latest:
        if not json_path:
            json_path = latest[0]
        if not csv_path:
            csv_path = latest[1]

    if not json_path:
        json_path = materialize_curve_fitting_json(memory)

    from app.services.composition_fallback import is_placeholder_composition

    composition_blocked_reason = None
    comp_path = memory.get_var("ml_auto_composition_path")
    last_curve_comp = memory.get_var("curve_fitting_last_composition_file")
    if not comp_path and last_curve_comp:
        if is_placeholder_composition(str(last_curve_comp)):
            composition_blocked_reason = (
                "The curve-fitting composition file is a non-authoritative placeholder; "
                "supply the real plate map before composition-dependent modeling."
            )
        else:
            comp_path = last_curve_comp
    if comp_path:
        resolved = resolve_results_path(str(comp_path))
        if resolved:
            comp_path = str(resolved)
        elif Path(str(comp_path)).is_file():
            comp_path = str(Path(comp_path).resolve())
        else:
            comp_path = str(comp_path)
    elif not composition_blocked_reason:
        from app.tools.ml_automation import find_composition_csv

        comp_path = find_composition_csv(get_results_dir())
        if comp_path and is_placeholder_composition(str(comp_path)):
            comp_path = None
            composition_blocked_reason = (
                "Only a non-authoritative placeholder composition file was found."
            )

    if json_path:
        memory.set_var("ml_auto_json_path", json_path)
    if csv_path:
        memory.set_var("ml_auto_csv_path", csv_path)
    if comp_path:
        memory.set_var("ml_auto_composition_path", comp_path)
    elif composition_blocked_reason:
        memory.set_var("ml_auto_composition_path", None)
        memory.set_var("ml_composition_blocked_reason", composition_blocked_reason)

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "composition_path": comp_path,
        "composition_blocked_reason": composition_blocked_reason,
    }
