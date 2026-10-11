"""Curve fitting previews, serialized results, and plot paths for the web UI."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, List, Optional

import pandas as pd

from app.tools.paths import get_runtime_root
from app.utils.json_safe import to_jsonable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _plot_api_url(well: str, read: Optional[str] = None) -> str:
    q = f"well={quote(well)}"
    if read is not None and str(read).strip():
        q += f"&read={quote(str(read).strip())}"
    return f"/agents/curve-fitting/plot?{q}"


def _resolve_data_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return p.resolve()
    bases = (_PROJECT_ROOT, get_runtime_root(), Path.cwd())
    seen: set[Path] = set()
    for base in bases:
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)
        candidate = (base / path_str).resolve()
        if candidate.is_file():
            return candidate
    return None


def _enrich_wells_plot_urls(wells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure plot_url is set whenever a plot file exists (or plot_path was recorded)."""
    enriched: List[Dict[str, Any]] = []
    for item in wells:
        if not isinstance(item, dict):
            continue
        well = dict(item)
        if well.get("plot_url"):
            enriched.append(well)
            continue
        plot_path = well.get("plot_path")
        if not plot_path:
            files = well.get("files")
            if isinstance(files, dict):
                plot_path = files.get("fitting_plot")
        if plot_path and _resolve_data_path(str(plot_path)):
            well["plot_path"] = str(plot_path)
            well["plot_url"] = _plot_api_url(
                str(well.get("well_name", "")),
                well.get("read"),
            )
        elif plot_path:
            well["plot_path"] = str(plot_path)
            well["plot_url"] = _plot_api_url(
                str(well.get("well_name", "")),
                well.get("read"),
            )
        enriched.append(well)
    return enriched


def preview_table_file(path_str: str, *, max_rows: int = 20) -> Dict[str, Any]:
    """Return a small tabular preview for CSV or Excel."""
    path = _resolve_data_path(path_str)
    if not path:
        return {"error": f"File not found: {path_str}"}

    try:
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, nrows=max_rows)
            total_rows = len(pd.read_excel(path))
        else:
            df = pd.read_csv(path, nrows=max_rows)
            total_rows = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1

        df = df.where(pd.notnull(df), None)
        columns = [str(c) for c in df.columns.tolist()]
        rows = to_jsonable(df.head(max_rows).values.tolist())
        return {
            "filename": path.name,
            "columns": columns,
            "rows": rows,
            "preview_row_count": len(rows),
            "total_rows": max(total_rows, 0),
            "column_count": len(columns),
        }
    except Exception as exc:
        return {"error": str(exc), "filename": path.name}


def _serialize_peak(peak: Any) -> Dict[str, Any]:
    return {
        "center": float(peak.center) if hasattr(peak, "center") and peak.center is not None else None,
        "height": float(peak.height) if hasattr(peak, "height") and peak.height is not None else None,
        "fwhm": float(peak.fwhm) if hasattr(peak, "fwhm") and peak.fwhm is not None else None,
    }


def _serialize_fit_result(fit: Any) -> Dict[str, Any]:
    if fit is None:
        return {"success": False}
    stats = getattr(fit, "stats", None)
    return {
        "success": bool(getattr(fit, "success", False)),
        "r2": float(stats.r2) if stats and hasattr(stats, "r2") else None,
        "rmse": float(stats.rmse) if stats and hasattr(stats, "rmse") else None,
        "redchi": float(stats.redchi) if stats and hasattr(stats, "redchi") else None,
        "peak_count": len(getattr(fit, "peaks", []) or []),
        "peaks": [_serialize_peak(p) for p in (getattr(fit, "peaks", None) or [])],
    }


def _serialize_well_result(item: Dict[str, Any]) -> Dict[str, Any]:
    well = str(item.get("well_name", ""))
    read = item.get("read", "")
    read_str = str(read) if read is not None and read != "" else ""
    plot_path = (item.get("files") or {}).get("fitting_plot")
    plot_url = _plot_api_url(well, read_str or None) if plot_path else None

    quality = to_jsonable(item.get("quality_assessment") or {})
    if not isinstance(quality, dict):
        quality = {}

    return {
        "well_name": well,
        "read": read_str or None,
        "fit": _serialize_fit_result(item.get("fit_result")),
        "quality_assessment": quality,
        "qa": to_jsonable(item.get("qa") or {}),
        "analysis_mode": str(item.get("analysis_mode") or "unknown"),
        "plot_url": plot_url,
        "plot_path": str(plot_path) if plot_path else None,
    }


def serialize_curve_fitting_results(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    wells_raw = raw.get("results") or []
    wells: List[Dict[str, Any]] = []
    for item in wells_raw:
        if isinstance(item, dict):
            wells.append(_serialize_well_result(item))
    wells = _enrich_wells_plot_urls(wells)

    summary = raw.get("summary") or {}
    if isinstance(summary, dict):
        total = summary.get("total_wells") or len(wells)
        successful = summary.get("successful_fits")
        if successful is None:
            successful = sum(1 for w in wells if w.get("fit", {}).get("success"))
        wells_analyzed = summary.get("wells_analyzed")
        summary_out = {
            "total_wells": int(total),
            "successful_fits": int(successful),
            "success_rate_pct": round(100.0 * int(successful) / max(int(total), 1), 1),
            "review_required": int(
                summary.get("review_required")
                if summary.get("review_required") is not None
                else sum(1 for w in wells if (w.get("qa") or {}).get("review_required"))
            ),
            "analysis_mode": summary.get("analysis_mode"),
            "wells_analyzed": to_jsonable(wells_analyzed) if wells_analyzed is not None else None,
        }
    else:
        summary_out = {
            "total_wells": len(wells),
            "successful_fits": sum(1 for w in wells if w.get("fit", {}).get("success")),
            "success_rate_pct": 0.0,
        }
        if wells:
            summary_out["success_rate_pct"] = round(
                100.0 * summary_out["successful_fits"] / max(summary_out["total_wells"], 1),
                1,
            )

    files = raw.get("files") or {}
    files_out = {}
    if isinstance(files, dict):
        for key, val in files.items():
            if val:
                files_out[key] = str(val)

    return to_jsonable(
        {
            "success": bool(raw.get("success", wells)),
            "error": raw.get("error"),
            "summary": summary_out,
            "wells": wells,
            "files": files_out,
            "jupyter_upload": raw.get("jupyter_upload"),
        }
    )


def get_curve_fitting_results(memory: Any) -> Dict[str, Any]:
    raw = memory.get_var("curve_fitting_results")
    # Already stored in API-safe form (has "wells" with "fit" dicts, not fit_result objects)
    if isinstance(raw, dict) and raw.get("wells") and not raw.get("results"):
        serialized = dict(raw)
        serialized["wells"] = _enrich_wells_plot_urls(list(raw.get("wells") or []))
    else:
        serialized = serialize_curve_fitting_results(raw)
    last_error = memory.get_var("curve_fitting_last_error")
    data_file = memory.get_var("curve_fitting_last_data_file")
    comp_file = memory.get_var("curve_fitting_last_composition_file")
    comp_provenance = memory.get_var("curve_fitting_composition_provenance")
    return {
        "has_results": serialized is not None and bool(serialized.get("wells")),
        "last_error": str(last_error) if last_error else None,
        "data_file": data_file,
        "composition_file": comp_file,
        "composition_provenance": to_jsonable(comp_provenance),
        "results": serialized,
    }


def get_curve_fitting_session(memory: Any) -> Dict[str, Any]:
    """Workflow + auto-run state for the Curve Fitting page."""
    from app.services.workflow_followups import WORKFLOW_STEP_PAGES, build_auto_curve_fitting_payload

    payload = build_auto_curve_fitting_payload(memory)
    return to_jsonable(
        {
            **get_curve_fitting_results(memory),
            "auto_run_pending": payload is not None,
            "auto_run_data_file": memory.get_var("auto_run_data_file"),
            "auto_run_comp_file": memory.get_var("auto_run_comp_file"),
            "auto_run_params": memory.get_var("auto_run_params") or {},
            "demo_workflow_running": bool(memory.get_var("demo_workflow_running")),
            "workflow_active": bool(memory.get_var("workflow_active")),
            "workflow_step": memory.get_var("workflow_step"),
            "auto_ml_after_curve_fitting": bool(
                memory.get_var("auto_ml_after_curve_fitting")
            ),
            "next_page": WORKFLOW_STEP_PAGES.get(
                str(memory.get_var("workflow_step") or "")
            ),
        }
    )


def find_plot_path(memory: Any, well: str, read: Optional[str] = None) -> Optional[Path]:
    raw = memory.get_var("curve_fitting_results")
    if not isinstance(raw, dict):
        return None

    read_norm = str(read).strip() if read is not None and str(read).strip() else ""
    items = raw.get("wells") or raw.get("results") or []
    matches: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or str(item.get("well_name")) != well:
            continue
        item_read = item.get("read", "")
        item_read_str = str(item_read).strip() if item_read is not None and str(item_read).strip() else ""
        if read_norm and item_read_str != read_norm:
            continue
        matches.append(item)

    if not matches:
        return None
    if not read_norm and len(matches) > 1:
        matches = [matches[0]]

    for item in matches:
        plot = item.get("plot_path") or (item.get("files") or {}).get("fitting_plot")
        if plot:
            resolved = _resolve_data_path(str(plot))
            if resolved:
                return resolved

    return _discover_plot_on_disk(well, read_norm or None)


def _discover_plot_on_disk(well: str, read: Optional[str]) -> Optional[Path]:
    """Locate a saved fitting PNG when session metadata paths are stale or relative."""
    candidates: List[str] = []
    if read:
        candidates.append(f"results/fit_results_{well}_read{read}.png")
    candidates.append(f"results/fit_results_{well}.png")
    for rel in candidates:
        resolved = _resolve_data_path(rel)
        if resolved:
            return resolved

    bases = (_PROJECT_ROOT, get_runtime_root(), Path.cwd())
    seen: set[Path] = set()
    for base in bases:
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)
        results_dir = base / "results"
        if not results_dir.is_dir():
            continue
        glob_pat = f"fit_results_{well}_read*.png" if read else f"fit_results_{well}*.png"
        hits = sorted(results_dir.glob(glob_pat))
        if read:
            exact = results_dir / f"fit_results_{well}_read{read}.png"
            if exact.is_file():
                return exact.resolve()
        if hits:
            return hits[0].resolve()
    return None
