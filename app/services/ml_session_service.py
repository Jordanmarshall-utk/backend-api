"""ML Models page session — parity with Streamlit pages/ml_models.py (automation path)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.curve_fitting_exports import (
    discover_curve_fitting_files,
    find_latest_curve_fitting_pair,
    resolve_results_path,
    sync_ml_paths_from_curve_fitting,
)
from app.services.workflow_service import ML_MODEL_OPTIONS
from app.services.reasoning_integration import reasoning_context_for_stage
from app.tools.ml_automation import (
    MODEL_DUAL_TORCH_GP,
    MODEL_MONTE_CARLO_TREE,
    MODEL_SINGLE_GP,
    find_composition_csv,
    inspect_peak_csv,
)
from app.tools.paths import get_results_dir
from app.utils.json_safe import to_jsonable

DEFAULT_ML_CONFIG: Dict[str, Any] = {
    "target": "peak_1_wavelength",
    "beta": 2.0,
    "n_candidates": 20,
    "kernel_type": "RBF",
    "acquisition_method": "UCB",
    "dual_gp": {
        "performance_target": "R_squared",
        "compute_instability": False,
        "lengthscale": 1.0,
        "noise_level": 1e-4,
        "beta": 2.0,
        "instability_threshold_percentile": 0.7,
        "use_multiplicative_adjustment": True,
        "instability_params": {
            "target_wavelength": 700.0,
            "wavelength_tolerance": 10.0,
            "degradation_weight": 0.4,
            "position_weight": 0.6,
            "multiple_peak_penalty": 0.5,
        },
    },
    "monte_carlo_tree": {
        "repo_path": os.environ.get("MONTE_CARLO_REPO_PATH", ""),
        "n_attempts": 500,
        "with_agent": False,
    },
}


def _basename(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return Path(path).name


def _discover_composition_files(results_dir: Optional[str] = None) -> List[str]:
    """Find composition CSVs under results/ and data/ (newest first)."""
    root = Path(results_dir or get_results_dir())
    hits: List[Path] = []
    search_roots = [root]
    data_dir = Path("data")
    if data_dir.is_dir():
        search_roots.append(data_dir.resolve())

    for base in search_roots:
        if not base.is_dir():
            continue
        for dirpath, _, filenames in os.walk(base):
            for name in filenames:
                if "composition" in name.lower() and name.lower().endswith(".csv"):
                    hits.append(Path(dirpath) / name)

    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.resolve()) for p in hits]


def list_result_files() -> Dict[str, Any]:
    results_dir = get_results_dir()
    json_abs, csv_abs = discover_curve_fitting_files(results_dir)
    comp_abs = _discover_composition_files(results_dir)
    return {
        "json_files": [_basename(p) for p in json_abs],
        "csv_files": [_basename(p) for p in csv_abs],
        "composition_files": [_basename(p) for p in comp_abs],
        "json_paths": json_abs,
        "csv_paths": csv_abs,
        "composition_paths": comp_abs,
        "results_dir": results_dir,
    }


def get_ml_session(memory: Any) -> Dict[str, Any]:
    sync_ml_paths_from_curve_fitting(memory)

    model_choice = memory.get_var("optimization_model_choice") or MODEL_SINGLE_GP
    if model_choice not in ML_MODEL_OPTIONS:
        model_choice = MODEL_SINGLE_GP

    raw_cfg = memory.get_var("ml_model_config")
    if not isinstance(raw_cfg, dict):
        cfg = dict(DEFAULT_ML_CONFIG)
    else:
        cfg = _deep_merge_dict(dict(DEFAULT_ML_CONFIG), raw_cfg)

    json_path = memory.get_var("ml_auto_json_path")
    csv_path = memory.get_var("ml_auto_csv_path")
    comp_path = memory.get_var("ml_auto_composition_path")

    if not json_path:
        latest = find_latest_curve_fitting_pair()
        if latest:
            json_path, csv_path = latest[0], latest[1] or csv_path

    if not comp_path:
        comp_path = find_composition_csv(
            get_results_dir(),
            extra_paths=[memory.get_var("curve_fitting_last_composition_file")],
        )

    files = list_result_files()
    gp_results = memory.get_var("gp_results")
    mc_results = memory.get_var("monte_carlo_results")

    json_ok = bool(json_path and Path(str(json_path)).is_file())
    csv_ok = bool(csv_path and Path(str(csv_path)).is_file())
    comp_ok = bool(comp_path and Path(str(comp_path)).is_file())

    csv_schema: Optional[Dict[str, Any]] = None
    if csv_ok:
        csv_schema = inspect_peak_csv(str(csv_path))

    try:
        from app.tools.ml_core import _HAS_TORCH
    except ImportError:
        _HAS_TORCH = False

    return to_jsonable(
        {
            "model_choice": model_choice,
            "models_requiring_composition": [MODEL_SINGLE_GP],
            "model_options": ML_MODEL_OPTIONS,
            "ml_model_config": cfg,
            "auto_ml_after_curve_fitting": bool(memory.get_var("auto_ml_after_curve_fitting")),
            "workflow_ml_model_choice": memory.get_var("workflow_ml_model_choice"),
            "json_path": json_path,
            "csv_path": csv_path,
            "composition_path": comp_path,
            "json_file": _basename(str(json_path)) if json_path else None,
            "csv_file": _basename(str(csv_path)) if csv_path else None,
            "composition_file": _basename(str(comp_path)) if comp_path else None,
            "latest_files": files,
            "has_curve_fitting_exports": json_ok,
            "has_composition_file": comp_ok,
            "has_gp_results": bool(gp_results),
            "gp_results": gp_results if isinstance(gp_results, dict) else None,
            "monte_carlo_results": mc_results if isinstance(mc_results, dict) else None,
            "csv_schema": csv_schema,
            "torch_available": bool(_HAS_TORCH),
            "ready": json_ok or csv_ok,
        }
    )


def _deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def patch_ml_session(
    memory: Any,
    *,
    model_choice: Optional[str] = None,
    ml_model_config: Optional[Dict[str, Any]] = None,
    auto_ml_after_curve_fitting: Optional[bool] = None,
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    composition_path: Optional[str] = None,
) -> Dict[str, Any]:
    if model_choice is not None:
        memory.set_var("optimization_model_choice", model_choice)
        memory.set_var("workflow_ml_model_choice", model_choice)
    if ml_model_config is not None:
        existing = memory.get_var("ml_model_config")
        if not isinstance(existing, dict):
            existing = dict(DEFAULT_ML_CONFIG)
        else:
            existing = _deep_merge_dict(dict(DEFAULT_ML_CONFIG), existing)
        memory.set_var(
            "ml_model_config",
            _deep_merge_dict(existing, ml_model_config),
        )
    if auto_ml_after_curve_fitting is not None:
        memory.set_var("auto_ml_after_curve_fitting", auto_ml_after_curve_fitting)
    if json_path is not None:
        resolved = resolve_results_path(json_path)
        memory.set_var("ml_auto_json_path", str(resolved) if resolved else json_path)
    if csv_path is not None:
        resolved = resolve_results_path(csv_path)
        memory.set_var("ml_auto_csv_path", str(resolved) if resolved else csv_path)
    if composition_path is not None:
        resolved = resolve_results_path(composition_path)
        memory.set_var(
            "ml_auto_composition_path",
            str(resolved) if resolved else composition_path,
        )
    return get_ml_session(memory)


def run_ml_automation(memory: Any, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from app.tools.ml_automation import run_automated_ml_model

    payload = payload or {}
    sync_ml_paths_from_curve_fitting(memory)
    session = get_ml_session(memory)
    model_choice = payload.get("model_choice") or session["model_choice"]
    reasoning_context = (
        str(payload.get("reasoning_context") or "")
        if "reasoning_context" in payload
        else reasoning_context_for_stage(memory, "ml_models")
    )
    reasoning_guard = {
        "context_applied": bool(reasoning_context),
        "scope": "model_selection_and_validation",
        "model_choice_preserved": True,
        "model_choice": str(model_choice),
        "numeric_data_modified": False,
    }
    memory.set_var("ml_reasoning_guard", reasoning_guard)
    def _resolve_file(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        resolved = resolve_results_path(str(path))
        if resolved:
            return str(resolved)
        p = Path(str(path))
        return str(p.resolve()) if p.is_file() else None

    json_path = _resolve_file(
        payload.get("json_path") or payload.get("results_json") or session.get("json_path")
    )
    csv_path = _resolve_file(
        payload.get("csv_path") or payload.get("results_csv") or session.get("csv_path")
    )
    composition_csv = _resolve_file(
        payload.get("composition_path")
        or payload.get("composition_csv")
        or session.get("composition_path")
    )

    if not json_path and not csv_path:
        sync = sync_ml_paths_from_curve_fitting(memory)
        json_path = sync.get("json_path")
        csv_path = sync.get("csv_path")
        composition_csv = composition_csv or sync.get("composition_path")

    raw_cfg = memory.get_var("ml_model_config")
    if not isinstance(raw_cfg, dict):
        cfg = dict(DEFAULT_ML_CONFIG)
    else:
        cfg = _deep_merge_dict(dict(DEFAULT_ML_CONFIG), raw_cfg)

    if model_choice == MODEL_DUAL_TORCH_GP and not csv_path:
        return to_jsonable(
            {
                "status": "error",
                "message": "Select or export a peak CSV before running dual-objective GP.",
                "result": {"success": False, "error": "Missing peak CSV."},
                "reasoning_guard": reasoning_guard,
                "session": get_ml_session(memory),
            }
        )

    if model_choice == MODEL_MONTE_CARLO_TREE:
        mc_cfg = cfg.get("monte_carlo_tree") if isinstance(cfg.get("monte_carlo_tree"), dict) else {}
        if not (mc_cfg.get("repo_path") or os.environ.get("MONTE_CARLO_REPO_PATH")):
            return to_jsonable(
                {
                    "status": "error",
                    "message": "Configure Monte Carlo Decision Tree repo path before running.",
                    "result": {"success": False, "error": "Missing monte_carlo_tree.repo_path"},
                    "reasoning_guard": reasoning_guard,
                    "session": get_ml_session(memory),
                }
            )

    result = run_automated_ml_model(
        model_choice=str(model_choice),
        json_path=json_path,
        csv_path=csv_path or None,
        composition_csv=composition_csv,
        auto_config=cfg,
    )

    if isinstance(result, dict) and result.get("success"):
        if model_choice in (MODEL_SINGLE_GP, MODEL_DUAL_TORCH_GP):
            memory.set_var("gp_results", result)
            memory.set_var("gp_results_available", True)
        elif model_choice == MODEL_MONTE_CARLO_TREE:
            memory.set_var("monte_carlo_results", result)
        if json_path:
            memory.set_var("ml_auto_json_path", json_path)
        if csv_path:
            memory.set_var("ml_auto_csv_path", csv_path)
        if composition_csv:
            memory.set_var("ml_auto_composition_path", composition_csv)

    return to_jsonable(
        {
            "status": "success" if result.get("success") else "error",
            "message": result.get("error") or "ML automation finished.",
            "result": result,
            "reasoning_guard": reasoning_guard,
            "session": get_ml_session(memory),
        }
    )
