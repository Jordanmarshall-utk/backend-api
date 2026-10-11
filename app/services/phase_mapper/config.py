"""Configuration loading and defaults."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "plate_reader_file": "",
        "plate_reader_files": [],
        "composition_map": "data/composition_map_template.csv",
        "xrd_labels": "",
        "output_dir": "outputs",
    },
    "plate": {"expected_wells": 96},
    "read_assignments": {},
    "read_type_keywords": {"absorbance": ["abs", "od"], "pl": ["pl", "emission"]},
    "thresholds": {
        "pl_low_signal": 100.0,
        "pl_saturation_fraction": 0.98,
        "pl_min_peak_prominence_fraction": 0.08,
        "absorbance_low_signal": 0.01,
        "absorbance_flat_std": 0.003,
        "abs_min_peak_prominence_fraction": 0.05,
        "broad_pl_fwhm_nm": 60.0,
        "broad_abs_edge_width_nm": 80.0,
        "neighbor_count": 5,
        "min_xrd_labels_for_training": 5,
        "exclude_low_confidence_xrd": False,
        "n_pca_components": 5,
    },
    "modeling": {"estimator": "extra_trees", "n_estimators": 300, "random_state": 42, "test_size": 0.3},
    "xrd_processing": {
        "reference_rules_file": "data/xrd_reference_rules_template.csv",
        "peak_prominence_fraction": 0.08,
        "secondary_peak_relative_height_min": 0.12,
        "secondary_extra_peak_count_min": 1,
        "auto_flag_secondary_from_extra_peaks": False,
        "practical_clean_extra_peak_count_max": 2,
        "practical_mixed_extra_peak_count_min": 3,
        "practical_require_primary_and_high_order": True,
        "clean_perovskite_min_peak_window_hits": 2,
        "primary_3d_peak_window_deg": [13.6, 14.8],
        "high_order_min_deg": 27.0,
        "high_order_max_deg": 30.0,
        "high_order_relative_height_min": 0.12,
        "high_order_split_peak_count": 2,
        "high_order_broad_fwhm_deg": 0.60,
        "high_order_peak_shift_threshold_deg": 0.20,
        "high_order_reference_peak_deg": None,
        "perovskite_peak_windows_deg": [(13.6, 14.8), (19.4, 20.8), (23.8, 25.2), (27.0, 30.0)],
        "reference_peak_tolerance_deg": 0.30,
        "reference_phases_deg": {},
    },
    "active_learning": {
        "recommend_n": 16,
        "diversity_lambda": 0.15,
        "weights": {
            "model_uncertainty": 0.30,
            "optical_heterogeneity": 0.15,
            "pl_abs_disagreement": 0.15,
            "optical_phase_complexity": 0.20,
            "composition_distance": 0.10,
            "optical_neighbor_outlier": 0.10,
        },
    },
}


def _merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULT_CONFIG, user_cfg)
    cfg["_config_dir"] = str(path.resolve().parent)
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg.get("_config_dir", ".")).joinpath(path)
