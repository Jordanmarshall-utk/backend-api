"""Reusable experiment profiles and generated per-well composition metadata."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .utils import COLS


DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent.parent / "data" / "experiment_profiles.yaml"


def load_experiment_profiles(path: str | Path = DEFAULT_PROFILES_PATH) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("profiles", raw)


def profile_options(path: str | Path = DEFAULT_PROFILES_PATH) -> dict[str, str]:
    profiles = load_experiment_profiles(path)
    return {key: str(value.get("display_name", key)) for key, value in profiles.items()}


def get_experiment_profile(profile_id: str, path: str | Path = DEFAULT_PROFILES_PATH) -> dict[str, Any]:
    profiles = load_experiment_profiles(path)
    if profile_id not in profiles:
        raise KeyError(f"Unknown experiment profile: {profile_id}")
    profile = deepcopy(profiles[profile_id])
    profile["profile_id"] = profile_id
    return profile


def identify_experiment_profile(
    candidate_names: list[str],
    path: str | Path = DEFAULT_PROFILES_PATH,
) -> dict[str, Any] | None:
    """Suggest a profile from notebook-backed workbook/library filename patterns."""
    profiles = load_experiment_profiles(path)
    normalized_names = [" ".join(str(name).lower().replace("_", " ").split()) for name in candidate_names if name]
    scores = []
    for profile_id, profile in profiles.items():
        matched = []
        for pattern in profile.get("filename_patterns", []) or []:
            normalized_pattern = " ".join(str(pattern).lower().replace("_", " ").split())
            if normalized_pattern and any(normalized_pattern in name for name in normalized_names):
                matched.append(str(pattern))
        if matched:
            scores.append(
                {
                    "profile_id": profile_id,
                    "display_name": str(profile.get("display_name", profile_id)),
                    "score": len(matched),
                    "matched_patterns": matched,
                    "source_notebook": str(profile.get("source_notebook", "")),
                }
            )
    if not scores:
        return None
    scores.sort(key=lambda item: (item["score"], len(max(item["matched_patterns"], key=len))), reverse=True)
    best = scores[0]
    tied = [item for item in scores if item["score"] == best["score"]]
    best["confidence"] = "high" if len(tied) == 1 else "ambiguous"
    return best


def _identity_column(prefix: str, value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
    return f"{prefix}_is_{safe}"


def build_profile_composition_table(profile_id: str, plate_id: str = "") -> pd.DataFrame:
    """Generate one model-ready metadata row per active well in an experiment profile."""
    profile = get_experiment_profile(profile_id)
    rows_to_use = [str(row).upper() for row in profile.get("rows", list("ABCDEFGH"))]
    additive_by_row = {str(k).upper(): float(v) for k, v in (profile.get("additive_percent_by_row") or {}).items()}
    host_stock_m = float(profile.get("host_stock_molarity", 0.3))
    total_volume_ul = float(profile.get("total_volume_ul", 50.0))
    additive_volume_ul = float(profile.get("additive_volume_ul", 5.0))
    two_d_precursor = str(profile["two_d_precursor"])
    three_d_precursor = str(profile.get("three_d_precursor", "FAPbI3"))
    additive = str(profile["additive"])
    all_profiles = load_experiment_profiles()
    two_d_identity_columns = {
        _identity_column("two_d_precursor", str(item["two_d_precursor"]))
        for item in all_profiles.values()
    }
    additive_identity_columns = {
        _identity_column("additive", str(item["additive"]))
        for item in all_profiles.values()
    }
    records = []

    for row in rows_to_use:
        additive_percent = float(additive_by_row.get(row, 0.0))
        row_additive_volume = additive_volume_ul if additive_percent > 0 else 0.0
        host_total_volume = total_volume_ul - row_additive_volume
        additive_final_m = host_stock_m * additive_percent / 100.0
        additive_stock_m = additive_final_m * total_volume_ul / row_additive_volume if row_additive_volume > 0 else 0.0

        for col in COLS:
            two_d_fraction = (12 - col) / 11.0
            fapbi3_fraction = 1.0 - two_d_fraction
            record = {
                "profile_id": profile_id,
                "plate_id": plate_id,
                "well": f"{row}{col}",
                "row": row,
                "col": col,
                "two_d_precursor": two_d_precursor,
                "three_d_precursor": three_d_precursor,
                "additive": additive if additive_percent > 0 else "none",
                "two_d_fraction": two_d_fraction,
                "fapbi3_fraction": fapbi3_fraction,
                "additive_percent": additive_percent,
                "additive_fraction": additive_percent / 100.0,
                "additive_final_molarity": additive_final_m,
                "additive_stock_molarity": additive_stock_m,
                "host_total_volume_ul": host_total_volume,
                "additive_volume_ul": row_additive_volume,
                "total_volume_ul": total_volume_ul,
                "two_d_volume_ul": host_total_volume * two_d_fraction,
                "fapbi3_volume_ul": host_total_volume * fapbi3_fraction,
                "two_d_stock_molarity": host_stock_m,
                "fapbi3_stock_molarity": host_stock_m,
                "two_d_spacer_stoichiometry": float(profile.get("two_d_spacer_stoichiometry", 1.0)),
                "composition_family_is_quasi_2d_additive": 1.0,
                _identity_column("two_d_precursor", two_d_precursor): 1.0,
                _identity_column("additive", additive): float(additive_percent > 0),
                "is_control": str(additive_percent == 0),
                "replicate_group": "",
                "notes": (
                    f"{two_d_precursor}:{two_d_fraction:.4f}; "
                    f"{three_d_precursor}:{fapbi3_fraction:.4f}; "
                    f"{additive}:{additive_percent:g}%"
                ),
            }
            for identity_col in two_d_identity_columns | additive_identity_columns:
                record.setdefault(identity_col, 0.0)
            records.append(record)
    return pd.DataFrame(records)
