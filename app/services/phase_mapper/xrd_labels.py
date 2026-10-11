"""XRD label loading and target definitions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import normalize_well

BINARY_TARGETS = [
    "xrd_clean_3d_perovskite",
    "xrd_mixed_phase",
    "xrd_secondary_phase_present",
    "xrd_PbX2_like_impurity_present",
    "xrd_RbPbI3_present",
    "xrd_DMAPbI3_present",
    "xrd_delta_phase_present",
    "xrd_poly_hexagonal_phase_present",
    "xrd_PbI2_present",
    "xrd_3d_phase_segregation_present",
    "xrd_peak_splitting_present",
    "xrd_primary_peak_splitting_present",
    "xrd_secondary_peak_splitting_present",
    "xrd_tertiary_peak_splitting_present",
    "xrd_high_order_peak_splitting_present",
    "xrd_200_peak_splitting_present",
    "xrd_octahedral_tilting_indicator",
]

REGRESSION_TARGETS = [
    "xrd_homogeneity_score",
    "xrd_primary_peak_position_deg_2theta",
    "xrd_secondary_peak_position_deg_2theta",
    "xrd_tertiary_peak_position_deg_2theta",
]

LEGACY_NUMERIC_COLUMNS = [
    "xrd_peak_broadening_score",
    "xrd_high_order_peak_broadening_score",
    "xrd_200_peak_broadening_score",
    "xrd_3d_peak_shift_score",
    "xrd_high_order_peak_shift_score",
    "xrd_200_peak_position_deg_2theta",
    "xrd_200_peak_fwhm_deg",
    "xrd_200_peak_count",
    "xrd_primary_peak_fwhm_deg",
    "xrd_secondary_peak_fwhm_deg",
    "xrd_tertiary_peak_fwhm_deg",
    "xrd_primary_peak_count",
    "xrd_secondary_peak_count",
    "xrd_tertiary_peak_count",
    "xrd_additional_peak_count_excluding_primary_3d",
    "xrd_extra_peak_count",
]

TEXT_LABEL_COLUMNS = [
    "xrd_key_2theta_regions_assessed",
    "xrd_primary_peak_label",
    "xrd_primary_peak_visual_shape",
    "xrd_secondary_peak_label",
    "xrd_secondary_peak_visual_shape",
    "xrd_tertiary_peak_label",
    "xrd_tertiary_peak_visual_shape",
    "xrd_extra_or_impurity_peak_2theta_list",
    "xrd_missing_or_suppressed_peak_2theta_list",
    "xrd_visual_phase_call",
    "xrd_optional_plane_or_hkl_notes",
    "xrd_reference_pattern_or_sample",
    "xrd_additional_peak_positions_excluding_primary_3d_deg",
    "xrd_extra_peak_positions_deg",
    "xrd_reference_peak_matches",
    "xrd_candidate_reference_phases",
]

LABEL_COLUMNS = BINARY_TARGETS + REGRESSION_TARGETS + [
    "xrd_assignment_confidence",
    "xrd_dominant_phase_label",
    "xrd_notes",
] + TEXT_LABEL_COLUMNS + LEGACY_NUMERIC_COLUMNS

CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.65, "low": 0.35}
VISUAL_PHASE_TO_BINARY = {
    "clean_3d": {"xrd_clean_3d_perovskite": 1, "xrd_mixed_phase": 0, "xrd_secondary_phase_present": 0, "xrd_3d_phase_segregation_present": 0},
    "mixed_phase": {"xrd_clean_3d_perovskite": 0, "xrd_mixed_phase": 1},
    "secondary_phase": {"xrd_clean_3d_perovskite": 0, "xrd_secondary_phase_present": 1},
    "phase_segregation": {"xrd_clean_3d_perovskite": 0, "xrd_3d_phase_segregation_present": 1},
}
SPLIT_SHAPES = {"split/doublet", "multiple"}


def _read_label_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        for preferred in ["XRD_labels_template", "XRD_labels", "Labels", "labels"]:
            if preferred in sheets:
                return sheets[preferred]
        for sheet in sheets.values():
            if "well" in sheet.columns:
                return sheet
        return next(iter(sheets.values()))
    return pd.read_csv(path)


def _fill_blank_numeric(df: pd.DataFrame, col: str, values: pd.Series) -> None:
    if col not in df.columns:
        df[col] = np.nan
    mask = df[col].isna() & values.notna()
    df.loc[mask, col] = values.loc[mask]


def _apply_visual_label_shortcuts(df: pd.DataFrame) -> None:
    if "xrd_visual_phase_call" in df.columns:
        visual_calls = df["xrd_visual_phase_call"].fillna("").astype(str).str.strip().str.lower()
        for phase_call, assignments in VISUAL_PHASE_TO_BINARY.items():
            rows = visual_calls == phase_call
            for col, value in assignments.items():
                _fill_blank_numeric(df, col, pd.Series(np.where(rows, value, np.nan), index=df.index))

    split_cols = [
        ("xrd_primary_peak_visual_shape", "xrd_primary_peak_splitting_present"),
        ("xrd_secondary_peak_visual_shape", "xrd_secondary_peak_splitting_present"),
        ("xrd_tertiary_peak_visual_shape", "xrd_tertiary_peak_splitting_present"),
    ]
    any_split = pd.Series(False, index=df.index)
    for shape_col, target_col in split_cols:
        if shape_col not in df.columns:
            continue
        shapes = df[shape_col].fillna("").astype(str).str.strip().str.lower()
        split_values = shapes.isin(SPLIT_SHAPES).astype(float)
        split_values.loc[shapes == ""] = np.nan
        _fill_blank_numeric(df, target_col, split_values)
        any_split |= shapes.isin(SPLIT_SHAPES)
    _fill_blank_numeric(df, "xrd_peak_splitting_present", pd.Series(np.where(any_split, 1, np.nan), index=df.index))

    if "xrd_extra_or_impurity_peak_2theta_list" in df.columns:
        extra_peaks = df["xrd_extra_or_impurity_peak_2theta_list"].fillna("").astype(str).str.strip()
        has_extra_peaks = extra_peaks.ne("")
        _fill_blank_numeric(df, "xrd_secondary_phase_present", pd.Series(np.where(has_extra_peaks, 1, np.nan), index=df.index))


def load_xrd_labels(path: str | Path | None) -> pd.DataFrame:
    """Load optional XRD labels; returns an empty compatible table if absent."""
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["well"] + LABEL_COLUMNS)
    path = Path(path)
    df = _read_label_file(path)
    if "well" not in df.columns:
        raise ValueError("XRD label file must contain a 'well' column.")
    df["well"] = df["well"].map(normalize_well)
    _apply_visual_label_shortcuts(df)
    for col in BINARY_TARGETS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in REGRESSION_TARGETS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in LEGACY_NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "xrd_assignment_confidence" not in df.columns:
        df["xrd_assignment_confidence"] = "medium"
    df["xrd_assignment_confidence"] = df["xrd_assignment_confidence"].fillna("medium").astype(str).str.lower()
    if "xrd_label_source" in df.columns:
        source = df["xrd_label_source"].fillna("").astype(str).str.lower()
        manual = source.str.startswith("manual") | source.str.contains("reviewed")
        df.loc[manual, "xrd_assignment_confidence"] = "high"
    df["xrd_sample_weight"] = df["xrd_assignment_confidence"].map(CONFIDENCE_WEIGHTS).fillna(0.65)
    for col in ["xrd_dominant_phase_label", "xrd_notes"] + TEXT_LABEL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df
