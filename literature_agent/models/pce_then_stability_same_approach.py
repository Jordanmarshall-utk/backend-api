#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCE -> Stability sequential modeling pipeline for the Perovskite Database.

Purpose
-------
This script keeps the modeling direction as:

    1) Predict PCE first using leakage-safe design/process/interface/device-context inputs.
    2) Use the same general feature-processing and grouped-CV approach to predict stability.
    3) For stability, compare:
       - design_only: stability from design/process/interface fields only
       - design_plus_predicted_pce: stability from design inputs + OOF predicted PCE
       - design_plus_measured_initial_performance: post-fabrication triage using measured initial JV/Stabilised metrics
       - physical_condition_features: empirical targets with explicit harsh-condition acceleration features
       - condition_normalized_hybrid: reference-condition degradation targets derived from a semi-empirical physical layer

This avoids using actual measured PCE as a pre-experiment stability feature unless explicitly in the
post-fabrication triage mode.

Colab usage
-----------
!pip install -q numpy pandas scipy scikit-learn xgboost matplotlib openpyxl
!python /content/pce_then_stability_same_approach.py --csv /content/Perovskite_database_content_all_data.csv

Outputs
-------
artifacts_pce_then_stability/
    pce/
    stability/
    model_comparison_pce_then_stability.csv
    README.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    INPUT_CSV: str = "Perovskite_database_content_all_data.csv"
    OUTPUT_DIR: str = "artifacts_pce_then_stability"

    # PCE target
    PCE_TARGET: str = "JV_default_PCE"
    PCE_FALLBACK_TARGET: str = "JV_reverse_scan_PCE"
    PCE_MIN: float = 5.0
    PCE_MAX: float = 30.0

    # Row completeness
    APPLY_ROW_COMPLETENESS_FILTER: bool = True
    MIN_ROW_COMPLETENESS: float = 0.20

    # CV/modeling
    N_SPLITS: int = 5
    RANDOM_STATE: int = 42
    MAX_CATEGORICAL_LEVELS: int = 80
    MIN_ROWS_REGRESSION: int = 300
    MIN_ROWS_CLASSIFICATION: int = 300
    MIN_POSITIVE_CLASS: int = 40
    N_ESTIMATORS: int = 650

    # PCE prediction feature mode
    # design_only excludes all direct JV/EQE/Stability/Outdoor/Stabilised outputs.
    PCE_FEATURE_MODE: str = "design_only"

    # Stability modes
    RUN_STABILITY_DESIGN_ONLY: bool = True
    RUN_STABILITY_PLUS_PREDICTED_PCE: bool = True
    RUN_STABILITY_PLUS_MEASURED_INITIAL_PERFORMANCE: bool = True

    # Stability target filters
    MIN_STABILITY_EXPOSURE_H_FOR_RETENTION_CLASS: float = 100.0
    STABILITY_RETENTION_THRESHOLD: float = 80.0

    # Physics-informed harsh-condition normalization. These are semi-empirical
    # acceleration assumptions, not a fully mechanistic degradation model.
    ENABLE_PHYSICAL_STABILITY_LAYER: bool = True
    PHYS_REF_TEMP_K: float = 300.0
    PHYS_REF_RH_PERCENT: float = 20.0
    PHYS_REF_LIGHT_SUN: float = 1.0
    PHYS_EA_EV: float = 0.60
    PHYS_RH_EXPONENT: float = 1.0
    PHYS_LIGHT_EXPONENT: float = 0.70
    PHYS_BETA: float = 1.0
    PHYS_MIN_EXPOSURE_H: float = 1.0
    PHYS_MIN_RETENTION_FRAC: float = 0.01
    PHYS_MAX_RETENTION_FRAC: float = 1.20

    # Interpretation output limits
    TOP_N_FEATURE_IMPORTANCE: int = 80

CFG = Config()

MISSING_TOKENS = {
    "", " ", "na", "n/a", "n.a.", "none", "null", "nan", "unknown",
    "-", "--", "—", "not available", "not reported", "nr", "n.r.", "not applicable",
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_name(s: str) -> str:
    s = str(s)
    s = s.replace("%", " percent ")
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def norm_missing(v):
    if v is None:
        return np.nan
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return np.nan
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in MISSING_TOKENS:
            return np.nan
    return v


def dataframe_map(df: pd.DataFrame, func) -> pd.DataFrame:
    try:
        return df.map(func)
    except Exception:
        return df.applymap(func)


def robust_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-16", "latin1", "cp1252"]
    last_error = None
    kwargs = dict(kwargs)
    kwargs.pop("low_memory", None)  # not allowed with python engine
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip", **kwargs)
            print(f"Loaded CSV with encoding={enc}, engine=python, on_bad_lines=skip: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            return df
        except Exception as e:
            last_error = e
            print(f"Failed encoding={enc}: {type(e).__name__}: {e}")
    raise RuntimeError(f"Could not read CSV. Last error: {last_error}")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = dataframe_map(df, norm_missing)
    return df


def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
        errors="coerce",
    )


TEXT_LIKE_FEATURE_HINTS = (
    "stack", "sequence", "architecture", "formula", "composition", "material",
    "compound", "solvent", "additive", "cation", "anion", "halide", "dopant",
    "substrate", "etl", "htl", "contact", "electrode", "method", "process",
    "condition", "atmosphere", "reference", "doi", "journal", "title",
)


def numeric_like_ratio(s: pd.Series) -> float:
    """Return the fraction of non-missing values that look numeric as whole values."""
    raw = s.dropna().astype(str).str.strip()
    raw = raw[~raw.str.lower().isin({"", "nan", "none", "null", "missing"})]
    if raw.empty:
        return 0.0

    # Accept numbers, percentages, and simple numeric values with units
    # ("100 h", "25 C", ">20%", "1 sun"). Do not accept strings where the
    # number is embedded inside chemical/device text, e.g. "TiO2" or stacks.
    numeric_pattern = re.compile(
        r"^\s*[<>~=≈≤≥+-]*\s*"
        r"[-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?"
        r"\s*(?:%|[A-Za-z°µμ/^\-]+(?:\s*[A-Za-z°µμ/^\-]+)?)?\s*$"
    )
    matches = raw.map(lambda x: bool(numeric_pattern.match(x)) and bool(re.search(r"\d", x)))
    return float(matches.mean())


def is_text_like_feature_name(name: str) -> bool:
    norm = normalize_name(name)
    return any(hint in norm for hint in TEXT_LIKE_FEATURE_HINTS)


def coerce_numeric_frame(X):
    if isinstance(X, pd.DataFrame):
        return X.apply(to_numeric_series)
    return pd.DataFrame(X).apply(to_numeric_series).to_numpy()


def find_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_group_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Ref_DOI_number", "Ref_DOI", "DOI", "doi", "Reference_DOI", "Ref_doi", "Ref_ID", "Ref_ID_temp",
    ]
    col = find_first_existing(df, candidates)
    if col:
        return col
    doi_like = [c for c in df.columns if "doi" in normalize_name(c)]
    return doi_like[0] if doi_like else None


def safe_json_dump(obj, path: Path) -> None:
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        return str(x)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=conv)


PHYSICAL_TARGETS = [
    "target_phys_log_k_ref",
    "target_phys_log1p_T80_ref_hours",
    "target_phys_log_k_obs",
]

PHYSICAL_STRESS_FEATURES = [
    "PHYS_AF_temperature",
    "PHYS_AF_humidity",
    "PHYS_AF_light",
    "PHYS_AF_total",
    "PHYS_log_AF_total",
    "PHYS_missing_temperature_flag",
    "PHYS_missing_humidity_flag",
    "PHYS_missing_light_flag",
    "PHYS_condition_known_score",
]


# =============================================================================
# LEAKAGE / FEATURE RULES
# =============================================================================

def is_admin_or_provenance(col: str) -> bool:
    n = normalize_name(col)
    bad = [
        "ref_", "doi", "journal", "publication_date", "lead_author", "original_filename",
        "name_of_person_entering", "data_entered_by", "free_text_comment", "license", "url", "webpage",
        "raw_data", "filename", "comment",
    ]
    return any(b in n for b in bad)


def is_direct_performance_output(col: str) -> bool:
    raw = str(col)
    n = normalize_name(col)
    if raw.startswith("JV_") or raw.startswith("EQE_") or raw.startswith("Stabilised_performance"):
        return True
    perf_words = ["pce", "voc", "jsc", "fill_factor", "ff", "efficiency", "hysteresis", "mppt"]
    return any(w in n for w in perf_words) and (raw.startswith("JV") or raw.startswith("EQE") or raw.startswith("Stabilised"))


def is_stability_output(col: str) -> bool:
    raw = str(col)
    return raw.startswith("Stability_") or raw.startswith("Outdoor_")


def is_forbidden_for_pce(col: str, target_col: str) -> bool:
    if col == target_col:
        return False
    if is_direct_performance_output(col):
        return True
    if is_stability_output(col):
        return True
    # These are often proxies or downstream context.
    n = normalize_name(col)
    proxy = ["certified", "champion", "stabilized", "stabilised", "mppt", "reverse_scan", "forward_scan"]
    return any(p in n for p in proxy)


def is_forbidden_for_stability_design_only(col: str) -> bool:
    # This model-generated column is attached before the shared stability
    # inventory is constructed. Keep it out of the common baseline so it can
    # be added explicitly only by ``design_plus_predicted_pce``.
    if normalize_name(col) == "pce_model_oof_predicted_pce":
        return True
    # Exclude measured performance and stability outputs from the shared
    # non-target baseline. Retrospective modes add their allowed inputs below.
    if is_direct_performance_output(col):
        return True
    if is_stability_output(col):
        return True
    n = normalize_name(col)
    proxy = ["certified", "champion", "mppt", "reverse_scan", "forward_scan"]
    return any(p in n for p in proxy)


def allowed_initial_performance_cols(df: pd.DataFrame) -> List[str]:
    # Post-fabrication triage mode. Do not include end/stability results, but allow initial JV/Stabilised metrics.
    candidates = [
        "JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",
        "JV_reverse_scan_PCE", "JV_forward_scan_PCE",
        "Stabilised_performance_PCE", "Stabilised_performance_time",
    ]
    return [c for c in candidates if c in df.columns]


def build_base_feature_columns(df: pd.DataFrame, target: str, purpose: str, extra_exclude: Optional[Sequence[str]] = None) -> List[str]:
    extra_exclude = set(extra_exclude or [])
    cols = []
    for c in df.columns:
        if c == target or c in extra_exclude:
            continue
        if is_admin_or_provenance(c):
            continue
        if purpose == "pce":
            if is_forbidden_for_pce(c, target):
                continue
        elif purpose == "stability_design":
            if is_forbidden_for_stability_design_only(c):
                continue
        else:
            raise ValueError(f"Unknown purpose: {purpose}")
        cols.append(c)
    return cols


def infer_feature_blocks(columns: Sequence[str]) -> Dict[str, List[str]]:
    blocks = {"chemistry_architecture": [], "process": [], "interfaces": [], "device_context": [], "other": []}
    for c in columns:
        n = normalize_name(c)
        if any(k in n for k in ["perovskite_composition", "perovskite_a", "perovskite_b", "perovskite_c", "composition", "additive", "dopant", "bandgap", "dimensionality", "crystal_structure", "cell_architecture", "nip", "pin"]):
            blocks["chemistry_architecture"].append(c)
        elif any(k in n for k in ["solvent", "anneal", "deposition", "spin", "temperature", "time", "atmosphere", "humidity", "pressure", "quenching", "antisolvent", "concentration", "precursor", "storage", "aging"]):
            blocks["process"].append(c)
        elif any(k in n for k in ["etl", "htl", "electron_transport", "hole_transport", "contact", "backcontact", "sam", "c60", "spiro", "pedot", "substrate", "electrode", "interlayer"]):
            blocks["interfaces"].append(c)
        elif any(k in n for k in ["cell_area", "area", "flexible", "semitransparent", "module", "number_of_cells", "encapsulation", "device", "stack"]):
            blocks["device_context"].append(c)
        else:
            blocks["other"].append(c)
    return blocks


# =============================================================================
# ADDITIVE/PASSIVATOR/SPACER DESCRIPTOR FEATURES (NON-POLARIS)
# =============================================================================

KNOWN_ADDITIVES = {
    "bai": {"mw": 201.09, "chain_length": 4, "aromatic": 0, "iodide": 1, "chloride": 0, "long_chain": 0, "rp_like": 1},
    "butylammonium iodide": {"mw": 201.09, "chain_length": 4, "aromatic": 0, "iodide": 1, "chloride": 0, "long_chain": 0, "rp_like": 1},
    "oai": {"mw": 257.16, "chain_length": 8, "aromatic": 0, "iodide": 1, "chloride": 0, "long_chain": 1, "rp_like": 1},
    "octylammonium iodide": {"mw": 257.16, "chain_length": 8, "aromatic": 0, "iodide": 1, "chloride": 0, "long_chain": 1, "rp_like": 1},
    "peai": {"mw": 249.09, "chain_length": 2, "aromatic": 1, "iodide": 1, "chloride": 0, "long_chain": 0, "rp_like": 1},
    "phenethylammonium iodide": {"mw": 249.09, "chain_length": 2, "aromatic": 1, "iodide": 1, "chloride": 0, "long_chain": 0, "rp_like": 1},
    "pmai": {"mw": 235.07, "chain_length": 1, "aromatic": 1, "iodide": 1, "chloride": 0, "long_chain": 0, "rp_like": 1},
    "macl": {"mw": 67.52, "chain_length": 1, "aromatic": 0, "iodide": 0, "chloride": 1, "long_chain": 0, "rp_like": 0},
    "mai": {"mw": 158.97, "chain_length": 1, "aromatic": 0, "iodide": 1, "chloride": 0, "long_chain": 0, "rp_like": 0},
    "fai": {"mw": 171.97, "chain_length": 0, "aromatic": 0, "iodide": 1, "chloride": 0, "long_chain": 0, "rp_like": 0},
    "oami": {"mw": 381.38, "chain_length": 18, "aromatic": 0, "iodide": 1, "chloride": 0, "long_chain": 1, "rp_like": 1},
    "oleylammonium iodide": {"mw": 381.38, "chain_length": 18, "aromatic": 0, "iodide": 1, "chloride": 0, "long_chain": 1, "rp_like": 1},
}


def add_additive_descriptor_features(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [c for c in df.columns if any(k in normalize_name(c) for k in ["additive", "passivat", "spacer", "cation", "organic"])]
    text = pd.Series("", index=df.index, dtype=object)
    for c in cols:
        text = text.str.cat(df[c].fillna("").astype(str), sep=" ; ")
    low = text.str.lower()

    # Binary known additive flags
    for key in ["bai", "oai", "peai", "pmai", "macl", "mai", "fai", "oami"]:
        pattern = key
        if key == "mai":
            pattern = r"\bmai\b|methylammonium iodide"
        elif key == "fai":
            pattern = r"\bfai\b|formamidinium iodide"
        else:
            pattern = r"\b" + re.escape(key) + r"\b"
        df[f"ADD_DESC_has_{key}"] = low.str.contains(pattern, regex=True, na=False).astype(int)

    # Aggregate descriptors for known additives found in text
    desc_names = ["mw", "chain_length", "aromatic", "iodide", "chloride", "long_chain", "rp_like"]
    sums = {d: np.zeros(len(df), dtype=float) for d in desc_names}
    counts = np.zeros(len(df), dtype=float)
    token_count_rows = []

    for token, desc in KNOWN_ADDITIVES.items():
        mask = low.str.contains(re.escape(token), regex=True, na=False).to_numpy()
        n = int(mask.sum())
        if n:
            token_count_rows.append({"token": token, "matched_rows": n, **desc})
        counts += mask.astype(float)
        for d in desc_names:
            sums[d] += mask.astype(float) * float(desc[d])

    df["ADD_DESC_known_match_count"] = counts
    df["ADD_DESC_has_known_additive"] = (counts > 0).astype(int)
    for d in desc_names:
        df[f"ADD_DESC_mean_{d}"] = np.where(counts > 0, sums[d] / np.maximum(counts, 1), 0.0)
        df[f"ADD_DESC_sum_{d}"] = sums[d]

    # General text-based features
    df["ADD_DESC_text_has_additive_info"] = (text.str.len() > 5).astype(int)
    df["ADD_DESC_text_length"] = text.str.len().astype(float)
    df["ADD_DESC_text_mentions_iodide"] = low.str.contains("iodide|\bi\b", regex=True, na=False).astype(int)
    df["ADD_DESC_text_mentions_chloride"] = low.str.contains("chloride|\bcl\b", regex=True, na=False).astype(int)
    df["ADD_DESC_text_mentions_bromide"] = low.str.contains("bromide|\bbr\b", regex=True, na=False).astype(int)
    df["ADD_DESC_text_mentions_aromatic"] = low.str.contains("phenyl|benz|pyrid|thiophen|naphth|aromatic", regex=True, na=False).astype(int)

    pd.DataFrame(token_count_rows).sort_values("matched_rows", ascending=False).to_csv(out_dir / "additive_descriptor_token_counts.csv", index=False)
    return df


# =============================================================================
# TARGET CREATION
# =============================================================================

def prepare_pce_target(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    target = cfg.PCE_TARGET if cfg.PCE_TARGET in df.columns else cfg.PCE_FALLBACK_TARGET
    if target not in df.columns:
        raise ValueError(f"Could not find PCE target {cfg.PCE_TARGET} or fallback {cfg.PCE_FALLBACK_TARGET}")
    df[target] = to_numeric_series(df[target])
    before = len(df)
    df = df[df[target].notna()].copy()
    df = df[(df[target] >= cfg.PCE_MIN) & (df[target] <= cfg.PCE_MAX)].copy()
    print(f"PCE target filtering: {before:,} -> {len(df):,} rows using {target} in [{cfg.PCE_MIN}, {cfg.PCE_MAX}]")
    return df, target


def add_stability_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Make numeric copies where present.
    numeric_candidates = [
        "Stability_time_total_exposure",
        "Stability_PCE_initial_value",
        "Stability_PCE_end_of_experiment",
        "Stability_PCE_T80",
        "Stability_PCE_T95",
        "Stability_PCE_after_1000_h",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = to_numeric_series(df[c])

    if "Stability_time_total_exposure" in df.columns:
        df["target_stability_exposure_h"] = df["Stability_time_total_exposure"]
    else:
        df["target_stability_exposure_h"] = np.nan

    # Interpret end_of_experiment. If values mostly <= 1.5, convert fraction to percent.
    if "Stability_PCE_end_of_experiment" in df.columns:
        end = df["Stability_PCE_end_of_experiment"].copy()
        finite = end.dropna()
        if len(finite) and finite.quantile(0.95) <= 1.5:
            end = end * 100.0
        df["target_end_retention_pct"] = end
    else:
        df["target_end_retention_pct"] = np.nan

    if "Stability_PCE_initial_value" in df.columns and "Stability_PCE_end_of_experiment" in df.columns:
        init = df["Stability_PCE_initial_value"]
        end_raw = df["Stability_PCE_end_of_experiment"]
        # If end looks like PCE absolute and initial exists, derive retention; otherwise keep end-retention target above.
        abs_like = end_raw.dropna().quantile(0.95) > 2.0 if end_raw.notna().sum() else False
        if abs_like:
            derived = np.where((init > 0) & np.isfinite(init), (end_raw / init) * 100.0, np.nan)
            # Fill target only when reasonable.
            df["target_end_retention_pct"] = pd.Series(derived, index=df.index).where(pd.Series(derived, index=df.index).between(0, 150), df["target_end_retention_pct"])

    if "Stability_PCE_after_1000_h" in df.columns:
        after = df["Stability_PCE_after_1000_h"].copy()
        finite = after.dropna()
        if len(finite) and finite.quantile(0.95) <= 1.5:
            after = after * 100.0
        df["target_after_1000h_retention_pct"] = after
    else:
        df["target_after_1000h_retention_pct"] = np.nan

    if "Stability_PCE_T80" in df.columns:
        t80 = df["Stability_PCE_T80"]
        df["target_T80_hours"] = t80
        df["target_log1p_T80_hours"] = np.log1p(t80.where(t80 >= 0))
    else:
        df["target_T80_hours"] = np.nan
        df["target_log1p_T80_hours"] = np.nan

    if "Stability_PCE_T95" in df.columns:
        t95 = df["Stability_PCE_T95"]
        df["target_T95_hours"] = t95
        df["target_log1p_T95_hours"] = np.log1p(t95.where(t95 >= 0))
    else:
        df["target_T95_hours"] = np.nan
        df["target_log1p_T95_hours"] = np.nan

    # Classification labels.
    df["target_end_retention_ge80"] = np.where(df["target_end_retention_pct"].notna(), (df["target_end_retention_pct"] >= 80).astype(float), np.nan)
    df["target_end_retention_ge80_after_100h"] = np.where(
        df["target_end_retention_pct"].notna() & (df["target_stability_exposure_h"] >= 100),
        (df["target_end_retention_pct"] >= 80).astype(float),
        np.nan,
    )
    df["target_end_retention_ge80_after_500h"] = np.where(
        df["target_end_retention_pct"].notna() & (df["target_stability_exposure_h"] >= 500),
        (df["target_end_retention_pct"] >= 80).astype(float),
        np.nan,
    )
    df["target_T80_ge100h"] = np.where(df["target_T80_hours"].notna(), (df["target_T80_hours"] >= 100).astype(float), np.nan)
    df["target_T80_ge500h"] = np.where(df["target_T80_hours"].notna(), (df["target_T80_hours"] >= 500).astype(float), np.nan)
    df["target_after_1000h_ge80"] = np.where(df["target_after_1000h_retention_pct"].notna(), (df["target_after_1000h_retention_pct"] >= 80).astype(float), np.nan)

    target_cols = regression_targets() + classification_targets()
    rows = []
    for t in target_cols:
        s = df[t]
        row = {"target": t, "non_null": int(s.notna().sum()), "coverage_fraction": float(s.notna().mean())}
        if t in classification_targets():
            vc = s.dropna().value_counts()
            row["class_0"] = int(vc.get(0.0, 0))
            row["class_1"] = int(vc.get(1.0, 0))
        else:
            row["mean"] = float(s.mean(skipna=True)) if s.notna().sum() else np.nan
            row["median"] = float(s.median(skipna=True)) if s.notna().sum() else np.nan
        rows.append(row)
    return df, pd.DataFrame(rows)


def _parse_numeric_midpoint(v) -> float:
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) if np.isfinite(v) else np.nan
    s = str(v).strip().lower().replace(",", "")
    if not s or s in MISSING_TOKENS:
        return np.nan
    # Treat hyphens between two numbers as range separators without changing
    # genuinely negative values such as "-20 C".
    s = re.sub(r"(?<=\d)\s*[-\u2013\u2014]\s*(?=\d)", " to ", s)
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    nums = [float(x) for x in vals if x not in {"", ".", "+", "-"}]
    if not nums:
        return np.nan
    if "\u00b1" in s or "+/-" in s:
        return float(nums[0])
    # Use the midpoint for reported ranges such as "60-85 C" or "20 to 40%".
    return float(np.mean(nums[:2])) if len(nums) >= 2 else float(nums[0])


def _best_physical_column(df: pd.DataFrame, exact_names: Sequence[str], hints: Sequence[str]) -> Optional[str]:
    normalized = {normalize_name(c): c for c in df.columns}
    for name in exact_names:
        if normalize_name(name) in normalized:
            return normalized[normalize_name(name)]
    candidates = []
    condition_context = ("stability", "aging", "ageing", "test", "exposure", "outdoor", "lifetime")
    for c in df.columns:
        n = normalize_name(c)
        # Do not silently use fabrication/annealing conditions as harsh-test
        # conditions. Generic fallbacks must carry explicit stability context.
        if not any(token in n for token in condition_context):
            continue
        score = sum(1 for hint in hints if normalize_name(hint) in n)
        if score:
            context_score = sum(1 for token in condition_context if token in n)
            candidates.append((score + 3 * context_score, int(df[c].notna().sum()), c))
    return max(candidates, default=(0, 0, None))[2]


def _standardize_temperature(s: pd.Series) -> pd.Series:
    values = s.map(_parse_numeric_midpoint).astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values
    median = float(finite.median())
    if 250 <= median <= 400:
        return values
    if -50 <= median <= 150:
        return values + 273.15
    return pd.Series(np.nan, index=s.index, dtype=float)


def _standardize_humidity(s: pd.Series) -> pd.Series:
    values = s.map(_parse_numeric_midpoint).astype(float)
    finite = values[np.isfinite(values)]
    if not finite.empty and float(finite.quantile(0.95)) <= 1.0:
        values = values * 100.0
    return values.where(values.between(0, 100))


def _standardize_light_sun(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    unknown = pd.Series(1, index=s.index, dtype=int)
    for idx, raw in s.items():
        if pd.isna(raw):
            continue
        text = str(raw).strip().lower()
        if not text or text in MISSING_TOKENS or "dark" in text:
            continue
        # Remove dimensional exponents before parsing so the "2" in W/m2 or
        # mW/cm2 is not mistaken for the second endpoint of a numeric range.
        numeric_text = re.sub(r"(cm|m)\s*(?:\^?\s*[-+]?\s*2|\u00b2)", r"\1", text)
        val = _parse_numeric_midpoint(numeric_text)
        if not np.isfinite(val):
            continue
        if any(token in text for token in ["mw/cm", "mw cm", "mwcm"]):
            val = val / 100.0
        elif any(token in text for token in ["w/m2", "w m-2", "w m^-2", "w·m"]):
            val = val / 1000.0
        elif val > 10:
            # Literature database light-intensity values near 100 are normally mW/cm2.
            val = val / 100.0
        if 0 < val <= 20:
            out.at[idx] = float(val)
            unknown.at[idx] = 0
    return out, unknown


def _physical_scatter(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=180)
    ax.scatter(np.asarray(x)[mask], np.asarray(y)[mask], s=10, alpha=0.35)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _physical_sanity_checks(cfg: Config) -> List[str]:
    """Return warnings for physically inconsistent parameter behavior."""
    warnings_out = []
    kb_ev_k = 8.617333262e-5

    def temp_af(temp_k: float) -> float:
        exponent = (cfg.PHYS_EA_EV / kb_ev_k) * ((1.0 / cfg.PHYS_REF_TEMP_K) - (1.0 / temp_k))
        return float(np.exp(np.clip(exponent, -50, 50)))

    hotter_k = max(cfg.PHYS_REF_TEMP_K + 30.0, 330.0)
    if not temp_af(hotter_k) > temp_af(cfg.PHYS_REF_TEMP_K):
        warnings_out.append("SANITY CHECK FAILED: hotter temperature does not increase the temperature acceleration factor.")
    rh_low = 0.5 * cfg.PHYS_REF_RH_PERCENT
    rh_high = 2.0 * cfg.PHYS_REF_RH_PERCENT
    af_rh_low = 1.0 if rh_low <= cfg.PHYS_REF_RH_PERCENT else (rh_low / cfg.PHYS_REF_RH_PERCENT) ** cfg.PHYS_RH_EXPONENT
    af_rh_high = 1.0 if rh_high <= cfg.PHYS_REF_RH_PERCENT else (rh_high / cfg.PHYS_REF_RH_PERCENT) ** cfg.PHYS_RH_EXPONENT
    if not math.isclose(float(af_rh_low), 1.0, rel_tol=1e-9):
        warnings_out.append("SANITY CHECK FAILED: humidity below the reference condition should have AF=1.")
    if not af_rh_high >= af_rh_low:
        warnings_out.append("SANITY CHECK FAILED: higher humidity does not increase the humidity acceleration factor.")
    light_high = 2.0 * cfg.PHYS_REF_LIGHT_SUN
    af_light_high = (light_high / cfg.PHYS_REF_LIGHT_SUN) ** cfg.PHYS_LIGHT_EXPONENT
    if not af_light_high >= 1.0:
        warnings_out.append("SANITY CHECK FAILED: higher illumination does not increase the light acceleration factor.")
    return warnings_out


def add_physical_stability_layer(df: pd.DataFrame, cfg: Config, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.PHYS_REF_TEMP_K <= 0:
        raise ValueError("PHYS_REF_TEMP_K must be positive.")
    if cfg.PHYS_REF_RH_PERCENT <= 0:
        raise ValueError("PHYS_REF_RH_PERCENT must be positive.")
    if cfg.PHYS_REF_LIGHT_SUN <= 0:
        raise ValueError("PHYS_REF_LIGHT_SUN must be positive.")
    if cfg.PHYS_EA_EV < 0 or cfg.PHYS_RH_EXPONENT < 0 or cfg.PHYS_LIGHT_EXPONENT < 0 or cfg.PHYS_BETA <= 0:
        raise ValueError("Physical activation energy/exponents must be non-negative and PHYS_BETA must be positive.")

    physical_dir = ensure_dir(out_dir / "physical_layer")
    temp_col = _best_physical_column(
        df,
        ["Stability_temperature_range", "Stability_temperature_load_condition"],
        ["stability_temperature", "stability_temp", "aging_temperature", "test_temperature", "temperature"],
    )
    rh_col = _best_physical_column(
        df,
        ["Stability_relative_humidity_average_value", "Stability_relative_humidity_range", "Stability_relative_humidity_load_conditions"],
        ["stability_humidity", "relative_humidity", "humidity", "_rh"],
    )
    light_col = _best_physical_column(
        df,
        ["Stability_light_intensity"],
        ["stability_light_intensity", "light_intensity", "illumination", "irradiance", "suns", "sun", "mw_cm2"],
    )

    mapping = {"temperature": temp_col, "relative_humidity": rh_col, "light_intensity": light_col}
    safe_json_dump(mapping, physical_dir / "physical_condition_column_mapping.json")

    empty = pd.Series(np.nan, index=df.index, dtype=float)
    df["PHYS_test_temp_K"] = _standardize_temperature(df[temp_col]) if temp_col else empty.copy()
    df["PHYS_test_temp_C"] = df["PHYS_test_temp_K"] - 273.15
    df["PHYS_test_RH_percent"] = _standardize_humidity(df[rh_col]) if rh_col else empty.copy()
    if light_col:
        light_sun, light_unknown = _standardize_light_sun(df[light_col])
    else:
        light_sun = empty.copy()
        light_unknown = pd.Series(1, index=df.index, dtype=int)
    df["PHYS_test_light_sun"] = light_sun

    df["PHYS_missing_temperature_flag"] = df["PHYS_test_temp_K"].isna().astype(int)
    df["PHYS_missing_humidity_flag"] = df["PHYS_test_RH_percent"].isna().astype(int)
    df["PHYS_missing_light_flag"] = light_unknown
    known_count = 3 - (
        df["PHYS_missing_temperature_flag"]
        + df["PHYS_missing_humidity_flag"]
        + df["PHYS_missing_light_flag"]
    )
    df["PHYS_condition_known_score"] = known_count / 3.0

    kb_ev_k = 8.617333262e-5
    temp_for_af = df["PHYS_test_temp_K"].fillna(cfg.PHYS_REF_TEMP_K)
    rh_for_af = df["PHYS_test_RH_percent"].fillna(cfg.PHYS_REF_RH_PERCENT)
    light_for_af = df["PHYS_test_light_sun"].fillna(cfg.PHYS_REF_LIGHT_SUN)

    exponent = (cfg.PHYS_EA_EV / kb_ev_k) * ((1.0 / cfg.PHYS_REF_TEMP_K) - (1.0 / temp_for_af))
    df["PHYS_AF_temperature"] = np.exp(np.clip(exponent, -50, 50))
    df["PHYS_AF_humidity"] = np.where(
        rh_for_af <= cfg.PHYS_REF_RH_PERCENT,
        1.0,
        (rh_for_af / cfg.PHYS_REF_RH_PERCENT) ** cfg.PHYS_RH_EXPONENT,
    )
    df["PHYS_AF_light"] = (light_for_af / cfg.PHYS_REF_LIGHT_SUN) ** cfg.PHYS_LIGHT_EXPONENT
    df["PHYS_AF_total"] = df["PHYS_AF_temperature"] * df["PHYS_AF_humidity"] * df["PHYS_AF_light"]
    df["PHYS_log_AF_total"] = np.log(df["PHYS_AF_total"].where(df["PHYS_AF_total"] > 0))

    beta = max(float(cfg.PHYS_BETA), 1e-9)
    retention = pd.to_numeric(df.get("target_end_retention_pct", empty), errors="coerce") / 100.0
    exposure = pd.to_numeric(df.get("target_stability_exposure_h", empty), errors="coerce")
    t80 = pd.to_numeric(df.get("target_T80_hours", empty), errors="coerce")
    retention_valid = (
        retention.between(cfg.PHYS_MIN_RETENTION_FRAC, cfg.PHYS_MAX_RETENTION_FRAC)
        & (retention < 1.0)
        & (exposure >= cfg.PHYS_MIN_EXPOSURE_H)
    )
    t80_valid = t80 >= cfg.PHYS_MIN_EXPOSURE_H
    k_from_retention = ((-np.log(retention.where(retention_valid))) ** (1.0 / beta)) / exposure
    k_from_t80 = ((-math.log(0.80)) ** (1.0 / beta)) / t80
    df["PHYS_k_obs_h_inv"] = k_from_retention.where(retention_valid, k_from_t80.where(t80_valid))
    source = pd.Series(pd.NA, index=df.index, dtype="object")
    source.loc[t80_valid] = "T80"
    source.loc[retention_valid] = "end_retention"
    df["PHYS_k_obs_source"] = source

    conditions_known = df["PHYS_condition_known_score"] == 1.0
    physical_valid = (
        conditions_known
        & np.isfinite(df["PHYS_AF_total"])
        & (df["PHYS_AF_total"] > 0)
        & np.isfinite(df["PHYS_k_obs_h_inv"])
        & (df["PHYS_k_obs_h_inv"] > 0)
    )
    df["phys_k_ref_h_inv"] = (df["PHYS_k_obs_h_inv"] / df["PHYS_AF_total"]).where(physical_valid)
    df["target_phys_log_k_obs"] = np.log(df["PHYS_k_obs_h_inv"].where(df["PHYS_k_obs_h_inv"] > 0))
    df["target_phys_log_k_ref"] = np.log(df["phys_k_ref_h_inv"].where(df["phys_k_ref_h_inv"] > 0))
    t80_numerator = (-math.log(0.80)) ** (1.0 / beta)
    df["target_phys_T80_ref_hours"] = t80_numerator / df["phys_k_ref_h_inv"]
    df["target_phys_log1p_T80_ref_hours"] = np.log1p(df["target_phys_T80_ref_hours"])

    condition_coverage = pd.DataFrame([
        {"field": "standardized_temperature", "source_column": temp_col, "non_null": int(df["PHYS_test_temp_K"].notna().sum()), "coverage_fraction": float(df["PHYS_test_temp_K"].notna().mean())},
        {"field": "standardized_relative_humidity", "source_column": rh_col, "non_null": int(df["PHYS_test_RH_percent"].notna().sum()), "coverage_fraction": float(df["PHYS_test_RH_percent"].notna().mean())},
        {"field": "standardized_light_intensity", "source_column": light_col, "non_null": int(df["PHYS_test_light_sun"].notna().sum()), "coverage_fraction": float(df["PHYS_test_light_sun"].notna().mean())},
        {"field": "observed_degradation_rate", "source_column": "retention/exposure or T80", "non_null": int(df["PHYS_k_obs_h_inv"].notna().sum()), "coverage_fraction": float(df["PHYS_k_obs_h_inv"].notna().mean())},
        {"field": "all_conditions_known", "source_column": "combined", "non_null": int(conditions_known.sum()), "coverage_fraction": float(conditions_known.mean())},
        {"field": "valid_condition_normalized_target", "source_column": "combined", "non_null": int(physical_valid.sum()), "coverage_fraction": float(physical_valid.mean())},
    ])
    condition_coverage.to_csv(physical_dir / "physical_condition_coverage.csv", index=False)
    pd.DataFrame([{
        "total_rows": len(df),
        "missing_temperature": int(df["PHYS_missing_temperature_flag"].sum()),
        "missing_humidity": int(df["PHYS_missing_humidity_flag"].sum()),
        "missing_light": int(df["PHYS_missing_light_flag"].sum()),
        "missing_observed_degradation_rate": int(df["PHYS_k_obs_h_inv"].isna().sum()),
        "valid_condition_normalized_target": int(physical_valid.sum()),
    }]).to_csv(physical_dir / "physical_target_exclusion_summary.csv", index=False)

    audit_rows = []
    for target in PHYSICAL_TARGETS:
        s = pd.to_numeric(df[target], errors="coerce")
        audit_rows.append({
            "target": target,
            "non_null": int(s.notna().sum()),
            "coverage_fraction": float(s.notna().mean()),
            "mean": float(s.mean(skipna=True)) if s.notna().sum() else np.nan,
            "median": float(s.median(skipna=True)) if s.notna().sum() else np.nan,
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(physical_dir / "physical_stability_target_audit.csv", index=False)
    preview_cols = [
        c for c in [
            temp_col, rh_col, light_col, "target_end_retention_pct", "target_stability_exposure_h",
            "target_T80_hours", "PHYS_test_temp_K", "PHYS_test_RH_percent", "PHYS_test_light_sun",
            "PHYS_AF_temperature", "PHYS_AF_humidity", "PHYS_AF_light", "PHYS_AF_total",
            "PHYS_k_obs_h_inv", "PHYS_k_obs_source", "phys_k_ref_h_inv", *PHYSICAL_TARGETS,
        ] if c and c in df.columns
    ]
    df[preview_cols].to_csv(physical_dir / "physical_targets_preview.csv", index=False)
    safe_json_dump({
        "reference_temperature_K": cfg.PHYS_REF_TEMP_K,
        "reference_RH_percent": cfg.PHYS_REF_RH_PERCENT,
        "reference_light_sun": cfg.PHYS_REF_LIGHT_SUN,
        "activation_energy_eV": cfg.PHYS_EA_EV,
        "humidity_exponent": cfg.PHYS_RH_EXPONENT,
        "light_exponent": cfg.PHYS_LIGHT_EXPONENT,
        "weibull_beta": cfg.PHYS_BETA,
        "physical_target_rows": int(physical_valid.sum()),
        "condition_mapping": mapping,
    }, physical_dir / "physical_model_parameters.json")

    warnings_out = [
        "Physics-informed harsh-condition normalization uses semi-empirical Arrhenius, humidity, and light acceleration factors.",
        "It is not a fully mechanistic chemistry-of-failure model.",
        f"Rows with all three standardized conditions: {int(conditions_known.sum())}/{len(df)}.",
        f"Rows with valid condition-normalized physical targets: {int(physical_valid.sum())}/{len(df)}.",
    ]
    warnings_out.extend(_physical_sanity_checks(cfg))
    invalid_af = ~(np.isfinite(df["PHYS_AF_total"]) & (df["PHYS_AF_total"] > 0))
    if int(invalid_af.sum()):
        warnings_out.append(f"WARNING: {int(invalid_af.sum())} rows produced a non-positive or non-finite total acceleration factor.")
    if int(physical_valid.sum()) < cfg.MIN_ROWS_REGRESSION:
        warnings_out.append("Physical-target coverage is below the configured minimum regression sample count; hybrid models may be skipped.")
    (physical_dir / "physical_layer_warnings.txt").write_text("\n".join(warnings_out) + "\n", encoding="utf-8")

    finite_af = df["PHYS_AF_total"][np.isfinite(df["PHYS_AF_total"]) & (df["PHYS_AF_total"] > 0)]
    if len(finite_af):
        fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=180)
        ax.hist(np.log10(finite_af), bins=40, alpha=0.8)
        ax.set_xlabel("log10(total acceleration factor)")
        ax.set_ylabel("Count")
        ax.set_title("Harsh-condition acceleration-factor distribution")
        fig.tight_layout()
        fig.savefig(physical_dir / "AF_total_distribution.png", bbox_inches="tight")
        plt.close(fig)
    _physical_scatter(df["PHYS_AF_total"], df["PHYS_k_obs_h_inv"], "AF total", "Observed degradation rate (h^-1)", "Observed degradation rate vs harsh-condition acceleration", physical_dir / "k_obs_vs_AF_total.png")
    _physical_scatter(df["PHYS_k_obs_h_inv"], df["phys_k_ref_h_inv"], "Observed degradation rate (h^-1)", "Reference-condition degradation rate (h^-1)", "Observed vs condition-normalized degradation rate", physical_dir / "k_obs_vs_k_ref.png")
    return df, audit


def regression_targets(include_physical: bool = False) -> List[str]:
    targets = [
        "target_end_retention_pct",
        "target_after_1000h_retention_pct",
        "target_log1p_T80_hours",
        "target_log1p_T95_hours",
    ]
    return targets + PHYSICAL_TARGETS if include_physical else targets


def classification_targets() -> List[str]:
    return [
        "target_end_retention_ge80",
        "target_end_retention_ge80_after_100h",
        "target_end_retention_ge80_after_500h",
        "target_T80_ge100h",
        "target_T80_ge500h",
        "target_after_1000h_ge80",
    ]


# =============================================================================
# COMPLETENESS FILTER
# =============================================================================

def apply_row_completeness(df: pd.DataFrame, feature_cols: List[str], cfg: Config, out_dir: Path) -> pd.DataFrame:
    if not cfg.APPLY_ROW_COMPLETENESS_FILTER:
        return df
    allowed = [c for c in feature_cols if c in df.columns and not c.startswith("ADD_DESC_")]
    if not allowed:
        return df
    comp = df[allowed].notna().mean(axis=1)
    audit = pd.DataFrame({"row_index": df.index, "input_fill_ratio": comp})
    audit.to_csv(out_dir / "row_completeness_audit.csv", index=False)
    before = len(df)
    kept = df[comp >= cfg.MIN_ROW_COMPLETENESS].copy()
    print(f"Row completeness filter: kept {len(kept):,}/{before:,} rows at >= {cfg.MIN_ROW_COMPLETENESS:.0%} input fill.")
    safe_json_dump({"before": before, "after": len(kept), "threshold": cfg.MIN_ROW_COMPLETENESS, "counted_columns": len(allowed)}, out_dir / "row_completeness_filter.json")
    return kept


# =============================================================================
# MODELING
# =============================================================================

def split_num_cat(X: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            # Try numeric conversion only for columns whose values are numeric
            # as whole values. This keeps chemical formulas and device-stack
            # strings such as "TiO2" out of the median-imputed numeric block.
            numeric = to_numeric_series(s)
            non_missing = int(s.notna().sum())
            converted = int(numeric.notna().sum())
            looks_numeric = numeric_like_ratio(s)
            if (
                non_missing > 0
                and not is_text_like_feature_name(c)
                and looks_numeric >= 0.80
                and converted >= max(20, 0.80 * non_missing)
            ):
                X[c] = numeric
                num_cols.append(c)
            else:
                nunique = s.astype(str).nunique(dropna=True)
                if nunique <= cfg.MAX_CATEGORICAL_LEVELS:
                    cat_cols.append(c)
                else:
                    # Use a shortened text/category representation for very high cardinality fields.
                    X[c] = s.fillna("MISSING").astype(str).str.slice(0, 80)
                    cat_cols.append(c)
    return num_cols, cat_cols


def make_preprocessor(X: pd.DataFrame, cfg: Config) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X_work = X.copy()
    num_cols, cat_cols = split_num_cat(X_work, cfg)
    try:
        num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        cat_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    except TypeError:
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=5)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("to_numeric", FunctionTransformer(coerce_numeric_frame, validate=False)),
                ("imputer", num_imputer),
                ("scaler", StandardScaler(with_mean=False)),
            ]), num_cols),
            ("cat", Pipeline([("imputer", cat_imputer), ("onehot", ohe)]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, num_cols, cat_cols


def make_model(task: str, cfg: Config):
    if HAS_XGB:
        if task == "regression":
            return XGBRegressor(
                n_estimators=cfg.N_ESTIMATORS,
                max_depth=5,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=3,
                reg_lambda=3.0,
                objective="reg:squarederror",
                random_state=cfg.RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            )
        return XGBClassifier(
            n_estimators=max(250, cfg.N_ESTIMATORS // 2),
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=cfg.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        if task == "regression":
            return ExtraTreesRegressor(
                n_estimators=max(250, cfg.N_ESTIMATORS // 2),
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=cfg.RANDOM_STATE,
                n_jobs=-1,
            )
        return ExtraTreesClassifier(
            n_estimators=max(250, cfg.N_ESTIMATORS // 2),
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=cfg.RANDOM_STATE,
            n_jobs=-1,
        )


def make_cv(y: pd.Series, groups: Optional[pd.Series], cfg: Config):
    n = len(y)
    if groups is not None and groups.notna().nunique() >= cfg.N_SPLITS:
        return GroupKFold(n_splits=cfg.N_SPLITS).split(np.zeros(n), y, groups.astype(str).fillna("NO_GROUP"))
    return KFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE).split(np.zeros(n), y)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if name == "num":
                names.extend([f"num__{c}" for c in cols])
            elif name == "cat":
                try:
                    ohe = trans.named_steps["onehot"]
                    names.extend(list(ohe.get_feature_names_out(cols)))
                except Exception:
                    names.extend([f"cat__{c}" for c in cols])
        return names


def plot_pred_vs_actual(y_true, y_pred, title: str, path: Path, ylabel: str = "Predicted"):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    rmse = rmse_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=180)
    ax.scatter(y_true, y_pred, s=9, alpha=0.35)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    pad = 0.05 * (hi - lo) if hi > lo else 1
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], linestyle="--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.04, 0.96, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nn = {len(y_true):,}", transform=ax.transAxes, va="top", ha="left", bbox=dict(boxstyle="round", alpha=0.15))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def rmse_score(y_true, y_pred) -> float:
    """Version-stable RMSE for sklearn installs without squared=False."""
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def plot_classification_scores(y_true, y_score, title: str, path: Path):
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = np.asarray(y_true)[mask]
    y_score = np.asarray(y_score)[mask]
    if len(y_true) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=180)
    ax.hist(y_score[y_true == 0], bins=30, alpha=0.55, label="class 0")
    ax.hist(y_score[y_true == 1], bins=30, alpha=0.55, label="class 1")
    ax.set_xlabel("Predicted probability / score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train_oof_model(df: pd.DataFrame, feature_cols: List[str], target_col: str, task: str, group_col: Optional[str], cfg: Config, out_dir: Path, label: str) -> Tuple[pd.DataFrame, Dict, Optional[Pipeline]]:
    out_dir = ensure_dir(out_dir)
    data = df[feature_cols + [target_col] + ([group_col] if group_col and group_col in df.columns else [])].copy()
    data[target_col] = to_numeric_series(data[target_col])
    data = data[data[target_col].notna()].copy()
    if task == "classification":
        data = data[data[target_col].isin([0, 1, 0.0, 1.0])].copy()
        vc = data[target_col].value_counts()
        if len(data) < cfg.MIN_ROWS_CLASSIFICATION or vc.min() < cfg.MIN_POSITIVE_CLASS:
            return pd.DataFrame(), {"label": label, "target": target_col, "task": task, "status": "skipped_insufficient_class_support", "n": len(data)}, None
    else:
        if len(data) < cfg.MIN_ROWS_REGRESSION:
            return pd.DataFrame(), {"label": label, "target": target_col, "task": task, "status": "skipped_insufficient_rows", "n": len(data)}, None

    X = data[feature_cols].copy()
    y = data[target_col].astype(float)
    groups = data[group_col] if group_col and group_col in data.columns else None

    oof_pred = np.full(len(data), np.nan)
    oof_score = np.full(len(data), np.nan)
    fold_rows = []
    cv = list(make_cv(y, groups, cfg))
    print(f"\nTraining {label} | target={target_col} | task={task} | rows={len(data):,} | features={len(feature_cols):,}")

    for fold, (tr, te) in enumerate(cv, start=1):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        pre, _, _ = make_preprocessor(X_tr, cfg)
        model = make_model(task, cfg)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        if task == "classification":
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                pred_score = pipe.predict_proba(X_te)[:, 1]
            else:
                pred_score = pipe.predict(X_te)
            pred = (pred_score >= 0.5).astype(float)
            oof_score[te] = pred_score
            oof_pred[te] = pred
            metrics = {
                "fold": fold,
                "accuracy": accuracy_score(y_te, pred),
                "balanced_accuracy": balanced_accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred, zero_division=0),
            }
            if y_te.nunique() == 2:
                metrics["roc_auc"] = roc_auc_score(y_te, pred_score)
            print(f"  fold {fold}: bal_acc={metrics['balanced_accuracy']:.3f}, f1={metrics['f1']:.3f}")
        else:
            pred = pipe.predict(X_te)
            oof_pred[te] = pred
            metrics = {
                "fold": fold,
                "r2": r2_score(y_te, pred),
                "rmse": rmse_score(y_te, pred),
                "mae": mean_absolute_error(y_te, pred),
            }
            print(f"  fold {fold}: R2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")
        fold_rows.append(metrics)

    # Fit full model for interpretation / deployment.
    pre, _, _ = make_preprocessor(X, cfg)
    model = make_model(task, cfg)
    full_pipe = Pipeline([("pre", pre), ("model", model)])
    full_pipe.fit(X, y)

    pred_df = data[[target_col] + ([group_col] if group_col and group_col in data.columns else [])].copy()
    pred_df["oof_pred"] = oof_pred
    if task == "classification":
        pred_df["oof_score"] = oof_score
    pred_df.to_csv(out_dir / f"{label}_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"{label}_fold_metrics.csv", index=False)

    if task == "regression":
        overall = {
            "label": label, "target": target_col, "task": task, "status": "trained", "n": len(data), "n_features_raw": len(feature_cols),
            "oof_r2": r2_score(y, oof_pred),
            "oof_rmse": rmse_score(y, oof_pred),
            "oof_mae": mean_absolute_error(y, oof_pred),
        }
        plot_pred_vs_actual(y.to_numpy(), oof_pred, f"{label}: OOF predicted vs actual", out_dir / f"{label}_oof_pred_vs_actual.png")
    else:
        pred_cls = (oof_score >= 0.5).astype(float)
        overall = {
            "label": label, "target": target_col, "task": task, "status": "trained", "n": len(data), "n_features_raw": len(feature_cols),
            "oof_accuracy": accuracy_score(y, pred_cls),
            "oof_balanced_accuracy": balanced_accuracy_score(y, pred_cls),
            "oof_f1": f1_score(y, pred_cls, zero_division=0),
        }
        if y.nunique() == 2:
            overall["oof_roc_auc"] = roc_auc_score(y, oof_score)
        plot_classification_scores(y.to_numpy(), oof_score, f"{label}: OOF score distributions", out_dir / f"{label}_oof_score_distribution.png")

    # Feature importance, if supported.
    try:
        names = get_feature_names(full_pipe.named_steps["pre"])
        importances = getattr(full_pipe.named_steps["model"], "feature_importances_", None)
        if importances is not None and len(importances) == len(names):
            imp = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
            imp.to_csv(out_dir / f"{label}_feature_importance.csv", index=False)
            top = imp.head(cfg.TOP_N_FEATURE_IMPORTANCE)
            fig, ax = plt.subplots(figsize=(8.0, max(4, 0.18 * len(top))), dpi=180)
            ax.barh(top["feature"][::-1], top["importance"][::-1])
            ax.set_xlabel("Importance")
            ax.set_title(f"{label}: top feature importances")
            fig.tight_layout()
            fig.savefig(out_dir / f"{label}_feature_importance_top.png", bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"  Feature importance skipped for {label}: {e}")

    return pred_df, overall, full_pipe


# =============================================================================
# HOLDOUT EVALUATION FOR PCE
# =============================================================================

def run_group_holdout(df: pd.DataFrame, feature_cols: List[str], target_col: str, group_col: Optional[str], cfg: Config, out_dir: Path, label: str) -> Dict:
    data = df[feature_cols + [target_col] + ([group_col] if group_col and group_col in df.columns else [])].copy()
    data[target_col] = to_numeric_series(data[target_col])
    data = data[data[target_col].notna()].copy()
    X = data[feature_cols].copy()
    y = data[target_col].astype(float)
    if group_col and group_col in data.columns and data[group_col].notna().nunique() > 10:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=cfg.RANDOM_STATE)
        tr, te = next(splitter.split(X, y, groups=data[group_col].astype(str).fillna("NO_GROUP")))
    else:
        tr, te = train_test_split(np.arange(len(data)), test_size=0.20, random_state=cfg.RANDOM_STATE)
    pre, _, _ = make_preprocessor(X.iloc[tr].copy(), cfg)
    model = make_model("regression", cfg)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X.iloc[tr], y.iloc[tr])
    pred_train = pipe.predict(X.iloc[tr])
    pred_test = pipe.predict(X.iloc[te])
    plot_pred_vs_actual(y.iloc[tr].to_numpy(), pred_train, f"{label}: train predicted vs actual", out_dir / f"{label}_train_pred_vs_actual.png")
    plot_pred_vs_actual(y.iloc[te].to_numpy(), pred_test, f"{label}: grouped holdout predicted vs actual", out_dir / f"{label}_group_holdout_pred_vs_actual.png")
    hold = data.iloc[te][[target_col] + ([group_col] if group_col and group_col in data.columns else [])].copy()
    hold["pred"] = pred_test
    hold.to_csv(out_dir / f"{label}_group_holdout_predictions.csv", index=False)
    metrics = {
        "label": label,
        "holdout_n": len(te),
        "holdout_r2": r2_score(y.iloc[te], pred_test),
        "holdout_rmse": rmse_score(y.iloc[te], pred_test),
        "holdout_mae": mean_absolute_error(y.iloc[te], pred_test),
    }
    safe_json_dump(metrics, out_dir / f"{label}_group_holdout_metrics.json")
    return metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def stability_field_coverage(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).startswith("Stability_") or str(c).startswith("Outdoor_") or str(c).startswith("Stabilised_performance")]
    rows = []
    for c in cols:
        rows.append({"column": c, "non_null": int(df[c].notna().sum()), "coverage_fraction": float(df[c].notna().mean()), "dtype": str(df[c].dtype)})
    cov = pd.DataFrame(rows).sort_values("non_null", ascending=False)
    cov.to_csv(out_dir / "stability_field_coverage.csv", index=False)
    return cov


def write_readme(out_dir: Path, cfg: Config, group_col: Optional[str], pce_target: str):
    txt = f"""
PCE -> Stability sequential modeling pipeline complete.

What this run does:
1. Predicts PCE first using leakage-safe design/process/interface/device-context inputs.
2. Builds stability targets from Stability_* fields.
3. Predicts stability with the same grouped-CV/preprocessing/modeling approach.
4. Preserves the original empirical stability baseline.
5. Optionally adds a physics-informed harsh-condition layer using temperature, humidity, and illumination acceleration factors.
6. Compares stability modes:
   - design_only: pre-experiment stability prediction from design/process/interface fields.
   - design_plus_predicted_pce: pre-experiment stability prediction plus OOF predicted PCE.
   - design_plus_measured_initial_performance: post-fabrication triage using measured initial JV/Stabilised metrics.
   - physical_condition_features: empirical prediction with explicit harsh-test acceleration factors.
   - condition_normalized_hybrid: predicts degradation rate and T80 normalized to reference conditions.

Important interpretation:
- For pre-experiment prediction, use design_only or design_plus_predicted_pce.
- Do not use measured JV_default_PCE as a pre-experiment feature because that is only available after device fabrication.
- design_plus_measured_initial_performance is valid for deciding which fabricated devices are likely stable, not for designing before making them.
- The physical layer is semi-empirical, not a fully mechanistic chemistry-of-failure model.
- Its default reference condition is {cfg.PHYS_REF_TEMP_K:.2f} K, {cfg.PHYS_REF_RH_PERCENT:.1f}% RH, and {cfg.PHYS_REF_LIGHT_SUN:.2f} sun.
- Its default assumptions are Ea={cfg.PHYS_EA_EV:.3f} eV, humidity exponent={cfg.PHYS_RH_EXPONENT:.3f}, light exponent={cfg.PHYS_LIGHT_EXPONENT:.3f}, beta={cfg.PHYS_BETA:.3f}.
- Condition-normalized targets are created only when temperature, humidity, illumination, and an observed degradation target are all available.

Detected group/DOI column: {group_col}
PCE target used: {pce_target}
PCE range filter: {cfg.PCE_MIN} to {cfg.PCE_MAX}
Row completeness filter: {cfg.APPLY_ROW_COMPLETENESS_FILTER}, threshold={cfg.MIN_ROW_COMPLETENESS}

Open first:
- pce/pce_direct_oof_predictions.csv
- pce/pce_direct_oof_pred_vs_actual.png
- stability/stability_target_audit.csv
- stability/stability_model_metrics_summary.csv
- stability/physical_layer/physical_model_parameters.json
- stability/physical_layer/physical_condition_coverage.csv
- stability/physical_layer/physical_target_exclusion_summary.csv
- stability/physical_layer/physical_stability_target_audit.csv
- stability/physical_vs_empirical_model_comparison.csv
- model_comparison_pce_then_stability.csv
""".strip()
    (out_dir / "README.txt").write_text(txt, encoding="utf-8")


def main(cfg: Config = CFG):
    out_dir = ensure_dir(cfg.OUTPUT_DIR)
    pce_dir = ensure_dir(out_dir / "pce")
    stability_dir = ensure_dir(out_dir / "stability")
    safe_json_dump(asdict(cfg), out_dir / "config.json")
    script_path = Path(__file__).resolve()
    run_environment = {
        "model_backend": "xgboost" if HAS_XGB else "sklearn_extra_trees",
        "python_version": platform.python_version(),
        "input_csv": cfg.INPUT_CSV,
        "physical_stability_layer_enabled": cfg.ENABLE_PHYSICAL_STABILITY_LAYER,
        "model_script_path": str(script_path),
        "model_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    }
    safe_json_dump(run_environment, out_dir / "run_environment.json")

    print("=" * 90)
    print("PCE -> STABILITY SEQUENTIAL PIPELINE START")
    print("=" * 90)

    df_raw = robust_read_csv(cfg.INPUT_CSV)
    df_raw = clean_df(df_raw)
    group_col = find_group_col(df_raw)
    safe_json_dump({
        "input_csv": cfg.INPUT_CSV,
        "input_rows": len(df_raw),
        "input_columns": len(df_raw.columns),
        "detected_group_column": group_col,
    }, out_dir / "dataset_profile.json")
    print(f"Detected group column: {group_col}")

    # Add additive descriptors before feature selection.
    df_raw = add_additive_descriptor_features(df_raw, out_dir)

    # Stability field audit on raw data.
    cov = stability_field_coverage(df_raw, stability_dir)

    # Prepare PCE dataset and target.
    df_pce, pce_target = prepare_pce_target(df_raw.copy(), cfg)

    # Build PCE features and apply completeness based on PCE design inputs.
    pce_feature_cols = build_base_feature_columns(df_pce, pce_target, purpose="pce")
    # Include the new additive descriptor features.
    add_cols = [c for c in df_pce.columns if c.startswith("ADD_DESC_")]
    for c in add_cols:
        if c not in pce_feature_cols and c != pce_target:
            pce_feature_cols.append(c)
    pd.Series(pce_feature_cols, name="feature_column").to_csv(pce_dir / "pce_feature_columns.csv", index=False)
    blocks = infer_feature_blocks(pce_feature_cols)
    safe_json_dump({k: len(v) for k, v in blocks.items()}, pce_dir / "pce_feature_block_counts.json")

    df_pce = apply_row_completeness(df_pce, pce_feature_cols, cfg, pce_dir)

    # Step 1: Predict PCE.
    pce_pred_df, pce_metrics, pce_model = train_oof_model(
        df_pce, pce_feature_cols, pce_target, "regression", group_col, cfg, pce_dir, "pce_direct"
    )
    holdout_metrics = run_group_holdout(df_pce, pce_feature_cols, pce_target, group_col, cfg, pce_dir, "pce_direct")
    pce_metrics.update({f"group_{k}": v for k, v in holdout_metrics.items() if k != "label"})
    safe_json_dump(pce_metrics, pce_dir / "pce_direct_metrics.json")

    # Attach OOF predicted PCE back to raw rows for stability. Align by original index where possible.
    # pce_pred_df has same index as the filtered model data only if we preserved index; use a safer re-run map.
    # Here the pred_df index is the filtered data index in order, so assign by its index.
    df_raw["PCE_MODEL_oof_predicted_PCE"] = np.nan
    if len(pce_pred_df):
        df_raw.loc[pce_pred_df.index, "PCE_MODEL_oof_predicted_PCE"] = pce_pred_df["oof_pred"].to_numpy()

    # For rows without OOF PCE because they were outside the PCE target filter, fit PCE model on valid PCE rows and predict where possible.
    try:
        missing_mask = df_raw["PCE_MODEL_oof_predicted_PCE"].isna()
        if pce_model is not None and missing_mask.any():
            # Only columns in pce_feature_cols; if missing because raw lacks, fill with nan.
            X_missing = df_raw.loc[missing_mask, pce_feature_cols].copy()
            df_raw.loc[missing_mask, "PCE_MODEL_oof_predicted_PCE"] = pce_model.predict(X_missing)
    except Exception as e:
        print(f"Could not fill non-OOF PCE predictions for all rows: {e}")

    # Step 2: Make stability targets and train stability models.
    df_stab, target_audit = add_stability_targets(df_raw.copy())
    target_audit.to_csv(stability_dir / "stability_target_audit.csv", index=False)
    physical_audit = pd.DataFrame()
    if cfg.ENABLE_PHYSICAL_STABILITY_LAYER:
        df_stab, physical_audit = add_physical_stability_layer(df_stab, cfg, stability_dir)
    df_stab.to_csv(stability_dir / "dataset_with_pce_predictions_and_stability_targets_preview.csv", index=False)

    physical_cols = [c for c in df_stab.columns if c.startswith("PHYS_") or c.startswith("phys_") or c.startswith("target_phys_")]
    stability_target_cols = regression_targets(include_physical=True) + classification_targets() + [
        "target_T80_hours", "target_T95_hours", "target_stability_exposure_h"
    ] + physical_cols

    base_stability_features = build_base_feature_columns(df_stab, "__stability_target_placeholder__", purpose="stability_design", extra_exclude=stability_target_cols)
    for c in add_cols:
        if c in df_stab.columns and c not in base_stability_features:
            base_stability_features.append(c)

    mode_feature_sets = {}
    if cfg.RUN_STABILITY_DESIGN_ONLY:
        mode_feature_sets["design_only"] = list(base_stability_features)
    if cfg.RUN_STABILITY_PLUS_PREDICTED_PCE:
        feats = list(base_stability_features)
        if "PCE_MODEL_oof_predicted_PCE" not in feats:
            feats.append("PCE_MODEL_oof_predicted_PCE")
        mode_feature_sets["design_plus_predicted_pce"] = feats
    if cfg.RUN_STABILITY_PLUS_MEASURED_INITIAL_PERFORMANCE:
        feats = list(base_stability_features)
        for c in allowed_initial_performance_cols(df_stab):
            if c not in feats:
                feats.append(c)
        mode_feature_sets["design_plus_measured_initial_performance"] = feats

    all_metrics = []
    for mode, feats in mode_feature_sets.items():
        mode_dir = ensure_dir(stability_dir / mode)
        pd.Series(feats, name="feature_column").to_csv(mode_dir / "feature_columns.csv", index=False)
        safe_json_dump({k: len(v) for k, v in infer_feature_blocks(feats).items()}, mode_dir / "feature_block_counts.json")

        # Apply row completeness per stability feature mode, but do not drop stability-target rows globally before target availability.
        df_mode = apply_row_completeness(df_stab.copy(), feats, cfg, mode_dir)

        for t in regression_targets():
            pred_df, metrics, _ = train_oof_model(df_mode, feats, t, "regression", group_col, cfg, mode_dir, f"{mode}__{t}")
            metrics["mode"] = mode
            metrics["model_family"] = "empirical_baseline"
            metrics["target_type"] = "empirical_raw"
            all_metrics.append(metrics)
        for t in classification_targets():
            pred_df, metrics, _ = train_oof_model(df_mode, feats, t, "classification", group_col, cfg, mode_dir, f"{mode}__{t}")
            metrics["mode"] = mode
            metrics["model_family"] = "empirical_baseline"
            metrics["target_type"] = "empirical_classification"
            all_metrics.append(metrics)

    if cfg.ENABLE_PHYSICAL_STABILITY_LAYER:
        physical_feature_dir = ensure_dir(stability_dir / "physical_condition_features")
        physical_feature_cols = list(base_stability_features)
        for c in PHYSICAL_STRESS_FEATURES:
            if c in df_stab.columns and c not in physical_feature_cols:
                physical_feature_cols.append(c)
        pd.Series(physical_feature_cols, name="feature_column").to_csv(physical_feature_dir / "feature_columns.csv", index=False)
        df_physical_features = apply_row_completeness(df_stab.copy(), physical_feature_cols, cfg, physical_feature_dir)
        for t in regression_targets():
            _, metrics, _ = train_oof_model(
                df_physical_features, physical_feature_cols, t, "regression", group_col, cfg,
                physical_feature_dir, f"physical_condition_features__{t}",
            )
            metrics["mode"] = "physical_condition_features"
            metrics["model_family"] = "physical_condition_features"
            metrics["target_type"] = "empirical_raw"
            all_metrics.append(metrics)
        if "target_phys_log_k_obs" in df_stab.columns:
            _, metrics, _ = train_oof_model(
                df_physical_features, physical_feature_cols, "target_phys_log_k_obs", "regression", group_col, cfg,
                physical_feature_dir, "physical_condition_features__target_phys_log_k_obs",
            )
            metrics["mode"] = "physical_condition_features"
            metrics["model_family"] = "physical_condition_features"
            metrics["target_type"] = "physical_observed_rate"
            all_metrics.append(metrics)

        hybrid_dir = ensure_dir(stability_dir / "condition_normalized_hybrid")
        hybrid_feature_cols = list(base_stability_features)
        pd.Series(hybrid_feature_cols, name="feature_column").to_csv(hybrid_dir / "feature_columns.csv", index=False)
        df_hybrid = apply_row_completeness(df_stab.copy(), hybrid_feature_cols, cfg, hybrid_dir)
        for t in ["target_phys_log_k_ref", "target_phys_log1p_T80_ref_hours"]:
            pred_df, metrics, _ = train_oof_model(
                df_hybrid, hybrid_feature_cols, t, "regression", group_col, cfg,
                hybrid_dir, f"condition_normalized_hybrid__{t}",
            )
            metrics["mode"] = "condition_normalized_hybrid"
            metrics["model_family"] = "condition_normalized_hybrid"
            metrics["target_type"] = "condition_normalized_physical"
            all_metrics.append(metrics)
            if not pred_df.empty:
                plot_pred_vs_actual(
                    pred_df[t].to_numpy(),
                    pred_df["oof_pred"].to_numpy(),
                    f"Condition-normalized hybrid: {t}",
                    ensure_dir(stability_dir / "physical_layer") / f"{t}_pred_vs_actual.png",
                )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(stability_dir / "stability_model_metrics_summary.csv", index=False)
    if cfg.ENABLE_PHYSICAL_STABILITY_LAYER:
        metrics_df[metrics_df["model_family"].isin([
            "empirical_baseline", "physical_condition_features", "condition_normalized_hybrid"
        ])].to_csv(stability_dir / "physical_vs_empirical_model_comparison.csv", index=False)

    # Combined comparison.
    comparison_rows = []
    comparison_rows.append({"model_family": "PCE", "mode": "design_only", "target_type": "PCE", "target": pce_target, **pce_metrics})
    for _, r in metrics_df.iterrows():
        comparison_rows.append({"model_family": "stability", **r.to_dict()})
    comp = pd.DataFrame(comparison_rows)
    comp.to_csv(out_dir / "model_comparison_pce_then_stability.csv", index=False)

    # Recommendation text.
    rec_lines = []
    rec_lines.append("Recommended interpretation of this run")
    rec_lines.append("======================================")
    rec_lines.append("")
    rec_lines.append("1. Use the PCE model as the first-stage design/performance surrogate.")
    rec_lines.append("2. Use design_plus_predicted_pce to test whether predicted PCE helps stability prediction without using measured PCE leakage.")
    rec_lines.append("3. Use design_plus_measured_initial_performance only as a post-fabrication triage model.")
    rec_lines.append("4. For the first stability target, prioritize the trained target with the largest support and best grouped-CV metric.")
    if cfg.ENABLE_PHYSICAL_STABILITY_LAYER:
        rec_lines.append("5. Compare empirical_baseline with physical_condition_features and condition_normalized_hybrid before interpreting harsh-condition stability.")
    rec_lines.append("")
    trained = metrics_df[metrics_df.get("status", "") == "trained"].copy() if not metrics_df.empty else pd.DataFrame()
    if not trained.empty:
        rec_lines.append("Top trained stability models by available metric:")
        show = trained.copy()
        if "oof_r2" in show.columns:
            show["rank_metric"] = show["oof_r2"]
        if "oof_roc_auc" in show.columns:
            show["rank_metric"] = show["rank_metric"].fillna(show["oof_roc_auc"] if "rank_metric" in show else show["oof_roc_auc"])
        if "rank_metric" not in show.columns:
            show["rank_metric"] = np.nan
        keep = [c for c in ["mode", "target", "task", "n", "oof_r2", "oof_rmse", "oof_roc_auc", "oof_balanced_accuracy", "rank_metric"] if c in show.columns]
        rec_lines.append(show.sort_values("rank_metric", ascending=False)[keep].head(12).to_string(index=False))
    (out_dir / "stability_modeling_recommendation.txt").write_text("\n".join(rec_lines), encoding="utf-8")

    write_readme(out_dir, cfg, group_col, pce_target)

    print("=" * 90)
    print("PIPELINE COMPLETE")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CFG.INPUT_CSV, help="Path to Perovskite Database CSV")
    parser.add_argument("--out", default=CFG.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--n-estimators", type=int, default=CFG.N_ESTIMATORS)
    parser.add_argument("--min-completeness", type=float, default=CFG.MIN_ROW_COMPLETENESS)
    parser.add_argument("--disable-physical-stability-layer", action="store_true", help="Run only the original empirical PCE/stability models")
    parser.add_argument("--phys-ref-temp-k", type=float, default=CFG.PHYS_REF_TEMP_K)
    parser.add_argument("--phys-ref-rh-percent", type=float, default=CFG.PHYS_REF_RH_PERCENT)
    parser.add_argument("--phys-ref-light-sun", type=float, default=CFG.PHYS_REF_LIGHT_SUN)
    parser.add_argument("--phys-ea-ev", type=float, default=CFG.PHYS_EA_EV, help="Arrhenius activation energy in eV")
    parser.add_argument("--phys-rh-exponent", type=float, default=CFG.PHYS_RH_EXPONENT)
    parser.add_argument("--phys-light-exponent", type=float, default=CFG.PHYS_LIGHT_EXPONENT)
    parser.add_argument("--phys-beta", type=float, default=CFG.PHYS_BETA, help="Generalized exponential/Weibull degradation beta")
    args = parser.parse_args()
    CFG.INPUT_CSV = args.csv
    CFG.OUTPUT_DIR = args.out
    CFG.N_ESTIMATORS = args.n_estimators
    CFG.MIN_ROW_COMPLETENESS = args.min_completeness
    CFG.ENABLE_PHYSICAL_STABILITY_LAYER = not args.disable_physical_stability_layer
    CFG.PHYS_REF_TEMP_K = args.phys_ref_temp_k
    CFG.PHYS_REF_RH_PERCENT = args.phys_ref_rh_percent
    CFG.PHYS_REF_LIGHT_SUN = args.phys_ref_light_sun
    CFG.PHYS_EA_EV = args.phys_ea_ev
    CFG.PHYS_RH_EXPONENT = args.phys_rh_exponent
    CFG.PHYS_LIGHT_EXPONENT = args.phys_light_exponent
    CFG.PHYS_BETA = args.phys_beta
    main(CFG)
