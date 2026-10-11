"""Composition validation, descriptor calculation, and neighbor outlier features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .utils import WELLS_96, minmax, normalize_well, warn, well_row_col

# Approximate effective radii used only as relative composition descriptors.
# They are not exact structural values for any particular perovskite lattice.
A_SITE_COLUMNS = ["Cs_fraction", "Rb_fraction", "FA_fraction", "DMA_fraction"]
X_SITE_COLUMNS = ["I_fraction", "Br_fraction", "Cl_fraction"]
A_RADII = {"Cs_fraction": 1.67, "Rb_fraction": 1.52, "FA_fraction": 2.53, "DMA_fraction": 2.72}
X_RADII = {"I_fraction": 2.20, "Br_fraction": 1.96, "Cl_fraction": 1.81}
GENERIC_COMPOSITION_COLUMNS = [
    "two_d_fraction",
    "fapbi3_fraction",
    "additive_fraction",
    "additive_final_molarity",
    "two_d_stock_molarity",
    "fapbi3_stock_molarity",
    "two_d_spacer_stoichiometry",
    "composition_family_is_quasi_2d_additive",
    "shared_fa_host_fraction",
    "shared_iodide_fraction",
    "shared_bromide_fraction",
    "shared_layered_precursor_fraction",
    "shared_3d_precursor_fraction",
    "shared_additive_fraction",
]


def entropy(values) -> float:
    """Shannon entropy for non-negative fractions."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0:
        return 0.0
    v = v / v.sum()
    return float(-np.sum(v * np.log(v)))


def load_composition_map(path: str | Path, cfg: dict[str, Any], warnings_list: list[str] | None = None) -> pd.DataFrame:
    """Load and validate the 96-well composition map."""
    df = pd.read_csv(path)
    if "well" not in df.columns:
        raise ValueError("Composition map must contain a 'well' column.")
    df["well"] = df["well"].map(normalize_well)
    if "row" not in df.columns or "col" not in df.columns:
        rows_cols = df["well"].map(well_row_col)
        df["row"] = [rc[0] for rc in rows_cols]
        df["col"] = [rc[1] for rc in rows_cols]
    expected = int(cfg.get("plate", {}).get("expected_wells", 96))
    if expected == 96 and len(set(df["well"])) != 96:
        warn(f"Composition map has {len(set(df['well']))} unique wells, expected 96.", warnings_list)
    missing = set(WELLS_96) - set(df["well"])
    if expected == 96 and missing:
        warn(f"Composition map is missing wells: {sorted(missing)[:8]}{'...' if len(missing) > 8 else ''}", warnings_list)
    for col in A_SITE_COLUMNS + X_SITE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    a_sum = df[A_SITE_COLUMNS].sum(axis=1)
    x_sum = df[X_SITE_COLUMNS].sum(axis=1)
    if ((a_sum - 1).abs() > 0.05).any():
        warn("Some wells have A-site fractions far from 1.0; continuing with warnings.", warnings_list)
    if ((x_sum - 1).abs() > 0.05).any():
        warn("Some wells have X-site fractions far from 1.0; continuing with warnings.", warnings_list)
    return df


def apply_library_composition_substitutions(
    comp: pd.DataFrame,
    cfg: dict[str, Any],
    warnings_list: list[str] | None = None,
) -> pd.DataFrame:
    """Apply library-specific cation substitutions before descriptors are calculated."""
    df = comp.copy()
    experiment = cfg.get("experiment", {}) or {}
    library_name = str(experiment.get("library_slug") or experiment.get("library_name") or "").strip()
    normalized_library = "".join(ch for ch in library_name.lower() if ch.isalnum())
    substitutions = experiment.get("composition_substitutions") or {}

    # The Rb library uses the DMA plate mixing geometry, with Rb replacing DMA.
    if not substitutions and normalized_library == "rb":
        substitutions = {"DMA_fraction": "Rb_fraction"}

    applied = []
    for source, target in substitutions.items():
        if source not in df.columns:
            warn(f"Composition substitution source column not found: {source}", warnings_list)
            continue
        if target not in df.columns:
            df[target] = 0.0
        source_values = pd.to_numeric(df[source], errors="coerce").fillna(0.0)
        target_values = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
        df[target] = target_values + source_values
        df[source] = 0.0
        applied.append(f"{source}->{target}")

        if "notes" in df.columns:
            source_name = source.removesuffix("_fraction")
            target_name = target.removesuffix("_fraction")
            df["notes"] = df["notes"].fillna("").astype(str).str.replace(source_name, target_name, regex=False)

    if applied:
        df["composition_substitutions_applied"] = ", ".join(applied)
        warn(
            f"Applied {library_name or 'library'} composition substitution(s): {', '.join(applied)}.",
            warnings_list,
        )
    return df


def compute_composition_descriptors(comp: pd.DataFrame) -> pd.DataFrame:
    """Compute A-site and X-site composition descriptors."""
    df = comp.copy()
    for col in A_SITE_COLUMNS + X_SITE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df["A_site_sum"] = df[A_SITE_COLUMNS].sum(axis=1)
    df["X_site_sum"] = df[X_SITE_COLUMNS].sum(axis=1)
    eps = 1e-9
    df["Cs_to_FA_ratio"] = df["Cs_fraction"] / (df["FA_fraction"] + eps)
    df["Rb_to_FA_ratio"] = df["Rb_fraction"] / (df["FA_fraction"] + eps)
    df["Cs_to_Rb_ratio"] = df["Cs_fraction"] / (df["Rb_fraction"] + eps)
    df["DMA_to_FA_ratio"] = df["DMA_fraction"] / (df["FA_fraction"] + eps)
    df["I_to_Br_ratio"] = df["I_fraction"] / (df["Br_fraction"] + eps)
    df["Br_to_Cl_ratio"] = df["Br_fraction"] / (df["Cl_fraction"] + eps)
    df["small_cation_fraction"] = df["Cs_fraction"].fillna(0) + df["Rb_fraction"].fillna(0)
    df["large_organic_cation_fraction"] = df["FA_fraction"].fillna(0) + df["DMA_fraction"].fillna(0)
    df["inorganic_to_organic_A_ratio"] = df["small_cation_fraction"] / (df["large_organic_cation_fraction"] + eps)
    df["DMA_to_Rb_ratio"] = df["DMA_fraction"] / (df["Rb_fraction"] + eps)
    df["A_site_entropy"] = df[A_SITE_COLUMNS].apply(entropy, axis=1)
    df["X_site_entropy"] = df[X_SITE_COLUMNS].apply(entropy, axis=1)
    a_vals = df[list(A_RADII)].to_numpy(dtype=float)
    a_r = np.array(list(A_RADII.values()))
    a_sum = np.nansum(a_vals, axis=1)
    a_w = np.divide(a_vals, a_sum[:, None], out=np.zeros_like(a_vals), where=a_sum[:, None] != 0)
    df["A_site_mean_radius"] = a_w @ a_r
    df["A_site_radius_variance"] = np.sum(a_w * (a_r - df["A_site_mean_radius"].to_numpy()[:, None]) ** 2, axis=1)
    x_vals = df[list(X_RADII)].to_numpy(dtype=float)
    x_r = np.array(list(X_RADII.values()))
    x_sum = np.nansum(x_vals, axis=1)
    x_w = np.divide(x_vals, x_sum[:, None], out=np.zeros_like(x_vals), where=x_sum[:, None] != 0)
    df["X_site_mean_radius"] = x_w @ x_r
    df["X_site_radius_variance"] = np.sum(x_w * (x_r - df["X_site_mean_radius"].to_numpy()[:, None]) ** 2, axis=1)
    if "two_d_fraction" in df.columns and "fapbi3_fraction" in df.columns:
        df["two_d_to_fapbi3_ratio"] = pd.to_numeric(df["two_d_fraction"], errors="coerce") / (
            pd.to_numeric(df["fapbi3_fraction"], errors="coerce") + eps
        )
    quasi_2d = pd.to_numeric(
        df.get("composition_family_is_quasi_2d_additive", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0) > 0.5
    fapbi3 = pd.to_numeric(df.get("fapbi3_fraction", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    two_d = pd.to_numeric(df.get("two_d_fraction", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    additive = pd.to_numeric(df.get("additive_fraction", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    df["shared_fa_host_fraction"] = np.where(quasi_2d, fapbi3, pd.to_numeric(df["FA_fraction"], errors="coerce").fillna(0.0))
    df["shared_iodide_fraction"] = np.where(quasi_2d, 1.0, pd.to_numeric(df["I_fraction"], errors="coerce").fillna(0.0))
    df["shared_bromide_fraction"] = np.where(quasi_2d, 0.0, pd.to_numeric(df["Br_fraction"], errors="coerce").fillna(0.0))
    df["shared_layered_precursor_fraction"] = np.where(quasi_2d, two_d, 0.0)
    df["shared_3d_precursor_fraction"] = np.where(quasi_2d, fapbi3, 1.0)
    df["shared_additive_fraction"] = np.where(quasi_2d, additive, 0.0)
    return df


def composition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    identity_cols = [
        col
        for col in df.columns
        if col.startswith("two_d_precursor_is_") or col.startswith("additive_is_")
    ]
    cols = A_SITE_COLUMNS + X_SITE_COLUMNS + GENERIC_COMPOSITION_COLUMNS + identity_cols
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    matrix = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for col in matrix.columns:
        lo, hi = matrix[col].min(), matrix[col].max()
        if np.isfinite(lo) and np.isfinite(hi) and not np.isclose(lo, hi):
            matrix[col] = (matrix[col] - lo) / (hi - lo)
        else:
            matrix[col] = 0.0
    return matrix


def add_neighbor_outlier_features(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compare optical features to nearest composition neighbors."""
    df = table.copy()
    if len(df) < 3:
        df["optical_neighbor_outlier_score"] = 0.0
        return df
    mat = composition_matrix(df)
    dist = pairwise_distances(mat)
    k = min(int(cfg.get("thresholds", {}).get("neighbor_count", 5)), len(df) - 1)
    feature_cols = [c for c in df.columns if any(key in c for key in ["pl_main_peak_nm", "pl_FWHM_nm", "abs_edge_nm", "abs_edge_width_nm", "optical_heterogeneity_score_total"])]
    scores = []
    for i in range(len(df)):
        nn = np.argsort(dist[i])[1 : k + 1]
        parts = []
        for col in feature_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            scale = np.nanstd(vals)
            if not np.isfinite(vals.iloc[i]) or not np.isfinite(scale) or scale < 1e-12:
                continue
            parts.append(abs(vals.iloc[i] - np.nanmean(vals.iloc[nn])) / scale)
        scores.append(float(np.nanmean(parts)) if parts else 0.0)
    df["optical_neighbor_outlier_score"] = minmax(pd.Series(scores, index=df.index))
    return df


def composition_distance_from_labeled(table: pd.DataFrame, labeled_wells: set[str]) -> pd.Series:
    """Distance to nearest XRD-labeled composition, scaled 0..1."""
    if not labeled_wells:
        return pd.Series(1.0, index=table.index)
    mat = composition_matrix(table)
    id_col = "sample_key" if "sample_key" in table.columns else "well"
    labeled_mask = table[id_col].isin(labeled_wells)
    if labeled_mask.sum() == 0:
        return pd.Series(1.0, index=table.index)
    dist = pairwise_distances(mat, mat[labeled_mask])
    nearest = dist.min(axis=1)
    return minmax(pd.Series(nearest, index=table.index))
