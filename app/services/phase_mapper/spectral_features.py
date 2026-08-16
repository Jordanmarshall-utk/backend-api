"""Spectral feature extraction for PL, absorbance, cross-modal QC, and PCA."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
from sklearn.decomposition import PCA

from .utils import minmax, safe_ratio

HC_EV_NM = 1239.841984


def _clean_spectrum(wavelength_nm, signal) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    return x[order], y[order]


def _smooth(y: np.ndarray) -> np.ndarray:
    if len(y) >= 9:
        win = min(len(y) - (1 - len(y) % 2), 15)
        if win >= 5 and win % 2 == 1:
            return savgol_filter(y, win, 2)
    return y


def _fwhm(x: np.ndarray, y: np.ndarray, peak_idx: int) -> float:
    try:
        widths = peak_widths(y, [peak_idx], rel_height=0.5)
        left = np.interp(widths[2][0], np.arange(len(x)), x)
        right = np.interp(widths[3][0], np.arange(len(x)), x)
        return float(abs(right - left))
    except Exception:
        return np.nan


def extract_pl_features(wavelength_nm, signal, cfg: dict[str, Any] | None = None) -> dict[str, float | int]:
    """Extract defensible PL features from one spectrum."""
    cfg = cfg or {}
    th = cfg.get("thresholds", cfg)
    x, y = _clean_spectrum(wavelength_nm, signal)
    if len(x) < 3:
        return {"pl_low_signal_flag": 1, "pl_saturated_like_flag": 0, "pl_failed_flag": 1}
    ys = _smooth(y)
    y0 = ys - np.nanmin(ys)
    max_y = float(np.nanmax(y0)) if len(y0) else np.nan
    prominence = max(max_y * th.get("pl_min_peak_prominence_fraction", 0.08), 1e-12)
    peaks, props = find_peaks(y0, prominence=prominence)
    if len(peaks) == 0:
        peak_idx = int(np.nanargmax(y0))
        peaks = np.array([peak_idx])
        prominences = np.array([max_y])
    else:
        prominences = props.get("prominences", y0[peaks])
    order = np.argsort(y0[peaks])[::-1]
    peaks = peaks[order]
    main = int(peaks[0])
    secondary = int(peaks[1]) if len(peaks) > 1 else None
    area = float(np.trapezoid(np.clip(y0, 0, None), x))
    centroid = float(np.sum(x * np.clip(y0, 0, None)) / max(np.sum(np.clip(y0, 0, None)), 1e-12))
    width = float(np.sqrt(np.sum(((x - centroid) ** 2) * np.clip(y0, 0, None)) / max(np.sum(np.clip(y0, 0, None)), 1e-12)))
    left = y0[x < x[main]]
    right = y0[x > x[main]]
    asym = float(abs(np.sum(left) - np.sum(right)) / max(np.sum(y0), 1e-12))
    sec_ratio = safe_ratio(float(y0[secondary]), float(y0[main]), 0.0) if secondary is not None else 0.0
    fwhm = _fwhm(x, y0, main)
    hetero = (
        min(len(peaks) - 1, 4) / 4
        + min((fwhm if np.isfinite(fwhm) else 0) / th.get("broad_pl_fwhm_nm", 60.0), 2) / 2
        + min(sec_ratio, 1.5) / 1.5
        + min(asym, 1.0)
    ) / 4
    return {
        "pl_main_peak_nm": float(x[main]),
        "pl_main_peak_eV": float(HC_EV_NM / x[main]),
        "pl_FWHM_nm": float(fwhm),
        "pl_peak_count": int(len(peaks)),
        "pl_secondary_peak_nm": float(x[secondary]) if secondary is not None else np.nan,
        "pl_secondary_main_intensity_ratio": float(sec_ratio),
        "pl_integrated_intensity": area,
        "pl_weighted_centroid_nm": centroid,
        "pl_spectral_width_nm": width,
        "pl_asymmetry_score": asym,
        "pl_heterogeneity_score": float(hetero),
        "pl_low_signal_flag": int(max_y < th.get("pl_low_signal", 100.0)),
        "pl_saturated_like_flag": int(np.nanmax(y) >= th.get("pl_saturation_fraction", 0.98) * np.nanmax(y) and np.sum(y >= np.nanmax(y)) > 2),
        "pl_failed_flag": 0,
    }


def extract_absorbance_features(wavelength_nm, signal, cfg: dict[str, Any] | None = None) -> dict[str, float | int]:
    """Extract absorbance edge and feature descriptors from one spectrum."""
    cfg = cfg or {}
    th = cfg.get("thresholds", cfg)
    x, y = _clean_spectrum(wavelength_nm, signal)
    if len(x) < 5:
        return {"abs_low_signal_flag": 1, "abs_flat_signal_flag": 1, "abs_failed_edge_flag": 1}
    ys = _smooth(y)
    yr = np.nanmax(ys) - np.nanmin(ys)
    yn = (ys - np.nanmin(ys)) / max(yr, 1e-12)
    deriv = np.gradient(yn, x)
    abs_deriv = np.abs(deriv)
    edge_idx = int(np.nanargmax(abs_deriv))
    edge_nm = float(x[edge_idx])
    edge_slope = float(deriv[edge_idx])
    lo_idx = np.argmin(np.abs(yn - 0.1))
    hi_idx = np.argmin(np.abs(yn - 0.9))
    edge_width = float(abs(x[hi_idx] - x[lo_idx]))
    d_prom = max(np.nanmax(abs_deriv) * 0.25, 1e-12)
    d_peaks, _ = find_peaks(abs_deriv, prominence=d_prom)
    prominence = max(yr * th.get("abs_min_peak_prominence_fraction", 0.05), 1e-12)
    peaks, _ = find_peaks(ys - np.nanmin(ys), prominence=prominence)
    order = np.argsort(ys[peaks])[::-1] if len(peaks) else []
    peaks = peaks[order] if len(peaks) else np.array([], dtype=int)
    main = int(peaks[0]) if len(peaks) else int(np.nanargmax(ys))
    secondary = int(peaks[1]) if len(peaks) > 1 else None
    sec_ratio = safe_ratio(float(ys[secondary] - np.nanmin(ys)), float(ys[main] - np.nanmin(ys)), 0.0) if secondary is not None else 0.0
    hetero = (
        min(len(d_peaks) - 1, 4) / 4
        + min(len(peaks) - 1, 4) / 4
        + min(sec_ratio, 1.5) / 1.5
        + min(edge_width / th.get("broad_abs_edge_width_nm", 80.0), 2) / 2
    ) / 4
    return {
        "abs_edge_nm": edge_nm,
        "abs_edge_eV": float(HC_EV_NM / edge_nm),
        "abs_edge_slope": edge_slope,
        "abs_derivative_edge_count": int(len(d_peaks)),
        "abs_edge_width_nm": edge_width,
        "abs_peak_count": int(len(peaks)),
        "abs_main_feature_nm": float(x[main]),
        "abs_secondary_feature_nm": float(x[secondary]) if secondary is not None else np.nan,
        "abs_secondary_main_ratio": float(sec_ratio),
        "abs_integrated_absorbance": float(np.trapezoid(ys, x)),
        "abs_heterogeneity_score": float(hetero),
        "abs_low_signal_flag": int(yr < th.get("absorbance_low_signal", 0.01)),
        "abs_flat_signal_flag": int(np.nanstd(ys) < th.get("absorbance_flat_std", 0.003)),
        "abs_failed_edge_flag": 0,
    }


def extract_features_per_read(long_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Extract PL/abs features for every read/well and pivot to one row per well."""
    rows = []
    for (read_name, read_type, well), g in long_df.groupby(["read_name", "read_type", "well"], dropna=False):
        if read_type == "pl":
            feats = extract_pl_features(g["wavelength_nm"], g["signal"], cfg)
        elif read_type == "absorbance":
            feats = extract_absorbance_features(g["wavelength_nm"], g["signal"], cfg)
        else:
            continue
        feats.update({"well": well, "read_name": read_name, "read_type": read_type})
        rows.append(feats)
    if not rows:
        return pd.DataFrame(columns=["well"])
    per = pd.DataFrame(rows)
    wide_parts = []
    for read_name, g in per.groupby("read_name"):
        drop = ["read_name", "read_type"]
        tmp = g.drop(columns=drop).set_index("well")
        tmp = tmp.add_prefix(f"{read_name}_")
        wide_parts.append(tmp)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    return add_cross_modal_features(wide)


def add_cross_modal_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add aggregate PL/absorbance disagreement and QC features."""
    df = features.copy()
    pl_hetero = [c for c in df.columns if c.endswith("pl_heterogeneity_score")]
    abs_hetero = [c for c in df.columns if c.endswith("abs_heterogeneity_score")]
    pl_peak = [c for c in df.columns if c.endswith("pl_main_peak_nm")]
    pl_e = [c for c in df.columns if c.endswith("pl_main_peak_eV")]
    pl_int = [c for c in df.columns if c.endswith("pl_integrated_intensity")]
    abs_edge = [c for c in df.columns if c.endswith("abs_edge_nm")]
    abs_e = [c for c in df.columns if c.endswith("abs_edge_eV")]
    pl_counts = [c for c in df.columns if c.endswith("pl_peak_count")]
    pl_secondary_ratios = [c for c in df.columns if c.endswith("pl_secondary_main_intensity_ratio")]
    pl_widths = [c for c in df.columns if c.endswith("pl_FWHM_nm")]
    pl_asymmetry = [c for c in df.columns if c.endswith("pl_asymmetry_score")]
    abs_counts = [c for c in df.columns if c.endswith("abs_peak_count")]
    abs_edge_counts = [c for c in df.columns if c.endswith("abs_derivative_edge_count")]
    abs_edge_widths = [c for c in df.columns if c.endswith("abs_edge_width_nm")]
    abs_secondary_ratios = [c for c in df.columns if c.endswith("abs_secondary_main_ratio")]
    qc = [c for c in df.columns if c.endswith("_flag")]
    df["mean_PL_heterogeneity_score"] = df[pl_hetero].mean(axis=1) if pl_hetero else 0.0
    df["mean_absorbance_heterogeneity_score"] = df[abs_hetero].mean(axis=1) if abs_hetero else 0.0
    df["optical_heterogeneity_score_total"] = df[["mean_PL_heterogeneity_score", "mean_absorbance_heterogeneity_score"]].mean(axis=1)
    df["PL_peak_range_nm"] = df[pl_peak].max(axis=1) - df[pl_peak].min(axis=1) if pl_peak else np.nan
    df["PL_peak_std_nm"] = df[pl_peak].std(axis=1) if pl_peak else np.nan
    df["abs_edge_range_nm"] = df[abs_edge].max(axis=1) - df[abs_edge].min(axis=1) if abs_edge else np.nan
    df["abs_edge_std_nm"] = df[abs_edge].std(axis=1) if abs_edge else np.nan
    if pl_e and abs_e:
        stokes = []
        for pc in pl_e:
            for ac in abs_e:
                stokes.append(df[ac] - df[pc])
        df["stokes_shift_like_eV_mean"] = pd.concat(stokes, axis=1).mean(axis=1)
        df["pl_abs_disagreement_score"] = minmax(df["stokes_shift_like_eV_mean"].abs()) + 0.5 * minmax(df.get("abs_edge_range_nm", 0)) + 0.5 * minmax(df.get("PL_peak_range_nm", 0))
    else:
        df["stokes_shift_like_eV_mean"] = np.nan
        df["pl_abs_disagreement_score"] = 0.0
    top = next((c for c in pl_peak if "TopPL" in c), None)
    bottom = next((c for c in pl_peak if "BottomPL" in c), None)
    df["top_bottom_PL_peak_difference_nm"] = df[top] - df[bottom] if top and bottom else np.nan
    top_i = next((c for c in pl_int if "TopPL" in c), None)
    bot_i = next((c for c in pl_int if "BottomPL" in c), None)
    df["top_bottom_PL_intensity_ratio"] = df[top_i] / df[bot_i].replace(0, np.nan) if top_i and bot_i else np.nan
    df["measurement_QC_flag_count"] = df[qc].sum(axis=1) if qc else 0
    df["optical_max_PL_peak_count"] = df[pl_counts].max(axis=1) if pl_counts else 0.0
    df["optical_mean_PL_peak_count"] = df[pl_counts].mean(axis=1) if pl_counts else 0.0
    df["optical_max_PL_secondary_ratio"] = df[pl_secondary_ratios].max(axis=1) if pl_secondary_ratios else 0.0
    df["optical_max_PL_FWHM_nm"] = df[pl_widths].max(axis=1) if pl_widths else np.nan
    df["optical_max_PL_asymmetry_score"] = df[pl_asymmetry].max(axis=1) if pl_asymmetry else 0.0
    df["optical_max_abs_peak_count"] = df[abs_counts].max(axis=1) if abs_counts else 0.0
    df["optical_max_abs_edge_count"] = df[abs_edge_counts].max(axis=1) if abs_edge_counts else 0.0
    df["optical_max_abs_edge_width_nm"] = df[abs_edge_widths].max(axis=1) if abs_edge_widths else np.nan
    df["optical_max_abs_secondary_ratio"] = df[abs_secondary_ratios].max(axis=1) if abs_secondary_ratios else 0.0

    pl_complexity = (
        minmax(df["optical_max_PL_peak_count"].clip(lower=1) - 1)
        + minmax(df["optical_max_PL_secondary_ratio"])
        + minmax(df["optical_max_PL_FWHM_nm"])
        + minmax(df["optical_max_PL_asymmetry_score"])
    ) / 4
    abs_complexity = (
        minmax(df["optical_max_abs_peak_count"].clip(lower=1) - 1)
        + minmax(df["optical_max_abs_edge_count"].clip(lower=1) - 1)
        + minmax(df["optical_max_abs_edge_width_nm"])
        + minmax(df["optical_max_abs_secondary_ratio"])
    ) / 4
    df["optical_PL_phase_complexity_score"] = pl_complexity
    df["optical_abs_phase_complexity_score"] = abs_complexity
    df["optical_phase_complexity_score"] = (pl_complexity + abs_complexity) / 2
    df["optical_clean_3d_support_score"] = 1.0 - df["optical_phase_complexity_score"].clip(0.0, 1.0)
    return df


def add_spectral_pca_features(long_df: pd.DataFrame, features: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add normalized spectral-shape PCA scores per read."""
    if long_df.empty or features.empty:
        return features
    out = features.set_index("well").copy()
    n_components_cfg = int(cfg.get("thresholds", {}).get("n_pca_components", 5))
    for read_name, g in long_df.groupby("read_name"):
        wells = sorted(g["well"].dropna().unique())
        if len(wells) < 3:
            continue
        xmin, xmax = g["wavelength_nm"].min(), g["wavelength_nm"].max()
        grid = np.linspace(xmin, xmax, min(250, max(25, g["wavelength_nm"].nunique())))
        mat, used = [], []
        for well in wells:
            sg = g[g["well"] == well].sort_values("wavelength_nm")
            x, y = _clean_spectrum(sg["wavelength_nm"], sg["signal"])
            if len(x) < 3:
                continue
            yi = np.interp(grid, x, y)
            yi = yi - np.nanmin(yi)
            denom = np.linalg.norm(yi)
            mat.append(yi / denom if denom > 0 else yi)
            used.append(well)
        if len(mat) < 3:
            continue
        n_components = min(n_components_cfg, len(mat), len(grid))
        scores = PCA(n_components=n_components, random_state=cfg.get("modeling", {}).get("random_state", 42)).fit_transform(np.vstack(mat))
        for i in range(n_components):
            out.loc[used, f"{read_name}_PC{i+1}"] = scores[:, i]
    return out.reset_index()
