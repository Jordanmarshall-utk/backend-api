"""Active-learning and optical remeasurement recommendations."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .composition_features import composition_distance_from_labeled, composition_matrix
from .utils import minmax


def compute_xrd_priority(table: pd.DataFrame, labeled_wells: set[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """Combine uncertainty with optical phase evidence, coverage, and outliers."""
    df = table.copy()
    weights = cfg.get("active_learning", {}).get("weights", {})
    df["recommendation_model"] = "ExtraTrees baseline"
    df["recommendation_model_note"] = "Multitask NN predictions are diagnostic only and do not affect XRD priority."
    df["model_uncertainty_score"] = minmax(df.get("model_uncertainty_score", pd.Series(0, index=df.index)))
    df["optical_heterogeneity_score_total"] = minmax(df.get("optical_heterogeneity_score_total", pd.Series(0, index=df.index)))
    df["pl_abs_disagreement_score"] = minmax(df.get("pl_abs_disagreement_score", pd.Series(0, index=df.index)))
    df["optical_phase_complexity_score"] = minmax(df.get("optical_phase_complexity_score", pd.Series(0, index=df.index)))
    df["composition_distance_score"] = composition_distance_from_labeled(df, labeled_wells)
    df["optical_neighbor_outlier_score"] = minmax(df.get("optical_neighbor_outlier_score", pd.Series(0, index=df.index)))
    df["xrd_priority_score"] = (
        weights.get("model_uncertainty", 0.30) * df["model_uncertainty_score"]
        + weights.get("optical_heterogeneity", 0.15) * df["optical_heterogeneity_score_total"]
        + weights.get("pl_abs_disagreement", 0.15) * df["pl_abs_disagreement_score"]
        + weights.get("optical_phase_complexity", 0.20) * df["optical_phase_complexity_score"]
        + weights.get("composition_distance", 0.10) * df["composition_distance_score"]
        + weights.get("optical_neighbor_outlier", 0.10) * df["optical_neighbor_outlier_score"]
    )
    return df


def _reason(row: pd.Series) -> str:
    items = {
        "high model uncertainty": row.get("model_uncertainty_score", 0),
        "high optical heterogeneity": row.get("optical_heterogeneity_score_total", 0),
        "PL/abs disagreement": row.get("pl_abs_disagreement_score", 0),
        "optical evidence of phase complexity": row.get("optical_phase_complexity_score", 0),
        "composition coverage gap": row.get("composition_distance_score", 0),
        "optical outlier relative to neighbors": row.get("optical_neighbor_outlier_score", 0),
    }
    return "; ".join(k for k, _ in sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:2])


def select_diverse_recommendations(priority_table: pd.DataFrame, labeled_wells: set[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """Greedy batch selection with a composition diversity penalty."""
    n = int(cfg.get("active_learning", {}).get("recommend_n", 16))
    lam = float(cfg.get("active_learning", {}).get("diversity_lambda", 0.15))
    id_col = "sample_key" if "sample_key" in priority_table.columns else "well"
    candidates = priority_table[~priority_table[id_col].isin(labeled_wells)].copy()
    if candidates.empty:
        return candidates
    mat = composition_matrix(candidates)
    selected_idx: list[int] = []
    remaining = list(candidates.index)
    while remaining and len(selected_idx) < n:
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            base = float(candidates.loc[idx, "xrd_priority_score"])
            penalty = 0.0
            if selected_idx:
                d = pairwise_distances(mat.loc[[idx]], mat.loc[selected_idx]).min()
                penalty = lam * (1 - min(float(d), 1.0))
            score = base - penalty
            if score > best_score:
                best_idx, best_score = idx, score
        selected_idx.append(best_idx)
        remaining.remove(best_idx)
    rec = candidates.loc[selected_idx].copy()
    rec["reason_for_recommendation"] = rec.apply(_reason, axis=1)
    return rec.sort_values("xrd_priority_score", ascending=False)


def remeasurement_recommendations(table: pd.DataFrame) -> pd.DataFrame:
    """Rank wells likely to need optical remeasurement before spending XRD effort."""
    df = table.copy()
    flag_cols = [c for c in df.columns if any(s in c for s in ["low_signal_flag", "saturated_like_flag", "flat_signal_flag", "failed_edge_flag", "failed_flag"])]
    df["missing_spectra_score"] = df.filter(regex="(pl_main_peak_nm|abs_edge_nm)$").isna().mean(axis=1) if len(df.filter(regex="(pl_main_peak_nm|abs_edge_nm)$").columns) else 1.0
    df["optical_remeasurement_priority_score"] = minmax(df[flag_cols].sum(axis=1) if flag_cols else pd.Series(0, index=df.index)) + df["missing_spectra_score"]
    df["optical_remeasurement_priority_score"] = minmax(df["optical_remeasurement_priority_score"])
    def reason(row):
        bits = []
        if row.get("missing_spectra_score", 0) > 0:
            bits.append("missing or failed spectral features")
        for col in flag_cols:
            if row.get(col, 0) > 0:
                bits.append(col)
        return "; ".join(bits[:4]) or "no major optical QC issue"
    df["reason_for_remeasurement"] = df.apply(reason, axis=1)
    return df.sort_values("optical_remeasurement_priority_score", ascending=False)
