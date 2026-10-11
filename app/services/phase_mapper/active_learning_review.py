"""Run-to-run active-learning review tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .xrd_labels import BINARY_TARGETS, REGRESSION_TARGETS


ACTIVE_LEARNING_CYCLE_COLUMNS = [
    "recommended_in_run",
    "recommended_at",
    "sample_key",
    "library",
    "well",
    "row",
    "col",
    "task",
    "task_type",
    "actual_xrd_label",
    "before_baseline_prediction",
    "before_baseline_probability",
    "after_baseline_prediction",
    "after_baseline_probability",
    "before_baseline_correct",
    "after_baseline_correct",
    "before_multitask_mlp_prediction",
    "before_multitask_mlp_probability",
    "after_multitask_mlp_prediction",
    "after_multitask_mlp_probability",
    "before_multitask_mlp_correct",
    "after_multitask_mlp_correct",
    "before_xrd_priority_score",
    "after_xrd_priority_score",
]

ACTIVE_LEARNING_CYCLE_SUMMARY_COLUMNS = [
    "task",
    "task_type",
    "reviewed_recommended_labels",
    "before_baseline_predictions_available",
    "before_baseline_prediction_coverage",
    "before_baseline_accuracy",
    "after_baseline_predictions_available",
    "after_baseline_prediction_coverage",
    "after_baseline_accuracy",
    "baseline_accuracy_delta_after_minus_before",
    "before_multitask_mlp_predictions_available",
    "before_multitask_mlp_prediction_coverage",
    "before_multitask_mlp_accuracy",
    "after_multitask_mlp_predictions_available",
    "after_multitask_mlp_prediction_coverage",
    "after_multitask_mlp_accuracy",
    "multitask_mlp_accuracy_delta_after_minus_before",
    "before_baseline_mae",
    "after_baseline_mae",
    "baseline_mae_improvement_before_minus_after",
    "before_multitask_mlp_mae",
    "after_multitask_mlp_mae",
    "multitask_mlp_mae_improvement_before_minus_after",
]


def prediction_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pred_") or c.startswith("mtl_pred_")]


def actual_label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in BINARY_TARGETS + REGRESSION_TARGETS if c in df.columns]


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if "sample_key" in left.columns and "sample_key" in right.columns:
        return ["sample_key"]
    keys = [c for c in ["library", "well"] if c in left.columns and c in right.columns]
    if keys:
        return keys
    if "well" in left.columns and "well" in right.columns:
        return ["well"]
    return []


def _unique_existing_columns(columns: list[str], df: pd.DataFrame) -> list[str]:
    out = []
    seen = set()
    for col in columns:
        if col in df.columns and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def reviewed_label_summary(review: pd.DataFrame) -> pd.DataFrame:
    """Summarize prediction-vs-actual rows by target."""
    if review.empty or "task" not in review.columns:
        return pd.DataFrame()
    rows = []
    for (task, task_type), group in review.groupby(["task", "task_type"], dropna=False):
        row = {"task": task, "task_type": task_type, "reviewed_labels": len(group)}
        for prefix in ["baseline", "multitask_mlp"]:
            correct_col = f"{prefix}_correct"
            err_col = f"{prefix}_absolute_error"
            if correct_col in group.columns and group[correct_col].notna().any():
                row[f"{prefix}_accuracy_on_reviewed"] = group[correct_col].dropna().astype(bool).mean()
            if err_col in group.columns and group[err_col].notna().any():
                row[f"{prefix}_mae_on_reviewed"] = pd.to_numeric(group[err_col], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def build_active_learning_cycle_comparison(previous_recommendations: pd.DataFrame, current_predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare recommendation-time predictions with current reviewed XRD labels.

    This is the active-learning audit table:
      prediction before XRD measurement -> reviewed XRD label -> prediction after retraining.

    The before predictions come from saved recommendation tables from older runs,
    so they are not overwritten by the retrained model.
    """
    if previous_recommendations.empty or current_predictions.empty:
        return pd.DataFrame(columns=ACTIVE_LEARNING_CYCLE_COLUMNS)

    merge_keys = choose_merge_keys(previous_recommendations, current_predictions)
    if not merge_keys:
        return pd.DataFrame(columns=ACTIVE_LEARNING_CYCLE_COLUMNS)

    previous_keep = _unique_existing_columns(
        [
            *merge_keys,
            "recommended_in_run",
            "recommended_at",
            "reason_for_recommendation",
            "xrd_priority_score",
            "model_uncertainty_score",
            "multitask_mlp_uncertainty_score",
        ]
        + prediction_columns(previous_recommendations)
        ,
        previous_recommendations,
    )
    previous = previous_recommendations[previous_keep].copy()
    # A well can remain recommended across several iterations before XRD is
    # collected. Score it once using the latest prediction that was actually
    # saved before measurement, rather than inflating prospective sample counts.
    if "recommended_at" in previous.columns:
        previous["_recommended_at_sort"] = pd.to_datetime(previous["recommended_at"], errors="coerce")
        previous = previous.sort_values("_recommended_at_sort", na_position="first")
    previous = previous.drop_duplicates(subset=merge_keys, keep="last")
    previous = previous.drop(columns=["_recommended_at_sort"], errors="ignore")
    previous = previous.rename(
        columns={
            "xrd_priority_score": "before_xrd_priority_score",
            "model_uncertainty_score": "before_model_uncertainty_score",
            "multitask_mlp_uncertainty_score": "before_multitask_mlp_uncertainty_score",
            **{c: f"before_{c}" for c in prediction_columns(previous)},
        }
    )

    current_keep = _unique_existing_columns(
        [
            *merge_keys,
            "sample_key",
            "library",
            "well",
            "row",
            "col",
            "xrd_priority_score",
            "model_uncertainty_score",
            "multitask_mlp_uncertainty_score",
        ]
        + actual_label_columns(current_predictions)
        + prediction_columns(current_predictions)
        ,
        current_predictions,
    )
    current = current_predictions[current_keep].copy()
    current = current.rename(
        columns={
            "xrd_priority_score": "after_xrd_priority_score",
            "model_uncertainty_score": "after_model_uncertainty_score",
            "multitask_mlp_uncertainty_score": "after_multitask_mlp_uncertainty_score",
            **{c: f"after_{c}" for c in prediction_columns(current_predictions)},
        }
    )

    merged = previous.merge(current, on=merge_keys, how="inner")
    label_cols = [c for c in actual_label_columns(current_predictions) if c in merged.columns]
    if not label_cols:
        return pd.DataFrame(columns=ACTIVE_LEARNING_CYCLE_COLUMNS)

    rows = []
    id_cols = [c for c in ["sample_key", "library", "well", "row", "col"] if c in merged.columns]
    context_cols = [
        c
        for c in [
            "recommended_in_run",
            "recommended_at",
            "reason_for_recommendation",
            "before_xrd_priority_score",
            "before_model_uncertainty_score",
            "before_multitask_mlp_uncertainty_score",
            "after_xrd_priority_score",
            "after_model_uncertainty_score",
            "after_multitask_mlp_uncertainty_score",
        ]
        if c in merged.columns
    ]

    for _, row in merged.iterrows():
        base = {c: row.get(c) for c in id_cols + context_cols}
        for target in label_cols:
            actual = row.get(target)
            if pd.isna(actual):
                continue
            task_type = "classification" if target in BINARY_TARGETS else "regression"
            out = {**base, "task": target, "task_type": task_type, "actual_xrd_label": actual}
            before_pred = row.get(f"before_pred_{target}", np.nan)
            after_pred = row.get(f"after_pred_{target}", np.nan)
            before_mtl = row.get(f"before_mtl_pred_{target}", np.nan)
            after_mtl = row.get(f"after_mtl_pred_{target}", np.nan)
            out["before_baseline_prediction"] = before_pred
            out["after_baseline_prediction"] = after_pred
            out["before_multitask_mlp_prediction"] = before_mtl
            out["after_multitask_mlp_prediction"] = after_mtl

            for source_name, source_col in [
                ("before_baseline_probability", f"before_pred_{target}_prob"),
                ("after_baseline_probability", f"after_pred_{target}_prob"),
                ("before_multitask_mlp_probability", f"before_mtl_pred_{target}_prob"),
                ("after_multitask_mlp_probability", f"after_mtl_pred_{target}_prob"),
            ]:
                if source_col in merged.columns:
                    out[source_name] = row.get(source_col)

            if task_type == "classification":
                actual_int = int(round(float(actual)))
                if pd.notna(before_pred):
                    out["before_baseline_correct"] = int(round(float(before_pred))) == actual_int
                if pd.notna(after_pred):
                    out["after_baseline_correct"] = int(round(float(after_pred))) == actual_int
                if pd.notna(before_mtl):
                    out["before_multitask_mlp_correct"] = int(round(float(before_mtl))) == actual_int
                if pd.notna(after_mtl):
                    out["after_multitask_mlp_correct"] = int(round(float(after_mtl))) == actual_int
            else:
                if pd.notna(before_pred):
                    out["before_baseline_absolute_error"] = abs(float(before_pred) - float(actual))
                if pd.notna(after_pred):
                    out["after_baseline_absolute_error"] = abs(float(after_pred) - float(actual))
                if pd.notna(before_mtl):
                    out["before_multitask_mlp_absolute_error"] = abs(float(before_mtl) - float(actual))
                if pd.notna(after_mtl):
                    out["after_multitask_mlp_absolute_error"] = abs(float(after_mtl) - float(actual))
            rows.append(out)
    return pd.DataFrame(rows, columns=ACTIVE_LEARNING_CYCLE_COLUMNS if not rows else None)


def active_learning_cycle_summary(cycle: pd.DataFrame) -> pd.DataFrame:
    """Summarize before-vs-after performance on newly reviewed recommendation wells."""
    if cycle.empty or "task" not in cycle.columns:
        return pd.DataFrame(columns=ACTIVE_LEARNING_CYCLE_SUMMARY_COLUMNS)
    rows = []
    for (task, task_type), group in cycle.groupby(["task", "task_type"], dropna=False):
        row = {"task": task, "task_type": task_type, "reviewed_recommended_labels": len(group)}
        if task_type == "classification":
            for prefix in ["baseline", "multitask_mlp"]:
                before_col = f"before_{prefix}_correct"
                after_col = f"after_{prefix}_correct"
                before_available = int(group[before_col].notna().sum()) if before_col in group.columns else 0
                after_available = int(group[after_col].notna().sum()) if after_col in group.columns else 0
                row[f"before_{prefix}_predictions_available"] = before_available
                row[f"before_{prefix}_prediction_coverage"] = before_available / len(group)
                row[f"after_{prefix}_predictions_available"] = after_available
                row[f"after_{prefix}_prediction_coverage"] = after_available / len(group)
                if before_col in group.columns and group[before_col].notna().any():
                    row[f"before_{prefix}_accuracy"] = group[before_col].dropna().astype(bool).mean()
                if after_col in group.columns and group[after_col].notna().any():
                    row[f"after_{prefix}_accuracy"] = group[after_col].dropna().astype(bool).mean()
                if f"before_{prefix}_accuracy" in row and f"after_{prefix}_accuracy" in row:
                    row[f"{prefix}_accuracy_delta_after_minus_before"] = row[f"after_{prefix}_accuracy"] - row[f"before_{prefix}_accuracy"]
        else:
            for prefix in ["baseline", "multitask_mlp"]:
                before_col = f"before_{prefix}_absolute_error"
                after_col = f"after_{prefix}_absolute_error"
                if before_col in group.columns and group[before_col].notna().any():
                    row[f"before_{prefix}_mae"] = pd.to_numeric(group[before_col], errors="coerce").mean()
                if after_col in group.columns and group[after_col].notna().any():
                    row[f"after_{prefix}_mae"] = pd.to_numeric(group[after_col], errors="coerce").mean()
                if f"before_{prefix}_mae" in row and f"after_{prefix}_mae" in row:
                    row[f"{prefix}_mae_improvement_before_minus_after"] = row[f"before_{prefix}_mae"] - row[f"after_{prefix}_mae"]
        rows.append(row)
    return pd.DataFrame(rows, columns=ACTIVE_LEARNING_CYCLE_SUMMARY_COLUMNS if not rows else None)
