"""Multitask first-pass modeling for XRD-derived phase labels."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .xrd_labels import BINARY_TARGETS, REGRESSION_TARGETS


NON_FEATURE_PREFIXES = ("xrd_",)
NON_FEATURE_COLUMNS = {
    "well",
    "row",
    "col",
    "library",
    "sample_key",
    "is_control",
    "replicate_group",
    "notes",
    "profile_id",
    "plate_id",
    "two_d_precursor",
    "three_d_precursor",
    "additive",
    "composition_substitutions_applied",
    "read_label",
    "read_name",
}


def configured_targets(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return independently modeled targets after applying configured exclusions."""
    default_excluded = [
        "xrd_3d_phase_segregation_present",
        "xrd_peak_splitting_present",
        "xrd_200_peak_splitting_present",
    ]
    excluded = set(cfg.get("modeling", {}).get("excluded_targets", default_excluded) or [])
    return (
        [target for target in BINARY_TARGETS if target not in excluded],
        [target for target in REGRESSION_TARGETS if target not in excluded],
    )


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Numeric feature columns excluding XRD targets and identifiers."""
    cols = []
    for col in table.columns:
        if col in NON_FEATURE_COLUMNS or col.startswith(NON_FEATURE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(table[col]):
            cols.append(col)
    return cols


def feature_family_columns(table: pd.DataFrame) -> dict[str, list[str]]:
    """Split model inputs into optical-only, composition-only, and combined families."""
    combined = feature_columns(table)
    optical_tokens = (
        "pl_", "abs_", "optical_", "stokes_", "measurement_qc_",
        "top_bottom_", "mean_pl_", "mean_absorbance_",
    )
    optical_prefixes = ("toppl", "bottompl", "abs_", "pl_", "read")

    def is_optical(col: str) -> bool:
        lower = col.lower()
        return (
            lower.startswith(optical_prefixes)
            or any(token in lower for token in optical_tokens)
            or "_pc" in lower
        )

    optical = [col for col in combined if is_optical(col)]
    composition = [col for col in combined if col not in optical]
    return {"composition_only": composition, "optical_only": optical, "combined": combined}


def compare_feature_family_models(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Evaluate whether optical measurements add predictive value beyond composition."""
    families = feature_family_columns(table)
    rows: list[dict[str, Any]] = []
    min_labels = int(cfg.get("thresholds", {}).get("min_xrd_labels_for_training", 5))
    estimator_kind = cfg.get("modeling", {}).get("estimator", "extra_trees")
    test_size = float(cfg.get("modeling", {}).get("test_size", 0.3))
    random_state = int(cfg.get("modeling", {}).get("random_state", 42))
    targets = [
        target
        for target in ["xrd_clean_3d_perovskite", "xrd_mixed_phase", "xrd_secondary_phase_present"]
        if target in table.columns
    ]
    for target in targets:
        mask = table[target].notna()
        if cfg.get("thresholds", {}).get("exclude_low_confidence_xrd", False) and "xrd_assignment_confidence" in table.columns:
            mask &= table["xrd_assignment_confidence"].astype(str).str.lower() != "low"
        y = pd.to_numeric(table.loc[mask, target], errors="coerce")
        if len(y) < max(8, min_labels) or y.nunique(dropna=True) < 2 or y.value_counts().min() < 2:
            continue
        train_idx, test_idx = train_test_split(
            y.index,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        for family, candidates in families.items():
            feats = _observed_features(table.loc[train_idx], candidates)
            if not feats:
                continue
            model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", _make_estimator(estimator_kind, "classification", cfg))])
            weights = (
                pd.to_numeric(table.loc[train_idx, "xrd_sample_weight"], errors="coerce").fillna(1.0)
                if "xrd_sample_weight" in table.columns
                else None
            )
            fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
            model.fit(table.loc[train_idx, feats], table.loc[train_idx, target], **fit_kwargs)
            pred = model.predict(table.loc[test_idx, feats])
            rows.append(
                {
                    "task": target,
                    "feature_family": family,
                    "n_labels": len(y),
                    "n_features": len(feats),
                    "evaluation_scope": "same_held_out_rows_across_feature_families",
                    **classification_metric_row(table.loc[test_idx, target], pred),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    combined = out[out["feature_family"].eq("combined")][["task", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "combined_balanced_accuracy"}
    )
    out = out.merge(combined, on="task", how="left")
    out["balanced_accuracy_delta_vs_combined"] = out["balanced_accuracy"] - out["combined_balanced_accuracy"]
    return out


def _make_estimator(kind: str, task: str, cfg: dict[str, Any]):
    n = int(cfg.get("modeling", {}).get("n_estimators", 300))
    rs = int(cfg.get("modeling", {}).get("random_state", 42))
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=n, random_state=rs, class_weight="balanced") if task == "classification" else RandomForestRegressor(n_estimators=n, random_state=rs)
    return ExtraTreesClassifier(n_estimators=n, random_state=rs, class_weight="balanced") if task == "classification" else ExtraTreesRegressor(n_estimators=n, random_state=rs)


def _tree_regression_uncertainty(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    imputed = model.named_steps["imputer"].transform(x)
    est = model.named_steps["model"]
    preds = np.vstack([tree.predict(imputed) for tree in est.estimators_])
    return np.std(preds, axis=0)


def _align_prediction_features(predict_table: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    return predict_table.reindex(columns=feats)


def _observed_features(table: pd.DataFrame, feats: list[str], min_observed: int = 1) -> list[str]:
    """Keep features with enough real observations and variation in training rows."""
    observed = []
    varying = []
    for col in feats:
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce").dropna()
        if len(values) >= min_observed:
            observed.append(col)
            if values.nunique() > 1:
                varying.append(col)
    # Tiny first-cycle datasets can legitimately have no varying descriptor.
    return varying or observed


def classification_metric_row(truth, pred) -> dict[str, Any]:
    """Imbalance-aware binary classification metrics and confusion counts."""
    truth = pd.Series(truth).astype(int)
    pred = pd.Series(pred, index=truth.index).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    majority_accuracy = float(truth.value_counts(normalize=True).max()) if len(truth) else np.nan
    return {
        "accuracy": accuracy_score(truth, pred),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
        "positive_precision": precision_score(truth, pred, pos_label=1, zero_division=0),
        "positive_recall": recall_score(truth, pred, pos_label=1, zero_division=0),
        "positive_f1": f1_score(truth, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(truth, pred, pos_label=0, zero_division=0),
        "majority_class_accuracy": majority_accuracy,
        "accuracy_above_majority": accuracy_score(truth, pred) - majority_accuracy,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def train_multitask_models(table: pd.DataFrame, cfg: dict[str, Any], predict_table: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train one model per available XRD target and predict the requested wells."""
    predict_table = table if predict_table is None else predict_table
    feats = feature_columns(table)
    id_cols = ["sample_key"] if "sample_key" in predict_table.columns else [c for c in ["library", "well"] if c in predict_table.columns]
    predictions = predict_table[id_cols].copy()
    predictions["model_uncertainty_score"] = 0.0
    metrics_rows = []
    importance_rows = []
    models = {}
    if not feats:
        return models, predictions, pd.DataFrame(metrics_rows), pd.DataFrame(importance_rows)
    min_labels = int(cfg.get("thresholds", {}).get("min_xrd_labels_for_training", 5))
    estimator_kind = cfg.get("modeling", {}).get("estimator", "extra_trees")
    uncertainties = []

    binary_targets, regression_targets = configured_targets(cfg)
    for target in binary_targets + regression_targets:
        if target not in table.columns:
            continue
        task = "classification" if target in binary_targets else "regression"
        mask = table[target].notna()
        if cfg.get("thresholds", {}).get("exclude_low_confidence_xrd", False) and "xrd_assignment_confidence" in table.columns:
            mask &= table["xrd_assignment_confidence"].astype(str).str.lower() != "low"
        n_labels = int(mask.sum())
        if n_labels < min_labels:
            metrics_rows.append({"task": target, "task_type": task, "n_labels": n_labels, "status": "skipped_too_few_labels", "warning": "Metrics require enough reviewed labels for a held-out test set."})
            continue
        target_feats = _observed_features(table.loc[mask], feats)
        if not target_feats:
            metrics_rows.append({"task": target, "task_type": task, "n_labels": n_labels, "status": "skipped_no_observed_features", "warning": "No optical or composition features were observed for the reviewed labels."})
            continue
        x = table.loc[mask, target_feats]
        y = table.loc[mask, target]
        weights = (
            pd.to_numeric(table.loc[mask, "xrd_sample_weight"], errors="coerce").fillna(1.0)
            if "xrd_sample_weight" in table.columns
            else None
        )
        if task == "classification" and y.nunique(dropna=True) < 2:
            metrics_rows.append({"task": target, "task_type": task, "n_labels": n_labels, "status": "skipped_one_class", "warning": "Only one class present."})
            continue
        if task == "regression" and y.nunique(dropna=True) < 2:
            metrics_rows.append({"task": target, "task_type": task, "n_labels": n_labels, "status": "skipped_constant_target", "warning": "Only one target value present."})
            continue
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", _make_estimator(estimator_kind, task, cfg))])
        test_size = float(cfg.get("modeling", {}).get("test_size", 0.3))
        can_split = n_labels >= 8 and (task == "regression" or y.value_counts().min() >= 2)
        if can_split:
            stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
            train_idx, test_idx = train_test_split(x.index, test_size=test_size, random_state=cfg.get("modeling", {}).get("random_state", 42), stratify=stratify)
            fit_kwargs = {"model__sample_weight": weights.loc[train_idx]} if weights is not None else {}
            model.fit(table.loc[train_idx, target_feats], table.loc[train_idx, target], **fit_kwargs)
            pred_test = model.predict(table.loc[test_idx, target_feats])
            y_test = table.loc[test_idx, target]
            if task == "classification":
                metrics_rows.append(
                    {
                        "task": target,
                        "task_type": task,
                        "n_labels": n_labels,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "status": "trained",
                        **classification_metric_row(y_test, pred_test),
                        "warning": "Held-out metrics are not reliable for small labeled datasets.",
                    }
                )
            else:
                mse = mean_squared_error(y_test, pred_test)
                metrics_rows.append(
                    {
                        "task": target,
                        "task_type": task,
                        "n_labels": n_labels,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "status": "trained",
                        "rmse": float(np.sqrt(mse)),
                        "mae": mean_absolute_error(y_test, pred_test),
                        "r2": r2_score(y_test, pred_test),
                        "warning": "Held-out metrics are not reliable for small labeled datasets.",
                    }
                )
        else:
            metrics_rows.append(
                {
                    "task": target,
                    "task_type": task,
                    "n_labels": n_labels,
                    "status": "trained_no_holdout",
                    "warning": "Model was fit, but there were not enough labels/classes for a held-out performance estimate.",
                }
            )
        fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
        model.fit(x, y, **fit_kwargs)
        models[target] = model
        all_x = _align_prediction_features(predict_table, target_feats)
        if task == "classification":
            proba = model.predict_proba(all_x)
            classes = list(model.named_steps["model"].classes_)
            prob_one = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(predict_table))
            predictions[f"pred_{target}_prob"] = prob_one
            predictions[f"pred_{target}"] = (prob_one >= 0.5).astype(int)
            unc = 1 - np.max(proba, axis=1)
        else:
            pred = model.predict(all_x)
            predictions[f"pred_{target}"] = pred
            unc = _tree_regression_uncertainty(model, all_x)
        uncertainties.append(pd.Series(unc, index=predict_table.index))
        est = model.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            for col, val in zip(target_feats, est.feature_importances_):
                importance_rows.append({"task": target, "feature": col, "importance": float(val)})
    if uncertainties:
        predictions["model_uncertainty_score"] = pd.concat(uncertainties, axis=1).mean(axis=1)
        m = predictions["model_uncertainty_score"]
        if m.max() > m.min():
            predictions["model_uncertainty_score"] = (m - m.min()) / (m.max() - m.min())
    return models, predictions, pd.DataFrame(metrics_rows), pd.DataFrame(importance_rows)


def leave_one_library_out_tree_metrics(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Measure transfer by training on other libraries and testing one unseen library."""
    columns = [
        "model_family", "held_out_library", "task", "task_type", "n_train_labels", "n_test_labels", "status",
        "accuracy", "balanced_accuracy", "weighted_f1", "positive_precision", "positive_recall", "positive_f1",
        "specificity", "majority_class_accuracy", "accuracy_above_majority", "true_negative", "false_positive",
        "false_negative", "true_positive", "rmse", "mae", "r2",
    ]
    if "library" not in table.columns or table["library"].dropna().nunique() < 2:
        return pd.DataFrame(columns=columns)
    feats = feature_columns(table)
    if not feats:
        return pd.DataFrame(columns=columns)
    rows = []
    min_labels = int(cfg.get("thresholds", {}).get("min_xrd_labels_for_training", 5))
    estimator_kind = cfg.get("modeling", {}).get("estimator", "extra_trees")
    for held_out in sorted(table["library"].dropna().astype(str).unique()):
        binary_targets, regression_targets = configured_targets(cfg)
        for target in binary_targets + regression_targets:
            if target not in table.columns:
                continue
            task = "classification" if target in binary_targets else "regression"
            train_mask = table[target].notna() & (table["library"].astype(str) != held_out)
            test_mask = table[target].notna() & (table["library"].astype(str) == held_out)
            train = table.loc[train_mask].copy()
            test = table.loc[test_mask].copy()
            if len(train) < min_labels or len(test) < 2:
                continue
            y_train = pd.to_numeric(train[target], errors="coerce")
            y_test = pd.to_numeric(test[target], errors="coerce")
            if y_train.nunique(dropna=True) < 2:
                continue
            transfer_feats = _observed_features(train, feats)
            if not transfer_feats:
                continue
            model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", _make_estimator(estimator_kind, task, cfg))])
            weights = (
                pd.to_numeric(train["xrd_sample_weight"], errors="coerce").fillna(1.0)
                if "xrd_sample_weight" in train.columns
                else pd.Series(1.0, index=train.index)
            )
            model.fit(train[transfer_feats], y_train, model__sample_weight=weights)
            pred = model.predict(_align_prediction_features(test, transfer_feats))
            row = {
                "model_family": "per_target_tree_baseline",
                "held_out_library": held_out,
                "task": target,
                "task_type": task,
                "n_train_labels": len(train),
                "n_test_labels": len(test),
                "status": "leave_one_library_out",
            }
            if task == "classification":
                row.update(classification_metric_row(y_test, pred))
            else:
                mse = mean_squared_error(y_test, pred)
                row.update({"rmse": float(np.sqrt(mse)), "mae": mean_absolute_error(y_test, pred), "r2": r2_score(y_test, pred)})
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def build_out_of_fold_tree_review_predictions(model_training: pd.DataFrame, current_table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Predict reviewed current-run XRD labels with models that did not train on that sample.

    This guards against the very optimistic 0/1 probabilities that tree ensembles
    can produce when asked to predict rows they just trained on.
    """
    feats = feature_columns(model_training)
    if not feats:
        return pd.DataFrame(columns=["task", "task_type", "actual_xrd_label", "baseline_prediction", "baseline_probability", "baseline_correct", "evaluation_method"])

    min_labels = int(cfg.get("thresholds", {}).get("min_xrd_labels_for_training", 5))
    estimator_kind = cfg.get("modeling", {}).get("estimator", "extra_trees")
    random_state = int(cfg.get("modeling", {}).get("random_state", 42))
    max_splits = int(cfg.get("modeling", {}).get("review_oof_splits", 5))
    id_cols = [c for c in ["sample_key", "library", "well", "row", "col"] if c in current_table.columns]
    rows: list[dict[str, Any]] = []

    binary_targets, regression_targets = configured_targets(cfg)
    for target in binary_targets + regression_targets:
        if target not in model_training.columns or target not in current_table.columns:
            continue
        task = "classification" if target in binary_targets else "regression"
        current_y = pd.to_numeric(current_table[target], errors="coerce")
        current_labeled = current_table[current_y.notna()].copy()
        if current_labeled.empty:
            continue
        current_labeled[target] = current_y.loc[current_labeled.index]
        if len(current_labeled) < 2:
            continue

        if task == "classification":
            counts = current_labeled[target].value_counts(dropna=True)
            if len(counts) < 2 or counts.min() < 2:
                continue
            n_splits = min(max_splits, len(current_labeled), int(counts.min()))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_iter = splitter.split(current_labeled, current_labeled[target])
        else:
            n_splits = min(max_splits, len(current_labeled))
            if n_splits < 2:
                continue
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_iter = splitter.split(current_labeled)

        for fold, (_, val_pos) in enumerate(split_iter, start=1):
            val = current_labeled.iloc[val_pos].copy()
            train_mask = model_training[target].notna()
            if cfg.get("thresholds", {}).get("exclude_low_confidence_xrd", False) and "xrd_assignment_confidence" in model_training.columns:
                train_mask &= model_training["xrd_assignment_confidence"].astype(str).str.lower() != "low"
            if "sample_key" in model_training.columns and "sample_key" in val.columns:
                train_mask &= ~model_training["sample_key"].isin(val["sample_key"])
            elif "well" in model_training.columns and "well" in val.columns:
                train_mask &= ~model_training["well"].isin(val["well"])

            train = model_training.loc[train_mask].copy()
            n_train = len(train)
            if n_train < min_labels:
                continue
            y_train = pd.to_numeric(train[target], errors="coerce")
            valid_train = y_train.notna()
            train = train.loc[valid_train]
            y_train = y_train.loc[valid_train]
            if task == "classification" and y_train.nunique(dropna=True) < 2:
                continue
            if task == "regression" and y_train.nunique(dropna=True) < 2:
                continue

            fold_feats = _observed_features(train, feats)
            if not fold_feats:
                continue
            model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", _make_estimator(estimator_kind, task, cfg))])
            weights = (
                pd.to_numeric(train["xrd_sample_weight"], errors="coerce").fillna(1.0)
                if "xrd_sample_weight" in train.columns
                else None
            )
            fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
            model.fit(train[fold_feats], y_train, **fit_kwargs)
            x_val = _align_prediction_features(val, fold_feats)
            pred = model.predict(x_val)

            if task == "classification":
                proba = model.predict_proba(x_val)
                classes = list(model.named_steps["model"].classes_)
                prob_one = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(val))
            else:
                prob_one = np.full(len(val), np.nan)

            for i, (_, val_row) in enumerate(val.iterrows()):
                actual = val_row[target]
                out = {c: val_row.get(c) for c in id_cols}
                out.update(
                    {
                        "task": target,
                        "task_type": task,
                        "actual_xrd_label": actual,
                        "baseline_prediction": pred[i],
                        "n_train_labels": n_train,
                        "fold": fold,
                        "evaluation_method": "out_of_fold_current_review_label",
                    }
                )
                if task == "classification":
                    out["baseline_probability"] = float(prob_one[i])
                    out["baseline_correct"] = int(round(float(pred[i]))) == int(round(float(actual)))
                else:
                    out["baseline_absolute_error"] = abs(float(pred[i]) - float(actual))
                rows.append(out)

    return pd.DataFrame(rows)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def _normalizer_from_table(table: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_raw = table[feats].to_numpy(dtype=float)
    med = table[feats].median(numeric_only=True).reindex(feats).fillna(0.0).to_numpy(dtype=float)
    x_imp = np.where(np.isfinite(x_raw), x_raw, med)
    mean = np.mean(x_imp, axis=0)
    std = np.std(x_imp, axis=0)
    std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
    return med, mean, std


def _normalize_features(table: pd.DataFrame, feats: list[str], med: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_raw = _align_prediction_features(table, feats).to_numpy(dtype=float)
    x_imp = np.where(np.isfinite(x_raw), x_raw, med)
    return (x_imp - mean) / std


def _available_shared_targets(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    min_labels = int(cfg.get("modeling", {}).get("multitask_mlp", {}).get("min_labeled_rows", cfg.get("thresholds", {}).get("min_xrd_labels_for_training", 5)))
    rows: list[dict[str, Any]] = []
    cls_targets: list[str] = []
    reg_targets: list[str] = []
    confidence_mask = pd.Series(True, index=table.index)
    if cfg.get("thresholds", {}).get("exclude_low_confidence_xrd", False) and "xrd_assignment_confidence" in table.columns:
        confidence_mask = table["xrd_assignment_confidence"].astype(str).str.lower() != "low"

    binary_targets, regression_targets = configured_targets(cfg)
    for target in binary_targets:
        if target not in table.columns:
            continue
        y = pd.to_numeric(table[target], errors="coerce")
        mask = y.notna() & confidence_mask
        n_labels = int(mask.sum())
        if n_labels < min_labels:
            rows.append({"task": target, "task_type": "classification", "n_labels": n_labels, "status": "skipped_too_few_labels"})
            continue
        values = sorted(y.loc[mask].dropna().unique())
        if len(values) < 2:
            rows.append({"task": target, "task_type": "classification", "n_labels": n_labels, "status": "skipped_one_class"})
            continue
        cls_targets.append(target)

    for target in regression_targets:
        if target not in table.columns:
            continue
        y = pd.to_numeric(table[target], errors="coerce")
        mask = y.notna() & confidence_mask
        n_labels = int(mask.sum())
        if n_labels < min_labels:
            rows.append({"task": target, "task_type": "regression", "n_labels": n_labels, "status": "skipped_too_few_labels"})
            continue
        if y.loc[mask].nunique(dropna=True) < 2:
            rows.append({"task": target, "task_type": "regression", "n_labels": n_labels, "status": "skipped_constant_target"})
            continue
        reg_targets.append(target)
    return cls_targets, reg_targets, rows


def _target_matrices(table: pd.DataFrame, cls_targets: list[str], reg_targets: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_cls = np.column_stack([pd.to_numeric(table[t], errors="coerce").to_numpy(dtype=float) for t in cls_targets]) if cls_targets else np.zeros((len(table), 0))
    m_cls = np.isfinite(y_cls)
    y_cls = np.where(m_cls, y_cls, 0.0)

    y_reg_raw = np.column_stack([pd.to_numeric(table[t], errors="coerce").to_numpy(dtype=float) for t in reg_targets]) if reg_targets else np.zeros((len(table), 0))
    m_reg = np.isfinite(y_reg_raw)
    reg_mean = np.zeros(len(reg_targets), dtype=float)
    reg_std = np.ones(len(reg_targets), dtype=float)
    y_reg = np.zeros_like(y_reg_raw)
    for j in range(len(reg_targets)):
        vals = y_reg_raw[m_reg[:, j], j]
        if len(vals):
            reg_mean[j] = float(np.mean(vals))
            reg_std[j] = float(np.std(vals))
            if reg_std[j] <= 1e-12 or not np.isfinite(reg_std[j]):
                reg_std[j] = 1.0
            y_reg[m_reg[:, j], j] = (vals - reg_mean[j]) / reg_std[j]
    return y_cls, m_cls, y_reg, m_reg, reg_mean, reg_std


def _fit_shared_mlp(
    x: np.ndarray,
    y_cls: np.ndarray,
    m_cls: np.ndarray,
    y_reg: np.ndarray,
    m_reg: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    mcfg = cfg.get("modeling", {}).get("multitask_mlp", {})
    rs = int(cfg.get("modeling", {}).get("random_state", 42))
    rng = np.random.default_rng(rs)
    hidden = int(mcfg.get("hidden_units", 32))
    epochs = int(mcfg.get("epochs", 600))
    lr = float(mcfg.get("learning_rate", 0.01))
    l2 = float(mcfg.get("l2", 1e-4))

    n_features = x.shape[1]
    n_outputs = y_cls.shape[1] + y_reg.shape[1]
    w1 = rng.normal(0, 1 / np.sqrt(max(1, n_features)), size=(n_features, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 1 / np.sqrt(max(1, hidden)), size=(hidden, n_outputs))
    b2 = np.zeros(n_outputs)
    cls_n = y_cls.shape[1]

    for _ in range(epochs):
        h_pre = x @ w1 + b1
        h = np.tanh(h_pre)
        out = h @ w2 + b2
        grad_out = np.zeros_like(out)

        if cls_n:
            p = _sigmoid(out[:, :cls_n])
            denom = np.maximum(m_cls.sum(axis=0), 1.0)
            grad_out[:, :cls_n] = ((p - y_cls) * m_cls) / denom

        if y_reg.shape[1]:
            reg_slice = slice(cls_n, cls_n + y_reg.shape[1])
            denom = np.maximum(m_reg.sum(axis=0), 1.0)
            grad_out[:, reg_slice] = ((out[:, reg_slice] - y_reg) * m_reg) / denom

        grad_out /= max(1, n_outputs)
        gw2 = h.T @ grad_out + l2 * w2
        gb2 = grad_out.sum(axis=0)
        gh = grad_out @ w2.T
        gh_pre = gh * (1.0 - h**2)
        gw1 = x.T @ gh_pre + l2 * w1
        gb1 = gh_pre.sum(axis=0)

        w1 -= lr * gw1
        b1 -= lr * gb1
        w2 -= lr * gw2
        b2 -= lr * gb2

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _predict_shared_mlp(model: dict[str, np.ndarray], x: np.ndarray, n_cls: int) -> tuple[np.ndarray, np.ndarray]:
    h = np.tanh(x @ model["w1"] + model["b1"])
    out = h @ model["w2"] + model["b2"]
    cls = _sigmoid(out[:, :n_cls]) if n_cls else np.zeros((len(x), 0))
    reg = out[:, n_cls:] if out.shape[1] > n_cls else np.zeros((len(x), 0))
    return cls, reg


def train_shared_multitask_mlp(table: pd.DataFrame, cfg: dict[str, Any], predict_table: pd.DataFrame | None = None) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train a true shared-representation multitask MLP for comparison with the tree baseline.

    The current production predictions still come from the per-target tree models.
    This learner shares one hidden layer across all available XRD targets and uses
    masks so partially labeled targets can contribute without requiring every row
    to have every XRD label.
    """
    predict_table = table if predict_table is None else predict_table
    feats = feature_columns(table)
    id_cols = ["sample_key"] if "sample_key" in predict_table.columns else [c for c in ["library", "well"] if c in predict_table.columns]
    predictions = predict_table[id_cols].copy()
    metrics_rows: list[dict[str, Any]] = []
    if not feats:
        return {}, predictions, pd.DataFrame([{"model_family": "multitask_mlp", "task": "all", "status": "skipped_no_features"}])

    cls_targets, reg_targets, skipped_rows = _available_shared_targets(table, cfg)
    metrics_rows.extend({"model_family": "multitask_mlp", **row, "warning": "Shared multitask learner skipped this target."} for row in skipped_rows)
    if not cls_targets and not reg_targets:
        if not metrics_rows:
            metrics_rows.append({"model_family": "multitask_mlp", "task": "all", "status": "no_trainable_targets"})
        return {}, predictions, pd.DataFrame(metrics_rows)

    y_cls, m_cls, y_reg, m_reg, reg_mean, reg_std = _target_matrices(table, cls_targets, reg_targets)
    labeled_any = (m_cls.any(axis=1) if m_cls.size else False) | (m_reg.any(axis=1) if m_reg.size else False)
    feats = _observed_features(table.iloc[np.where(labeled_any)[0]], feats)
    if not feats:
        metrics_rows.append({"model_family": "multitask_mlp", "task": "all", "status": "skipped_no_observed_features", "warning": "No optical or composition features were observed for the reviewed labels."})
        return {}, predictions, pd.DataFrame(metrics_rows)
    med, mean, std = _normalizer_from_table(table, feats)
    x_all = _normalize_features(table, feats, med, mean, std)
    labeled_idx = np.where(labeled_any)[0]
    mcfg = cfg.get("modeling", {}).get("multitask_mlp", {})
    rs = int(cfg.get("modeling", {}).get("random_state", 42))
    rng = np.random.default_rng(rs)
    validation_fraction = float(mcfg.get("validation_fraction", cfg.get("modeling", {}).get("test_size", 0.3)))
    can_holdout = len(labeled_idx) >= 8 and validation_fraction > 0

    if can_holdout:
        shuffled = labeled_idx.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * validation_fraction)))
        val_idx = shuffled[:n_val]
        train_idx = np.setdiff1d(np.arange(len(table)), val_idx)
        holdout_model = _fit_shared_mlp(x_all[train_idx], y_cls[train_idx], m_cls[train_idx], y_reg[train_idx], m_reg[train_idx], cfg)
        val_cls, val_reg_z = _predict_shared_mlp(holdout_model, x_all[val_idx], len(cls_targets))

        for j, target in enumerate(cls_targets):
            mask = m_cls[val_idx, j]
            n_test = int(mask.sum())
            n_labels = int(m_cls[:, j].sum())
            if n_test >= 2 and len(np.unique(y_cls[val_idx][mask, j])) >= 2:
                pred = (val_cls[mask, j] >= 0.5).astype(int)
                truth = y_cls[val_idx][mask, j].astype(int)
                metrics_rows.append({"model_family": "multitask_mlp", "task": target, "task_type": "classification", "n_labels": n_labels, "n_test": n_test, "status": "trained", **classification_metric_row(truth, pred), "warning": "Held-out metrics are not reliable for small labeled datasets."})
            else:
                metrics_rows.append({"model_family": "multitask_mlp", "task": target, "task_type": "classification", "n_labels": n_labels, "n_test": n_test, "status": "trained_no_holdout_metric", "warning": "Validation split did not contain enough labeled classes for this target."})

        for j, target in enumerate(reg_targets):
            mask = m_reg[val_idx, j]
            n_test = int(mask.sum())
            n_labels = int(m_reg[:, j].sum())
            if n_test >= 2:
                pred = val_reg_z[mask, j] * reg_std[j] + reg_mean[j]
                truth_z = y_reg[val_idx][mask, j]
                truth = truth_z * reg_std[j] + reg_mean[j]
                mse = mean_squared_error(truth, pred)
                metrics_rows.append({"model_family": "multitask_mlp", "task": target, "task_type": "regression", "n_labels": n_labels, "n_test": n_test, "status": "trained", "rmse": float(np.sqrt(mse)), "mae": mean_absolute_error(truth, pred), "r2": r2_score(truth, pred), "warning": "Held-out metrics are not reliable for small labeled datasets."})
            else:
                metrics_rows.append({"model_family": "multitask_mlp", "task": target, "task_type": "regression", "n_labels": n_labels, "n_test": n_test, "status": "trained_no_holdout_metric", "warning": "Validation split did not contain enough labels for this target."})
    else:
        for target in cls_targets:
            metrics_rows.append({"model_family": "multitask_mlp", "task": target, "task_type": "classification", "n_labels": int(m_cls[:, cls_targets.index(target)].sum()), "status": "trained_no_holdout", "warning": "Not enough labeled rows for a held-out shared-model estimate."})
        for target in reg_targets:
            metrics_rows.append({"model_family": "multitask_mlp", "task": target, "task_type": "regression", "n_labels": int(m_reg[:, reg_targets.index(target)].sum()), "status": "trained_no_holdout", "warning": "Not enough labeled rows for a held-out shared-model estimate."})

    model = _fit_shared_mlp(x_all, y_cls, m_cls, y_reg, m_reg, cfg)
    x_pred = _normalize_features(predict_table, feats, med, mean, std)
    cls_pred, reg_pred_z = _predict_shared_mlp(model, x_pred, len(cls_targets))
    uncertainties = []

    for j, target in enumerate(cls_targets):
        prob = cls_pred[:, j]
        predictions[f"mtl_pred_{target}_prob"] = prob
        predictions[f"mtl_pred_{target}"] = (prob >= 0.5).astype(int)
        uncertainties.append(pd.Series(1.0 - np.maximum(prob, 1.0 - prob), index=predict_table.index))

    for j, target in enumerate(reg_targets):
        pred = reg_pred_z[:, j] * reg_std[j] + reg_mean[j]
        predictions[f"mtl_pred_{target}"] = pred

    if uncertainties:
        predictions["multitask_mlp_uncertainty_score"] = pd.concat(uncertainties, axis=1).mean(axis=1)
        m = predictions["multitask_mlp_uncertainty_score"]
        if m.max() > m.min():
            predictions["multitask_mlp_uncertainty_score"] = (m - m.min()) / (m.max() - m.min())
    else:
        predictions["multitask_mlp_uncertainty_score"] = 0.0

    model_package = {
        "model": model,
        "feature_columns": feats,
        "classification_targets": cls_targets,
        "regression_targets": reg_targets,
        "feature_median": med,
        "feature_mean": mean,
        "feature_std": std,
        "regression_target_mean": reg_mean,
        "regression_target_std": reg_std,
    }
    return model_package, predictions, pd.DataFrame(metrics_rows)


def compare_model_metrics(baseline_metrics: pd.DataFrame, multitask_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a side-by-side task metric table for the tree baseline and shared MLP."""
    if baseline_metrics.empty and multitask_metrics.empty:
        return pd.DataFrame()
    base = baseline_metrics.copy()
    if "task" not in base.columns:
        return pd.DataFrame()
    base["model_family"] = base.get("model_family", "per_target_tree_baseline")
    mt = multitask_metrics.copy()
    if "task" not in mt.columns:
        return pd.DataFrame()
    rows = []
    tasks = sorted(set(base["task"].dropna().astype(str)) | set(mt["task"].dropna().astype(str)))
    for task in tasks:
        b = base[base["task"].astype(str) == task].tail(1)
        m = mt[mt["task"].astype(str) == task].tail(1)
        row: dict[str, Any] = {"task": task}
        for prefix, frame in [("baseline", b), ("multitask_mlp", m)]:
            if frame.empty:
                continue
            rec = frame.iloc[0].to_dict()
            for col in ["task_type", "status", "n_labels", "n_test", "accuracy", "balanced_accuracy", "weighted_f1", "positive_precision", "positive_recall", "positive_f1", "specificity", "majority_class_accuracy", "accuracy_above_majority", "true_negative", "false_positive", "false_negative", "true_positive", "rmse", "mae", "r2", "warning"]:
                if col in rec:
                    row[f"{prefix}_{col}"] = rec[col]
        task_type = row.get("baseline_task_type") or row.get("multitask_mlp_task_type")
        if task_type == "classification" and pd.notna(row.get("baseline_weighted_f1")) and pd.notna(row.get("multitask_mlp_weighted_f1")):
            row["winner_by_primary_metric"] = "multitask_mlp" if row["multitask_mlp_weighted_f1"] > row["baseline_weighted_f1"] else "baseline"
            row["primary_metric_delta_mtl_minus_baseline"] = row["multitask_mlp_weighted_f1"] - row["baseline_weighted_f1"]
        elif task_type == "regression" and pd.notna(row.get("baseline_rmse")) and pd.notna(row.get("multitask_mlp_rmse")):
            row["winner_by_primary_metric"] = "multitask_mlp" if row["multitask_mlp_rmse"] < row["baseline_rmse"] else "baseline"
            row["primary_metric_delta_mtl_minus_baseline"] = row["baseline_rmse"] - row["multitask_mlp_rmse"]
        rows.append(row)
    return pd.DataFrame(rows)
