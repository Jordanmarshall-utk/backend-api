"""End-to-end pipeline orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .active_learning import compute_xrd_priority, remeasurement_recommendations, select_diverse_recommendations
from .composition_features import (
    add_neighbor_outlier_features,
    apply_library_composition_substitutions,
    compute_composition_descriptors,
    load_composition_map,
)
from .config import resolve_path
from .experiment_profiles import build_profile_composition_table
from .modeling import (
    build_out_of_fold_tree_review_predictions,
    compare_model_metrics,
    compare_feature_family_models,
    configured_targets,
    feature_columns,
    leave_one_library_out_tree_metrics,
    train_multitask_models,
    train_shared_multitask_mlp,
)
from .plate_reader import parse_plate_reader_file
from .plotting import make_standard_heatmaps
from .spectral_features import add_spectral_pca_features, extract_features_per_read
from .utils import WELLS_96, ensure_output_dir, save_run_summary, well_row_col
from .xrd_labels import BINARY_TARGETS, REGRESSION_TARGETS, load_xrd_labels


def _empty_optical_table() -> pd.DataFrame:
    rows = []
    for well in WELLS_96:
        r, c = well_row_col(well)
        rows.append({"well": well, "row": r, "col": c, "optical_heterogeneity_score_total": 0.0, "pl_abs_disagreement_score": 0.0, "measurement_QC_flag_count": 0})
    return pd.DataFrame(rows)


def _add_sample_key(df: pd.DataFrame, library_name: str = "") -> pd.DataFrame:
    out = df.copy()
    if "library" not in out.columns:
        out["library"] = library_name
    out["library"] = out["library"].fillna("").astype(str)
    out["sample_key"] = out["library"].where(out["library"] != "", "library") + "::" + out["well"].astype(str)
    return out


def _load_prior_training_tables(cfg: dict[str, Any], warnings_list: list[str]) -> list[pd.DataFrame]:
    tables = []
    for value in cfg.get("paths", {}).get("training_tables", []) or []:
        path = resolve_path(cfg, value)
        if not path or not path.exists():
            warnings_list.append(f"Pooled training table not found: {value}")
            continue
        table = pd.read_csv(path)
        if "well" not in table.columns:
            warnings_list.append(f"Skipped pooled training table without well column: {path}")
            continue
        if "library" not in table.columns:
            table["library"] = path.parent.name
        tables.append(_add_sample_key(table))
    return tables


def _merge_xrd_labels(table: pd.DataFrame, xrd: pd.DataFrame, library_name: str = "") -> pd.DataFrame:
    """Merge reviewed XRD labels without erasing labels from other libraries."""
    if xrd.empty:
        out = table.copy()
        if "xrd_sample_weight" not in out.columns:
            out["xrd_sample_weight"] = np.nan
        return out

    labels = xrd.copy()
    if library_name and "library" not in labels.columns:
        labels["library"] = library_name
    if library_name and "library" in labels.columns:
        labels["library"] = labels["library"].fillna("").astype(str).replace("", library_name)
    if "xrd_library" not in labels.columns and "library" in labels.columns:
        labels["xrd_library"] = labels["library"]

    out = table.copy()
    merge_keys = ["library", "well"] if "library" in labels.columns and "library" in out.columns else ["well"]
    label_cols = [c for c in labels.columns if c.startswith("xrd_")]
    labels = labels[merge_keys + label_cols].drop_duplicates(subset=merge_keys, keep="last")
    existing_xrd_cols = [c for c in label_cols if c in out.columns]
    merged = out.merge(labels, on=merge_keys, how="left", suffixes=("", "__reviewed"))
    for col in existing_xrd_cols:
        reviewed_col = f"{col}__reviewed"
        if reviewed_col in merged.columns:
            merged[col] = merged[reviewed_col].combine_first(merged[col])
            merged = merged.drop(columns=reviewed_col)
    if "xrd_sample_weight" not in merged.columns:
        merged["xrd_sample_weight"] = np.nan
    return merged


def _xrd_label_coverage(table: pd.DataFrame) -> pd.DataFrame:
    targets = [c for c in BINARY_TARGETS + REGRESSION_TARGETS if c in table.columns]
    rows = []
    if "library" not in table.columns:
        table = table.copy()
        table["library"] = ""
    for library, group in table.groupby("library", dropna=False):
        labeled = group[targets].notna().any(axis=1) if targets else pd.Series(False, index=group.index)
        rows.append(
            {
                "library": library,
                "rows": len(group),
                "xrd_labeled_rows": int(labeled.sum()),
                "unique_labeled_wells": int(group.loc[labeled, "well"].nunique()) if "well" in group.columns else int(labeled.sum()),
            }
        )
    return pd.DataFrame(rows)


def _pooled_training_audit(
    table_before_dedup: pd.DataFrame,
    table_after_dedup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe the unique cross-library samples used to train the models."""
    targets = [c for c in BINARY_TARGETS + REGRESSION_TARGETS if c in table_after_dedup.columns]
    model_features = feature_columns(table_after_dedup)
    duplicate_columns = ["sample_key", "library", "well", "profile_id", "_training_source", "duplicate_resolution"]
    duplicate_rows = pd.DataFrame(columns=duplicate_columns)
    if "sample_key" in table_before_dedup.columns:
        duplicate_mask = table_before_dedup.duplicated(subset=["sample_key"], keep=False)
        duplicate_rows = table_before_dedup.loc[
            duplicate_mask,
            [c for c in ["sample_key", "library", "well", "profile_id", "_training_source"] if c in table_before_dedup.columns],
        ].copy()
        if not duplicate_rows.empty:
            duplicate_rows["duplicate_resolution"] = "latest row kept for model training"
        duplicate_rows = duplicate_rows.reindex(columns=duplicate_columns)

    rows = []
    grouped = table_after_dedup.groupby("library", dropna=False) if "library" in table_after_dedup.columns else [("", table_after_dedup)]
    for library, group in grouped:
        labeled = group[targets].notna().any(axis=1) if targets else pd.Series(False, index=group.index)
        feature_values = group[model_features].apply(pd.to_numeric, errors="coerce") if model_features else pd.DataFrame(index=group.index)
        any_feature = feature_values.notna().any(axis=1) if model_features else pd.Series(False, index=group.index)
        profile_ids = (
            sorted(group["profile_id"].dropna().astype(str).loc[lambda s: s.str.strip() != ""].unique())
            if "profile_id" in group.columns
            else []
        )
        duplicate_count = 0
        if not duplicate_rows.empty and "library" in duplicate_rows.columns:
            duplicate_count = int((duplicate_rows["library"].astype(str) == str(library)).sum())
        rows.append(
            {
                "library": library,
                "profile_id": ",".join(profile_ids),
                "unique_training_samples": int(group["sample_key"].nunique()) if "sample_key" in group.columns else len(group),
                "xrd_labeled_samples": int(labeled.sum()),
                "samples_with_model_features": int(any_feature.sum()),
                "usable_numeric_feature_columns": len(model_features),
                "duplicate_rows_found_before_dedup": duplicate_count,
                "training_rule": "latest unique library::well sample",
            }
        )
    return pd.DataFrame(rows), duplicate_rows


def _target_redundancy_audit(table: pd.DataFrame) -> pd.DataFrame:
    """Find targets whose overlapping reviewed labels are identical."""
    targets = [c for c in BINARY_TARGETS + REGRESSION_TARGETS if c in table.columns]
    rows = []
    for i, left in enumerate(targets):
        for right in targets[i + 1 :]:
            pair = table[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if pair.empty:
                continue
            identical = bool(np.allclose(pair[left], pair[right]))
            rows.append(
                {
                    "target_a": left,
                    "target_b": right,
                    "overlapping_labeled_rows": len(pair),
                    "identical_on_overlap": identical,
                    "correlation": pair[left].corr(pair[right]) if len(pair) > 1 else np.nan,
                    "review_recommendation": (
                        "Review whether these should remain separate targets."
                        if identical and len(pair) >= 5
                        else ""
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "target_a",
            "target_b",
            "overlapping_labeled_rows",
            "identical_on_overlap",
            "correlation",
            "review_recommendation",
        ],
    )


def _xrd_confidence_audit(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize confidence provenance and weights for labeled XRD samples."""
    targets = [c for c in BINARY_TARGETS + REGRESSION_TARGETS if c in table.columns]
    labeled = table[targets].notna().any(axis=1) if targets else pd.Series(False, index=table.index)
    subset = table.loc[labeled].copy()
    if subset.empty:
        return pd.DataFrame(columns=["library", "xrd_assignment_confidence", "labeled_samples", "mean_sample_weight", "training_use"])
    if "library" not in subset.columns:
        subset["library"] = ""
    if "xrd_assignment_confidence" not in subset.columns:
        subset["xrd_assignment_confidence"] = "unspecified"
    confidence = subset["xrd_assignment_confidence"].fillna("unspecified").astype(str).str.lower()
    subset["xrd_assignment_confidence"] = confidence.replace("", "unspecified")
    subset["_weight"] = pd.to_numeric(subset.get("xrd_sample_weight", 1.0), errors="coerce").fillna(1.0)
    out = (
        subset.groupby(["library", "xrd_assignment_confidence"], dropna=False)
        .agg(labeled_samples=("well", "count"), mean_sample_weight=("_weight", "mean"))
        .reset_index()
    )
    out["training_use"] = np.where(
        out["xrd_assignment_confidence"].eq("low"),
        "included with reduced weight unless excluded in run settings",
        "included",
    )
    return out


def _targeted_xrd_review_queue(current_table: pd.DataFrame, oof_review: pd.DataFrame) -> pd.DataFrame:
    """Build a short, high-value review queue instead of requiring every rule label to be checked."""
    if current_table.empty:
        return pd.DataFrame()
    named_targets = [
        "xrd_PbX2_like_impurity_present",
        "xrd_RbPbI3_present",
        "xrd_DMAPbI3_present",
        "xrd_delta_phase_present",
        "xrd_poly_hexagonal_phase_present",
        "xrd_PbI2_present",
    ]
    disagreements: dict[str, list[str]] = {}
    if not oof_review.empty and "baseline_correct" in oof_review.columns:
        wrong = oof_review[oof_review["baseline_correct"].eq(False)]
        for sample_key, group in wrong.groupby("sample_key", dropna=False):
            disagreements[str(sample_key)] = sorted(group["task"].dropna().astype(str).unique())

    rows = []
    for _, row in current_table.iterrows():
        source = str(row.get("xrd_label_source", "") or "")
        confidence = str(row.get("xrd_assignment_confidence", "") or "").lower()
        manually_reviewed = source.lower().startswith("manual") or "reviewed" in source.lower() or confidence == "high"
        reasons = []
        priority = 0
        if pd.to_numeric(pd.Series([row.get("xrd_clean_3d_perovskite")]), errors="coerce").fillna(0).iloc[0] == 1:
            reasons.append("positive clean-3D rule call")
            priority = max(priority, 2)
        if pd.to_numeric(pd.Series([row.get("xrd_mixed_phase")]), errors="coerce").fillna(0).iloc[0] == 1:
            reasons.append("positive mixed-phase rule call")
            priority = max(priority, 3)
        named_positive = [
            target
            for target in named_targets
            if pd.to_numeric(pd.Series([row.get(target)]), errors="coerce").fillna(0).iloc[0] == 1
        ]
        if named_positive:
            reasons.append("automatic named-phase proposal: " + ", ".join(named_positive))
            priority = max(priority, 4)
        sample_key = str(row.get("sample_key", ""))
        if sample_key in disagreements:
            reasons.append("OOF model disagrees: " + ", ".join(disagreements[sample_key]))
            priority = max(priority, 4)
        rule_review_value = row.get("xrd_rule_review_required", False)
        rule_review_required = False if pd.isna(rule_review_value) else bool(rule_review_value)
        if rule_review_required:
            reasons.append("ambiguous practical XRD rule")
            priority = max(priority, 2)
        source_files = str(row.get("xrd_source_file", "") or "")
        if ";" in source_files:
            reasons.append("multiple XRD scans available; check replicate agreement")
            priority = max(priority, 2)
        if not reasons:
            continue
        rows.append(
            {
                "review_priority": priority,
                "review_tier": "required" if priority >= 4 else "spot_check",
                "review_status": (
                    "already_manually_reviewed"
                    if manually_reviewed
                    else "required_manual_review"
                    if priority >= 4
                    else "optional_spot_check"
                ),
                "sample_key": sample_key,
                "library": row.get("library", ""),
                "well": row.get("well", ""),
                "review_reason": "; ".join(reasons),
                "xrd_clean_3d_perovskite": row.get("xrd_clean_3d_perovskite", np.nan),
                "xrd_mixed_phase": row.get("xrd_mixed_phase", np.nan),
                "xrd_assignment_confidence": confidence,
                "xrd_label_source": source,
                "xrd_rule_status": row.get("xrd_rule_status", ""),
                "xrd_base_rule_status": row.get("xrd_base_rule_status", row.get("xrd_rule_status", "")),
                "xrd_base_rule_clean_3d_perovskite": row.get("xrd_base_rule_clean_3d_perovskite", np.nan),
                "xrd_base_rule_mixed_phase": row.get("xrd_base_rule_mixed_phase", np.nan),
                "xrd_final_label_basis": row.get("xrd_final_label_basis", ""),
                "xrd_source_file": source_files,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["review_status", "review_priority", "library", "well"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _model_readiness_summary(metrics: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Translate evaluation tables into practical screening-readiness guidance."""
    if metrics.empty or "task" not in metrics.columns:
        return pd.DataFrame()
    rows = []
    for _, metric in metrics.iterrows():
        task = str(metric.get("task", ""))
        metric_status = str(metric.get("status", ""))
        task_transfer = transfer[transfer["task"].astype(str).eq(task)] if not transfer.empty and "task" in transfer.columns else pd.DataFrame()
        holdout_balanced = pd.to_numeric(pd.Series([metric.get("balanced_accuracy")]), errors="coerce").iloc[0]
        holdout_recall = pd.to_numeric(pd.Series([metric.get("positive_recall")]), errors="coerce").iloc[0]
        transfer_balanced = pd.to_numeric(task_transfer.get("balanced_accuracy", pd.Series(dtype=float)), errors="coerce")
        transfer_recall = pd.to_numeric(task_transfer.get("positive_recall", pd.Series(dtype=float)), errors="coerce")
        worst_transfer_balanced = float(transfer_balanced.min()) if transfer_balanced.notna().any() else np.nan
        worst_transfer_recall = float(transfer_recall.min()) if transfer_recall.notna().any() else np.nan
        if not metric_status.startswith("trained"):
            status = "not_trainable_yet"
            guidance = "Not enough reviewed class variation or labeled rows to train this target yet."
        elif np.isfinite(worst_transfer_balanced) and (worst_transfer_balanced < 0.60 or (np.isfinite(worst_transfer_recall) and worst_transfer_recall < 0.25)):
            status = "new_library_anchor_xrd_required"
            guidance = "Use for prioritization only until this new chemistry has an initial representative XRD anchor set."
        elif np.isfinite(holdout_balanced) and holdout_balanced >= 0.65 and np.isfinite(holdout_recall) and holdout_recall >= 0.30:
            status = "screening_ready_within_supported_libraries"
            guidance = "Useful for ranking wells in represented chemistries; continue prospective before-XRD validation."
        else:
            status = "early_model_keep_collecting_xrd"
            guidance = "Use cautiously for ranking; collect targeted positives, uncertainties, and chemistry anchors."
        rows.append(
            {
                "task": task,
                "model_status": metric_status,
                "readiness_status": status,
                "holdout_balanced_accuracy": holdout_balanced,
                "holdout_positive_recall": holdout_recall,
                "worst_library_transfer_balanced_accuracy": worst_transfer_balanced,
                "worst_library_transfer_positive_recall": worst_transfer_recall,
                "guidance": guidance,
            }
        )
    return pd.DataFrame(rows)


def _modeled_target_status(cfg: dict[str, Any]) -> pd.DataFrame:
    binary, regression = configured_targets(cfg)
    modeled = set(binary + regression)
    excluded = set(BINARY_TARGETS + REGRESSION_TARGETS) - modeled
    rows = []
    for target in BINARY_TARGETS + REGRESSION_TARGETS:
        rows.append(
            {
                "task": target,
                "task_type": "classification" if target in BINARY_TARGETS else "regression",
                "modeling_status": "modeled" if target in modeled else "excluded_redundant_alias",
                "reason": "" if target in modeled else "Retained in labels/audits, but excluded as a currently redundant target.",
            }
        )
    return pd.DataFrame(rows)


def _xrd_prediction_actual_review(table: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic in-sample prediction overlay for reviewed XRD labels."""
    targets = [c for c in BINARY_TARGETS + REGRESSION_TARGETS if c in table.columns]
    id_cols = [c for c in ["sample_key", "library", "well", "row", "col"] if c in table.columns]
    context_cols = [
        c
        for c in [
            "xrd_priority_score",
            "model_uncertainty_score",
            "multitask_mlp_uncertainty_score",
            "optical_heterogeneity_score_total",
            "pl_abs_disagreement_score",
            "composition_distance_score",
            "optical_neighbor_outlier_score",
        ]
        if c in table.columns
    ]
    rows = []
    for _, row in table.iterrows():
        base = {c: row.get(c) for c in id_cols + context_cols}
        for target in targets:
            actual = row.get(target)
            if pd.isna(actual):
                continue
            task_type = "classification" if target in BINARY_TARGETS else "regression"
            out = {
                **base,
                "task": target,
                "task_type": task_type,
                "actual_xrd_label": actual,
                "evaluation_scope": "diagnostic_in_sample_reviewed_well",
                "valid_for_performance_scoring": False,
            }
            baseline_pred = row.get(f"pred_{target}", np.nan)
            mtl_pred = row.get(f"mtl_pred_{target}", np.nan)
            out["baseline_prediction"] = baseline_pred
            out["multitask_mlp_prediction"] = mtl_pred
            prob_col = f"pred_{target}_prob"
            mtl_prob_col = f"mtl_pred_{target}_prob"
            if prob_col in table.columns:
                out["baseline_probability"] = row.get(prob_col)
            if mtl_prob_col in table.columns:
                out["multitask_mlp_probability"] = row.get(mtl_prob_col)
            if task_type == "classification":
                if pd.notna(baseline_pred):
                    out["baseline_correct"] = int(round(float(baseline_pred))) == int(round(float(actual)))
                if pd.notna(mtl_pred):
                    out["multitask_mlp_correct"] = int(round(float(mtl_pred))) == int(round(float(actual)))
            else:
                if pd.notna(baseline_pred):
                    out["baseline_absolute_error"] = abs(float(baseline_pred) - float(actual))
                if pd.notna(mtl_pred):
                    out["multitask_mlp_absolute_error"] = abs(float(mtl_pred) - float(actual))
            rows.append(out)
    return pd.DataFrame(rows)


def run_pipeline(cfg: dict[str, Any], config_path: Path | None = None) -> dict[str, Path]:
    """Run the full command-line pipeline and save all expected outputs."""
    out = ensure_output_dir(resolve_path(cfg, cfg.get("paths", {}).get("output_dir")) or "outputs")
    library_name = cfg.get("experiment", {}).get("library_name", "")
    warnings_list: list[str] = []
    summary_rows = []

    configured_plate_files = cfg.get("paths", {}).get("plate_reader_files") or []
    if not configured_plate_files and cfg.get("paths", {}).get("plate_reader_file"):
        configured_plate_files = [cfg.get("paths", {}).get("plate_reader_file")]
    plate_paths = [
        path
        for path in (resolve_path(cfg, value) for value in configured_plate_files)
        if path and path.exists()
    ]
    if plate_paths:
        long_parts = []
        summary_parts = []
        for plate_path in plate_paths:
            parsed_long, parsed_summary = parse_plate_reader_file(plate_path, cfg)
            parsed_long = parsed_long.copy()
            parsed_summary = parsed_summary.copy()
            parsed_long["source_file"] = plate_path.name
            parsed_summary["source_file"] = plate_path.name
            if len(plate_paths) > 1:
                source_prefix = "".join(ch if ch.isalnum() else "_" for ch in plate_path.stem).strip("_")
                parsed_long["read_name"] = source_prefix + "__" + parsed_long["read_name"].astype(str)
                parsed_summary["read_name"] = source_prefix + "__" + parsed_summary["read_name"].astype(str)
            long_parts.append(parsed_long)
            summary_parts.append(parsed_summary)
        long_df = pd.concat(long_parts, ignore_index=True, sort=False)
        read_summary = pd.concat(summary_parts, ignore_index=True, sort=False)
    else:
        long_df = pd.DataFrame(columns=["read_num", "read_label", "read_name", "read_type", "well", "wavelength_nm", "signal", "source_file"])
        read_summary = pd.DataFrame(columns=["read_num", "read_label", "read_name", "read_type", "n_wells_detected", "n_wavelength_points", "wavelength_min_nm", "wavelength_max_nm", "source_file"])
        warnings_list.append("No plate-reader file provided or found; continuing with composition-only outputs.")
    read_summary.to_csv(out / "parsed_reads_summary.csv", index=False)
    long_df.to_csv(out / "spectral_data_long_format.csv", index=False)

    if not long_df.empty:
        optical = extract_features_per_read(long_df, cfg)
        optical = add_spectral_pca_features(long_df, optical, cfg)
    else:
        optical = _empty_optical_table()
    if library_name:
        optical.insert(0, "library", library_name)
    optical.to_csv(out / "optical_features_per_well.csv", index=False)

    profile_id = str(cfg.get("experiment", {}).get("profile_id", "")).strip()
    comp_path = resolve_path(cfg, cfg.get("paths", {}).get("composition_map"))
    if profile_id:
        comp = build_profile_composition_table(profile_id, plate_id=library_name)
        warnings_list.append(f"Generated composition metadata from experiment profile: {profile_id}.")
    elif comp_path and comp_path.exists():
        comp = load_composition_map(comp_path, cfg, warnings_list)
    else:
        rows = []
        for well in WELLS_96:
            r, c = well_row_col(well)
            rows.append({"well": well, "row": r, "col": c, "Cs_fraction": np.nan, "Rb_fraction": np.nan, "FA_fraction": np.nan, "DMA_fraction": np.nan, "I_fraction": np.nan, "Br_fraction": np.nan, "Cl_fraction": np.nan, "is_control": "", "replicate_group": "", "notes": ""})
        comp = pd.DataFrame(rows)
        warnings_list.append("No composition map found; created empty 96-well composition table.")
    comp = apply_library_composition_substitutions(comp, cfg, warnings_list)
    comp_desc = compute_composition_descriptors(comp)
    if library_name:
        comp_desc.insert(0, "library", library_name)
    comp_desc = _add_sample_key(comp_desc, library_name)
    comp_desc.to_csv(out / "composition_descriptors_per_well.csv", index=False)

    xrd_path = resolve_path(cfg, cfg.get("paths", {}).get("xrd_labels"))
    xrd = load_xrd_labels(xrd_path)
    training = comp_desc.merge(optical.drop(columns=[c for c in ["row", "col", "library"] if c in optical.columns]), on="well", how="left")
    training = _merge_xrd_labels(training, xrd, library_name)
    training = _add_sample_key(training, library_name)
    training = add_neighbor_outlier_features(training, cfg)
    training.to_csv(out / "multitask_training_table.csv", index=False)

    prior_training = _load_prior_training_tables(cfg, warnings_list)
    if not xrd.empty:
        prior_training = [_merge_xrd_labels(table, xrd) for table in prior_training]
    for table in prior_training:
        table["_training_source"] = "latest saved prior library"
    training_for_pool = training.copy()
    training_for_pool["_training_source"] = "current library run"
    model_training_before_dedup = (
        pd.concat(prior_training + [training_for_pool], ignore_index=True, sort=False)
        if prior_training
        else training_for_pool
    )
    model_training = model_training_before_dedup.copy()
    if "sample_key" in model_training.columns:
        model_training = model_training.drop_duplicates(subset=["sample_key"], keep="last")
        model_training = model_training.sort_values("sample_key").reset_index(drop=True)
    profile_numeric_prefixes = ("two_d_precursor_is_", "additive_is_")
    profile_numeric_columns = {
        "two_d_fraction",
        "fapbi3_fraction",
        "additive_fraction",
        "additive_percent",
        "additive_final_molarity",
        "additive_stock_molarity",
        "host_total_volume_ul",
        "additive_volume_ul",
        "total_volume_ul",
        "two_d_volume_ul",
        "fapbi3_volume_ul",
        "two_d_stock_molarity",
        "fapbi3_stock_molarity",
        "two_d_spacer_stoichiometry",
        "composition_family_is_quasi_2d_additive",
    }
    for col in model_training.columns:
        if col in profile_numeric_columns or col.startswith(profile_numeric_prefixes):
            model_training[col] = pd.to_numeric(model_training[col], errors="coerce").fillna(0.0)
    training_scope, duplicate_audit = _pooled_training_audit(model_training_before_dedup, model_training)
    training_scope.to_csv(out / "pooled_training_scope_by_library.csv", index=False)
    duplicate_audit.to_csv(out / "pooled_training_duplicate_audit.csv", index=False)
    _target_redundancy_audit(model_training).to_csv(out / "xrd_target_redundancy_audit.csv", index=False)
    _xrd_confidence_audit(model_training).to_csv(out / "xrd_label_confidence_audit.csv", index=False)
    _modeled_target_status(cfg).to_csv(out / "modeled_target_status.csv", index=False)
    _xrd_label_coverage(model_training).to_csv(out / "xrd_label_coverage_by_library.csv", index=False)
    oof_review = build_out_of_fold_tree_review_predictions(model_training, training, cfg)
    oof_review.to_csv(out / "xrd_out_of_fold_prediction_review.csv", index=False)
    _targeted_xrd_review_queue(training, oof_review).to_csv(out / "xrd_targeted_manual_review_queue.csv", index=False)

    models, predictions, metrics, importances = train_multitask_models(model_training, cfg, predict_table=training)
    if metrics.empty:
        metrics = pd.DataFrame([{"task": "all", "status": "no_models_trained", "warning": "No XRD labels or too few labels for supervised training."}])
    metrics.to_csv(out / "model_metrics.csv", index=False)
    compare_feature_family_models(model_training, cfg).to_csv(out / "feature_family_model_comparison.csv", index=False)
    importances.to_csv(out / "feature_importance_by_task.csv", index=False)
    transfer_metrics = leave_one_library_out_tree_metrics(model_training, cfg)
    transfer_metrics.to_csv(out / "leave_one_library_out_metrics.csv", index=False)
    _model_readiness_summary(metrics, transfer_metrics).to_csv(out / "model_readiness_summary.csv", index=False)

    shared_models = {}
    shared_predictions = pd.DataFrame()
    shared_metrics = pd.DataFrame()
    comparison = pd.DataFrame()
    if cfg.get("modeling", {}).get("compare_multitask_mlp", False):
        shared_models, shared_predictions, shared_metrics = train_shared_multitask_mlp(model_training, cfg, predict_table=training)
        if shared_metrics.empty:
            shared_metrics = pd.DataFrame([{"model_family": "multitask_mlp", "task": "all", "status": "no_models_trained", "warning": "No trainable shared multitask targets."}])
        shared_metrics.to_csv(out / "multitask_mlp_metrics.csv", index=False)
        shared_predictions.to_csv(out / "multitask_mlp_predictions_all_wells.csv", index=False)
        comparison = compare_model_metrics(metrics, shared_metrics)
        comparison.to_csv(out / "model_comparison_metrics.csv", index=False)

    merge_keys = [c for c in ["sample_key"] if c in training.columns and c in predictions.columns] or ["well"]
    all_table = training.merge(predictions, on=merge_keys, how="left", suffixes=("", "_model"))
    if not shared_predictions.empty:
        shared_merge_keys = [c for c in ["sample_key"] if c in all_table.columns and c in shared_predictions.columns] or ["well"]
        all_table = all_table.merge(shared_predictions, on=shared_merge_keys, how="left", suffixes=("", "_mtl_model"))
    if "model_uncertainty_score_model" in all_table.columns:
        all_table["model_uncertainty_score"] = all_table["model_uncertainty_score_model"].fillna(all_table.get("model_uncertainty_score", 0))
        all_table = all_table.drop(columns=["model_uncertainty_score_model"])
    elif "model_uncertainty_score" not in all_table.columns:
        all_table["model_uncertainty_score"] = 0.0

    target_cols = [c for c in BINARY_TARGETS + REGRESSION_TARGETS if c in xrd.columns]
    if not xrd.empty and target_cols:
        labeled_xrd = xrd.loc[xrd[target_cols].notna().any(axis=1)].copy()
        if "well" in labeled_xrd.columns:
            valid_well = labeled_xrd["well"].notna() & labeled_xrd["well"].astype(str).str.strip().isin(WELLS_96)
            dropped = int((~valid_well).sum())
            if dropped:
                warnings_list.append(f"Dropped {dropped} XRD label row(s) without a valid 96-well ID from labeled-well counts.")
            labeled_xrd = labeled_xrd.loc[valid_well].copy()
        labeled_xrd = _add_sample_key(labeled_xrd, library_name)
        labeled_wells = set(labeled_xrd["sample_key"])
    else:
        labeled_wells = set()
    priority = compute_xrd_priority(all_table, labeled_wells, cfg)
    id_col = "sample_key" if "sample_key" in priority.columns else "well"
    priority["has_reviewed_xrd_label"] = priority[id_col].isin(labeled_wells)
    priority["prediction_role"] = np.where(
        priority["has_reviewed_xrd_label"],
        "training_or_reviewed_xrd_label",
        "unlabeled_prediction_candidate",
    )
    recs = select_diverse_recommendations(priority, labeled_wells, cfg)
    remeasure = remeasurement_recommendations(priority)
    priority.to_csv(out / "model_predictions_all_wells.csv", index=False)
    candidate_predictions = priority[~priority["has_reviewed_xrd_label"]].copy()
    candidate_predictions.to_csv(out / "model_predictions_unlabeled_candidate_wells.csv", index=False)
    _xrd_prediction_actual_review(priority).to_csv(out / "xrd_prediction_vs_actual_review.csv", index=False)
    rec_cols = [c for c in ["library", "well", "row", "col", "Cs_fraction", "Rb_fraction", "FA_fraction", "DMA_fraction", "I_fraction", "Br_fraction", "Cl_fraction", "xrd_priority_score", "recommendation_model", "recommendation_model_note", "model_uncertainty_score", "optical_phase_complexity_score", "optical_heterogeneity_score_total", "pl_abs_disagreement_score", "composition_distance_score", "optical_neighbor_outlier_score"] if c in recs.columns]
    pred_cols = [c for c in recs.columns if c.startswith("pred_") or c.startswith("mtl_pred_")]
    recs[rec_cols + pred_cols + ["reason_for_recommendation"]].to_csv(out / "xrd_recommendations_next_wells.csv", index=False)
    re_cols = [c for c in ["well", "row", "col", "optical_remeasurement_priority_score", "missing_spectra_score", "measurement_QC_flag_count", "reason_for_remeasurement"] if c in remeasure.columns]
    remeasure[re_cols].to_csv(out / "optical_remeasurement_recommendations.csv", index=False)

    make_standard_heatmaps(remeasure, out)

    summary_rows.extend(
        [
            {"item": "plate_reader_files", "value": len(plate_paths), "detail": "; ".join(path.name for path in plate_paths) if plate_paths else "missing"},
            {"item": "library", "value": library_name, "detail": "Separate runs/libraries should use separate names because well IDs repeat."},
            {"item": "spectral_rows", "value": len(long_df), "detail": ""},
            {"item": "composition_wells", "value": comp_desc["well"].nunique(), "detail": ""},
            {"item": "xrd_labeled_wells", "value": len(labeled_wells), "detail": ""},
            {"item": "pooled_training_rows", "value": len(model_training), "detail": "Current run plus selected prior library training tables."},
            {"item": "pooled_training_libraries", "value": ",".join(sorted(model_training.get("library", pd.Series(dtype=str)).dropna().astype(str).unique())), "detail": ""},
            {"item": "pooled_duplicate_rows_removed", "value": max(0, len(model_training_before_dedup) - len(model_training)), "detail": "Historical iterations are retained for performance tracking, but models use only the latest unique library::well row."},
            {"item": "models_trained", "value": len(models), "detail": "Supervised models skipped if fewer than configured minimum labels."},
            {"item": "shared_multitask_models_trained", "value": len(shared_models.get("classification_targets", [])) + len(shared_models.get("regression_targets", [])) if shared_models else 0, "detail": "True shared multitask MLP targets trained for comparison; ExtraTrees remains the baseline recommendation model."},
        ]
    )
    summary_rows.extend({"item": "warning", "value": w, "detail": ""} for w in warnings_list)
    save_run_summary(out / "run_summary.csv", summary_rows)
    return {"output_dir": out}
