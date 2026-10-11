"""Conservative reference-pattern and composition-based XRD label proposals."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


PHASE_TARGET_RULES = {
    "rbpbi3": ["xrd_RbPbI3_present"],
    "dmapbi3": ["xrd_DMAPbI3_present"],
    "pbi2": ["xrd_PbI2_present", "xrd_PbX2_like_impurity_present"],
}


def _normalized_phase(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _phase_targets(phase: str) -> list[str]:
    normalized = _normalized_phase(phase)
    targets = list(PHASE_TARGET_RULES.get(normalized, []))
    if "delta" in normalized:
        targets.append("xrd_delta_phase_present")
    if "poly" in normalized or "hexagonal" in normalized:
        targets.append("xrd_poly_hexagonal_phase_present")
    return list(dict.fromkeys(targets))


def _composition_compatibility(phase: str, composition: pd.Series | None, minimum_fraction: float) -> tuple[bool, str]:
    """Reject only phases that require a composition species known to be absent."""
    if composition is None:
        return True, "composition unavailable; no chemistry gate applied"

    normalized = _normalized_phase(phase)
    checks = [
        ("rb", "Rb_fraction"),
        ("dma", "DMA_fraction"),
        ("cs", "Cs_fraction"),
    ]
    for token, column in checks:
        if token not in normalized or column not in composition.index:
            continue
        fraction = pd.to_numeric(pd.Series([composition.get(column)]), errors="coerce").iloc[0]
        if pd.notna(fraction) and float(fraction) < minimum_fraction:
            return False, f"{column}={float(fraction):.3f} is below {minimum_fraction:.3f}"

    if ("fa" in normalized or "fapbi3" in normalized) and "shared_fa_host_fraction" in composition.index:
        fraction = pd.to_numeric(pd.Series([composition.get("shared_fa_host_fraction")]), errors="coerce").iloc[0]
        if pd.notna(fraction) and float(fraction) < minimum_fraction:
            return False, f"shared_fa_host_fraction={float(fraction):.3f} is below {minimum_fraction:.3f}"
    return True, "composition compatible"


def _greedy_reference_matches(sample_peaks: pd.DataFrame, reference_peaks: pd.DataFrame, default_tolerance: float) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for sample_index, sample in sample_peaks.iterrows():
        for reference_index, reference in reference_peaks.iterrows():
            tolerance = pd.to_numeric(pd.Series([reference.get("tolerance")]), errors="coerce").iloc[0]
            tolerance = min(default_tolerance, float(tolerance)) if pd.notna(tolerance) and float(tolerance) > 0 else default_tolerance
            delta = float(sample["two_theta"]) - float(reference["reference_two_theta"])
            if abs(delta) <= tolerance:
                candidates.append(
                    {
                        "sample_index": int(sample_index),
                        "reference_index": int(reference_index),
                        "sample_two_theta": float(sample["two_theta"]),
                        "reference_two_theta": float(reference["reference_two_theta"]),
                        "abs_delta": abs(delta),
                        "tolerance": tolerance,
                        "closeness": max(0.0, 1.0 - abs(delta) / tolerance),
                        "sample_weight": float(sample["sample_weight"]),
                        "reference_weight": float(reference["reference_weight"]),
                    }
                )

    matches: list[dict[str, float]] = []
    used_sample: set[int] = set()
    used_reference: set[int] = set()
    for candidate in sorted(candidates, key=lambda row: (row["abs_delta"], -row["reference_weight"], -row["sample_weight"])):
        if candidate["sample_index"] in used_sample or candidate["reference_index"] in used_reference:
            continue
        used_sample.add(candidate["sample_index"])
        used_reference.add(candidate["reference_index"])
        matches.append(candidate)
    return matches


def score_reference_patterns(
    peak_table: pd.DataFrame,
    references: pd.DataFrame,
    composition: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score complete fitted scan patterns against enabled reference phases."""
    auto_cfg = (cfg or {}).get("xrd_auto_assignment", {})
    tolerance = float(auto_cfg.get("match_tolerance_deg", 0.28))
    minimum_fraction = float(auto_cfg.get("composition_presence_min_fraction", 0.01))
    maximum_reference_peaks = int(auto_cfg.get("maximum_reference_peaks", 12))
    minimum_reference_intensity = float(auto_cfg.get("minimum_reference_relative_intensity", 5.0))

    if peak_table.empty or references.empty:
        return pd.DataFrame()

    refs = references.copy()
    selected = refs["selected"].fillna(True).astype(bool) if "selected" in refs.columns else pd.Series(True, index=refs.index)
    visible = refs["visible"].fillna(True).astype(bool) if "visible" in refs.columns else pd.Series(True, index=refs.index)
    refs = refs[selected & visible]
    refs["reference_two_theta"] = pd.to_numeric(refs["reference_two_theta"], errors="coerce")
    refs["relative_intensity"] = pd.to_numeric(refs.get("relative_intensity"), errors="coerce")
    refs = refs.dropna(subset=["reference_two_theta", "phase"])

    peaks = peak_table.copy()
    peaks["two_theta"] = pd.to_numeric(peaks.get("two_theta"), errors="coerce")
    peaks["area"] = pd.to_numeric(peaks.get("area"), errors="coerce")
    peaks["height"] = pd.to_numeric(peaks.get("height"), errors="coerce")
    peaks["fwhm"] = pd.to_numeric(peaks.get("fwhm"), errors="coerce")
    peaks = peaks.dropna(subset=["two_theta"])
    if "status" in peaks.columns:
        peaks = peaks[peaks["status"].fillna("").astype(str) != "ignored_noise"]

    composition_lookup: dict[tuple[str, str], pd.Series] = {}
    if composition is not None and not composition.empty:
        comp = composition.copy()
        if "library" not in comp.columns:
            comp["library"] = ""
        composition_lookup = {
            (str(row.get("library", "")), str(row.get("well", ""))): row
            for _, row in comp.iterrows()
        }

    rows: list[dict[str, Any]] = []
    group_cols = [col for col in ["library", "sample_id", "well", "source_file"] if col in peaks.columns]
    for group_key, sample_group in peaks.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(group_cols, group_key))
        sample_group = sample_group.copy()
        sample_group["sample_weight"] = sample_group["area"].where(sample_group["area"] > 0)
        sample_group["sample_weight"] = sample_group["sample_weight"].fillna(sample_group["height"] * sample_group["fwhm"])
        sample_group["sample_weight"] = sample_group["sample_weight"].fillna(sample_group["height"]).fillna(1.0).clip(lower=0.0)
        sample_total = float(sample_group["sample_weight"].sum()) or 1.0

        scan_min = float(sample_group["two_theta"].min())
        scan_max = float(sample_group["two_theta"].max())
        comp_row = composition_lookup.get((str(identity.get("library", "")), str(identity.get("well", ""))))
        if comp_row is None:
            comp_row = composition_lookup.get(("", str(identity.get("well", ""))))

        for phase, phase_refs in refs.groupby("phase", dropna=True):
            phase_refs = phase_refs[(phase_refs["reference_two_theta"] >= scan_min - tolerance) & (phase_refs["reference_two_theta"] <= scan_max + tolerance)].copy()
            if phase_refs.empty:
                continue
            phase_refs["reference_weight"] = phase_refs["relative_intensity"].fillna(100.0)
            phase_refs = phase_refs[phase_refs["reference_weight"] >= minimum_reference_intensity]
            phase_refs = phase_refs.nlargest(maximum_reference_peaks, "reference_weight").copy()
            if phase_refs.empty:
                continue
            reference_total = float(phase_refs["reference_weight"].sum()) or 1.0
            matches = _greedy_reference_matches(sample_group, phase_refs, tolerance)
            other_reference_positions = refs.loc[refs["phase"].astype(str) != str(phase), "reference_two_theta"].to_numpy(dtype=float)
            unique_matches = [
                match
                for match in matches
                if not np.any(np.abs(other_reference_positions - match["reference_two_theta"]) <= tolerance)
            ]
            matched_reference_weight = sum(match["reference_weight"] for match in matches)
            matched_sample_weight = sum(match["sample_weight"] for match in matches)
            coverage = len(matches) / len(phase_refs)
            intensity_coverage = matched_reference_weight / reference_total
            sample_explained = matched_sample_weight / sample_total
            mean_closeness = float(np.mean([match["closeness"] for match in matches])) if matches else 0.0
            pattern_score = 0.35 * coverage + 0.35 * intensity_coverage + 0.20 * mean_closeness + 0.10 * sample_explained
            compatible, composition_reason = _composition_compatibility(str(phase), comp_row, minimum_fraction)
            rows.append(
                {
                    **identity,
                    "phase": str(phase),
                    "mapped_xrd_targets": ";".join(_phase_targets(str(phase))),
                    "composition_compatible": compatible,
                    "composition_reason": composition_reason,
                    "reference_peaks_considered": int(len(phase_refs)),
                    "matched_reference_peaks": int(len(matches)),
                    "unique_matched_reference_peaks": int(len(unique_matches)),
                    "reference_peak_coverage": coverage,
                    "reference_intensity_coverage": intensity_coverage,
                    "sample_peak_area_explained": sample_explained,
                    "mean_match_closeness": mean_closeness,
                    "pattern_score": pattern_score,
                    "matched_peak_pairs": "; ".join(
                        f"{match['sample_two_theta']:.3f}->{match['reference_two_theta']:.3f}"
                        for match in matches
                    ),
                }
            )
    return pd.DataFrame(rows)


def propose_reference_labels(
    phase_scores: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert strong, composition-compatible reference scores into reviewable labels."""
    auto_cfg = (cfg or {}).get("xrd_auto_assignment", {})
    minimum_matches = int(auto_cfg.get("minimum_matched_reference_peaks", 2))
    minimum_score = float(auto_cfg.get("minimum_pattern_score", 0.52))
    minimum_coverage = float(auto_cfg.get("minimum_reference_peak_coverage", 0.20))
    minimum_unique_matches = int(auto_cfg.get("minimum_unique_reference_peaks", 1))
    ambiguity_margin = float(auto_cfg.get("ambiguity_margin", 0.08))
    if phase_scores.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = [col for col in ["library", "well"] if col in phase_scores.columns]
    for group_key, group in phase_scores.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(group_cols, group_key))
        eligible = group[
            group["composition_compatible"].fillna(False).astype(bool)
            & (pd.to_numeric(group["matched_reference_peaks"], errors="coerce").fillna(0) >= minimum_matches)
            & (pd.to_numeric(group["unique_matched_reference_peaks"], errors="coerce").fillna(0) >= minimum_unique_matches)
            & (pd.to_numeric(group["pattern_score"], errors="coerce").fillna(0) >= minimum_score)
            & (pd.to_numeric(group["reference_peak_coverage"], errors="coerce").fillna(0) >= minimum_coverage)
        ].sort_values("pattern_score", ascending=False)
        eligible = eligible.drop_duplicates(subset=["phase"], keep="first")
        proposals: dict[str, float] = {}
        accepted_phases: list[str] = []
        ambiguous_phases: list[str] = []
        for target, target_group in eligible.assign(
            _targets=eligible["mapped_xrd_targets"].fillna("").astype(str).str.split(";")
        ).explode("_targets").groupby("_targets", dropna=False):
            if not target:
                continue
            target_group = target_group.sort_values("pattern_score", ascending=False)
            best = target_group.iloc[0]
            runner_up_score = float(target_group.iloc[1]["pattern_score"]) if len(target_group) > 1 else -np.inf
            if float(best["pattern_score"]) - runner_up_score < ambiguity_margin:
                ambiguous_phases.extend(target_group.head(2)["phase"].astype(str).tolist())
                continue
            proposals[str(target)] = 1.0
            accepted_phases.append(str(best["phase"]))

        if proposals:
            proposals["xrd_secondary_phase_present"] = 1.0
            proposals["xrd_mixed_phase"] = 1.0
            proposals["xrd_clean_3d_perovskite"] = 0.0

        rows.append(
            {
                **identity,
                **proposals,
                "xrd_auto_assignment_confidence": "medium" if proposals else "low",
                "xrd_auto_assignment_status": "proposed_needs_review" if proposals else "no_confident_reference_label",
                "xrd_auto_assigned_phases": "; ".join(dict.fromkeys(accepted_phases)),
                "xrd_auto_ambiguous_phases": "; ".join(dict.fromkeys(ambiguous_phases)),
                "xrd_auto_assignment_note": (
                    "Automatic reference-pattern and composition proposal; manual XRD review remains authoritative."
                ),
            }
        )
    return pd.DataFrame(rows)


def merge_proposals_into_draft_labels(draft_labels: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    """Merge stronger reference proposals into the automatic rule-based draft."""
    if draft_labels.empty or proposals.empty:
        return draft_labels.copy()
    out = draft_labels.copy()
    proposal_work = proposals.copy()
    merge_keys = [col for col in ["xrd_library", "well"] if col in out.columns]
    if "xrd_library" in merge_keys and "library" in proposal_work.columns:
        proposal_work = proposal_work.rename(columns={"library": "xrd_library"})
    if not merge_keys:
        merge_keys = ["well"]
    metadata_cols = [col for col in proposal_work.columns if col.startswith("xrd_auto_")]
    target_cols = [
        col
        for col in proposal_work.columns
        if col.startswith("xrd_") and col not in metadata_cols and col not in merge_keys
    ]
    merged = out.merge(proposal_work[merge_keys + target_cols + metadata_cols], on=merge_keys, how="left", suffixes=("", "__auto"))
    for col in target_cols:
        auto_col = f"{col}__auto"
        if auto_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[auto_col].combine_first(merged[col])
            else:
                merged[col] = merged[auto_col]
            merged = merged.drop(columns=auto_col)
    for col in metadata_cols:
        auto_col = f"{col}__auto"
        if auto_col in merged.columns:
            merged[col] = merged[auto_col]
            merged = merged.drop(columns=auto_col)
    has_proposal = merged.get("xrd_auto_assignment_status", pd.Series("", index=merged.index)).eq("proposed_needs_review")
    merged.loc[has_proposal, "xrd_assignment_confidence"] = "low"
    merged.loc[has_proposal, "xrd_notes"] = merged.loc[has_proposal, "xrd_auto_assignment_note"]
    merged.loc[has_proposal, "xrd_final_label_basis"] = "reference_pattern_override_needs_review"
    merged["xrd_label_source"] = np.where(
        has_proposal,
        "automatic_reference_pattern_and_composition_proposal",
        merged.get("xrd_label_source", ""),
    )
    return merged
