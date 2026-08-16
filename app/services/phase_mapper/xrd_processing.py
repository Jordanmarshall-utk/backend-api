"""Raw XRD CSV parsing and conservative feature extraction."""
from __future__ import annotations

import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .utils import normalize_well


WELL_RE = re.compile(r"\b([A-H](?:1[0-2]|[1-9]))\b", re.IGNORECASE)
DEFAULT_PEROVSKITE_PEAK_WINDOWS = [(13.6, 14.8), (19.4, 20.8), (23.8, 25.2), (27.0, 30.0)]
DEFAULT_PRIMARY_3D_PEAK_WINDOW = (13.6, 14.8)
DEFAULT_REFERENCE_PEAK_TOLERANCE_DEG = 0.30


def infer_well_from_filename(path: str | Path) -> str:
    """Infer a 96-well ID from names like A11.csv, A11 N2.csv, or Rb A11.csv."""
    match = WELL_RE.search(Path(path).stem)
    return normalize_well(match.group(1)) if match else ""


def read_xrd_csv(path: str | Path) -> pd.DataFrame:
    """Read an instrument CSV that contains a [Scan points] Angle/Intensity table."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "[scan points]":
            header_idx = idx + 1
            break
    if header_idx is None:
        raise ValueError(f"No [Scan points] table found in {path}")
    df = pd.read_csv(path, skiprows=header_idx, skipinitialspace=True)
    if "Angle" not in df.columns or "Intensity" not in df.columns:
        raise ValueError(f"XRD CSV must contain Angle and Intensity columns: {path}")
    out = df[["Angle", "Intensity"]].copy()
    out.columns = ["two_theta_deg", "intensity"]
    out["two_theta_deg"] = pd.to_numeric(out["two_theta_deg"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    return out.dropna().sort_values("two_theta_deg").reset_index(drop=True)


def read_xrdml(path: str | Path) -> pd.DataFrame:
    """Read a PANalytical-style XRDML scan into two-theta/intensity columns."""
    path = Path(path)
    root = ET.parse(path).getroot()
    positions = None
    for element in root.iter():
        if element.tag.endswith("positions") and element.attrib.get("axis") == "2Theta":
            positions = element
            break
    counts = next((element for element in root.iter() if element.tag.endswith("counts")), None)
    if positions is None or counts is None or not counts.text:
        raise ValueError(f"XRDML file must contain 2Theta positions and counts: {path}")

    start = next((element for element in positions if element.tag.endswith("startPosition")), None)
    end = next((element for element in positions if element.tag.endswith("endPosition")), None)
    if start is None or end is None or start.text is None or end.text is None:
        raise ValueError(f"XRDML file must contain start/end 2Theta positions: {path}")

    y = np.array([float(value) for value in counts.text.split()], dtype=float)
    x = np.linspace(float(start.text), float(end.text), len(y))
    return pd.DataFrame({"two_theta_deg": x, "intensity": y})


def read_xrd_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".xrdml":
        return read_xrdml(path)
    return read_xrd_csv(path)


def _smooth(y: np.ndarray) -> np.ndarray:
    if len(y) < 9:
        return y
    window = min(31, len(y) if len(y) % 2 else len(y) - 1)
    window = max(5, window)
    return savgol_filter(y, window_length=window, polyorder=2)


def _peak_table(x: np.ndarray, y: np.ndarray, prominence_fraction: float) -> pd.DataFrame:
    baseline = np.nanpercentile(y, 5)
    ys = _smooth(np.maximum(y - baseline, 0))
    max_y = float(np.nanmax(ys)) if len(ys) else 0.0
    if max_y <= 0:
        return pd.DataFrame(columns=["two_theta_deg", "height", "relative_height", "fwhm_deg"])
    peaks, props = find_peaks(ys, prominence=max(max_y * prominence_fraction, 1e-9))
    if len(peaks) == 0:
        peaks = np.array([int(np.nanargmax(ys))])
    widths = peak_widths(ys, peaks, rel_height=0.5)
    step = float(np.nanmedian(np.diff(x))) if len(x) > 1 else np.nan
    rows = []
    for peak_idx, width_points in zip(peaks, widths[0]):
        height = float(ys[peak_idx])
        rows.append(
            {
                "two_theta_deg": float(x[peak_idx]),
                "height": height,
                "relative_height": height / max_y if max_y else np.nan,
                "fwhm_deg": float(width_points * step) if np.isfinite(step) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("height", ascending=False).reset_index(drop=True)


def _peak_positions_from_text(text: object) -> list[float]:
    if pd.isna(text) or text == "":
        return []
    return [float(chunk) for chunk in str(text).split(";") if chunk.strip()]


def _in_any_window(value: float, windows: list[tuple[float, float]]) -> bool:
    return any(lo <= value <= hi for lo, hi in windows)


def _window_hits(positions: list[float], windows: list[tuple[float, float]]) -> int:
    return sum(any(lo <= pos <= hi for pos in positions) for lo, hi in windows)


def _numeric_group_series(group: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in group.columns:
        return pd.Series(default, index=group.index)
    return pd.to_numeric(group[col], errors="coerce")


def _resolve_cfg_path(cfg: dict[str, Any], value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg.get("_config_dir", ".")).joinpath(path)


def _load_reference_rules_file(cfg: dict[str, Any], xrd_cfg: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_cfg_path(cfg, xrd_cfg.get("reference_rules_file"))
    if not path or not path.exists():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        references = payload.get("references", []) if isinstance(payload, dict) else []
        reference_phases: dict[str, dict[str, list[float]]] = {}
        tolerances: list[float] = []
        for reference in references:
            if not isinstance(reference, dict) or reference.get("visible") is False:
                continue
            phase = str(reference.get("name") or "").strip()
            if not phase:
                continue
            positions: list[float] = []
            for peak in reference.get("peaks", []):
                if not isinstance(peak, dict) or peak.get("selected") is False:
                    continue
                value = pd.to_numeric(pd.Series([peak.get("two_theta")]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                positions.append(float(value))
                tolerance = pd.to_numeric(pd.Series([peak.get("tolerance")]), errors="coerce").iloc[0]
                if not pd.isna(tolerance):
                    tolerances.append(float(tolerance))
            if positions:
                reference_phases[phase] = {"peaks_deg": positions}
        out: dict[str, Any] = {"reference_phases_deg": reference_phases}
        if tolerances:
            out["reference_peak_tolerance_deg"] = float(np.median(tolerances))
        return out

    rules = pd.read_csv(path)
    enabled = rules.get("enabled", True)
    if not isinstance(enabled, bool):
        enabled = enabled.fillna(True).astype(str).str.lower().isin({"1", "true", "yes", "y"})
        rules = rules.loc[enabled].copy()
    reference_phases: dict[str, dict[str, list[float]]] = {}
    for phase, group in rules.groupby("phase_label", dropna=True):
        positions = pd.to_numeric(group.get("peak_2theta_deg"), errors="coerce").dropna().astype(float).tolist()
        if positions:
            reference_phases[str(phase)] = {"peaks_deg": positions}
    out: dict[str, Any] = {"reference_phases_deg": reference_phases}
    tolerance = pd.to_numeric(rules.get("tolerance_deg"), errors="coerce").dropna()
    if len(tolerance):
        out["reference_peak_tolerance_deg"] = float(tolerance.median())
    primary_rows = rules.loc[rules.get("match_role", pd.Series("", index=rules.index)).fillna("").astype(str).str.lower().eq("primary_3d")]
    primary_positions = pd.to_numeric(primary_rows.get("peak_2theta_deg"), errors="coerce").dropna()
    if len(primary_positions):
        half_width = float(out.get("reference_peak_tolerance_deg", DEFAULT_REFERENCE_PEAK_TOLERANCE_DEG))
        center = float(primary_positions.median())
        out["primary_3d_peak_window_deg"] = [center - half_width, center + half_width]
    return out


def _xrd_processing_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = cfg or {}
    xrd_cfg = dict(cfg.get("xrd_processing", {}))
    rules_cfg = _load_reference_rules_file(cfg, xrd_cfg)
    if rules_cfg:
        for key, value in rules_cfg.items():
            if key == "reference_phases_deg":
                merged = dict(xrd_cfg.get("reference_phases_deg", {}))
                merged.update(value)
                xrd_cfg[key] = merged
            else:
                xrd_cfg[key] = value
    return xrd_cfg


def _reference_phase_peaks(xrd_cfg: dict[str, Any]) -> dict[str, list[float]]:
    raw = xrd_cfg.get("reference_phases_deg", xrd_cfg.get("reference_phases", {}))
    phases: dict[str, list[float]] = {}
    if not isinstance(raw, dict):
        return phases
    for phase, spec in raw.items():
        if isinstance(spec, dict):
            values = spec.get("peaks_deg", spec.get("peaks", []))
        else:
            values = spec
        peaks = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float).tolist()
        if peaks:
            phases[str(phase)] = peaks
    return phases


def _reference_matches(
    positions: list[float],
    reference_phases: dict[str, list[float]],
    tolerance_deg: float,
) -> tuple[str, str]:
    match_rows: list[tuple[float, str, float, float]] = []
    phase_hits: dict[str, list[float]] = {}
    for position in positions:
        for phase, reference_positions in reference_phases.items():
            for reference_position in reference_positions:
                delta = position - reference_position
                if abs(delta) <= tolerance_deg:
                    match_rows.append((position, phase, reference_position, delta))
                    phase_hits.setdefault(phase, []).append(abs(delta))

    if not match_rows:
        return "", ""

    match_text = "; ".join(
        f"{position:.3f}->{phase}@{reference_position:.3f}({delta:+.3f})"
        for position, phase, reference_position, delta in sorted(match_rows, key=lambda item: (item[0], item[1], abs(item[3])))
    )
    phase_text = "; ".join(
        f"{phase}({len(deltas)} peaks, mean_abs_delta={np.mean(deltas):.3f})"
        for phase, deltas in sorted(phase_hits.items(), key=lambda item: (-len(item[1]), np.mean(item[1]), item[0]))
    )
    return match_text, phase_text


def summarize_xrd_pattern(path: str | Path, library: str = "", cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract review-oriented XRD features without assigning chemistry-specific impurities."""
    xrd_cfg = _xrd_processing_cfg(cfg)
    df = read_xrd_file(path)
    x = df["two_theta_deg"].to_numpy(dtype=float)
    y = df["intensity"].to_numpy(dtype=float)
    peaks = _peak_table(x, y, float(xrd_cfg.get("peak_prominence_fraction", 0.08)))
    high_min = float(xrd_cfg.get("high_order_min_deg", 27.0))
    high_max = float(xrd_cfg.get("high_order_max_deg", 30.0))
    high = peaks[(peaks["two_theta_deg"] >= high_min) & (peaks["two_theta_deg"] <= high_max)].copy()
    strong_high = high[high["relative_height"] >= float(xrd_cfg.get("high_order_relative_height_min", 0.12))]
    strongest = peaks.head(5)
    strong_peak_min = float(xrd_cfg.get("secondary_peak_relative_height_min", 0.12))
    perovskite_windows = xrd_cfg.get("perovskite_peak_windows_deg", DEFAULT_PEROVSKITE_PEAK_WINDOWS)
    perovskite_windows = [(float(lo), float(hi)) for lo, hi in perovskite_windows]
    primary_3d_window = xrd_cfg.get("primary_3d_peak_window_deg", DEFAULT_PRIMARY_3D_PEAK_WINDOW)
    primary_3d_window = (float(primary_3d_window[0]), float(primary_3d_window[1]))
    strong_peaks = peaks[peaks["relative_height"] >= strong_peak_min]
    additional_peaks = strong_peaks[~strong_peaks["two_theta_deg"].apply(lambda val: primary_3d_window[0] <= float(val) <= primary_3d_window[1])]
    extra_peaks = strong_peaks[~strong_peaks["two_theta_deg"].apply(lambda val: _in_any_window(float(val), perovskite_windows))]
    reference_phases = _reference_phase_peaks(xrd_cfg)
    reference_tolerance = float(xrd_cfg.get("reference_peak_tolerance_deg", DEFAULT_REFERENCE_PEAK_TOLERANCE_DEG))
    reference_matches, candidate_phases = _reference_matches(
        additional_peaks["two_theta_deg"].astype(float).tolist(),
        reference_phases,
        reference_tolerance,
    )

    return {
        "source_file": str(path),
        "library": library,
        "sample_id": Path(path).stem,
        "well": infer_well_from_filename(path),
        "scan_min_deg": float(np.nanmin(x)) if len(x) else np.nan,
        "scan_max_deg": float(np.nanmax(x)) if len(x) else np.nan,
        "max_intensity": float(np.nanmax(y)) if len(y) else np.nan,
        "background_p05": float(np.nanpercentile(y, 5)) if len(y) else np.nan,
        "detected_peak_count": int(len(peaks)),
        "top_peak_positions_deg": ";".join(f"{v:.3f}" for v in strongest["two_theta_deg"]),
        "top_peak_relative_heights": ";".join(f"{v:.3f}" for v in strongest["relative_height"]),
        "high_order_window_min_deg": high_min,
        "high_order_window_max_deg": high_max,
        "high_order_peak_count": int(len(strong_high)),
        "high_order_main_peak_position_deg": float(high.iloc[0]["two_theta_deg"]) if len(high) else np.nan,
        "high_order_main_peak_fwhm_deg": float(high.iloc[0]["fwhm_deg"]) if len(high) else np.nan,
        "high_order_peak_positions_deg": ";".join(f"{v:.3f}" for v in strong_high["two_theta_deg"]),
        "primary_3d_peak_present": int(
            any(primary_3d_window[0] <= float(val) <= primary_3d_window[1] for val in strong_peaks["two_theta_deg"])
        ),
        "perovskite_peak_window_hits": int(_window_hits(strong_peaks["two_theta_deg"].tolist(), perovskite_windows)),
        "additional_peak_count_excluding_primary_3d": int(len(additional_peaks)),
        "additional_peak_positions_excluding_primary_3d_deg": ";".join(f"{v:.3f}" for v in additional_peaks["two_theta_deg"]),
        "extra_peak_count": int(len(extra_peaks)),
        "extra_peak_positions_deg": ";".join(f"{v:.3f}" for v in extra_peaks["two_theta_deg"]),
        "reference_peak_matches": reference_matches,
        "candidate_reference_phases": candidate_phases,
    }


def summarize_xrd_directories(directories: list[str | Path], cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    for directory in directories:
        directory = Path(directory)
        library = directory.name.replace(" XRD", "")
        paths = sorted(list(directory.glob("*.csv")) + list(directory.glob("*.xrdml")))
        for path in paths:
            rows.append(summarize_xrd_pattern(path, library=library, cfg=cfg))
    return pd.DataFrame(rows)


def summarize_xrd_files(files: list[str | Path], library: str = "uploaded_raw", cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(file) for file in files):
        rows.append(summarize_xrd_pattern(path, library=library, cfg=cfg))
    return pd.DataFrame(rows)


def make_draft_label_table(features: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Create a conservative labels draft from raw features for human review."""
    xrd_cfg = _xrd_processing_cfg(cfg)
    broad_fwhm = float(xrd_cfg.get("high_order_broad_fwhm_deg", 0.35))
    split_count = int(xrd_cfg.get("high_order_split_peak_count", 2))
    clean_min_hits = int(xrd_cfg.get("clean_perovskite_min_peak_window_hits", 2))
    secondary_extra_min = int(xrd_cfg.get("secondary_extra_peak_count_min", 1))
    auto_flag_secondary = bool(xrd_cfg.get("auto_flag_secondary_from_extra_peaks", False))
    clean_extra_max = int(xrd_cfg.get("practical_clean_extra_peak_count_max", 2))
    mixed_extra_min = int(xrd_cfg.get("practical_mixed_extra_peak_count_min", clean_extra_max + 1))
    require_primary_and_high_order = bool(xrd_cfg.get("practical_require_primary_and_high_order", True))
    primary_window = xrd_cfg.get("primary_3d_peak_window_deg", DEFAULT_PRIMARY_3D_PEAK_WINDOW)
    primary_window = (float(primary_window[0]), float(primary_window[1]))
    peak_shift_reference = xrd_cfg.get("high_order_reference_peak_deg")
    peak_shift_threshold = float(xrd_cfg.get("high_order_peak_shift_threshold_deg", 0.2))
    rows = []
    group_cols = ["library", "well"] if "library" in features.columns else ["well"]
    for _, group in features.groupby(group_cols, dropna=False):
        high_order_count = int(_numeric_group_series(group, "high_order_peak_count").fillna(0).max())
        main_peak = _numeric_group_series(group, "high_order_main_peak_position_deg", np.nan).dropna()
        main_fwhm = _numeric_group_series(group, "high_order_main_peak_fwhm_deg", np.nan).dropna()
        window_hits = int(_numeric_group_series(group, "perovskite_peak_window_hits").fillna(0).max())
        extra_peak_count = int(_numeric_group_series(group, "extra_peak_count").fillna(0).max())
        extra_positions = []
        for value in group.get("extra_peak_positions_deg", pd.Series(dtype=object)):
            extra_positions.extend(_peak_positions_from_text(value))
        additional_positions = []
        for value in group.get("additional_peak_positions_excluding_primary_3d_deg", pd.Series(dtype=object)):
            additional_positions.extend(_peak_positions_from_text(value))
        reference_matches = sorted(set(group.get("reference_peak_matches", pd.Series(dtype=object)).dropna().astype(str)))
        candidate_phases = sorted(set(group.get("candidate_reference_phases", pd.Series(dtype=object)).dropna().astype(str)))
        has_primary_3d = bool(int(_numeric_group_series(group, "primary_3d_peak_present").fillna(0).max()))
        has_high_order_3d = high_order_count >= 1
        has_perovskite = (
            has_primary_3d and has_high_order_3d
            if require_primary_and_high_order
            else window_hits >= clean_min_hits
        )
        has_secondary = extra_peak_count >= secondary_extra_min
        has_many_extra_peaks = extra_peak_count >= mixed_extra_min
        has_split = high_order_count >= split_count
        max_fwhm = float(main_fwhm.max()) if len(main_fwhm) else np.nan
        has_broadening = np.isfinite(max_fwhm) and max_fwhm >= broad_fwhm
        mean_peak = float(main_peak.mean()) if len(main_peak) else np.nan
        if peak_shift_reference is None or not np.isfinite(mean_peak):
            peak_shift_score = np.nan
            has_peak_shift = False
        else:
            peak_shift_score = abs(mean_peak - float(peak_shift_reference))
            has_peak_shift = peak_shift_score >= peak_shift_threshold
        is_clean = has_perovskite and extra_peak_count <= clean_extra_max and not has_split and not has_broadening
        is_clear_mixed = has_split or has_many_extra_peaks
        rule_confidence = "medium" if is_clean or is_clear_mixed else "low"
        if is_clean:
            rule_status = "clear_practical_clean_3d_rule"
            rule_reason = (
                f"Primary 3D peak near 14 deg and high-order peak near 28 deg detected; "
                f"{extra_peak_count} meaningful extra peaks (allowed <= {clean_extra_max})."
            )
        elif is_clear_mixed:
            rule_status = "clear_practical_mixed_rule"
            reasons = []
            if has_split:
                reasons.append("high-order peak splitting")
            if has_many_extra_peaks:
                reasons.append(f"{extra_peak_count} meaningful extra peaks")
            rule_reason = "Not clean / mixed rule: " + " and ".join(reasons) + "."
        else:
            rule_status = "ambiguous_rule_needs_review"
            rule_reason = (
                "Pattern does not satisfy the practical clean-3D rule and does not have "
                f"the clear mixed-pattern threshold of {mixed_extra_min} extra peaks; manual review recommended."
            )
        rows.append(
            {
                "well": group["well"].iloc[0],
                "xrd_clean_3d_perovskite": 1 if is_clean else 0,
                "xrd_mixed_phase": 1 if is_clear_mixed else np.nan if has_secondary else 0,
                "xrd_secondary_phase_present": 1 if has_secondary and auto_flag_secondary else np.nan if has_secondary else 0,
                "xrd_3d_phase_segregation_present": 1 if has_split else 0,
                "xrd_peak_splitting_present": 1 if has_split else 0,
                "xrd_200_peak_splitting_present": 1 if has_split else 0,
                "xrd_octahedral_tilting_indicator": 1 if has_peak_shift else 0 if peak_shift_reference is not None and np.isfinite(mean_peak) else np.nan,
                "xrd_peak_broadening_score": max_fwhm,
                "xrd_200_peak_broadening_score": max_fwhm,
                "xrd_3d_peak_shift_score": peak_shift_score,
                "xrd_200_peak_position_deg_2theta": mean_peak,
                "xrd_200_peak_fwhm_deg": max_fwhm,
                "xrd_200_peak_count": high_order_count,
                "xrd_extra_peak_count": extra_peak_count,
                "xrd_extra_peak_positions_deg": ";".join(f"{pos:.3f}" for pos in sorted(set(extra_positions))),
                "xrd_additional_peak_count_excluding_primary_3d": int(_numeric_group_series(group, "additional_peak_count_excluding_primary_3d").fillna(0).max()),
                "xrd_additional_peak_positions_excluding_primary_3d_deg": ";".join(f"{pos:.3f}" for pos in sorted(set(additional_positions))),
                "xrd_reference_peak_matches": "; ".join(value for value in reference_matches if value),
                "xrd_candidate_reference_phases": "; ".join(value for value in candidate_phases if value),
                "xrd_rule_confidence": rule_confidence,
                "xrd_rule_status": rule_status,
                "xrd_rule_reason": rule_reason,
                "xrd_rule_primary_3d_peak_present": int(has_primary_3d),
                "xrd_rule_high_order_3d_peak_present": int(has_high_order_3d),
                "xrd_rule_review_required": bool(rule_confidence == "low"),
                "xrd_source_file": "; ".join(group["source_file"].astype(str)),
                "xrd_library": group["library"].iloc[0] if "library" in group.columns else "",
                "xrd_sample_id": "; ".join(group["sample_id"].astype(str)),
            }
        )
    aggregated = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "well": aggregated["well"],
            "xrd_clean_3d_perovskite": aggregated["xrd_clean_3d_perovskite"],
            "xrd_mixed_phase": aggregated["xrd_mixed_phase"],
            "xrd_base_rule_clean_3d_perovskite": aggregated["xrd_clean_3d_perovskite"],
            "xrd_base_rule_mixed_phase": aggregated["xrd_mixed_phase"],
            "xrd_secondary_phase_present": aggregated["xrd_secondary_phase_present"],
            "xrd_PbX2_like_impurity_present": np.nan,
            "xrd_RbPbI3_present": np.nan,
            "xrd_DMAPbI3_present": np.nan,
            "xrd_delta_phase_present": np.nan,
            "xrd_poly_hexagonal_phase_present": np.nan,
            "xrd_PbI2_present": np.nan,
            "xrd_3d_phase_segregation_present": aggregated["xrd_3d_phase_segregation_present"],
            "xrd_peak_splitting_present": aggregated["xrd_peak_splitting_present"],
            "xrd_200_peak_splitting_present": aggregated["xrd_200_peak_splitting_present"],
            "xrd_octahedral_tilting_indicator": aggregated["xrd_octahedral_tilting_indicator"],
            "xrd_homogeneity_score": np.nan,
            "xrd_peak_broadening_score": aggregated["xrd_peak_broadening_score"],
            "xrd_200_peak_broadening_score": aggregated["xrd_200_peak_broadening_score"],
            "xrd_3d_peak_shift_score": aggregated["xrd_3d_peak_shift_score"],
            "xrd_200_peak_position_deg_2theta": aggregated["xrd_200_peak_position_deg_2theta"],
            "xrd_200_peak_fwhm_deg": aggregated["xrd_200_peak_fwhm_deg"],
            "xrd_200_peak_count": aggregated["xrd_200_peak_count"],
            "xrd_assignment_confidence": aggregated["xrd_rule_confidence"],
            "xrd_dominant_phase_label": np.where(aggregated["xrd_clean_3d_perovskite"] == 1, "3D perovskite-like", "mixed/needs review"),
            "xrd_notes": aggregated["xrd_rule_reason"],
            "xrd_label_source": "automatic_practical_xrd_rule",
            "xrd_rule_status": aggregated["xrd_rule_status"],
            "xrd_base_rule_status": aggregated["xrd_rule_status"],
            "xrd_rule_reason": aggregated["xrd_rule_reason"],
            "xrd_base_rule_reason": aggregated["xrd_rule_reason"],
            "xrd_final_label_basis": "practical_xrd_rule",
            "xrd_rule_primary_3d_peak_present": aggregated["xrd_rule_primary_3d_peak_present"],
            "xrd_rule_high_order_3d_peak_present": aggregated["xrd_rule_high_order_3d_peak_present"],
            "xrd_rule_review_required": aggregated["xrd_rule_review_required"],
            "xrd_source_file": aggregated["xrd_source_file"],
            "xrd_library": aggregated["xrd_library"],
            "xrd_sample_id": aggregated["xrd_sample_id"],
            "xrd_additional_peak_count_excluding_primary_3d": aggregated["xrd_additional_peak_count_excluding_primary_3d"],
            "xrd_additional_peak_positions_excluding_primary_3d_deg": aggregated["xrd_additional_peak_positions_excluding_primary_3d_deg"],
            "xrd_extra_peak_count": aggregated["xrd_extra_peak_count"],
            "xrd_extra_peak_positions_deg": aggregated["xrd_extra_peak_positions_deg"],
            "xrd_reference_peak_matches": aggregated["xrd_reference_peak_matches"],
            "xrd_candidate_reference_phases": aggregated["xrd_candidate_reference_phases"],
        }
    )
    return out.sort_values(["xrd_library", "well", "xrd_sample_id"]).reset_index(drop=True)
