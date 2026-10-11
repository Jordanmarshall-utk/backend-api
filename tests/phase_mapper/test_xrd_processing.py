from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from app.services.phase_mapper.xrd_processing import make_draft_label_table, summarize_xrd_pattern
from app.services.phase_mapper.xrd_peak_fitting import (
    apply_assignment_candidates,
    apply_best_assignments,
    attach_peak_fit_summary,
    collect_assignment_candidates,
    fit_xrd_files,
    load_reference_library,
    refit_xrd_file_with_peak_table,
    save_reference_library_json,
    summarize_assignments,
    williamson_hall_rows,
)


def _write_xrd_csv(path: Path, peaks: list[tuple[float, float]]) -> None:
    x = np.linspace(10.0, 40.0, 1501)
    y = np.full_like(x, 5.0)
    for center, height in peaks:
        y += height * np.exp(-0.5 * ((x - center) / 0.045) ** 2)
    rows = ["[Scan points]", "Angle,Intensity"]
    rows.extend(f"{angle:.4f},{intensity:.6f}" for angle, intensity in zip(x, y))
    path.write_text("\n".join(rows), encoding="utf-8")


def test_raw_xrd_reference_matches_ignore_primary_3d_peak(tmp_path: Path):
    xrd_path = tmp_path / "A1.csv"
    _write_xrd_csv(xrd_path, [(14.04, 1000), (20.08, 260), (25.42, 320), (31.31, 240)])
    cfg = {
        "xrd_processing": {
            "peak_prominence_fraction": 0.05,
            "secondary_peak_relative_height_min": 0.10,
            "primary_3d_peak_window_deg": [13.6, 14.8],
            "reference_peak_tolerance_deg": 0.20,
            "reference_phases_deg": {
                "FA_3D_alpha_perovskite": {"peaks_deg": [14.06, 20.07]},
                "PbI2": {"peaks_deg": [12.70, 25.41, 38.67]},
                "metallic_Pb": {"peaks_deg": [31.30, 36.30]},
            },
        }
    }

    summary = summarize_xrd_pattern(xrd_path, library="DMA", cfg=cfg)

    assert "14.040" not in summary["additional_peak_positions_excluding_primary_3d_deg"]
    assert "FA_3D_alpha_perovskite" in summary["candidate_reference_phases"]
    assert "PbI2" in summary["candidate_reference_phases"]
    assert "metallic_Pb" in summary["candidate_reference_phases"]
    assert "25.420->PbI2@25.410" in summary["reference_peak_matches"]


def test_draft_labels_leave_secondary_for_manual_review_by_default():
    features = pd.DataFrame(
        [
            {
                "source_file": "A1.csv",
                "library": "DMA",
                "sample_id": "A1",
                "well": "A1",
                "high_order_peak_count": 1,
                "high_order_main_peak_position_deg": 28.2,
                "high_order_main_peak_fwhm_deg": 0.2,
                "perovskite_peak_window_hits": 2,
                "additional_peak_count_excluding_primary_3d": 2,
                "additional_peak_positions_excluding_primary_3d_deg": "20.070;25.410",
                "extra_peak_count": 1,
                "extra_peak_positions_deg": "25.410",
                "reference_peak_matches": "25.410->PbI2@25.410(+0.000)",
                "candidate_reference_phases": "PbI2(1 peaks, mean_abs_delta=0.000)",
            }
        ]
    )

    labels = make_draft_label_table(features)
    row = labels.iloc[0]

    assert pd.isna(row["xrd_secondary_phase_present"])
    assert row["xrd_extra_peak_positions_deg"] == "25.410"
    assert "PbI2" in row["xrd_candidate_reference_phases"]
    assert "manual" in row["xrd_notes"]


def test_practical_rule_allows_two_extra_peaks_but_three_is_mixed():
    common = {
        "library": "DMA",
        "high_order_peak_count": 1,
        "high_order_main_peak_position_deg": 28.2,
        "high_order_main_peak_fwhm_deg": 0.2,
        "primary_3d_peak_present": 1,
        "perovskite_peak_window_hits": 2,
        "additional_peak_count_excluding_primary_3d": 4,
        "additional_peak_positions_excluding_primary_3d_deg": "20.0;24.0;28.2;31.0",
        "reference_peak_matches": "",
        "candidate_reference_phases": "",
    }
    features = pd.DataFrame(
        [
            {**common, "source_file": "A1.csv", "sample_id": "A1", "well": "A1", "extra_peak_count": 2, "extra_peak_positions_deg": "10;11"},
            {**common, "source_file": "A2.csv", "sample_id": "A2", "well": "A2", "extra_peak_count": 3, "extra_peak_positions_deg": "10;11;12"},
        ]
    )

    labels = make_draft_label_table(features).set_index("well")

    assert labels.loc["A1", "xrd_clean_3d_perovskite"] == 1
    assert labels.loc["A1", "xrd_base_rule_clean_3d_perovskite"] == 1
    assert labels.loc["A1", "xrd_final_label_basis"] == "practical_xrd_rule"
    assert labels.loc["A1", "xrd_assignment_confidence"] == "medium"
    assert labels.loc["A2", "xrd_clean_3d_perovskite"] == 0
    assert labels.loc["A2", "xrd_mixed_phase"] == 1
    assert labels.loc["A2", "xrd_assignment_confidence"] == "medium"


def test_reference_rules_csv_drives_raw_xrd_matching(tmp_path: Path):
    rules_path = tmp_path / "rules.csv"
    pd.DataFrame(
        [
            {
                "enabled": True,
                "phase_label": "3D_perovskite_primary",
                "match_role": "primary_3d",
                "peak_2theta_deg": 14.0,
                "tolerance_deg": 0.25,
            },
            {
                "enabled": True,
                "phase_label": "custom_phase",
                "match_role": "reference_hint",
                "peak_2theta_deg": 22.2,
                "tolerance_deg": 0.25,
            },
        ]
    ).to_csv(rules_path, index=False)
    xrd_path = tmp_path / "B2.csv"
    _write_xrd_csv(xrd_path, [(14.02, 1000), (22.18, 300)])

    summary = summarize_xrd_pattern(
        xrd_path,
        cfg={"_config_dir": str(tmp_path), "xrd_processing": {"reference_rules_file": "rules.csv"}},
    )

    assert "14.020" not in summary["additional_peak_positions_excluding_primary_3d_deg"]
    assert "custom_phase" in summary["candidate_reference_phases"]


def test_confirmed_references_json_drives_raw_xrd_matching(tmp_path: Path):
    rules_path = tmp_path / "confirmed_references.json"
    rules_path.write_text(
        json.dumps(
            {
                "version": 1,
                "references": [
                    {
                        "name": "Coworker Phase",
                        "visible": True,
                        "peaks": [
                            {"two_theta": 22.2, "selected": True, "tolerance": 0.25},
                            {"two_theta": 30.0, "selected": False, "tolerance": 0.25},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    xrd_path = tmp_path / "B3.csv"
    _write_xrd_csv(xrd_path, [(14.02, 1000), (22.18, 300)])

    summary = summarize_xrd_pattern(
        xrd_path,
        cfg={"_config_dir": str(tmp_path), "xrd_processing": {"reference_rules_file": "confirmed_references.json"}},
    )

    assert "Coworker Phase" in summary["candidate_reference_phases"]
    assert "30.000" not in summary["reference_peak_matches"]


@pytest.mark.skip(
    reason="requires the bundled external XRD peak-fitting GUI app "
    "(external/XRD fitting/), which the design doc lists as a Non-goal "
    "with its porting decision deferred"
)
def test_bundled_peak_fitting_outputs_summary_and_peak_table(tmp_path: Path):
    xrd_path = tmp_path / "C3.csv"
    _write_xrd_csv(xrd_path, [(14.02, 1000), (28.25, 500)])

    peaks, curves, fit_summary = fit_xrd_files([xrd_path], library="DMA")

    assert not peaks.empty
    assert not curves.empty
    assert fit_summary.iloc[0]["peak_fit_status"] == "fitted"
    assert fit_summary.iloc[0]["well"] == "C3"
    assert fit_summary.iloc[0]["peak_fit_peak_count"] >= 2
    assert "peak_fit_positions_deg" in fit_summary.columns

    features = pd.DataFrame([{"source_file": str(xrd_path), "well": "C3"}])
    merged = attach_peak_fit_summary(features, fit_summary)
    assert "peak_fit_r2" in merged.columns
    assert merged.iloc[0]["peak_fit_status"] == "fitted"


@pytest.mark.skip(
    reason="requires the bundled external XRD peak-fitting GUI app "
    "(external/XRD fitting/), which the design doc lists as a Non-goal "
    "with its porting decision deferred"
)
def test_refit_xrd_file_with_reviewed_peak_table(tmp_path: Path):
    xrd_path = tmp_path / "D4.csv"
    _write_xrd_csv(xrd_path, [(14.02, 1000), (28.25, 500)])
    reviewed = pd.DataFrame(
        [
            {"peak_id": 1, "two_theta": 14.0, "height": 900.0, "fwhm": 0.12, "shape": "pseudo_voigt", "status": "confirmed"},
            {"peak_id": 2, "two_theta": 28.2, "height": 450.0, "fwhm": 0.12, "shape": "pseudo_voigt", "status": "confirmed"},
        ]
    )

    peaks, curves, summary = refit_xrd_file_with_peak_table(xrd_path, reviewed)

    assert summary["peak_fit_status"] == "fitted"
    assert summary["well"] == "D4"
    assert len(peaks) == 2
    assert not curves.empty


def test_reference_assignment_selects_best_priority_then_delta(tmp_path: Path):
    ref_path = tmp_path / "confirmed_references.json"
    ref_path.write_text(
        json.dumps(
            {
                "references": [
                    {
                        "name": "High Priority Phase",
                        "visible": True,
                        "priority": 10,
                        "peaks": [{"two_theta": 14.04, "selected": True, "tolerance": 0.3}],
                    },
                    {
                        "name": "Low Priority Closer Phase",
                        "visible": True,
                        "priority": 50,
                        "peaks": [{"two_theta": 14.02, "selected": True, "tolerance": 0.3}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    refs = load_reference_library(ref_path)
    peaks = pd.DataFrame(
        [
            {
                "library": "DMA",
                "sample_id": "A1",
                "well": "A1",
                "source_file": "A1.csv",
                "peak_id": 1,
                "two_theta": 14.021,
                "height": 1000.0,
                "fwhm": 0.1,
                "area": 100.0,
                "status": "fitted",
                "phase": "",
                "hkl": "",
            }
        ]
    )

    candidates = collect_assignment_candidates(peaks, refs, active_phases=["High Priority Phase", "Low Priority Closer Phase"])
    best = candidates[candidates["is_best"]].iloc[0]

    assert best["candidate_phase"] == "High Priority Phase"

    assigned, assignment_table = apply_best_assignments(peaks, candidates, overwrite=True)
    summary = summarize_assignments(assigned, refs)

    assert assigned.iloc[0]["phase"] == "High Priority Phase"
    assert assignment_table.iloc[0]["phase"] == "High Priority Phase"
    assert summary.iloc[0]["phase"] == "High Priority Phase"
    assert summary.iloc[0]["coverage"] == "1/1"


def test_reference_assignment_handles_empty_numeric_phase_column():
    peaks = pd.DataFrame(
        [
            {
                "library": "DMA",
                "sample_id": "A1",
                "well": "A1",
                "source_file": "A1.csv",
                "peak_id": 1,
                "two_theta": 14.02,
                "height": 1000.0,
                "fwhm": 0.1,
                "area": 100.0,
                "phase": np.nan,
                "hkl": np.nan,
                "status": np.nan,
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "library": "DMA",
                "sample_id": "A1",
                "well": "A1",
                "source_file": "A1.csv",
                "peak_id": 1,
                "sample_two_theta": 14.02,
                "candidate_phase": "3D alpha phase",
                "reference_two_theta": 14.0,
                "delta_two_theta": 0.02,
                "hkl": "",
                "priority": 50,
                "score": 0.9,
                "is_best": True,
            }
        ]
    )

    assigned, _ = apply_best_assignments(peaks, candidates, overwrite=True)

    assert assigned.iloc[0]["phase"] == "3D alpha phase"


def test_selected_reference_assignment_can_override_best_candidate():
    peaks = pd.DataFrame(
        [
            {
                "library": "DMA",
                "sample_id": "A1",
                "well": "A1",
                "source_file": "A1.csv",
                "peak_id": 1,
                "two_theta": 14.02,
                "height": 1000.0,
                "fwhm": 0.1,
                "area": 100.0,
                "phase": "",
                "hkl": "",
                "status": "auto",
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "library": "DMA",
                "sample_id": "A1",
                "well": "A1",
                "source_file": "A1.csv",
                "peak_id": 1,
                "sample_two_theta": 14.02,
                "candidate_phase": "Best Phase",
                "reference_two_theta": 14.01,
                "delta_two_theta": 0.01,
                "hkl": "100",
                "priority": 10,
                "score": 0.95,
                "is_best": True,
            },
            {
                "library": "DMA",
                "sample_id": "A1",
                "well": "A1",
                "source_file": "A1.csv",
                "peak_id": 1,
                "sample_two_theta": 14.02,
                "candidate_phase": "Manual Choice",
                "reference_two_theta": 14.04,
                "delta_two_theta": -0.02,
                "hkl": "001",
                "priority": 50,
                "score": 0.90,
                "is_best": False,
            },
        ]
    )

    assigned, assignment_table = apply_assignment_candidates(peaks, candidates.iloc[[1]], overwrite=True)

    assert assigned.iloc[0]["phase"] == "Manual Choice"
    assert assigned.iloc[0]["hkl"] == "001"
    assert assignment_table.iloc[0]["phase"] == "Manual Choice"


def test_reference_library_json_roundtrip_and_wh_summary(tmp_path: Path):
    refs = pd.DataFrame(
        [
            {
                "phase": "3D alpha phase",
                "reference_two_theta": 14.0,
                "hkl": "100",
                "relative_intensity": 100.0,
                "tolerance": 0.25,
                "selected": True,
                "visible": True,
                "priority": 10,
                "color": "#123456",
                "source_type": "manual",
            }
        ]
    )
    path = tmp_path / "refs.json"
    save_reference_library_json(refs, path)
    loaded = load_reference_library(path)

    assert loaded.iloc[0]["phase"] == "3D alpha phase"
    assert loaded.iloc[0]["reference_two_theta"] == 14.0
    assert bool(loaded.iloc[0]["selected"]) is True

    peaks = pd.DataFrame(
        [
            {"phase": "3D alpha phase", "two_theta": 14.0, "fwhm": 0.12, "peak_id": 1},
            {"phase": "3D alpha phase", "two_theta": 28.1, "fwhm": 0.18, "peak_id": 2},
        ]
    )
    points, summary = williamson_hall_rows(peaks)

    assert len(points) == 2
    assert summary.iloc[0]["phase"] == "3D alpha phase"
    assert summary.iloc[0]["num_points"] == 2
