import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.agents.reasoning.reasoning_service import ReasoningService
from app.agents.reasoning.schemas import TaskPacket
from app.services.composition_fallback import (
    composition_from_worklist,
    composition_provenance,
    ensure_composition_csv,
    is_placeholder_composition,
)
from app.tools.fitting_agent import (
    PeakFitResult,
    PeakFitStats,
    PeakGuess,
    _build_composite_model,
    assess_fitting_quality,
    build_fit_qa,
    deterministic_peak_guesses,
)
from scripts.rerun_curve_fitting_with_qa import _detect_measured_wells, _parse_wells


def _fit_result(*, center: float, r2: float = 0.99, rmse: float = 1.0) -> PeakFitResult:
    return PeakFitResult(
        success=r2 >= 0.92,
        stats=PeakFitStats(
            r2=r2,
            rmse=rmse,
            aic=1.0,
            bic=2.0,
            redchi=12345.0,
            nfev=4,
        ),
        best_params={},
        peaks=[PeakGuess(center=center, height=10.0, fwhm=20.0)],
        baseline=0.0,
        report="",
        model_kind="gaussian",
        lmfit_result=None,
    )


def test_peak_center_bounds_are_clipped_to_measured_domain():
    x = np.arange(450.0, 871.0)
    y = np.exp(-0.5 * ((x - 865.0) / 10.0) ** 2)
    _, params = _build_composite_model(
        x,
        y,
        [PeakGuess(center=900.0, height=1.0, fwhm=20.0)],
        baseline=0.0,
    )
    center = params["p0_center"]
    assert center.value == 870.0
    assert center.min >= 450.0
    assert center.max <= 870.0


def test_deterministic_guesses_ignore_acquisition_endpoints():
    x = np.arange(450.0, 871.0)
    y = 100.0 * np.exp(-0.5 * ((x - 525.0) / 8.0) ** 2)
    y[-1] = 1000.0
    guesses = deterministic_peak_guesses(x, y, max_peaks=3)
    assert guesses.peaks
    assert all(450.0 < peak.center < 870.0 for peak in guesses.peaks)
    assert any(abs(peak.center - 525.0) <= 1.0 for peak in guesses.peaks)


def test_qa_flags_boundary_peaks_and_normalizes_rmse():
    x = np.arange(450.0, 871.0)
    y = np.linspace(0.0, 100.0, len(x))
    qa = build_fit_qa(_fit_result(center=868.0, rmse=5.0), x, y)
    assert qa["review_required"] is True
    assert "boundary_adjacent_peak" in qa["review_reasons"]
    assert qa["outside_range_centers_nm"] == []
    assert qa["normalized_rmse"] == 0.05


def test_weighted_residual_variance_is_not_labeled_calibrated_chi_square():
    assessment = assess_fitting_quality(_fit_result(center=525.0))
    assert "chi2" not in assessment
    assert "diagnostic only" in assessment["weighted_residual_variance"]


def test_missing_composition_generates_non_authoritative_placeholder(tmp_path, monkeypatch):
    spectral = tmp_path / "spectral.csv"
    spectral.write_text("Plate Type,96 WELL PLATE\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.services.composition_fallback.get_user_data_dir",
        lambda: str(tmp_path / "user-data"),
    )

    composition = ensure_composition_csv(str(spectral))
    frame = pd.read_csv(composition, index_col=0)
    provenance = composition_provenance(composition)

    assert frame.index.tolist() == ["composition_unavailable"]
    assert len(frame.columns) == 96
    assert is_placeholder_composition(composition) is True
    assert provenance["authoritative"] is False
    assert provenance["safe_for_composition_modeling"] is False
    assert json.loads(open(provenance["metadata_path"], encoding="utf-8").read())["well_count"] == 96


def test_actual_mixing_worklist_can_be_normalized_into_composition_map(tmp_path):
    worklist = tmp_path / "mixing.csv"
    worklist.write_text(
        "Well,C60_uL,PEA_uL,FA_uL,Solvent_uL\n"
        "A1,10,20,20,50\n"
        "A2,25,25,0,50\n",
        encoding="utf-8",
    )
    output = tmp_path / "composition.csv"
    composition_from_worklist(str(worklist), str(output))
    frame = pd.read_csv(output, index_col=0)
    provenance = composition_provenance(str(output))

    assert frame.loc["C60", "A1"] == 10.0
    assert frame.loc["PEA", "A1"] == 20.0
    assert frame.loc["FA", "A1"] == 20.0
    assert frame.loc["Solvent", "A1"] == 50.0
    assert frame["A2"].sum() == 100.0
    assert provenance["status"] == "derived_from_worklist"
    assert provenance["safe_for_composition_modeling"] is True


def test_measured_well_detection_excludes_blank_export_columns(monkeypatch):
    tensor = np.asarray(
        [
            [
                [1.0, np.nan, 1.0],
                [2.0, np.nan, 2.0],
                [3.0, np.nan, np.nan],
            ]
        ]
    )

    class DummyCurveFitting:
        def __init__(self, _config):
            pass

        def curate_dataset(self):
            return {"tensor": tensor, "wells": ["A1", "A2", "B1"]}

    monkeypatch.setattr(
        "scripts.rerun_curve_fitting_with_qa.build_agent_config",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "scripts.rerun_curve_fitting_with_qa.CurveFitting",
        DummyCurveFitting,
    )

    assert _detect_measured_wells(Path("spectra.csv"), "composition.csv") == ["A1"]
    assert _parse_wells("A2, A1, A2") == ["A1", "A2"]


def test_curve_fitting_bypasses_reasoning_council():
    packet = TaskPacket(
        domain_agent="Curve Fitting Agent",
        stage="curve_fitting",
        objective="Run deterministic curve fitting for uploaded spectra",
    )
    assert ReasoningService.needs_reasoning(packet) is False
