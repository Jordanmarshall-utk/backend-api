"""Fast, deterministic data fixture for the POLARIS presentation tour."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from app.services.analysis_service import _hypothesis_fingerprint, _hypothesis_text
from app.services.workflow_service import run_demo_workflow


WELLS = [*[f"A{i}" for i in range(1, 13)], *[f"B{i}" for i in range(1, 4)]]
R2_VALUES = [
    0.99985,
    0.99992,
    0.99987,
    0.99992,
    0.99990,
    0.99992,
    0.99992,
    0.99990,
    0.99991,
    0.99992,
    0.99993,
    0.99993,
    0.99991,
    0.99991,
    0.99992,
]


def _results_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _phase_peaks(pea_fraction: float) -> List[Dict[str, float]]:
    """Representative iodide RP/3D phase peaks for a composition-series replay."""
    candidates = [
        (815.0, 520.0 * (1.0 - pea_fraction) ** 0.55 + 15.0, 40.0),
        (700.0, 230.0 * math.exp(-0.5 * ((pea_fraction - 0.34) / 0.15) ** 2), 34.0),
        (619.0, 285.0 * math.exp(-0.5 * ((pea_fraction - 0.55) / 0.16) ** 2), 28.0),
        (574.0, 310.0 * math.exp(-0.5 * ((pea_fraction - 0.73) / 0.15) ** 2), 24.0),
        (522.0, 430.0 * pea_fraction**1.65, 20.0),
    ]
    peaks = [
        {"center": center, "height": height, "fwhm": fwhm}
        for center, height, fwhm in candidates
        if height >= 42.0
    ]
    return sorted(peaks, key=lambda peak: peak["height"], reverse=True)


def _write_source_data(results_dir: Path) -> Dict[str, Path]:
    """Write raw spectra and compositions from the same model used by the replay."""
    demo_dir = results_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    spectral_path = demo_dir / "demo_spectral_data.csv"
    composition_path = demo_dir / "demo_composition.csv"

    with spectral_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        blank_tail = [""] * len(WELLS)
        writer.writerows(
            [
                ["Software Version", "3.08.01", *blank_tail],
                ["Experiment File Path", "POLARIS presentation replay", *blank_tail],
                ["Plate Number", "1", *blank_tail],
                ["Plate Type", "96 WELL PLATE (Use plate lid)", *blank_tail],
                ["Read", "Fluorescence Spectrum", *blank_tail],
                ["", "Full Plate", *blank_tail],
                ["Read 1:EM Spectrum", "", *blank_tail],
                ["", "", *blank_tail],
                ["", "Wavelength", *WELLS],
            ]
        )
        peak_sets = [_phase_peaks(index / (len(WELLS) - 1)) for index in range(len(WELLS))]
        for wavelength in range(450, 901, 2):
            row: List[Any] = ["", wavelength]
            for well_index, peaks in enumerate(peak_sets):
                signal = 10.0 + 0.35 * math.sin((wavelength + well_index * 7) / 13.0)
                for peak in peaks:
                    sigma = peak["fwhm"] / 2.355
                    signal += peak["height"] * math.exp(
                        -0.5 * ((wavelength - peak["center"]) / sigma) ** 2
                    )
                row.append(f"{signal:.4f}")
            writer.writerow(row)

    with composition_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *WELLS])
        fractions = [index / (len(WELLS) - 1) for index in range(len(WELLS))]
        writer.writerow(["PEA2PbI4", *[f"{fraction * 100:.6f}" for fraction in fractions]])
        writer.writerow(["FAPbI3", *[f"{(1.0 - fraction) * 100:.6f}" for fraction in fractions]])

    return {"spectral": spectral_path, "composition": composition_path}


def _write_fit_plot(path: Path, peaks: List[Dict[str, float]], well: str) -> None:
    points: List[str] = []
    for pixel_x in range(72, 702, 7):
        wavelength = 400.0 + ((pixel_x - 72) / 630.0) * 500.0
        signal = 10.0
        for peak in peaks:
            sigma = peak["fwhm"] / 2.355
            signal += peak["height"] * math.exp(-0.5 * ((wavelength - peak["center"]) / sigma) ** 2)
        pixel_y = 318.0 - min(255.0, signal * 0.5)
        points.append(f"{pixel_x},{pixel_y:.1f}")

    dominant = peaks[0]
    marker_lines = []
    for peak in peaks:
        peak_x = 72 + ((peak["center"] - 400) / 500) * 630
        marker_lines.append(
            f'<line x1="{peak_x:.1f}" y1="48" x2="{peak_x:.1f}" y2="318" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="7 7" opacity="0.7"/>'
        )
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="390" viewBox="0 0 760 390">',
                '<rect width="760" height="390" rx="18" fill="#0b1220"/>',
                '<line x1="72" y1="318" x2="702" y2="318" stroke="#64748b"/>',
                '<line x1="72" y1="48" x2="72" y2="318" stroke="#64748b"/>',
                f'<polyline points="{" ".join(points)}" fill="none" stroke="#38bdf8" stroke-width="4"/>',
                *marker_lines,
                f'<text x="72" y="28" fill="#e2e8f0" font-family="sans-serif" font-size="18">{well}: {len(peaks)} fitted phase peak(s), dominant at {dominant["center"]:.0f} nm</text>',
                '<text x="340" y="360" fill="#94a3b8" font-family="sans-serif" font-size="14">Wavelength (nm)</text>',
                '</svg>',
            ]
        ),
        encoding="utf-8",
    )


def _curve_results(results_dir: Path) -> Dict[str, Any]:
    wells: List[Dict[str, Any]] = []
    for index, well in enumerate(WELLS):
        pea_fraction = index / (len(WELLS) - 1)
        peaks = _phase_peaks(pea_fraction)
        rmse = 1.35 + 0.24 * ((index % 4) / 3.0)
        plot_path = results_dir / f"presentation_fit_{well}.svg"
        _write_fit_plot(plot_path, peaks, well)
        wells.append(
            {
                "well_name": well,
                "read": "1",
                "fit": {
                    "success": True,
                    "r2": R2_VALUES[index],
                    "rmse": rmse,
                    "redchi": rmse**2,
                    "peak_count": len(peaks),
                    "peaks": peaks,
                },
                "quality_assessment": {
                    "quality": "Excellent",
                    "message": "High-confidence fit for the representative presentation dataset.",
                },
                "plot_url": f"/agents/curve-fitting/plot?well={well}&read=1",
                "plot_path": str(plot_path),
            }
        )

    json_path = results_dir / "presentation_peak_fit_results.json"
    csv_path = results_dir / "presentation_peak_fit_export.csv"
    payload: Dict[str, Any] = {
        "success": True,
        "error": None,
        "summary": {
            "total_wells": len(WELLS),
            "successful_fits": len(WELLS),
            "success_rate_pct": 100.0,
            "wells_analyzed": WELLS,
        },
        "wells": wells,
        "files": {"json_results": str(json_path), "csv_export": str(csv_path)},
        "jupyter_upload": None,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    peak_fields = ["Wavelength", "Intensity", "FWHM", "Area", "LeftArea", "RightArea", "Asymmetry"]
    header = ["Read", "Composition", "Well", "R_squared", "Total_Quality_Peaks"]
    for peak_number in range(1, 6):
        header.extend([f"Peak_{peak_number}_{field}" for field in peak_fields])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, item in enumerate(wells):
            fit = item["fit"]
            pea_fraction = index / (len(wells) - 1)
            row: List[Any] = [
                "1",
                f"PEA2PbI4={pea_fraction:.6f}; FAPbI3={1.0 - pea_fraction:.6f}",
                item["well_name"],
                f'{fit["r2"]:.6f}',
                len(fit["peaks"]),
            ]
            for peak in fit["peaks"][:5]:
                sigma = peak["fwhm"] / 2.355
                area = peak["height"] * sigma * math.sqrt(2.0 * math.pi)
                row.extend(
                    [
                        f'{peak["center"]:.4f}',
                        f'{peak["height"]:.4f}',
                        f'{peak["fwhm"]:.4f}',
                        f"{area:.4f}",
                        f"{area / 2.0:.4f}",
                        f"{area / 2.0:.4f}",
                        "1.0000",
                    ]
                )
            row.extend([""] * ((5 - len(fit["peaks"][:5])) * len(peak_fields)))
            writer.writerow(row)
    return payload


def _gp_results(results_dir: Path) -> Dict[str, Any]:
    candidate_specs = []
    for pea_fraction in [0.70, 0.73, 0.55, 0.84, 0.87]:
        peaks = _phase_peaks(pea_fraction)
        prediction = peaks[0]["center"]
        runner_up_ratio = peaks[1]["height"] / peaks[0]["height"] if len(peaks) > 1 else 0.0
        uncertainty = 8.0 + 18.0 * runner_up_ratio
        candidate_specs.append((pea_fraction, prediction, uncertainty))
    candidate_specs.sort(key=lambda item: item[2], reverse=True)
    candidates: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []
    for rank, (pea_fraction, prediction, uncertainty) in enumerate(candidate_specs, start=1):
        acquisition = uncertainty
        candidates.append(
            {
                "rank": rank,
                "candidate": f"candidate_{rank}",
                "predicted_value": prediction,
                "uncertainty": uncertainty,
                "acquisition_score": acquisition,
            }
        )
        suggestions.append(
            {
                "rank": rank,
                "compositions": {"PEA2PbI4": pea_fraction, "FAPbI3": 1.0 - pea_fraction},
                "predicted_value": prediction,
                "uncertainty": uncertainty,
                "acquisition_score": acquisition,
            }
        )

    training_predictions = [
        _phase_peaks(index / (len(WELLS) - 1))[0]["center"] for index in range(len(WELLS))
    ]
    prediction_mean = sum(training_predictions) / len(training_predictions)
    prediction_std = math.sqrt(
        sum((prediction - prediction_mean) ** 2 for prediction in training_predictions)
        / len(training_predictions)
    )

    plot_path = results_dir / "presentation_gp_ucb_acquisition.html"
    plot_path.write_text(
        """<!doctype html><html><body style="margin:0;background:#0b1220;color:#e2e8f0;font-family:sans-serif">
<svg viewBox="0 0 760 390" width="100%" role="img" aria-label="Representative GP acquisition plot">
<line x1="72" y1="320" x2="700" y2="320" stroke="#64748b"/><line x1="72" y1="45" x2="72" y2="320" stroke="#64748b"/>
<path d="M72 55 L490 55 L490 180 L545 180 L545 255 L700 255" fill="none" stroke="#22d3ee" stroke-width="5"/>
<path d="M72 38 L470 38 L470 150 L565 150 L565 228 L700 228 L700 282 L525 282 L525 210 L510 210 L510 75 L72 75 Z" fill="#22d3ee" opacity=".16"/>
<circle cx="512" cy="180" r="9" fill="#fbbf24"/><text x="260" y="125" font-size="17" fill="#fbbf24">Highest-uncertainty phase boundary</text>
<text x="285" y="365" font-size="14" fill="#94a3b8">PEA2PbI4 composition fraction</text>
</svg></body></html>""",
        encoding="utf-8",
    )
    return {
        "success": True,
        "model_type": "Precomputed single-objective GP replay",
        "kernel": "RBF",
        "kernel_type": "RBF",
        "target": "peak_1_wavelength",
        "target_name": "peak_1_wavelength",
        "acquisition_method": "Maximum predictive uncertainty",
        "beta": None,
        "top_candidates": candidates,
        "suggested_compositions": suggestions,
        "predictions": {"mean": prediction_mean, "std": prediction_std},
        "uncertainty_stats": {
            "mean": sum(item[2] for item in candidate_specs) / len(candidate_specs),
            "max": max(item[2] for item in candidate_specs),
        },
        "cv_score": 0.91,
        "cv_std": 0.04,
        "gp_ucb_acquisition_plot": str(plot_path),
    }


def load_presentation_demo(memory: Any) -> Dict[str, Any]:
    """Load a complete representative workflow without running fitting or ML models."""
    base_result = run_demo_workflow(memory, auto_fit=False)
    results_dir = _results_dir()
    source_files = _write_source_data(results_dir)
    curve_results = _curve_results(results_dir)
    gp_results = _gp_results(results_dir)

    composition_path = source_files["composition"]
    spectral_path = source_files["spectral"]
    json_path = Path(curve_results["files"]["json_results"])
    csv_path = Path(curve_results["files"]["csv_export"])
    memory.set_var("curve_fitting_results", curve_results)
    memory.set_var("curve_fitting_last_composition_file", str(composition_path))
    memory.set_var("curve_fitting_last_data_file", str(spectral_path))
    memory.set_var("ml_auto_json_path", str(json_path))
    memory.set_var("ml_auto_csv_path", str(csv_path))
    memory.set_var("ml_auto_composition_path", str(composition_path))
    memory.set_var("gp_results", gp_results)
    memory.set_var("gp_results_available", True)
    memory.set_var("gp_suggested_compositions", gp_results["suggested_compositions"])

    memory.set_var(
        "document_experiment",
        {
            "document_id": "presentation-experiment-protocol",
            "markdown": """# Experimental protocol

## Research objective

Resolve composition-dependent 2D, quasi-2D, and 3D photoluminescence features in a 15-well PEA2PbI4/FAPbI3 series.

## Plate and controls

- Use wells A1-A12 and B1-B3 on a 96-well plate.
- Include replicate wells near predicted phase-transition regions.
- Hold deposition, excitation, integration time, and temperature constant.

## Measurement and release gate

Collect emission from 450-900 nm, retain the raw spectra, and pass only validated files to multi-peak curve fitting. The worklist and plate layout are released only after required materials, volumes, and controls are present.
""",
            "pdf_url": None,
        },
    )

    report = """# Closed-loop analysis replay

## Goal / hypothesis evaluation

This presentation uses a fixed, literature-grounded representative dataset. It demonstrates the POLARIS workflow and does not constitute evidence for a physical materials claim.

## Results analysis

All 15 wells produced successful phase-resolved fits. FA-rich compositions retain long-wavelength emission near 815 nm, while increasing PEA-rich, lower-dimensional character introduces shorter-wavelength features near 619, 574, and 522 nm. The dominant peak can switch between phases rather than moving as one linear peak.

## Model-guided next step

The Gaussian-process model identifies the composition regions where the dominant emitting phase changes and ranks those uncertain transition regions for the next iteration. Uncertainty remains visible so the recommendation can be evaluated rather than accepted as a black-box result.

## Closed-loop conclusion

The replay demonstrates the full loop: hypothesis, experiment design, spectral fitting, predictive modeling, interpretation, and selection of the next candidate. A live run would replace these representative outputs with instrument measurements and newly computed models.
"""
    recommendations = [
        "Validate the suggested phase-transition compositions with new instrument measurements.",
        "Include replicate wells and process controls before drawing a scientific conclusion.",
        "Feed measured results back into POLARIS to update the next-candidate ranking.",
    ]
    hypothesis = _hypothesis_text(memory)
    memory.set_var("analysis_full_report", report)
    memory.set_var("analysis_recommendations", recommendations)
    memory.set_var("analysis_hypothesis_fingerprint", _hypothesis_fingerprint(hypothesis))
    memory.set_var("analysis_results", report)
    memory.set_var("analysis_ready", True)
    memory.insert_interaction(
        "assistant",
        "Resolved composition-dependent PL features near 522, 574, 619, and 815 nm across 15 wells.",
        "curve_fitting_results",
        "curve fitting",
    )
    memory.insert_interaction(
        "assistant",
        "Ranked candidate compositions by predictive uncertainty around the dominant-phase transitions.",
        "candidate_ranking",
        "ml_models",
    )
    memory.insert_interaction(
        "assistant",
        "Combined the hypothesis, protocol, fitted spectra, and model uncertainty into the next-experiment recommendation.",
        "analysis_report",
        "analysis",
    )

    response = dict(base_result) if isinstance(base_result, dict) else {"success": True}
    response.update(
        {
            "success": True,
            "presentation_ready": True,
            "precomputed": True,
            "message": "Representative presentation data loaded without running long computations.",
            "source_files": {key: str(value) for key, value in source_files.items()},
            "summary": {
                "wells": len(WELLS),
                "successful_fits": len(WELLS),
                "peak_range_nm": [522.0, 815.0],
                "gp_candidates": len(gp_results["top_candidates"]),
            },
        }
    )
    return response
