"""Run deterministic spectral fitting and package a scientifically auditable deliverable."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.agents.curve_fitting_agent import CurveFittingAgent
from app.services.composition_fallback import (
    composition_from_worklist,
    composition_provenance,
    ensure_composition_csv,
)
from app.tools.fitting_agent import CurveFitting, build_agent_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join(values: list[Any]) -> str:
    return "; ".join(str(value) for value in values)


def _well_sort_key(well: str) -> tuple[str, int]:
    return well[:1], int(well[1:])


def _detect_measured_wells(data_path: Path, composition_path: str) -> list[str]:
    """Return wells with actual numeric measurements, excluding blank plate columns."""
    config = build_agent_config(
        data_csv=str(data_path),
        composition_csv=composition_path,
        read_selection="all",
        fill_na_value=np.nan,
    )
    curated = CurveFitting(config).curate_dataset()
    tensor = np.asarray(curated["tensor"], dtype=float)
    wells = list(curated["wells"])
    measured = [
        well
        for index, well in enumerate(wells)
        if int(np.isfinite(tensor[:, :, index]).sum()) >= 3
    ]
    return sorted(measured, key=_well_sort_key)


def _parse_wells(value: str | None) -> list[str] | None:
    if not value:
        return None
    wells = [part.strip().upper() for part in value.split(",") if part.strip()]
    return sorted(dict.fromkeys(wells), key=_well_sort_key)


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        fit = item["fit_result"]
        qa = item.get("qa") or {}
        peaks = fit.peaks or []
        plot_path = Path((item.get("files") or {}).get("fitting_plot", ""))
        rows.append(
            {
                "well": item["well_name"],
                "read": item.get("read"),
                "success": bool(fit.success),
                "review_required": bool(qa.get("review_required")),
                "review_reasons": _join(qa.get("review_reasons") or []),
                "r2": float(fit.stats.r2),
                "rmse": float(fit.stats.rmse),
                "normalized_rmse": qa.get("normalized_rmse"),
                "weighted_residual_variance": float(fit.stats.redchi),
                "weighted_residual_variance_interpretation": qa.get(
                    "weighted_residual_variance_interpretation"
                ),
                "peak_count": len(peaks),
                "peak_centers_nm": _join([round(float(p.center), 3) for p in peaks]),
                "peak_heights": _join([round(float(p.height), 3) for p in peaks]),
                "peak_fwhm_nm": _join(
                    [round(float(p.fwhm), 3) if p.fwhm is not None else "" for p in peaks]
                ),
                "outside_range_centers_nm": _join(qa.get("outside_range_centers_nm") or []),
                "boundary_adjacent_centers_nm": _join(
                    qa.get("boundary_adjacent_centers_nm") or []
                ),
                "bound_pegged_centers_nm": _join(qa.get("bound_pegged_centers_nm") or []),
                "analysis_mode": item.get("analysis_mode", "deterministic"),
                "plot_file": plot_path.name if plot_path.name else None,
            }
        )
    return sorted(rows, key=lambda row: _well_sort_key(str(row["well"])))


def _write_composition_template(path: Path, wells: list[str]) -> None:
    frame = pd.DataFrame(
        "",
        index=["C60", "PEA", "FA"],
        columns=wells,
    )
    frame.index.name = "material"
    frame.to_csv(path)


def _write_mixing_worklist_template(path: Path, wells: list[str]) -> None:
    pd.DataFrame(
        {
            "Well": wells,
            "C60_uL": "",
            "PEA_uL": "",
            "FA_uL": "",
        }
    ).to_csv(path, index=False)


def _write_run_summary(
    path: Path,
    *,
    data_path: Path,
    frame: pd.DataFrame,
    plot_count: int,
    provenance: dict[str, Any],
    r2_target: float,
    review_r2_threshold: float,
) -> None:
    successful = frame[frame["success"] == True]  # noqa: E712
    failed = frame[frame["success"] != True]  # noqa: E712
    review = frame[frame["review_required"] == True]  # noqa: E712
    outside = frame[frame["outside_range_centers_nm"].fillna("") != ""]
    boundary = frame[frame["boundary_adjacent_centers_nm"].fillna("") != ""]
    accepted = frame[frame["review_required"] != True]  # noqa: E712
    peak_counts = Counter(successful["peak_count"].astype(int).tolist())
    lines = [
        "POLARIS Curve Fitting Validated Rerun",
        f"Dataset: {data_path.name}",
        "Analysis mode: deterministic (LLM advisory calls disabled)",
        f"Fit success threshold: R2 >= {r2_target:.3f}",
        f"QA review threshold: R2 >= {review_r2_threshold:.3f}",
        "Spectral center constraint: every fitted center must remain inside the measured domain",
        f"Measured wells returned: {len(frame)}",
        f"Successful fits: {len(successful)}",
        f"Unsuccessful fits: {len(failed)}",
        f"Unsuccessful wells: {_join(failed['well'].tolist()) or 'none'}",
        f"Wells requiring QA review: {len(review)}",
        f"Review wells: {_join(review['well'].tolist()) or 'none'}",
        f"Wells accepted without spectral QA flags: {len(accepted)}",
        f"Wells with out-of-range centers: {len(outside)}",
        f"Wells with boundary-adjacent centers: {len(boundary)}",
        f"Plots packaged: {plot_count}",
        f"Composition status: {provenance.get('status')}",
        f"Composition authoritative: {provenance.get('authoritative')}",
        f"Composition safe for modeling: {provenance.get('safe_for_composition_modeling')}",
        "",
        "Successful-fit R2 statistics:",
        f"- mean: {successful['r2'].mean():.5f}" if len(successful) else "- mean: n/a",
        f"- median: {successful['r2'].median():.5f}" if len(successful) else "- median: n/a",
        f"- range: {successful['r2'].min():.5f} to {successful['r2'].max():.5f}"
        if len(successful)
        else "- range: n/a",
        "",
        "Peak-count distribution among successful fits:",
    ]
    lines.extend(f"- {count} peaks: {peak_counts[count]} wells" for count in sorted(peak_counts))
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "- Weighted residual variance is retained as an optimizer diagnostic only; it is not a calibrated reduced chi-square because measurement uncertainties were not supplied.",
            "- Boundary-adjacent centers remain in the review queue because their full line shape is not observed.",
            "- Composition-dependent interpretation and ML remain blocked until the real plate map replaces the provided template.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.composition and args.worklist:
        raise ValueError("Use either --composition or --worklist, not both")

    if args.worklist and not args.composition:
        composition_path = composition_from_worklist(
            str(args.worklist.resolve()),
            str(output_dir / "composition_from_mixing_worklist.csv"),
        )
    else:
        composition_path = ensure_composition_csv(
            str(data_path),
            str(args.composition.resolve()) if args.composition else None,
        )
    provenance = composition_provenance(composition_path)
    requested_wells = _parse_wells(args.wells)
    measured_wells = requested_wells or _detect_measured_wells(data_path, composition_path)
    if not measured_wells:
        raise RuntimeError("No wells with numeric spectral measurements were detected")

    agent = CurveFittingAgent()
    result = agent.run_curve_fitting(
        data_csv_path=str(data_path),
        composition_csv_path=composition_path,
        wells_to_analyze=measured_wells,
        reads_to_analyze="all",
        read_type="em_spectrum",
        max_peaks=args.max_peaks,
        r2_target=args.r2_target,
        max_attempts=args.max_attempts,
        save_plots=True,
        api_delay_seconds=0.0,
        auto_trigger=False,
        use_llm=False,
        output_dir=str(output_dir),
    )
    if not result.get("success"):
        raise RuntimeError(str(result.get("error") or "Curve fitting failed"))

    results = result.get("results") or []
    rows = _summary_rows(results)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Curve fitting produced no well rows")

    # Recompute review flags at the requested reporting threshold without
    # changing the fit-success definition.
    low_review_r2 = frame["r2"] < args.review_r2_threshold
    frame.loc[low_review_r2, "review_required"] = True
    frame.loc[low_review_r2, "review_reasons"] = frame.loc[
        low_review_r2, "review_reasons"
    ].map(
        lambda value: "; ".join(
            dict.fromkeys(
                [part for part in str(value).split("; ") if part]
                + ["r2_below_review_threshold"]
            )
        )
    )

    summary_path = output_dir / "combined_well_fit_summary.csv"
    review_path = output_dir / "qa_review_queue.csv"
    frame.to_csv(summary_path, index=False)
    frame[frame["review_required"] == True].to_csv(review_path, index=False)  # noqa: E712

    native_json = Path(result["files"]["json_results"]).resolve()
    native_csv = Path(result["files"]["csv_export"]).resolve()
    combined_json = output_dir / "combined_curve_fitting_results.json"
    combined_csv = output_dir / "combined_peak_fit_export.csv"
    if native_json != combined_json:
        shutil.copy2(native_json, combined_json)
    if native_csv != combined_csv:
        shutil.copy2(native_csv, combined_csv)

    source_copy = output_dir / f"source_{data_path.name.replace(' ', '_')}"
    if data_path != source_copy:
        shutil.copy2(data_path, source_copy)

    composition_copy = output_dir / "composition_map_used.csv"
    composition_frame = pd.read_csv(composition_path)
    identity_columns = [
        column for column in composition_frame.columns if column not in measured_wells
    ]
    if provenance.get("authoritative"):
        identity_columns = identity_columns[:1]
    else:
        identity_columns = [composition_frame.columns[0]]
    composition_columns = identity_columns + [
        well for well in measured_wells if well in composition_frame.columns
    ]
    composition_frame.loc[:, composition_columns].to_csv(composition_copy, index=False)
    packaged_metadata_path = output_dir / "composition_map_used.metadata.json"
    packaged_provenance = dict(provenance)
    packaged_provenance["well_count"] = len(measured_wells)
    packaged_provenance["well_columns"] = measured_wells
    packaged_provenance["path"] = str(composition_copy)
    packaged_provenance["metadata_path"] = str(packaged_metadata_path)
    packaged_metadata = {
        key: value
        for key, value in packaged_provenance.items()
        if key not in {"path", "metadata_path"}
    }
    packaged_metadata_path.write_text(
        json.dumps(packaged_metadata, indent=2),
        encoding="utf-8",
    )
    if not provenance.get("authoritative"):
        _write_composition_template(
            output_dir / "composition_map_template_C60_PEA_FA.csv",
            frame["well"].tolist(),
        )
        _write_mixing_worklist_template(
            output_dir / "mixing_worklist_template_C60_PEA_FA.csv",
            frame["well"].tolist(),
        )

    plots_dir = output_dir / "plots"
    analysis_tmp = plots_dir / "_analysis_tmp"
    if analysis_tmp.is_dir():
        try:
            analysis_tmp.rmdir()
        except OSError:
            pass
    plot_files = sorted(plots_dir.glob("fit_results_*.png"))
    archive_base = output_dir / "plots"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=plots_dir))

    _write_run_summary(
        output_dir / "RUN_SUMMARY.txt",
        data_path=data_path,
        frame=frame,
        plot_count=len(plot_files),
        provenance=packaged_provenance,
        r2_target=args.r2_target,
        review_r2_threshold=args.review_r2_threshold,
    )

    manifest_files = sorted(
        (
            file
            for file in output_dir.iterdir()
            if file.is_file() and file.name != "RUN_MANIFEST.json"
        ),
        key=lambda file: file.name,
    )
    manifest = {
        "dataset": str(data_path),
        "output_directory": str(output_dir),
        "analysis_mode": "deterministic",
        "fit_r2_target": args.r2_target,
        "review_r2_threshold": args.review_r2_threshold,
        "max_peaks": args.max_peaks,
        "max_attempts": args.max_attempts,
        "composition_provenance": packaged_provenance,
        "well_count": len(frame),
        "measured_wells": measured_wells,
        "successful_fits": int(frame["success"].sum()),
        "review_required": int(frame["review_required"].sum()),
        "accepted_without_spectral_review": int((frame["review_required"] != True).sum()),  # noqa: E712
        "boundary_adjacent_center_wells": int(
            (frame["boundary_adjacent_centers_nm"].fillna("") != "").sum()
        ),
        "out_of_range_center_wells": int(
            (frame["outside_range_centers_nm"].fillna("") != "").sum()
        ),
        "plot_count": len(plot_files),
        "files": {
            file.name: {"sha256": _sha256(file), "bytes": file.stat().st_size}
            for file in manifest_files
            if file.is_file()
        },
    }
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if len(frame) != len(measured_wells):
        raise RuntimeError(
            f"Expected {len(measured_wells)} measured wells, received {len(frame)}"
        )
    if manifest["out_of_range_center_wells"]:
        raise RuntimeError("Validation failed: fitted centers escaped the measured domain")
    if len(plot_files) != len(frame):
        raise RuntimeError(
            f"Validation failed: {len(plot_files)} plots for {len(frame)} wells"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--composition", type=Path)
    parser.add_argument("--worklist", type=Path)
    parser.add_argument(
        "--wells",
        help="Optional comma-separated wells; defaults to numeric-data auto-detection",
    )
    parser.add_argument("--max-peaks", type=int, default=4)
    parser.add_argument("--r2-target", type=float, default=0.92)
    parser.add_argument("--review-r2-threshold", type=float, default=0.95)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
