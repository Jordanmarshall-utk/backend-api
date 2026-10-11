"""Monte Carlo/decision-tree optimization for spin-coated perovskite films.

The phase mapper chooses promising precursor compositions.  This module holds
composition fixed and searches processing conditions, then learns a process
response surface as completed spin-coating experiments are appended.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .plate_reader import parse_plate_reader_file
from .spectral_features import extract_pl_features


COMPOSITION_COLUMNS = (
    "Cs_fraction",
    "Rb_fraction",
    "FA_fraction",
    "DMA_fraction",
    "I_fraction",
    "Br_fraction",
    "Cl_fraction",
    "shared_layered_precursor_fraction",
    "shared_3d_precursor_fraction",
    "shared_additive_fraction",
)

NUMERIC_PROCESS_COLUMNS = (
    "concentration_M",
    "drop_volume_uL",
    "dispense_height_mm",
    "first_spin_speed_rpm",
    "first_spin_time_s",
    "spin_speed_rpm",
    "spin_time_s",
    "spin_acceleration_rpm_s",
    "antisolvent_volume_uL",
    "antisolvent_timing_before_end_s",
    "anneal_temperature_C",
    "anneal_time_s",
)

MODEL_COMPOSITION_COLUMNS = (
    "Cs_fraction",
    "Rb_fraction",
    "FA_fraction",
    "DMA_fraction",
    "I_fraction",
    "Br_fraction",
    "Cl_fraction",
)

MODEL_NUMERIC_COLUMNS = (
    *MODEL_COMPOSITION_COLUMNS,
    "anneal_temperature_C",
    "spin_speed_rpm",
)

HISTORY_ALIASES = {
    "Concentration_M": "concentration_M",
    "Drop_Volume_uL": "drop_volume_uL",
    "Dispense_Height_mm": "dispense_height_mm",
    "First_Spin_Speed_rpm": "first_spin_speed_rpm",
    "First_Spin_Time_s": "first_spin_time_s",
    "Spin_Speed_rpm": "spin_speed_rpm",
    "Spin_Time_s": "spin_time_s",
    "Spin_Acceleration_rpm_s": "spin_acceleration_rpm_s",
    "Antisolvent_Name": "antisolvent",
    "Antisolvent_Volume_uL": "antisolvent_volume_uL",
    "Antisolvent_Timing_Before_End_s": "antisolvent_timing_before_end_s",
    "Anneal_Temp_C": "anneal_temperature_C",
    "Anneal_Time_s": "anneal_time_s",
}

MEASUREMENT_WELLS = tuple(f"{row}{column}" for row in "ABCDE" for column in range(1, 8))


@dataclass(frozen=True)
class SpinCoatingSearchSpace:
    """Discrete, lab-editable processing search space."""

    concentration_M: Sequence[float] = (0.3,)
    drop_volume_uL: Sequence[int] = (40, 50, 60, 70, 80)
    dispense_height_mm: Sequence[float] = (10, 15, 20)
    first_spin_speed_rpm: Sequence[int] = (1000, 1500, 2000)
    first_spin_time_s: Sequence[int] = (5, 10, 15, 20)
    spin_speed_rpm: Sequence[int] = (3000, 4000, 4500, 5000, 5500, 6000)
    spin_time_s: Sequence[int] = (30, 45, 60, 90)
    spin_acceleration_rpm_s: Sequence[int] = (1000, 1500, 2000, 2500, 3000)
    antisolvent: Sequence[str] = ("Chlorobenzene", "Ethyl acetate", "None")
    antisolvent_volume_uL: Sequence[int] = (50, 75, 100, 125, 150)
    antisolvent_timing_before_end_s: Sequence[int] = (2, 4, 6, 8, 10, 12)
    anneal_temperature_C: Sequence[int] = (100, 120, 140, 145)
    anneal_time_s: Sequence[int] = (360, 480, 600, 720, 900, 1200)
    max_concentration_volume_product: float = 50.0
    min_main_spin_speed_rpm: int = 3000
    min_anneal_temperature_C: float = 95.0
    min_anneal_time_s: float = 300.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "SpinCoatingSearchSpace":
        if not values:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in known})


def _choice(rng: np.random.Generator, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("Every spin-coating search-space field must contain at least one value.")
    return values[int(rng.integers(0, len(values)))]


def _normalize_targets(targets: pd.DataFrame) -> pd.DataFrame:
    if targets is None or targets.empty:
        raise ValueError("At least one precursor composition target is required.")
    result = targets.copy()
    if "well" not in result:
        result["well"] = [f"target_{i + 1}" for i in range(len(result))]
    if "library" not in result:
        result["library"] = "spin_optimization"
    if "sample_key" not in result:
        result["sample_key"] = result["library"].astype(str) + "::" + result["well"].astype(str)
    for column in COMPOSITION_COLUMNS:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def build_spin_finalist_table(targets: pd.DataFrame, recommend_n: int = 8) -> pd.DataFrame:
    """Build reviewed, exploitation, exploration, and diversity finalist tracks."""
    result = _normalize_targets(targets)

    def numeric(column: str, default: float = 0.0) -> pd.Series:
        values = result.get(column, pd.Series(default, index=result.index))
        return pd.to_numeric(values, errors="coerce")

    predicted_clean = numeric("pred_xrd_clean_3d_perovskite_prob", 0.5).fillna(0.5).clip(0, 1)
    reviewed_clean_values = numeric("xrd_clean_3d_perovskite", np.nan)
    reviewed_clean = reviewed_clean_values.eq(1)
    reviewed = numeric("has_reviewed_xrd_label").eq(1) | reviewed_clean_values.notna()
    mixed = numeric("xrd_mixed_phase").fillna(0).eq(1)
    secondary = numeric("xrd_secondary_phase_present").fillna(0).eq(1)
    uncertainty = numeric("multitask_mlp_uncertainty_score")
    uncertainty = uncertainty.fillna(numeric("model_uncertainty_score")).fillna(0).clip(0, 1)
    homogeneity = numeric("xrd_homogeneity_score").fillna(0.5).clip(0, 1)
    optical_support = numeric("optical_clean_3d_support_score").fillna(0.5).clip(0, 1)
    clean_evidence = predicted_clean.where(~reviewed, reviewed_clean.astype(float))
    invalid_review = reviewed & (~reviewed_clean | mixed | secondary)

    result["phase_clean_evidence"] = clean_evidence
    result["phase_model_uncertainty"] = uncertainty
    result["reviewed_clean_3d_finalist"] = reviewed & reviewed_clean & ~mixed & ~secondary
    result["spin_finalist_score"] = (
        0.55 * clean_evidence
        + 0.15 * homogeneity
        + 0.10 * optical_support
        + 0.20 * (1.0 - uncertainty)
        - 0.50 * invalid_review.astype(float)
    ).clip(0, 1)

    result["finalist_track"] = np.select(
        [
            result["reviewed_clean_3d_finalist"],
            (~reviewed) & predicted_clean.ge(0.60) & uncertainty.ge(0.50),
            (~reviewed) & predicted_clean.ge(0.60),
        ],
        ["reviewed_clean_3d", "model_exploration", "model_exploitation"],
        default="hold_for_review",
    )

    comp = result.loc[:, COMPOSITION_COLUMNS].to_numpy(float)
    scale = np.ptp(comp, axis=0)
    scale[scale == 0] = 1.0
    normalized = comp / scale
    top_index = int(result["spin_finalist_score"].to_numpy().argmax())
    result["composition_distance_from_top"] = np.sqrt(
        ((normalized - normalized[top_index]) ** 2).sum(axis=1)
    )

    recommendation_reasons: dict[int, list[str]] = {int(i): [] for i in result.index}
    eligible = result[~invalid_review].copy()
    tracks = [
        ("reviewed_clean_3d", 3),
        ("model_exploitation", 2),
        ("model_exploration", 2),
    ]
    for track, count in tracks:
        chosen = eligible[eligible["finalist_track"] == track].nlargest(count, "spin_finalist_score")
        for index in chosen.index:
            recommendation_reasons[int(index)].append(track)
    diverse = eligible.nlargest(max(recommend_n * 3, recommend_n), "spin_finalist_score").nlargest(
        2, "composition_distance_from_top"
    )
    for index in diverse.index:
        recommendation_reasons[int(index)].append("composition_diversity")

    result["selection_reason"] = [
        "; ".join(recommendation_reasons[int(index)]) for index in result.index
    ]
    result["recommended_for_spin"] = result["selection_reason"].ne("")
    result = result.sort_values(
        ["recommended_for_spin", "reviewed_clean_3d_finalist", "spin_finalist_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result["spin_finalist_rank"] = np.arange(1, len(result) + 1)
    return result


def derive_spin_quality_score(history: pd.DataFrame) -> pd.DataFrame:
    """Fill quality using the original PL/FWHM/stability/peak objective."""
    result = history.copy()
    pl = pd.to_numeric(result.get("PL_Intensity", pd.Series(np.nan, index=result.index)), errors="coerce")
    fwhm = pd.to_numeric(result.get("PL_FWHM_nm", pd.Series(np.nan, index=result.index)), errors="coerce")
    stability = pd.to_numeric(result.get("Stability_Hours", pd.Series(np.nan, index=result.index)), errors="coerce")
    peak = pd.to_numeric(result.get("PL_Peak_nm", pd.Series(np.nan, index=result.index)), errors="coerce")
    derived = pl / 50.0 + 50.0 * (30.0 / fwhm.where(fwhm > 0))
    derived += 2.0 * stability.fillna(0.0)
    peak_deviation = (peak - 770.0).abs()
    peak_term = pd.Series(0.0, index=result.index)
    peak_term.loc[peak.notna() & peak_deviation.le(30.0)] = 7.5
    outside = peak.notna() & peak_deviation.gt(30.0)
    peak_term.loc[outside] = -15.0 * (peak_deviation.loc[outside] - 30.0) / 10.0
    derived += peak_term
    existing = pd.to_numeric(
        result.get("spin_quality_score", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    result["spin_quality_score"] = existing.fillna(derived)
    result["spin_quality_components_used"] = pd.concat(
        [pl.notna(), fwhm.notna(), stability.notna(), peak.notna()], axis=1
    ).sum(axis=1)
    return result


def simulate_initial_spin_results(
    batch: pd.DataFrame,
    random_state: int = 0,
) -> pd.DataFrame:
    """Create the original-style synthetic bootstrap iteration.

    These rows initialize the tree before real spin-coated-film measurements
    exist. They are always marked synthetic and must not be reported as data.
    """
    if batch is None or batch.empty:
        return pd.DataFrame()
    result = batch.copy()
    rng = np.random.default_rng(random_state)
    latent = np.clip(_cold_start_score(result), 0.0, 1.0)
    result["PL_Intensity"] = np.clip(
        1000.0 + 8000.0 * latent + rng.normal(0, 450, len(result)), 100, None
    )
    result["PL_FWHM_nm"] = np.clip(
        105.0 - 70.0 * latent + rng.normal(0, 5, len(result)), 20, 130
    )
    result["PL_Peak_nm"] = 770.0 + rng.normal(
        0, 8.0 + 12.0 * result["phase_mapper_model_uncertainty"].to_numpy(float), len(result)
    )
    result["Stability_Hours"] = np.clip(
        12.0 + 96.0 * latent + rng.normal(0, 8, len(result)), 1, None
    )
    result["result_source"] = "synthetic_bootstrap_not_measured"
    result = derive_spin_quality_score(result)
    return result


def process_spin_plate_reader(
    batch: pd.DataFrame,
    plate_reader_path: str,
    parser_config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach automatically extracted initial PL outcomes to a saved batch."""
    if batch is None or batch.empty:
        raise ValueError("A generated spin batch is required before processing optical data.")
    if "measurement_well" not in batch:
        raise ValueError("The generated batch is missing measurement_well assignments.")
    cfg = dict(parser_config or {})
    long_df, summary = parse_plate_reader_file(plate_reader_path, cfg)
    pl = long_df[long_df["read_type"].eq("pl")].copy()
    if pl.empty:
        raise ValueError(
            "No PL emission read was detected. Check the reader labels or assign the PL read in the parser configuration."
        )

    read_order = (
        pl[["read_num", "read_name"]]
        .drop_duplicates()
        .sort_values("read_num")
    )
    initial_names = read_order[
        read_order["read_name"].astype(str).str.contains("_t0", case=False, regex=False)
    ]["read_name"].tolist()
    if not initial_names:
        initial_names = [str(read_order.iloc[0]["read_name"])]
    initial = pl[pl["read_name"].astype(str).isin(initial_names)]

    rows = []
    for well, well_data in initial.groupby("well"):
        features = []
        for _, spectrum in well_data.groupby("read_name"):
            extracted = extract_pl_features(spectrum["wavelength_nm"], spectrum["signal"], cfg)
            extracted["PL_Intensity"] = float(
                pd.to_numeric(spectrum["signal"], errors="coerce").max()
            )
            features.append(extracted)
        feature_frame = pd.DataFrame(features)
        rows.append(
            {
                "measurement_well": well,
                "PL_Peak_nm": pd.to_numeric(feature_frame.get("pl_main_peak_nm"), errors="coerce").mean(),
                "PL_Intensity": pd.to_numeric(feature_frame.get("PL_Intensity"), errors="coerce").mean(),
                "PL_FWHM_nm": pd.to_numeric(feature_frame.get("pl_FWHM_nm"), errors="coerce").mean(),
                "optical_reads_averaged": len(feature_frame),
                "result_source": "measured_plate_reader",
            }
        )
    measured = pd.DataFrame(rows)
    result = batch.drop(
        columns=[
            "PL_Peak_nm",
            "PL_Intensity",
            "PL_FWHM_nm",
            "spin_quality_score",
            "spin_quality_components_used",
            "result_source",
        ],
        errors="ignore",
    ).merge(measured, on="measurement_well", how="left")
    result = derive_spin_quality_score(result)
    result["plate_reader_match_status"] = np.where(
        result["PL_Intensity"].notna(), "matched", "missing_optical_well"
    )
    return result, summary


def sample_candidates(
    targets: pd.DataFrame,
    search_space: SpinCoatingSearchSpace,
    n_samples: int = 5000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Monte Carlo sample recipes and apply the process decision tree."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    targets = _normalize_targets(targets)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}

    for _ in range(n_samples):
        target = targets.iloc[int(rng.integers(0, len(targets)))]
        antisolvent = str(_choice(rng, search_space.antisolvent))
        row = {
            "library": target["library"],
            "well": target["well"],
            "sample_key": target["sample_key"],
            **{column: float(target[column]) for column in COMPOSITION_COLUMNS},
            "phase_mapper_clean_3d_probability": float(
                target.get("pred_xrd_clean_3d_perovskite_prob", target.get("xrd_clean_3d_perovskite", 0.5))
            ),
            "phase_mapper_model_uncertainty": float(
                target.get(
                    "multitask_mlp_uncertainty_score",
                    target.get("model_uncertainty_score", 0.0),
                )
            ),
            "concentration_M": float(_choice(rng, search_space.concentration_M)),
            "drop_volume_uL": int(_choice(rng, search_space.drop_volume_uL)),
            "dispense_height_mm": float(_choice(rng, search_space.dispense_height_mm)),
            "first_spin_speed_rpm": int(_choice(rng, search_space.first_spin_speed_rpm)),
            "first_spin_time_s": int(_choice(rng, search_space.first_spin_time_s)),
            "spin_speed_rpm": int(_choice(rng, search_space.spin_speed_rpm)),
            "spin_time_s": int(_choice(rng, search_space.spin_time_s)),
            "spin_acceleration_rpm_s": int(_choice(rng, search_space.spin_acceleration_rpm_s)),
            "antisolvent": antisolvent,
            "antisolvent_volume_uL": 0 if antisolvent.lower() == "none" else int(_choice(rng, search_space.antisolvent_volume_uL)),
            "antisolvent_timing_before_end_s": 0 if antisolvent.lower() == "none" else int(_choice(rng, search_space.antisolvent_timing_before_end_s)),
            "anneal_temperature_C": int(_choice(rng, search_space.anneal_temperature_C)),
            "anneal_time_s": int(_choice(rng, search_space.anneal_time_s)),
        }

        reason = None
        if row["concentration_M"] * row["drop_volume_uL"] >= search_space.max_concentration_volume_product:
            reason = "concentration_volume_limit"
        elif row["spin_speed_rpm"] < search_space.min_main_spin_speed_rpm:
            reason = "main_spin_speed_too_low"
        elif row["anneal_temperature_C"] < search_space.min_anneal_temperature_C:
            reason = "anneal_temperature_too_low"
        elif row["anneal_time_s"] <= search_space.min_anneal_time_s:
            reason = "anneal_time_too_short"
        elif row["antisolvent_timing_before_end_s"] >= row["spin_time_s"] and antisolvent.lower() != "none":
            reason = "antisolvent_timing_outside_main_spin"

        if reason:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"No feasible recipes were sampled. Rejections: {rejection_counts}")
    result = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    result.attrs["rejection_counts"] = rejection_counts
    return result


def _prepare_history(history: pd.DataFrame | None, outcome_column: str) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    result = derive_spin_quality_score(history.rename(columns=HISTORY_ALIASES).copy())
    required = set(MODEL_COMPOSITION_COLUMNS) | {
        "spin_speed_rpm",
        "anneal_temperature_C",
        "antisolvent",
        outcome_column,
    }
    if not required.issubset(result.columns):
        return pd.DataFrame()
    for column in COMPOSITION_COLUMNS:
        if column not in result:
            result[column] = 0.0
    for column in (*COMPOSITION_COLUMNS, *NUMERIC_PROCESS_COLUMNS, outcome_column):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=[*MODEL_NUMERIC_COLUMNS, "antisolvent", outcome_column])
    if "spin_quality_components_used" in result:
        manual = pd.to_numeric(
            history.get(outcome_column, pd.Series(np.nan, index=history.index)),
            errors="coerce",
        ).notna()
        result = result[manual.reindex(result.index, fill_value=False) | result["spin_quality_components_used"].ge(2)]
    return result


def _feature_frame(frame: pd.DataFrame, antisolvent_levels: Sequence[str]) -> pd.DataFrame:
    features = frame.loc[:, MODEL_NUMERIC_COLUMNS].astype(float).copy()
    values = frame["antisolvent"].fillna("None").astype(str).str.lower()
    for level in antisolvent_levels:
        features[f"antisolvent__{level}"] = (values == level).astype(float)
    return features


def _cold_start_score(candidates: pd.DataFrame) -> np.ndarray:
    """Transparent initial heuristic until measured spin-coated films exist."""
    speed = candidates["spin_speed_rpm"].to_numpy(float)
    concentration = candidates["concentration_M"].to_numpy(float)
    volume = candidates["drop_volume_uL"].to_numpy(float)
    anneal = candidates["anneal_temperature_C"].to_numpy(float)
    anti = (candidates["antisolvent"].str.lower() != "none").to_numpy(float)
    phase_prior = candidates["phase_mapper_clean_3d_probability"].to_numpy(float)
    thickness_proxy = concentration * volume
    process_centering = (
        np.exp(-((speed - 4500.0) / 1700.0) ** 2)
        + np.exp(-((anneal - 120.0) / 35.0) ** 2)
        + np.exp(-((thickness_proxy - 18.0) / 12.0) ** 2)
    ) / 3.0
    return 0.65 * phase_prior + 0.25 * process_centering + 0.10 * anti


def rank_candidates(
    candidates: pd.DataFrame,
    history: pd.DataFrame | None = None,
    outcome_column: str = "spin_quality_score",
    exploration_weight: float = 0.25,
    phase_uncertainty_weight: float = 0.15,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank using predicted quality plus process- and phase-model uncertainty.

    ``prediction_uncertainty`` is learned only from completed spin-coated
    experiments. ``phase_mapper_model_uncertainty`` is inherited from the
    finalized library composition and remains a distinct exploration signal.
    """
    if not 0.0 <= exploration_weight <= 1.0:
        raise ValueError("exploration_weight must be between 0 and 1.")
    if not 0.0 <= phase_uncertainty_weight <= 1.0:
        raise ValueError("phase_uncertainty_weight must be between 0 and 1.")
    candidates = candidates.copy()
    clean_history = _prepare_history(history, outcome_column)
    diagnostics: dict[str, Any] = {
        "history_rows_used": int(len(clean_history)),
        "outcome_column": outcome_column,
    }

    if len(clean_history) >= 8 and clean_history[outcome_column].nunique() >= 2:
        levels = sorted(
            set(clean_history["antisolvent"].fillna("None").astype(str).str.lower())
            | set(candidates["antisolvent"].fillna("None").astype(str).str.lower())
        )
        x_train = _feature_frame(clean_history, levels)
        x_candidates = _feature_frame(candidates, levels)
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x_train, clean_history[outcome_column].to_numpy(float))
        candidate_matrix = x_candidates.to_numpy(dtype=float)
        tree_predictions = np.vstack([tree.predict(candidate_matrix) for tree in model.estimators_])
        predicted = tree_predictions.mean(axis=0)
        uncertainty = tree_predictions.std(axis=0)
        sources = clean_history.get("result_source", pd.Series("", index=clean_history.index)).astype(str)
        synthetic_only = sources.eq("synthetic_bootstrap_not_measured").all()
        if synthetic_only:
            # A tiny synthetic dataset often puts many top candidates in the
            # same terminal leaf. Retain a modest cold-start process prior so
            # the first real batch is not presented with a misleadingly flat
            # predicted score. Real measurements remove this regularization.
            prior = _cold_start_score(candidates)
            outcome_min = float(clean_history[outcome_column].min())
            outcome_range = float(clean_history[outcome_column].max() - outcome_min)
            prior_range = float(np.ptp(prior))
            scaled_prior = outcome_min + outcome_range * (
                (prior - float(np.min(prior))) / (prior_range if prior_range > 0 else 1.0)
            )
            predicted = 0.85 * predicted + 0.15 * scaled_prior
            diagnostics["model_status"] = "synthetic_bootstrap_tree_with_process_prior"
        else:
            diagnostics["model_status"] = "trained_decision_tree_ensemble"
    else:
        predicted = _cold_start_score(candidates)
        uncertainty = np.full(len(candidates), 0.5, dtype=float)
        diagnostics["model_status"] = "cold_start_physics_and_phase_prior"

    pred_range = np.ptp(predicted)
    unc_range = np.ptp(uncertainty)
    pred_norm = (predicted - np.min(predicted)) / (pred_range if pred_range > 0 else 1.0)
    unc_norm = (uncertainty - np.min(uncertainty)) / (unc_range if unc_range > 0 else 1.0)
    phase_uncertainty = (
        pd.to_numeric(candidates["phase_mapper_model_uncertainty"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .to_numpy(float)
    )
    process_acquisition = (1.0 - exploration_weight) * pred_norm + exploration_weight * unc_norm
    candidates["predicted_spin_quality"] = predicted
    candidates["prediction_uncertainty"] = uncertainty
    candidates["process_acquisition_score"] = process_acquisition
    candidates["acquisition_score"] = (
        (1.0 - phase_uncertainty_weight) * process_acquisition
        + phase_uncertainty_weight * phase_uncertainty
    )
    candidates["optimizer_model_status"] = diagnostics["model_status"]
    diagnostics["process_exploration_weight"] = float(exploration_weight)
    diagnostics["phase_mapper_uncertainty_weight"] = float(phase_uncertainty_weight)
    diagnostics["phase_uncertainty_source"] = (
        "multitask_mlp_uncertainty_score; fallback=model_uncertainty_score"
    )
    diagnostics["predicted_quality_range"] = [float(np.min(predicted)), float(np.max(predicted))]
    return candidates.sort_values("acquisition_score", ascending=False).reset_index(drop=True), diagnostics


def select_diverse_batch(ranked: pd.DataFrame, batch_size: int = 12) -> pd.DataFrame:
    """Select a process-diverse batch balanced across chosen compositions."""
    if ranked.empty or batch_size < 1:
        return ranked.head(0).copy()
    if batch_size > len(MEASUREMENT_WELLS):
        raise ValueError(
            f"The spin-film holder has {len(MEASUREMENT_WELLS)} positions (A1-E7); "
            "reduce the requested batch size."
        )
    selected: list[int] = []
    seen_recipes: set[tuple[Any, ...]] = set()
    recipe_columns = (
        "spin_speed_rpm",
        "spin_time_s",
        "antisolvent",
        "antisolvent_timing_before_end_s",
        "anneal_temperature_C",
        "anneal_time_s",
    )
    # Ranked candidates can otherwise be monopolized by the composition with
    # the strongest phase prior. Round-robin selection gives every composition
    # an experimental share while preserving acquisition order within it.
    sample_order = ranked["sample_key"].drop_duplicates().tolist()
    group_indices = {
        sample_key: ranked.index[ranked["sample_key"].eq(sample_key)].tolist()
        for sample_key in sample_order
    }
    positions = {sample_key: 0 for sample_key in sample_order}
    while len(selected) < batch_size:
        added_this_round = False
        for sample_key in sample_order:
            indices = group_indices[sample_key]
            while positions[sample_key] < len(indices):
                index = indices[positions[sample_key]]
                positions[sample_key] += 1
                row = ranked.loc[index]
                recipe = tuple(row[column] for column in recipe_columns)
                if recipe in seen_recipes:
                    continue
                selected.append(index)
                seen_recipes.add(recipe)
                added_this_round = True
                break
            if len(selected) >= batch_size:
                break
        if not added_this_round:
            break
    result = ranked.loc[selected].copy().reset_index(drop=True)
    result.insert(0, "experiment_id", [f"SPIN-{i + 1:03d}" for i in range(len(result))])
    result.insert(1, "measurement_well", list(MEASUREMENT_WELLS[: len(result)]))
    return result


def optimize_spin_coating(
    targets: pd.DataFrame,
    history: pd.DataFrame | None = None,
    search_space: SpinCoatingSearchSpace | Mapping[str, Any] | None = None,
    n_samples: int = 5000,
    batch_size: int = 12,
    outcome_column: str = "spin_quality_score",
    exploration_weight: float = 0.25,
    phase_uncertainty_weight: float = 0.15,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run sampling, decision-tree scoring, and diverse batch selection."""
    space = search_space if isinstance(search_space, SpinCoatingSearchSpace) else SpinCoatingSearchSpace.from_mapping(search_space)
    candidates = sample_candidates(targets, space, n_samples=n_samples, random_state=random_state)
    ranked, diagnostics = rank_candidates(
        candidates,
        history=history,
        outcome_column=outcome_column,
        exploration_weight=exploration_weight,
        phase_uncertainty_weight=phase_uncertainty_weight,
        random_state=random_state,
    )
    batch = select_diverse_batch(ranked, batch_size=batch_size)
    diagnostics.update(
        {
            "monte_carlo_attempts": int(n_samples),
            "feasible_unique_candidates": int(len(candidates)),
            "selected_batch_size": int(len(batch)),
            "rejection_counts": candidates.attrs.get("rejection_counts", {}),
        }
    )
    return batch, ranked, diagnostics
