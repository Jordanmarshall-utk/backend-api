import pytest
pytestmark = pytest.mark.skip(reason="spin-coating optimizer is a later sub-project")
import numpy as np
import pandas as pd

from app.services.phase_mapper.spin_coating_optimization import (
    SpinCoatingSearchSpace,
    build_spin_finalist_table,
    derive_spin_quality_score,
    optimize_spin_coating,
    process_spin_plate_reader,
    sample_candidates,
    simulate_initial_spin_results,
)


def test_plate_reader_results_are_matched_to_saved_batch(tmp_path):
    wavelengths = list(range(700, 841, 2))
    rows = [["Read 1:EM Spectrum", None, None], ["Wavelength", "A1", "A2"]]
    for wavelength in wavelengths:
        a1 = 1000 * np.exp(-0.5 * ((wavelength - 770) / 12) ** 2)
        a2 = 800 * np.exp(-0.5 * ((wavelength - 790) / 15) ** 2)
        rows.append([wavelength, a1, a2])
    path = tmp_path / "plate.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False)
    batch = pd.DataFrame(
        {
            "experiment_id": ["SPIN-001", "SPIN-002"],
            "measurement_well": ["A1", "A2"],
            "spin_speed_rpm": [4000, 4500],
            "annealing_temperature_C": [100, 100],
            "antisolvent": ["chlorobenzene", "chlorobenzene"],
        }
    )

    measured, summary = process_spin_plate_reader(batch, str(path), {})

    assert not summary.empty
    assert measured["plate_reader_match_status"].eq("matched").all()
    assert measured["PL_Peak_nm"].between(765, 795).all()
    assert measured["PL_Intensity"].notna().all()
    assert measured["PL_FWHM_nm"].notna().all()


def test_spin_holder_assignments_use_five_by_seven_layout():
    batch, _, _ = optimize_spin_coating(_targets(), n_samples=200, batch_size=35, random_state=4)
    assert batch["measurement_well"].tolist() == [
        f"{row}{column}" for row in "ABCDE" for column in range(1, 8)
    ]


def _targets():
    return pd.DataFrame(
        {
            "library": ["demo"],
            "well": ["A1"],
            "sample_key": ["demo::A1"],
            "FA_fraction": [0.8],
            "Cs_fraction": [0.2],
            "I_fraction": [1.0],
            "pred_xrd_clean_3d_perovskite_prob": [0.9],
            "multitask_mlp_uncertainty_score": [0.35],
        }
    )


def test_sampling_is_reproducible_and_feasible():
    space = SpinCoatingSearchSpace(
        concentration_M=(0.3,),
        drop_volume_uL=(40,),
        first_spin_speed_rpm=(1000,),
        first_spin_time_s=(10,),
        spin_speed_rpm=(4000,),
        spin_time_s=(45,),
        spin_acceleration_rpm_s=(2000,),
        antisolvent=("None",),
        antisolvent_volume_uL=(100,),
        antisolvent_timing_before_end_s=(6,),
        anneal_temperature_C=(120,),
        anneal_time_s=(600,),
    )
    first = sample_candidates(_targets(), space, n_samples=10, random_state=4)
    second = sample_candidates(_targets(), space, n_samples=10, random_state=4)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) >= 1
    assert set(first["antisolvent_volume_uL"]) == {0}


def test_optimizer_returns_cold_start_batch():
    batch, ranked, diagnostics = optimize_spin_coating(
        _targets(), n_samples=100, batch_size=6, random_state=3
    )
    assert len(batch) == 6
    assert ranked["acquisition_score"].is_monotonic_decreasing
    assert diagnostics["model_status"] == "cold_start_physics_and_phase_prior"
    assert set(batch["optimizer_model_status"]) == {"cold_start_physics_and_phase_prior"}
    assert set(batch["phase_mapper_model_uncertainty"]) == {0.35}


def test_optimizer_trains_on_completed_spin_history():
    _, seed_candidates, _ = optimize_spin_coating(
        _targets(), n_samples=100, batch_size=12, random_state=8
    )
    history = seed_candidates.head(20).copy()
    history["spin_quality_score"] = (
        history["spin_speed_rpm"] / 6000
        + history["anneal_temperature_C"] / 200
    )
    batch, _, diagnostics = optimize_spin_coating(
        _targets(), history=history, n_samples=100, batch_size=5, random_state=9
    )
    assert len(batch) == 5
    assert diagnostics["history_rows_used"] == 20
    assert diagnostics["model_status"] == "trained_decision_tree_ensemble"


def test_multitask_uncertainty_changes_cross_composition_acquisition():
    targets = pd.concat(
        [
            _targets().assign(sample_key="demo::A1", well="A1", multitask_mlp_uncertainty_score=0.0),
            _targets().assign(sample_key="demo::A2", well="A2", multitask_mlp_uncertainty_score=1.0),
        ],
        ignore_index=True,
    )
    batch, ranked, diagnostics = optimize_spin_coating(
        targets,
        n_samples=500,
        batch_size=20,
        exploration_weight=0.0,
        phase_uncertainty_weight=1.0,
        random_state=12,
    )
    assert set(batch["sample_key"]) == {"demo::A1", "demo::A2"}
    means = ranked.groupby("sample_key")["acquisition_score"].mean()
    assert means["demo::A2"] > means["demo::A1"]
    assert diagnostics["phase_mapper_uncertainty_weight"] == 1.0


def test_batch_is_balanced_across_selected_compositions():
    targets = pd.concat(
        [
            _targets().assign(sample_key="demo::A1", well="A1"),
            _targets().assign(sample_key="demo::A2", well="A2", pred_xrd_clean_3d_perovskite_prob=0.7),
            _targets().assign(sample_key="demo::A3", well="A3", pred_xrd_clean_3d_perovskite_prob=0.5),
        ],
        ignore_index=True,
    )
    batch, _, _ = optimize_spin_coating(targets, n_samples=600, batch_size=12, random_state=8)
    counts = batch["sample_key"].value_counts()
    assert set(counts.index) == {"demo::A1", "demo::A2", "demo::A3"}
    assert counts.max() - counts.min() <= 1


def test_default_antisolvents_exclude_toluene():
    assert set(SpinCoatingSearchSpace().antisolvent) == {
        "Chlorobenzene",
        "Ethyl acetate",
        "None",
    }


def test_synthetic_bootstrap_does_not_produce_flat_first_real_predictions():
    targets = pd.concat(
        [
            _targets().assign(sample_key="demo::A1", well="A1"),
            _targets().assign(sample_key="demo::A2", well="A2", Cs_fraction=0.4, FA_fraction=0.6),
        ],
        ignore_index=True,
    )
    seed_batch, _, _ = optimize_spin_coating(targets, n_samples=500, batch_size=12, random_state=3)
    synthetic = simulate_initial_spin_results(seed_batch, random_state=3)
    batch, _, diagnostics = optimize_spin_coating(
        targets, history=synthetic, n_samples=500, batch_size=12, random_state=4
    )
    assert diagnostics["model_status"] == "synthetic_bootstrap_tree_with_process_prior"
    assert batch["predicted_spin_quality"].nunique() > 1


def test_finalist_table_separates_reviewed_exploitation_and_exploration():
    targets = pd.DataFrame(
        {
            "library": ["q", "q", "q", "q"],
            "well": ["A1", "A2", "A3", "A4"],
            "sample_key": ["q::A1", "q::A2", "q::A3", "q::A4"],
            "FA_fraction": [0.7, 0.6, 0.5, 0.4],
            "Cs_fraction": [0.3, 0.4, 0.5, 0.6],
            "I_fraction": [1.0, 1.0, 0.8, 1.0],
            "Br_fraction": [0.0, 0.0, 0.2, 0.0],
            "has_reviewed_xrd_label": [True, False, False, True],
            "xrd_clean_3d_perovskite": [1.0, None, None, 0.0],
            "xrd_mixed_phase": [0.0, None, None, 1.0],
            "xrd_secondary_phase_present": [0.0, None, None, 0.0],
            "pred_xrd_clean_3d_perovskite_prob": [0.95, 0.90, 0.85, 0.80],
            "multitask_mlp_uncertainty_score": [0.10, 0.15, 0.80, 0.20],
        }
    )
    finalists = build_spin_finalist_table(targets)
    tracks = finalists.set_index("sample_key")["finalist_track"].to_dict()
    assert tracks["q::A1"] == "reviewed_clean_3d"
    assert tracks["q::A2"] == "model_exploitation"
    assert tracks["q::A3"] == "model_exploration"
    assert tracks["q::A4"] == "hold_for_review"
    assert not finalists.set_index("sample_key").loc["q::A4", "recommended_for_spin"]


def test_quality_score_is_derived_from_at_least_two_measured_components():
    history = pd.DataFrame(
        {
            "PL_Peak_nm": [770.0],
            "PL_Intensity": [5000.0],
            "PL_FWHM_nm": [30.0],
            "Stability_Hours": [48.0],
        }
    )
    scored = derive_spin_quality_score(history)
    assert scored.loc[0, "spin_quality_score"] > 0
    assert scored.loc[0, "spin_quality_components_used"] == 4


def test_simulated_bootstrap_is_labeled_and_trainable():
    batch, _, _ = optimize_spin_coating(_targets(), n_samples=200, batch_size=8, random_state=2)
    simulated = simulate_initial_spin_results(batch, random_state=2)
    assert len(simulated) == 8
    assert set(simulated["result_source"]) == {"synthetic_bootstrap_not_measured"}
    assert simulated["spin_quality_score"].notna().all()
    next_batch, _, diagnostics = optimize_spin_coating(
        _targets(), history=simulated, n_samples=200, batch_size=4, random_state=3
    )
    assert len(next_batch) == 4
    assert diagnostics["model_status"] == "synthetic_bootstrap_tree_with_process_prior"
