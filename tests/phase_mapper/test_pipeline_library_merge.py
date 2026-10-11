from pathlib import Path

import pandas as pd

from app.services.phase_mapper.pipeline import run_pipeline


def _base_cfg(output_dir: Path, labels_path: Path, library: str) -> dict:
    return {
        "paths": {
            "plate_reader_file": "",
            "composition_map": "",
            "xrd_labels": str(labels_path),
            "output_dir": str(output_dir),
        },
        "experiment": {"library_name": library},
        "plate": {"expected_wells": 96},
        "thresholds": {"min_xrd_labels_for_training": 5, "neighbor_count": 5, "n_pca_components": 2},
        "modeling": {"estimator": "extra_trees", "n_estimators": 5, "random_state": 42, "test_size": 0.3},
        "active_learning": {
            "recommend_n": 4,
            "diversity_lambda": 0.15,
            "weights": {
                "model_uncertainty": 0.30,
                "optical_heterogeneity": 0.15,
                "pl_abs_disagreement": 0.15,
                "optical_phase_complexity": 0.20,
                "composition_distance": 0.10,
                "optical_neighbor_outlier": 0.10,
            },
        },
    }


def test_xrd_labels_merge_by_library_and_well(tmp_path: Path):
    labels_path = tmp_path / "combined_labels.csv"
    pd.DataFrame(
        [
            {"library": "DMA", "well": "A1", "xrd_visual_phase_call": "clean_3d"},
            {"library": "Rb", "well": "A1", "xrd_visual_phase_call": "secondary_phase"},
        ]
    ).to_csv(labels_path, index=False)

    run_pipeline(_base_cfg(tmp_path / "dma_out", labels_path, "DMA"))
    run_pipeline(_base_cfg(tmp_path / "rb_out", labels_path, "Rb"))

    dma = pd.read_csv(tmp_path / "dma_out" / "multitask_training_table.csv")
    rb = pd.read_csv(tmp_path / "rb_out" / "multitask_training_table.csv")

    dma_a1 = dma.loc[dma["well"] == "A1"].iloc[0]
    rb_a1 = rb.loc[rb["well"] == "A1"].iloc[0]
    assert dma_a1["sample_key"] == "DMA::A1"
    assert rb_a1["sample_key"] == "Rb::A1"
    assert dma_a1["xrd_clean_3d_perovskite"] == 1
    assert dma_a1["xrd_secondary_phase_present"] == 0
    assert rb_a1["xrd_clean_3d_perovskite"] == 0
    assert rb_a1["xrd_secondary_phase_present"] == 1


def test_combined_xrd_labels_refresh_prior_library_training_rows(tmp_path: Path):
    labels_path = tmp_path / "combined_labels.csv"
    pd.DataFrame(
        [
            {"library": "DMA", "well": "A1", "xrd_visual_phase_call": "clean_3d"},
            {"library": "Rb", "well": "A1", "xrd_visual_phase_call": "secondary_phase"},
        ]
    ).to_csv(labels_path, index=False)
    prior_path = tmp_path / "rb_prior_training.csv"
    pd.DataFrame(
        [
            {"library": "Rb", "well": "A1", "sample_key": "Rb::A1", "row": "A", "col": 1, "Cs_fraction": 0.2, "Rb_fraction": 0.4, "FA_fraction": 0.4, "DMA_fraction": 0.0, "I_fraction": 0.8, "Br_fraction": 0.2},
        ]
    ).to_csv(prior_path, index=False)

    cfg = _base_cfg(tmp_path / "dma_with_prior_out", labels_path, "DMA")
    cfg["paths"]["training_tables"] = [str(prior_path)]
    cfg["thresholds"]["min_xrd_labels_for_training"] = 2
    cfg["modeling"]["compare_multitask_mlp"] = True
    cfg["modeling"]["multitask_mlp"] = {"hidden_units": 6, "epochs": 20, "learning_rate": 0.03, "min_labeled_rows": 2}
    run_pipeline(cfg)

    metrics = pd.read_csv(tmp_path / "dma_with_prior_out" / "model_metrics.csv")
    clean_metric = metrics.loc[metrics["task"] == "xrd_clean_3d_perovskite"].iloc[0]
    assert clean_metric["n_labels"] == 2
    assert clean_metric["status"] == "trained_no_holdout"

    review = pd.read_csv(tmp_path / "dma_with_prior_out" / "xrd_prediction_vs_actual_review.csv")
    assert "actual_xrd_label" in review.columns
    assert "baseline_prediction" in review.columns
    assert "multitask_mlp_prediction" in review.columns

    all_predictions = pd.read_csv(tmp_path / "dma_with_prior_out" / "model_predictions_all_wells.csv")
    candidate_predictions = pd.read_csv(tmp_path / "dma_with_prior_out" / "model_predictions_unlabeled_candidate_wells.csv")
    assert "has_reviewed_xrd_label" in all_predictions.columns
    assert "prediction_role" in all_predictions.columns
    assert not candidate_predictions["has_reviewed_xrd_label"].any()
    assert candidate_predictions["prediction_role"].eq("unlabeled_prediction_candidate").all()

    comparison = pd.read_csv(tmp_path / "dma_with_prior_out" / "model_comparison_metrics.csv")
    assert "baseline_status" in comparison.columns
    assert "multitask_mlp_status" in comparison.columns

    scope = pd.read_csv(tmp_path / "dma_with_prior_out" / "pooled_training_scope_by_library.csv")
    assert set(scope["library"]) == {"DMA", "Rb"}
    assert scope["training_rule"].eq("latest unique library::well sample").all()
    assert scope.set_index("library").loc["Rb", "xrd_labeled_samples"] == 1

    duplicate_audit = pd.read_csv(tmp_path / "dma_with_prior_out" / "pooled_training_duplicate_audit.csv")
    assert list(duplicate_audit.columns) == [
        "sample_key",
        "library",
        "well",
        "profile_id",
        "_training_source",
        "duplicate_resolution",
    ]


def test_current_library_labels_do_not_erase_prior_library_labels(tmp_path: Path):
    labels_path = tmp_path / "current_library_labels.csv"
    pd.DataFrame(
        [{"library": "Rb", "well": "A1", "xrd_visual_phase_call": "secondary_phase"}]
    ).to_csv(labels_path, index=False)
    prior_path = tmp_path / "dma_prior_training.csv"
    pd.DataFrame(
        [
            {
                "library": "DMA",
                "well": "A1",
                "sample_key": "DMA::A1",
                "row": "A",
                "col": 1,
                "FA_fraction": 0.8,
                "DMA_fraction": 0.2,
                "xrd_clean_3d_perovskite": 1,
                "xrd_secondary_phase_present": 0,
            }
        ]
    ).to_csv(prior_path, index=False)

    cfg = _base_cfg(tmp_path / "rb_with_dma_prior", labels_path, "Rb")
    cfg["paths"]["training_tables"] = [str(prior_path)]
    cfg["thresholds"]["min_xrd_labels_for_training"] = 2
    run_pipeline(cfg)

    coverage = pd.read_csv(tmp_path / "rb_with_dma_prior" / "xrd_label_coverage_by_library.csv")
    assert coverage.set_index("library").loc["DMA", "xrd_labeled_rows"] == 1
