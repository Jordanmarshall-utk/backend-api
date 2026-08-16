import numpy as np
import pandas as pd

from app.services.phase_mapper.modeling import build_out_of_fold_tree_review_predictions, compare_model_metrics, train_multitask_models, train_shared_multitask_mlp


def _toy_training_table() -> pd.DataFrame:
    rows = []
    for i in range(14):
        x = i / 13
        rows.append(
            {
                "sample_key": f"lib::A{i + 1}",
                "well": f"A{i + 1}",
                "FA_fraction": x,
                "I_fraction": 1.0 - 0.2 * x,
                "optical_heterogeneity_score_total": abs(x - 0.5),
                "xrd_clean_3d_perovskite": 1 if x > 0.45 else 0,
                "xrd_secondary_phase_present": 1 if i % 3 == 0 else 0,
                "xrd_homogeneity_score": 0.2 + 0.6 * x,
            }
        )
    table = pd.DataFrame(rows)
    table.loc[[1, 5, 9], "xrd_secondary_phase_present"] = np.nan
    table.loc[[2, 7], "xrd_homogeneity_score"] = np.nan
    return table


def test_shared_multitask_mlp_outputs_partial_label_predictions():
    table = _toy_training_table()
    cfg = {
        "thresholds": {"min_xrd_labels_for_training": 5},
        "modeling": {
            "random_state": 3,
            "test_size": 0.3,
            "multitask_mlp": {
                "hidden_units": 8,
                "epochs": 80,
                "learning_rate": 0.03,
                "validation_fraction": 0.3,
                "min_labeled_rows": 5,
            },
        },
    }

    model, predictions, metrics = train_shared_multitask_mlp(table, cfg, predict_table=table)

    assert model["classification_targets"]
    assert "xrd_clean_3d_perovskite" in model["classification_targets"]
    assert "xrd_homogeneity_score" in model["regression_targets"]
    assert "mtl_pred_xrd_clean_3d_perovskite_prob" in predictions.columns
    assert "mtl_pred_xrd_homogeneity_score" in predictions.columns
    assert "multitask_mlp_uncertainty_score" in predictions.columns
    assert predictions["mtl_pred_xrd_clean_3d_perovskite_prob"].between(0, 1).all()
    assert set(metrics["model_family"].dropna()) == {"multitask_mlp"}
    assert (metrics["status"].astype(str).str.startswith("trained")).any()


def test_metric_comparison_keeps_tree_baseline_side_by_side():
    table = _toy_training_table()
    cfg = {
        "thresholds": {"min_xrd_labels_for_training": 5},
        "modeling": {
            "estimator": "extra_trees",
            "n_estimators": 5,
            "random_state": 3,
            "test_size": 0.3,
            "multitask_mlp": {"hidden_units": 8, "epochs": 40, "learning_rate": 0.03, "min_labeled_rows": 5},
        },
    }

    _, _, baseline_metrics, _ = train_multitask_models(table, cfg, predict_table=table)
    _, _, mtl_metrics = train_shared_multitask_mlp(table, cfg, predict_table=table)
    comparison = compare_model_metrics(baseline_metrics, mtl_metrics)

    assert not comparison.empty
    assert "winner_by_primary_metric" in comparison.columns
    assert "baseline_status" in comparison.columns
    assert "multitask_mlp_status" in comparison.columns


def test_out_of_fold_tree_review_predictions_use_held_out_review_labels():
    table = _toy_training_table()
    cfg = {
        "thresholds": {"min_xrd_labels_for_training": 5},
        "modeling": {
            "estimator": "extra_trees",
            "n_estimators": 5,
            "random_state": 7,
            "review_oof_splits": 3,
        },
    }

    oof = build_out_of_fold_tree_review_predictions(table, table, cfg)

    clean = oof[oof["task"] == "xrd_clean_3d_perovskite"]
    assert not clean.empty
    assert "baseline_probability" in clean.columns
    assert "baseline_correct" in clean.columns
    assert clean["baseline_probability"].between(0, 1).all()
    assert clean["evaluation_method"].eq("out_of_fold_current_review_label").all()
