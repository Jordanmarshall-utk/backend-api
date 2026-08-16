import pandas as pd

from app.services.phase_mapper.modeling import compare_feature_family_models, feature_family_columns


def _table() -> pd.DataFrame:
    rows = []
    for idx in range(20):
        positive = idx >= 10
        rows.append(
            {
                "well": f"A{idx + 1}",
                "FA_fraction": idx / 19,
                "additive_fraction": (idx % 4) / 10,
                "TopPL_initial_pl_peak_count": 3 if positive else 1,
                "Abs_initial_abs_derivative_edge_count": 2 if positive else 1,
                "optical_phase_complexity_score": 0.9 if positive else 0.1,
                "xrd_clean_3d_perovskite": 0 if positive else 1,
                "xrd_mixed_phase": 1 if positive else 0,
            }
        )
    return pd.DataFrame(rows)


def test_feature_family_columns_separate_composition_and_optical_inputs():
    families = feature_family_columns(_table())
    assert "FA_fraction" in families["composition_only"]
    assert "TopPL_initial_pl_peak_count" in families["optical_only"]
    assert "optical_phase_complexity_score" in families["optical_only"]
    assert set(families["combined"]) == set(families["composition_only"] + families["optical_only"])


def test_feature_family_comparison_reports_all_three_models():
    cfg = {
        "thresholds": {"min_xrd_labels_for_training": 5},
        "modeling": {"estimator": "extra_trees", "test_size": 0.3, "random_state": 42, "n_estimators": 20},
    }
    comparison = compare_feature_family_models(_table(), cfg)
    assert set(comparison["feature_family"]) == {"composition_only", "optical_only", "combined"}
    assert set(comparison["task"]) == {"xrd_clean_3d_perovskite", "xrd_mixed_phase"}
