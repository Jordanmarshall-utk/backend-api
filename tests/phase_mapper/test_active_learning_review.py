import pandas as pd

from app.services.phase_mapper.active_learning_review import active_learning_cycle_summary, build_active_learning_cycle_comparison


def test_empty_cycle_comparison_has_headers_for_csv_roundtrip(tmp_path):
    cycle = build_active_learning_cycle_comparison(pd.DataFrame(), pd.DataFrame())
    assert cycle.empty
    assert "task" in cycle.columns
    path = tmp_path / "active_learning_cycle_comparison.csv"
    cycle.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert loaded.empty
    assert "actual_xrd_label" in loaded.columns


def test_active_learning_cycle_comparison_preserves_before_and_after_predictions():
    previous = pd.DataFrame(
        [
            {
                "sample_key": "Lib::A1",
                "well": "A1",
                "recommended_in_run": "run_1",
                "recommended_at": "2026-06-01T12:00:00",
                "xrd_priority_score": 0.91,
                "pred_xrd_clean_3d_perovskite": 0,
                "pred_xrd_clean_3d_perovskite_prob": 0.35,
                "mtl_pred_xrd_clean_3d_perovskite": 1,
                "mtl_pred_xrd_clean_3d_perovskite_prob": 0.62,
            }
        ]
    )
    current = pd.DataFrame(
        [
            {
                "sample_key": "Lib::A1",
                "library": "Lib",
                "well": "A1",
                "row": "A",
                "col": 1,
                "xrd_priority_score": 0.12,
                "xrd_clean_3d_perovskite": 1,
                "pred_xrd_clean_3d_perovskite": 1,
                "pred_xrd_clean_3d_perovskite_prob": 1.0,
                "mtl_pred_xrd_clean_3d_perovskite": 1,
                "mtl_pred_xrd_clean_3d_perovskite_prob": 0.84,
            }
        ]
    )

    cycle = build_active_learning_cycle_comparison(previous, current)

    assert len(cycle) == 1
    row = cycle.iloc[0]
    assert row["recommended_in_run"] == "run_1"
    assert row["actual_xrd_label"] == 1
    assert row["before_baseline_prediction"] == 0
    assert row["before_baseline_probability"] == 0.35
    assert row["after_baseline_prediction"] == 1
    assert row["after_baseline_probability"] == 1.0
    assert row["before_baseline_correct"] == False
    assert row["after_baseline_correct"] == True
    assert row["before_multitask_mlp_correct"] == True

    summary = active_learning_cycle_summary(cycle)
    assert summary.iloc[0]["before_baseline_accuracy"] == 0.0
    assert summary.iloc[0]["after_baseline_accuracy"] == 1.0
    assert summary.iloc[0]["before_baseline_predictions_available"] == 1
    assert summary.iloc[0]["before_baseline_prediction_coverage"] == 1.0


def test_active_learning_cycle_counts_repeated_recommendation_once_using_latest_prediction():
    previous = pd.DataFrame(
        [
            {
                "sample_key": "Lib::A1",
                "recommended_in_run": "run_1",
                "recommended_at": "2026-06-01T12:00:00",
                "pred_xrd_clean_3d_perovskite": 0,
            },
            {
                "sample_key": "Lib::A1",
                "recommended_in_run": "run_2",
                "recommended_at": "2026-06-02T12:00:00",
                "pred_xrd_clean_3d_perovskite": 1,
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "sample_key": "Lib::A1",
                "library": "Lib",
                "well": "A1",
                "xrd_clean_3d_perovskite": 1,
                "pred_xrd_clean_3d_perovskite": 1,
            }
        ]
    )
    cycle = build_active_learning_cycle_comparison(previous, current)
    assert len(cycle) == 1
    assert cycle.iloc[0]["recommended_in_run"] == "run_2"
    assert cycle.iloc[0]["before_baseline_correct"] == True
