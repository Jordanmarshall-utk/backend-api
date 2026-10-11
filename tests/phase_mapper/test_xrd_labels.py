from pathlib import Path

import pandas as pd

from app.services.phase_mapper.xrd_labels import load_xrd_labels


def test_visual_xrd_fields_populate_model_targets(tmp_path: Path):
    labels_path = tmp_path / "visual_labels.csv"
    pd.DataFrame(
        [
            {
                "well": "A1",
                "xrd_visual_phase_call": "secondary_phase",
                "xrd_primary_peak_visual_shape": "split/doublet",
                "xrd_extra_or_impurity_peak_2theta_list": "12.7; 25.4",
            },
            {
                "well": "A2",
                "xrd_visual_phase_call": "clean_3d",
                "xrd_primary_peak_visual_shape": "sharp",
            },
        ]
    ).to_csv(labels_path, index=False)

    labels = load_xrd_labels(labels_path).set_index("well")

    assert labels.loc["A1", "xrd_secondary_phase_present"] == 1
    assert labels.loc["A1", "xrd_primary_peak_splitting_present"] == 1
    assert labels.loc["A1", "xrd_peak_splitting_present"] == 1
    assert labels.loc["A2", "xrd_clean_3d_perovskite"] == 1
    assert labels.loc["A2", "xrd_secondary_phase_present"] == 0


def test_manual_review_source_is_highest_weight(tmp_path: Path):
    labels_path = tmp_path / "manual_labels.csv"
    pd.DataFrame(
        [
            {
                "well": "A1",
                "xrd_clean_3d_perovskite": 1,
                "xrd_assignment_confidence": "low",
                "xrd_label_source": "manual_review_upload",
            }
        ]
    ).to_csv(labels_path, index=False)

    labels = load_xrd_labels(labels_path).set_index("well")

    assert labels.loc["A1", "xrd_assignment_confidence"] == "high"
    assert labels.loc["A1", "xrd_sample_weight"] == 1.0
