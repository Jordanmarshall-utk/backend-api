import pandas as pd

from app.services.phase_mapper.active_learning import compute_xrd_priority, select_diverse_recommendations


def test_active_learning_score_ranking():
    df = pd.DataFrame(
        {
            "well": ["A1", "A2", "A3"],
            "row": ["A"] * 3,
            "col": [1, 2, 3],
            "Cs_fraction": [0.0, 0.5, 1.0],
            "Rb_fraction": [0.0, 0.0, 0.0],
            "FA_fraction": [1.0, 0.5, 0.0],
            "Br_fraction": [1.0, 0.5, 0.0],
            "Cl_fraction": [0.0, 0.5, 1.0],
            "model_uncertainty_score": [0.1, 1.0, 0.2],
            "optical_heterogeneity_score_total": [0.1, 1.0, 0.1],
            "pl_abs_disagreement_score": [0.1, 1.0, 0.1],
            "optical_phase_complexity_score": [0.1, 1.0, 0.1],
            "optical_neighbor_outlier_score": [0.0, 1.0, 0.0],
        }
    )
    cfg = {"active_learning": {"recommend_n": 2, "diversity_lambda": 0, "weights": {"model_uncertainty": 0.3, "optical_heterogeneity": 0.15, "pl_abs_disagreement": 0.15, "optical_phase_complexity": 0.2, "composition_distance": 0.1, "optical_neighbor_outlier": 0.1}}}
    priority = compute_xrd_priority(df, {"A1"}, cfg)
    recs = select_diverse_recommendations(priority, {"A1"}, cfg)
    assert recs.iloc[0]["well"] == "A2"
