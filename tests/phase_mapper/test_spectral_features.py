import numpy as np
import pandas as pd

from app.services.phase_mapper.spectral_features import add_cross_modal_features, extract_absorbance_features, extract_pl_features


def test_pl_feature_extraction_gaussian():
    x = np.linspace(500, 850, 400)
    y = 1000 * np.exp(-0.5 * ((x - 720) / 18) ** 2)
    feats = extract_pl_features(x, y, {"thresholds": {"pl_low_signal": 1, "pl_min_peak_prominence_fraction": 0.05}})
    assert abs(feats["pl_main_peak_nm"] - 720) < 2
    assert feats["pl_FWHM_nm"] > 30
    assert feats["pl_peak_count"] >= 1


def test_absorbance_edge_detection_synthetic_edge():
    x = np.linspace(400, 850, 500)
    y = 1 / (1 + np.exp((x - 620) / 8))
    feats = extract_absorbance_features(x, y, {"thresholds": {"absorbance_low_signal": 0.001}})
    assert abs(feats["abs_edge_nm"] - 620) < 5
    assert feats["abs_derivative_edge_count"] >= 1


def test_cross_modal_features_rank_multi_peak_optics_as_more_complex():
    features = pd.DataFrame(
        [
            {
                "TopPL_pl_peak_count": 1,
                "TopPL_pl_secondary_main_intensity_ratio": 0.0,
                "TopPL_pl_FWHM_nm": 30.0,
                "TopPL_pl_asymmetry_score": 0.05,
                "Abs_abs_peak_count": 1,
                "Abs_abs_derivative_edge_count": 1,
                "Abs_abs_edge_width_nm": 8.0,
                "Abs_abs_secondary_main_ratio": 0.0,
            },
            {
                "TopPL_pl_peak_count": 3,
                "TopPL_pl_secondary_main_intensity_ratio": 0.7,
                "TopPL_pl_FWHM_nm": 100.0,
                "TopPL_pl_asymmetry_score": 0.8,
                "Abs_abs_peak_count": 3,
                "Abs_abs_derivative_edge_count": 2,
                "Abs_abs_edge_width_nm": 45.0,
                "Abs_abs_secondary_main_ratio": 0.6,
            },
        ]
    )
    out = add_cross_modal_features(features)
    assert out.loc[1, "optical_phase_complexity_score"] > out.loc[0, "optical_phase_complexity_score"]
    assert out.loc[1, "optical_clean_3d_support_score"] < out.loc[0, "optical_clean_3d_support_score"]
