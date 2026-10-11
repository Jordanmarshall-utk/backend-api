import numpy as np
import pandas as pd

from app.services.phase_mapper.composition_features import apply_library_composition_substitutions, compute_composition_descriptors, entropy


def test_entropy_calculation():
    assert entropy([1, 0, 0]) == 0.0
    assert np.isclose(entropy([0.5, 0.5]), np.log(2))


def test_composition_descriptor_calculation():
    df = pd.DataFrame(
        {
            "well": ["A1"],
            "row": ["A"],
            "col": [1],
            "Cs_fraction": [0.2],
            "Rb_fraction": [0.1],
            "FA_fraction": [0.7],
            "Br_fraction": [0.8],
            "Cl_fraction": [0.2],
        }
    )
    out = compute_composition_descriptors(df)
    assert np.isclose(out.loc[0, "A_site_sum"], 1.0)
    assert np.isclose(out.loc[0, "X_site_sum"], 1.0)
    assert np.isclose(out.loc[0, "small_cation_fraction"], 0.3)
    assert out.loc[0, "A_site_mean_radius"] > 0


def test_rb_library_replaces_dma_fraction_before_descriptor_calculation():
    comp = pd.DataFrame(
        {
            "well": ["A1", "A2"],
            "Cs_fraction": [0.1, 0.2],
            "Rb_fraction": [0.0, 0.05],
            "FA_fraction": [0.7, 0.6],
            "DMA_fraction": [0.2, 0.15],
            "I_fraction": [1.0, 1.0],
            "Br_fraction": [0.0, 0.0],
            "Cl_fraction": [0.0, 0.0],
            "notes": ["DMAPbI3=20uL", "DMAPbI3=15uL"],
        }
    )

    mapped = apply_library_composition_substitutions(
        comp,
        {"experiment": {"library_name": "Rb", "library_slug": "Rb"}},
    )
    out = compute_composition_descriptors(mapped)

    assert np.allclose(out["DMA_fraction"], 0.0)
    assert np.allclose(out["Rb_fraction"], [0.2, 0.2])
    assert np.allclose(out["Rb_to_FA_ratio"], [0.2 / 0.7, 0.2 / 0.6])
    assert out["composition_substitutions_applied"].eq("DMA_fraction->Rb_fraction").all()
    assert out["notes"].str.contains("RbPbI3", regex=False).all()


def test_dma_library_keeps_dma_fraction_unchanged():
    comp = pd.DataFrame({"well": ["A1"], "Rb_fraction": [0.0], "DMA_fraction": [0.2]})
    mapped = apply_library_composition_substitutions(comp, {"experiment": {"library_name": "DMA"}})

    assert mapped.loc[0, "Rb_fraction"] == 0.0
    assert mapped.loc[0, "DMA_fraction"] == 0.2
