import numpy as np

from app.services.phase_mapper.experiment_profiles import build_profile_composition_table, identify_experiment_profile, profile_options


def test_full_additive_profile_generates_96_model_ready_wells():
    table = build_profile_composition_table("3ampi_macl_8row", plate_id="3AMPI_MACl")

    assert len(table) == 96
    assert table["well"].nunique() == 96
    assert table["two_d_precursor"].eq("3AMPIPbI4").all()
    assert np.isclose(table.loc[table["well"] == "A1", "two_d_fraction"].iloc[0], 1.0)
    assert np.isclose(table.loc[table["well"] == "A12", "fapbi3_fraction"].iloc[0], 1.0)
    assert table.loc[table["well"] == "H6", "additive_percent"].iloc[0] == 20
    assert np.isclose(table.loc[table["well"] == "H6", "additive_final_molarity"].iloc[0], 0.06)
    assert np.isclose(table.loc[table["well"] == "H6", "additive_stock_molarity"].iloc[0], 0.60)


def test_six_row_bda_bai_profile_only_generates_active_wells():
    table = build_profile_composition_table("bda_bai_6row")

    assert len(table) == 72
    assert set(table["row"]) == set("ABCDEF")
    assert table.loc[table["well"] == "F1", "additive_percent"].iloc[0] == 20
    assert "g1" not in {well.lower() for well in table["well"]}


def test_all_requested_additive_profiles_are_available():
    options = profile_options()
    assert {"bda_bai_6row", "bda_macl_8row", "bda_oai_8row", "3ampi_bai_8row", "3ampi_macl_8row"} <= set(options)


def test_notebook_backed_workbook_names_identify_correct_profiles():
    expected = {
        "RP Additive 95-120 degrees time evolution.xlsx": "bda_bai_6row",
        "050226_RP additive MACl 95 degrees.xlsx": "bda_macl_8row",
        "050226_RP additive MACl 95 degrees 2.xlsx": "bda_macl_8row",
        "0501626_RP additive OAI 95 degrees.xlsx": "bda_oai_8row",
        "05222026_RP additive 3AMP-FA with BAI 95 degrees.xlsx": "3ampi_bai_8row",
        "06122026 3AMP-FA RP additive MACl.xlsx": "3ampi_macl_8row",
    }
    for filename, profile_id in expected.items():
        identified = identify_experiment_profile([filename])
        assert identified is not None
        assert identified["profile_id"] == profile_id
        assert identified["confidence"] == "high"
