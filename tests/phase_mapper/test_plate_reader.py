from pathlib import Path

import pandas as pd

from app.services.phase_mapper.plate_reader import infer_read_name, infer_read_type, infer_sequence_read_names, parse_plate_reader_file


def test_parsing_of_read_assignments():
    cfg = {"read_assignments": {1: "Abs_initial"}, "read_type_keywords": {"absorbance": ["abs"], "pl": ["pl"]}}
    assert infer_read_name(1, "ignored", cfg["read_assignments"]) == "Abs_initial"
    assert infer_read_type("Abs_initial", cfg) == "absorbance"
    assert infer_read_type("TopPL_initial", cfg) == "pl"


def test_unassigned_gen5_reads_keep_unique_names_and_types():
    cfg = {"read_type_keywords": {"absorbance": ["abs"], "pl": ["pl"]}}
    assert infer_read_name(4, "EM Spectrum") == "EM_Spectrum_4"
    assert infer_read_name(5, "EM Spectrum") == "EM_Spectrum_5"
    assert infer_read_type("EM_Spectrum_4", cfg) == "pl"
    assert infer_read_type("Spectrum_1", cfg) == "absorbance"


def test_gen5_sequence_infers_absorbance_and_top_bottom_pl_timeline():
    blocks = [
        (1, "Spectrum", 0, 1),
        (2, "EM Spectrum", 1, 2),
        (3, "EM Spectrum", 2, 3),
        (4, "Spectrum", 3, 4),
        (5, "EM Spectrum", 4, 5),
        (6, "EM Spectrum", 5, 6),
    ]
    assert infer_sequence_read_names(blocks) == {
        1: "Abs_t0",
        2: "TopPL_t0",
        3: "BottomPL_t0",
        4: "Abs_t1",
        5: "TopPL_t1",
        6: "BottomPL_t1",
    }


def test_multisheet_workbook_uses_sheet_with_read_blocks(tmp_path: Path):
    path = tmp_path / "multi_sheet.xlsx"
    useful = pd.DataFrame(
        [
            ["Read 1:Spectrum", None, None],
            ["Wavelength", "A1", "A2"],
            [500, 1.0, 2.0],
            [501, 1.1, 2.1],
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([["metadata only"]]).to_excel(writer, sheet_name="Empty", index=False, header=False)
        useful.to_excel(writer, sheet_name="Useful", index=False, header=False)

    long_df, summary = parse_plate_reader_file(path, {"read_assignments": {}, "read_type_keywords": {}})
    assert len(summary) == 1
    assert summary.iloc[0]["read_type"] == "absorbance"
    assert set(long_df["well"]) == {"A1", "A2"}
