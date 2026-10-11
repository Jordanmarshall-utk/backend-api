from __future__ import annotations

import pytest

from app.tools.opentrons_draft import (
    build_opentrons_protocol,
    validate_opentrons_draft,
)


VALID_WORKLIST = """Well,Cs_uL,BDA_uL
A1,20,30
A2,40,0
"""


def test_generated_opentrons_protocol_is_portable_and_compiles():
    source = build_opentrons_protocol(
        csv_filename="worklist.csv",
        materials=["Cs_uL", "BDA_uL"],
    )

    compile(source, "opentrons_protocol.py", "exec")
    assert "C:\\Users" not in source
    assert "/var/lib/jupyter" not in source
    assert "top(z=-117)" not in source
    assert "new_tip='always'" in source
    assert "bottom(z=1)" in source


def test_validation_reports_syntax_and_worklist_safety():
    source = build_opentrons_protocol("worklist.csv", ["Cs_uL", "BDA_uL"])
    report = validate_opentrons_draft(source, VALID_WORKLIST)

    assert report.syntax_valid is True
    assert report.worklist_valid is True
    assert report.simulation_status in {"not_run", "unavailable"}
    assert report.execution_supported is False
    assert report.errors == []
    assert len(report.sha256) == 64


@pytest.mark.parametrize(
    "worklist, expected",
    [
        ("Well,Cs_uL\nA01,20\n", "canonical well name"),
        ("Well,Cs_uL\nA1,5\n", "minimum"),
        ("Well,Cs_uL\nA1,301\n", "maximum"),
        ("Well,Cs_uL,BDA_uL\nA1,200,200\n", "well capacity"),
        ("Well,Cs_uL\nA1,not-a-number\n", "numeric"),
    ],
)
def test_validation_rejects_unsafe_worklists(worklist: str, expected: str):
    source = build_opentrons_protocol("worklist.csv", ["Cs_uL", "BDA_uL"])
    report = validate_opentrons_draft(source, worklist)

    assert report.worklist_valid is False
    assert any(expected in message.lower() for message in report.errors)
    assert report.execution_supported is False


def test_generator_rejects_more_sources_than_the_configured_tube_rack():
    with pytest.raises(ValueError, match="15"):
        build_opentrons_protocol(
            "worklist.csv",
            [f"material_{index}_uL" for index in range(16)],
        )


def test_generated_protocol_rejects_dangerous_imports():
    report = validate_opentrons_draft(
        "import subprocess\nmetadata = {'apiLevel': '2.13'}\ndef run(protocol):\n    pass\n"
    )

    assert report.syntax_valid is True
    assert any("import" in message.lower() for message in report.errors)
    assert report.execution_supported is False
