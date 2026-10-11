"""Pure Opentrons draft generation and deterministic preflight validation.

This module never connects to, uploads to, or executes on laboratory hardware.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


MAX_TUBE_SOURCES = 15
MIN_P300_TRANSFER_UL = 20.0
MAX_P300_TRANSFER_UL = 300.0
DESTINATION_WELL_CAPACITY_UL = 360.0
_CANONICAL_96_WELL = re.compile(r"^[A-H](?:[1-9]|1[0-2])$")
_ALLOWED_IMPORT_ROOTS = {"csv", "pathlib", "opentrons"}


@dataclass(frozen=True)
class ValidationReport:
    status: str
    syntax_valid: bool
    worklist_valid: bool | None
    simulation_available: bool
    simulation_status: str
    execution_supported: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _material_mapping_lines(materials: Iterable[str]) -> str:
    return "\n".join(f"        {material!r}: {index}," for index, material in enumerate(materials))


def _validate_material_names(materials: Iterable[str]) -> list[str]:
    normalized = [str(material).strip() for material in materials]
    if not normalized or any(not material for material in normalized):
        raise ValueError("At least one non-empty material column is required.")
    if len(normalized) > MAX_TUBE_SOURCES:
        raise ValueError(
            f"The configured tube rack supports at most {MAX_TUBE_SOURCES} material sources."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Material column names must be unique.")
    return normalized


def build_opentrons_protocol(
    csv_filename: str,
    materials: Iterable[str],
    csv_path: str | Path = "",
) -> str:
    """Build a portable, fail-loud OT-2 protocol draft.

    The result is source code only. It is not evidence that an Opentrons simulation
    or a physical run has succeeded.
    """

    material_names = _validate_material_names(materials)
    filename = Path(str(csv_filename)).name
    if not filename:
        raise ValueError("A worklist CSV filename is required.")
    base_path = str(csv_path or "").strip()
    mapping_lines = _material_mapping_lines(material_names)

    return f'''from opentrons import protocol_api
import csv
from pathlib import Path

metadata = {{
    "protocolName": "POLARIS liquid-handling draft",
    "author": "POLARIS",
    "description": "Generated draft; simulation and human approval are required before execution.",
    "apiLevel": "2.13",
}}

MIN_TRANSFER_UL = {MIN_P300_TRANSFER_UL!r}
MAX_TRANSFER_UL = {MAX_P300_TRANSFER_UL!r}
MAX_WELL_TOTAL_UL = {DESTINATION_WELL_CAPACITY_UL!r}


def run(protocol: protocol_api.ProtocolContext):
    tiprack = protocol.load_labware("opentrons_96_tiprack_300ul", "1")
    tuberack = protocol.load_labware("opentrons_15_tuberack_nest_15ml_conical", "2")
    destination_plate = protocol.load_labware("corning_96_wellplate_360ul_flat", "3")
    pipette = protocol.load_instrument("p300_single_gen2", "left", tip_racks=[tiprack])
    pipette.flow_rate.aspirate = 30
    pipette.flow_rate.dispense = 150

    material_to_tube = {{
{mapping_lines}
    }}
    base_path = Path({base_path!r}) if {bool(base_path)!r} else Path.cwd()
    worklist_path = base_path / {filename!r}

    with worklist_path.open(mode="r", newline="", encoding="utf-8-sig") as csvfile:
        for row_number, row in enumerate(csv.DictReader(csvfile), start=2):
            well_id = str(row.get("Well") or "").strip().upper()
            if well_id not in destination_plate.wells_by_name():
                raise ValueError(f"Invalid destination well {{well_id!r}} on worklist row {{row_number}}")
            destination = destination_plate.wells_by_name()[well_id]
            transfers = []
            for material, tube_index in material_to_tube.items():
                raw_volume = str(row.get(material) or "").strip()
                if not raw_volume:
                    continue
                volume = float(raw_volume)
                if volume == 0:
                    continue
                if not MIN_TRANSFER_UL <= volume <= MAX_TRANSFER_UL:
                    raise ValueError(
                        f"{{material}} volume {{volume}} uL on row {{row_number}} is outside "
                        f"the validated {{MIN_TRANSFER_UL}}-{{MAX_TRANSFER_UL}} uL range"
                    )
                transfers.append((material, tube_index, volume))
            if sum(volume for _, _, volume in transfers) > MAX_WELL_TOTAL_UL:
                raise ValueError(f"Total volume exceeds destination capacity on row {{row_number}}")
            for material, tube_index, volume in transfers:
                protocol.comment(f"Transfer {{volume}} uL of {{material}} to {{well_id}}")
                pipette.transfer(
                    volume,
                    tuberack.wells()[tube_index].bottom(z=1),
                    destination,
                    new_tip='always',
                    blow_out=True,
                    blowout_location='destination well',
                )
'''


def _protocol_structure_errors(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    imported_roots: set[str] = set()
    has_run = False
    has_api_level = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            has_run = len(node.args.args) == 1
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    value = None
                has_api_level = isinstance(value, dict) and bool(value.get("apiLevel"))

    disallowed = sorted(imported_roots - _ALLOWED_IMPORT_ROOTS)
    if disallowed:
        errors.append(f"Disallowed protocol import(s): {', '.join(disallowed)}.")
    if not has_run:
        errors.append("Protocol must define run(protocol) with exactly one argument.")
    if not has_api_level:
        errors.append("Protocol metadata must define exactly one apiLevel.")
    return errors


def _worklist_errors(worklist_csv: str) -> list[str]:
    errors: list[str] = []
    reader = csv.DictReader(io.StringIO(str(worklist_csv or "")))
    headers = list(reader.fieldnames or [])
    if "Well" not in headers:
        return ["Worklist must contain a Well column."]
    materials = [header for header in headers if header != "Well"]
    if not materials:
        return ["Worklist must contain at least one material-volume column."]
    if len(materials) > MAX_TUBE_SOURCES:
        errors.append(f"Worklist exceeds the {MAX_TUBE_SOURCES}-source tube-rack capacity.")

    seen_wells: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        well = str(row.get("Well") or "").strip().upper()
        if not _CANONICAL_96_WELL.fullmatch(well):
            errors.append(
                f"Row {row_number} must use a canonical well name such as A1, not {well!r}."
            )
        elif well in seen_wells:
            errors.append(f"Duplicate destination well {well!r} on row {row_number}.")
        seen_wells.add(well)

        total = 0.0
        for material in materials:
            raw = str(row.get(material) or "").strip()
            if not raw:
                continue
            try:
                volume = float(raw)
            except ValueError:
                errors.append(f"Row {row_number} volume for {material!r} must be numeric.")
                continue
            if volume == 0:
                continue
            if volume < MIN_P300_TRANSFER_UL:
                errors.append(
                    f"Row {row_number} volume for {material!r} is below the P300 minimum."
                )
            elif volume > MAX_P300_TRANSFER_UL:
                errors.append(
                    f"Row {row_number} volume for {material!r} exceeds the P300 maximum."
                )
            total += volume
        if total > DESTINATION_WELL_CAPACITY_UL:
            errors.append(f"Row {row_number} total volume exceeds the destination well capacity.")
    return errors


def validate_opentrons_draft(
    protocol_source: str,
    worklist_csv: str | None = None,
) -> ValidationReport:
    """Perform deterministic preflight checks without executing protocol code."""

    source = str(protocol_source or "")
    checksum = hashlib.sha256(source.encode("utf-8")).hexdigest()
    errors: list[str] = []
    warnings: list[str] = []
    syntax_valid = False

    try:
        tree = ast.parse(source, filename="opentrons_protocol.py", mode="exec")
        compile(tree, "opentrons_protocol.py", "exec")
        syntax_valid = True
        errors.extend(_protocol_structure_errors(tree))
    except SyntaxError as exc:
        errors.append(f"Python syntax error at line {exc.lineno}: {exc.msg}.")

    worklist_valid: bool | None = None
    if worklist_csv is not None:
        worklist_findings = _worklist_errors(worklist_csv)
        errors.extend(worklist_findings)
        worklist_valid = not worklist_findings
    else:
        warnings.append("No worklist was supplied; liquid-transfer constraints were not checked.")

    simulation_available = importlib.util.find_spec("opentrons") is not None
    simulation_status = "not_run" if simulation_available else "unavailable"
    if not simulation_available:
        warnings.append("The official Opentrons simulator is not installed in this runtime.")

    status = "syntax_validated_draft" if syntax_valid and not errors else "invalid_draft"
    return ValidationReport(
        status=status,
        syntax_valid=syntax_valid,
        worklist_valid=worklist_valid,
        simulation_available=simulation_available,
        simulation_status=simulation_status,
        execution_supported=False,
        errors=errors,
        warnings=warnings,
        sha256=checksum,
    )
