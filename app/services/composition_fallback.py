"""Build a composition CSV when uploads omit one (required by curve fitting)."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.tools.fitting_agent import infer_wells_from_file_metadata
from app.tools.paths import get_user_data_dir

_WELL_RE = re.compile(r"^[A-H](?:[1-9]|1[0-2])$", re.I)
_R_RE = re.compile(r"^R\d+$", re.I)
_PLACEHOLDER_ROW = "composition_unavailable"


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def composition_provenance(composition_csv_path: str) -> Dict[str, Any]:
    """Return explicit provenance for a supplied or generated composition CSV."""
    path = Path(composition_csv_path).expanduser().resolve()
    metadata_path = _metadata_path(path)
    if metadata_path.is_file():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {**data, "path": str(path), "metadata_path": str(metadata_path)}
        except (OSError, ValueError):
            pass
    return {
        "status": "supplied",
        "authoritative": True,
        "safe_for_composition_modeling": True,
        "path": str(path),
        "metadata_path": None,
    }


def is_placeholder_composition(composition_csv_path: Optional[str]) -> bool:
    if not composition_csv_path:
        return False
    return composition_provenance(composition_csv_path).get("status") == "placeholder"


def composition_from_worklist(
    worklist_csv_path: str,
    output_path: str,
) -> str:
    """Derive a material-by-well percentage map from an actual mixing worklist."""
    source = Path(worklist_csv_path).expanduser().resolve()
    frame = pd.read_csv(source)
    well_candidates = [column for column in frame.columns if str(column).strip().lower() == "well"]
    if not well_candidates:
        raise ValueError("Mixing worklist must contain a Well column")
    well_column = well_candidates[0]
    material_columns = [
        column
        for column in frame.columns
        if column != well_column and str(column).strip().lower().endswith("_ul")
    ]
    if not material_columns:
        raise ValueError("Mixing worklist must contain material volume columns ending in _uL")

    wells = frame[well_column].astype(str).str.strip().str.upper()
    if wells.duplicated().any():
        duplicates = sorted(wells[wells.duplicated(keep=False)].unique().tolist())
        raise ValueError(f"Mixing worklist contains duplicate destination wells: {duplicates}")
    if not wells.map(lambda value: bool(_WELL_RE.match(value) or _R_RE.match(value))).all():
        raise ValueError("Mixing worklist contains invalid destination-well identifiers")

    volumes = frame[material_columns].apply(pd.to_numeric, errors="coerce")
    if volumes.isna().any().any() or (volumes < 0).any().any():
        raise ValueError("Mixing worklist material volumes must be finite, non-negative numbers")
    totals = volumes.sum(axis=1)
    if (totals <= 0).any():
        bad = wells[totals <= 0].tolist()
        raise ValueError(f"Mixing worklist rows must have positive total volume: {bad}")

    percentages = volumes.div(totals, axis=0).mul(100.0)
    material_names = [re.sub(r"_uL$", "", str(column), flags=re.I) for column in material_columns]
    composition = percentages.T
    composition.index = material_names
    composition.columns = wells.tolist()
    composition.index.name = "material"

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    composition.to_csv(destination)
    metadata = {
        "status": "derived_from_worklist",
        "authoritative": True,
        "safe_for_composition_modeling": True,
        "derivation": "Per-well material volumes normalized to 100 percent.",
        "source_worklist": str(source),
        "source_worklist_columns": [str(column) for column in material_columns],
        "well_count": len(wells),
        "well_columns": wells.tolist(),
    }
    _metadata_path(destination).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return str(destination)


def _wells_from_data_csv_header(data_csv_path: str, limit: int = 96) -> List[str]:
    wells: List[str] = []
    seen = set()
    try:
        with open(data_csv_path, encoding="utf-8", errors="ignore") as f:
            for _ in range(80):
                line = f.readline()
                if not line:
                    break
                for part in re.split(r"[,;\t]", line):
                    token = part.strip().strip('"').upper()
                    if not token or token in ("WAVELENGTH", "WL", "PLACEHOLDER", "NAN"):
                        continue
                    if _WELL_RE.match(token) or _R_RE.match(token):
                        if token not in seen:
                            seen.add(token)
                            wells.append(token)
                            if len(wells) >= limit:
                                return wells
    except OSError:
        pass
    return wells


def ensure_composition_csv(data_csv_path: str, composition_csv_path: Optional[str] = None) -> str:
    """
    Return a valid composition CSV path.

    When no map is supplied, generate only a structural placeholder containing
    the detected well columns.  No chemical identities or gradients are
    invented, and companion metadata prevents the placeholder from being used
    as an authoritative composition matrix downstream.
    """
    if composition_csv_path and str(composition_csv_path).strip():
        p = Path(composition_csv_path).expanduser()
        if p.is_file():
            return str(p.resolve())

    wells = infer_wells_from_file_metadata(data_csv_path) or _wells_from_data_csv_header(data_csv_path)
    if not wells:
        wells = [f"A{i}" for i in range(1, 9)]

    out_dir = Path(get_user_data_dir()) / "uploads" / "curve_fitting" / "generated" / uuid.uuid4().hex[:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "composition_auto.csv"
    placeholder = pd.DataFrame(
        [[float("nan") for _ in wells]],
        index=[_PLACEHOLDER_ROW],
        columns=wells,
    )
    placeholder.index.name = ""
    placeholder.to_csv(out_path)
    metadata = {
        "status": "placeholder",
        "authoritative": False,
        "safe_for_composition_modeling": False,
        "reason": "No composition map was supplied; only well identities were preserved.",
        "source_data_file": str(Path(data_csv_path).expanduser().resolve()),
        "well_count": len(wells),
        "well_columns": wells,
    }
    _metadata_path(out_path).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return str(out_path.resolve())
