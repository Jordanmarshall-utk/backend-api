"""Plate-reader parsing for CSV/Excel exports with one or more read blocks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import WELLS_96, normalize_well

READ_RE = re.compile(r"read\s*(\d+)\s*:?\s*(.*)", re.IGNORECASE)


def infer_read_name(read_num: int, read_label: str, assignments: dict | None = None) -> str:
    """Infer or assign a human-readable read name."""
    assignments = assignments or {}
    for key in (read_num, str(read_num)):
        if key in assignments:
            return str(assignments[key])
    label = (read_label or "").strip()
    if label:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")
        return f"{cleaned}_{read_num}" if cleaned else f"Read_{read_num}"
    return f"Read_{read_num}"


def infer_read_type(read_name: str, cfg: dict[str, Any]) -> str:
    """Return pl, absorbance, or unknown from configured keywords."""
    name = (read_name or "").lower()
    if name.startswith("abs_") or name.startswith("absorbance"):
        return "absorbance"
    if name.startswith("toppl_") or name.startswith("bottompl_"):
        return "pl"
    if "em_spectrum" in name or "em spectrum" in name:
        return "pl"
    if name.startswith("spectrum_") or name == "spectrum":
        return "absorbance"
    for kw in cfg.get("read_type_keywords", {}).get("absorbance", []):
        if kw.lower() in name:
            return "absorbance"
    for kw in cfg.get("read_type_keywords", {}).get("pl", []):
        if kw.lower() in name:
            return "pl"
    return "unknown"


def infer_sequence_read_names(blocks: list[tuple[int, str, int, int]]) -> dict[int, str]:
    """Infer absorbance and alternating top/bottom PL names from Gen5 read order."""
    names = {}
    time_index = 0
    emission_in_pair = 0
    for read_num, read_label, _, _ in blocks:
        label = (read_label or "").strip().lower()
        if "em spectrum" in label or label.startswith("em"):
            geometry = "TopPL" if emission_in_pair % 2 == 0 else "BottomPL"
            names[read_num] = f"{geometry}_t{time_index}"
            emission_in_pair += 1
            if emission_in_pair % 2 == 0:
                time_index += 1
        else:
            names[read_num] = f"Abs_t{time_index}"
            emission_in_pair = 0
    return names


def _read_raw_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        sheets = pd.read_excel(path, header=None, engine="openpyxl", sheet_name=None)
        if not sheets:
            return pd.DataFrame()
        # Some Gen5 exports keep an empty/aborted first sheet and the useful
        # read blocks on a later sheet. Use the populated sheet with most reads.
        return max(sheets.values(), key=lambda frame: len(_detect_blocks(frame)))
    return pd.read_csv(path, header=None)


def _is_wavelength_header(value: object) -> bool:
    text = str(value).strip().lower()
    return any(token in text for token in ["wavelength", "lambda", "nm"])


def _detect_blocks(raw: pd.DataFrame) -> list[tuple[int, str, int, int]]:
    """Find blocks as (read_num, label, header_row, end_row)."""
    read_markers: list[tuple[int, str, int]] = []
    for idx, row in raw.iterrows():
        joined = " ".join(str(x) for x in row.dropna().tolist())
        match = READ_RE.search(joined)
        if match:
            read_markers.append((int(match.group(1)), match.group(2).strip(), idx))

    blocks: list[tuple[int, str, int, int]] = []
    if read_markers:
        for i, (num, label, marker_row) in enumerate(read_markers):
            next_marker = read_markers[i + 1][2] if i + 1 < len(read_markers) else len(raw)
            header_row = None
            for r in range(marker_row + 1, next_marker):
                vals = raw.iloc[r].tolist()
                well_hits = sum(normalize_well(v) in WELLS_96 for v in vals)
                if any(_is_wavelength_header(v) for v in vals) or well_hits >= 2:
                    header_row = r
                    break
            if header_row is not None:
                blocks.append((num, label, header_row, next_marker))
        return blocks

    # Fallback: a single block with a wavelength column and well columns.
    for r in range(min(20, len(raw))):
        vals = raw.iloc[r].tolist()
        well_hits = sum(normalize_well(v) in WELLS_96 for v in vals)
        if any(_is_wavelength_header(v) for v in vals) or well_hits >= 2:
            return [(1, "", r, len(raw))]
    return []


def parse_plate_reader_file(path: str | Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse plate-reader CSV/XLSX into long-format spectra and a read summary."""
    path = Path(path)
    raw = _read_raw_table(path)
    blocks = _detect_blocks(raw)
    assignments = cfg.get("read_assignments", {})
    inferred_names = infer_sequence_read_names(blocks)
    long_rows: list[pd.DataFrame] = []
    summary: list[dict[str, Any]] = []

    for read_num, read_label, header_row, end_row in blocks:
        header = [str(x).strip() if not pd.isna(x) else "" for x in raw.iloc[header_row].tolist()]
        data = raw.iloc[header_row + 1 : end_row].copy()
        data.columns = header
        wavelength_col = next((c for c in data.columns if _is_wavelength_header(c)), data.columns[0])
        data[wavelength_col] = pd.to_numeric(data[wavelength_col], errors="coerce")
        data = data[data[wavelength_col].notna()]
        well_cols = [c for c in data.columns if normalize_well(c) in WELLS_96]
        read_name = infer_read_name(read_num, read_label, assignments) if read_num in assignments or str(read_num) in assignments else inferred_names.get(read_num, infer_read_name(read_num, read_label))
        read_type = infer_read_type(read_name, cfg)
        if well_cols:
            melted = data[[wavelength_col] + well_cols].melt(
                id_vars=wavelength_col, var_name="well", value_name="signal"
            )
            melted = melted.rename(columns={wavelength_col: "wavelength_nm"})
            melted["well"] = melted["well"].map(normalize_well)
            melted["signal"] = pd.to_numeric(melted["signal"], errors="coerce")
            melted["read_num"] = read_num
            melted["read_label"] = read_label
            melted["read_name"] = read_name
            melted["read_type"] = read_type
            long_rows.append(melted[["read_num", "read_label", "read_name", "read_type", "well", "wavelength_nm", "signal"]])
        summary.append(
            {
                "read_num": read_num,
                "read_label": read_label,
                "read_name": read_name,
                "read_type": read_type,
                "n_wells_detected": len(well_cols),
                "n_wavelength_points": int(data[wavelength_col].notna().sum()),
                "wavelength_min_nm": float(data[wavelength_col].min()) if len(data) else np.nan,
                "wavelength_max_nm": float(data[wavelength_col].max()) if len(data) else np.nan,
            }
        )

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=["read_num", "read_label", "read_name", "read_type", "well", "wavelength_nm", "signal"]
    )
    return long_df, pd.DataFrame(summary)
