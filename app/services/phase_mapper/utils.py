"""Utility helpers for plate maps, warnings, and numeric scaling."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))
WELLS_96 = [f"{r}{c}" for r in ROWS for c in COLS]


def normalize_well(value: object) -> str:
    """Return a normalized 96-well id such as A1."""
    if pd.isna(value):
        return ""
    text = str(value).strip().upper().replace(" ", "")
    if len(text) >= 2 and text[0] in ROWS:
        try:
            col = int(text[1:])
            return f"{text[0]}{col}"
        except ValueError:
            return text
    return text


def well_row_col(well: str) -> tuple[str, int]:
    well = normalize_well(well)
    return well[0], int(well[1:])


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def minmax(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0..1, preserving NaNs as 0."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        return pd.Series(0.0, index=series.index)
    return ((s - lo) / (hi - lo)).fillna(0.0)


def safe_ratio(num: float, den: float, default: float = np.nan) -> float:
    if den is None or pd.isna(den) or abs(den) < 1e-12:
        return default
    return num / den


def warn(message: str, warnings_list: list[str] | None = None) -> None:
    warnings.warn(message, stacklevel=2)
    if warnings_list is not None:
        warnings_list.append(message)


def save_run_summary(path: Path, rows: Iterable[dict]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def json_dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True)
