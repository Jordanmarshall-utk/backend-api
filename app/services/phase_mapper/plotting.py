"""96-well heatmap plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ROWS, normalize_well, well_row_col


def _first_matching_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = [c for c in df.columns if pattern in c]
        if matches:
            return matches[0]
    return None


def heatmap_plate(df: pd.DataFrame, value_col: str, out_path: Path, title: str | None = None) -> None:
    """Save an A-H by 1-12 plate heatmap."""
    if value_col not in df.columns:
        return
    grid = np.full((8, 12), np.nan)
    for _, row in df.iterrows():
        well = normalize_well(row.get("well", ""))
        try:
            r, c = well_row_col(well)
            grid[ROWS.index(r), c - 1] = pd.to_numeric(row.get(value_col), errors="coerce")
        except Exception:
            continue
    fig, ax = plt.subplots(figsize=(10, 5.4))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(12), labels=range(1, 13))
    ax.set_yticks(range(8), labels=ROWS)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title or value_col)
    for i in range(8):
        for j in range(12):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.2g}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_standard_heatmaps(table: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate requested standard heatmaps when columns are present."""
    out = Path(output_dir)
    mapping = {
        "heatmap_PL_main_peak_nm.png": _first_matching_column(table, ["pl_main_peak_nm"]),
        "heatmap_PL_FWHM_nm.png": _first_matching_column(table, ["pl_FWHM_nm"]),
        "heatmap_abs_edge_nm.png": _first_matching_column(table, ["abs_edge_nm"]),
        "heatmap_optical_heterogeneity_score_total.png": "optical_heterogeneity_score_total",
        "heatmap_pl_abs_disagreement_score.png": "pl_abs_disagreement_score",
        "heatmap_model_uncertainty_score.png": "model_uncertainty_score",
        "heatmap_xrd_priority_score.png": "xrd_priority_score",
        "heatmap_remeasurement_priority_score.png": "optical_remeasurement_priority_score",
        "heatmap_predicted_clean_3d_probability.png": "pred_xrd_clean_3d_perovskite_prob",
        "heatmap_predicted_mixed_phase_probability.png": "pred_xrd_mixed_phase_prob",
        "heatmap_predicted_homogeneity_score.png": "pred_xrd_homogeneity_score",
    }
    for filename, col in mapping.items():
        if col and col in table.columns:
            heatmap_plate(table, col, out / filename, col)
