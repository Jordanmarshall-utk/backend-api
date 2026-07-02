"""
Automated ML model execution after curve fitting completes.
This module handles automatic ML model training based on workflow configuration.
"""
from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Constants matching ml_models.py
MODEL_SINGLE_GP = "Single-objective GP (scikit-learn)"
MODEL_DUAL_TORCH_GP = "Dual-objective GP (PyTorch)"
MODEL_MONTE_CARLO_TREE = "Monte Carlo Decision Tree (external)"

_META_COLUMNS = frozenset(
    {"Read", "Composition", "Well", "composition_number", "iteration"}
)


def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """Numeric columns usable as GP targets/features (excludes metadata)."""
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if c not in _META_COLUMNS
    ]


def _deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _config_for_dual_gp(auto_config: Dict[str, Any]) -> Dict[str, Any]:
    nested = auto_config.get("dual_gp")
    flat = {k: v for k, v in auto_config.items() if k not in ("dual_gp", "monte_carlo_tree")}
    if isinstance(nested, dict):
        return {**nested, **flat}
    return flat


def _config_for_monte_carlo(auto_config: Dict[str, Any]) -> Dict[str, Any]:
    nested = auto_config.get("monte_carlo_tree")
    flat = {k: v for k, v in auto_config.items() if k not in ("dual_gp", "monte_carlo_tree")}
    if isinstance(nested, dict):
        return {**nested, **flat}
    return flat


def _default_performance_column(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    for name in ("R_squared", "r_squared", "R2", "r2"):
        if name in df.columns:
            return name
    for col in numeric_cols:
        low = col.lower()
        if "r_squared" in low or low == "r2":
            return col
    return numeric_cols[0]


def _default_stability_column(
    df: pd.DataFrame, numeric_cols: List[str], perf_col: str
) -> str:
    for col in numeric_cols:
        if col == perf_col:
            continue
        low = col.lower()
        if "stability" in low or "instability" in low:
            return col
    others = [c for c in numeric_cols if c != perf_col]
    return others[0] if others else perf_col


def _default_feature_columns(
    numeric_cols: List[str], perf_col: str, stab_col: str
) -> List[str]:
    exclude = {perf_col, stab_col, *(_META_COLUMNS & set(numeric_cols))}
    peak_feats = [
        c
        for c in numeric_cols
        if c.startswith("Peak_") and c not in exclude
    ]
    if peak_feats:
        return peak_feats
    return [c for c in numeric_cols if c not in exclude]


def can_compute_instability_from_csv(df: pd.DataFrame) -> bool:
    has_initial_final = (
        ("initial_peak_positions" in df.columns or "Peak_1_Wavelength" in df.columns)
        and (
            "initial_peak_intensities" in df.columns
            or "Peak_1_Intensity" in df.columns
        )
    )
    has_multiple_reads = "Read" in df.columns and df["Read"].nunique() > 1
    return bool(has_initial_final or has_multiple_reads)


def inspect_peak_csv(csv_path: str) -> Dict[str, Any]:
    """Return column metadata for dual-GP UI (Streamlit parity)."""
    path = Path(csv_path)
    if not path.is_file():
        return {"ok": False, "error": f"CSV not found: {csv_path}"}
    try:
        df = pd.read_csv(path, nrows=500)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if df.empty:
        return {"ok": False, "error": "CSV file is empty."}
    numeric_cols = _numeric_feature_columns(df)
    perf = _default_performance_column(df, numeric_cols) if numeric_cols else None
    stab = (
        _default_stability_column(df, numeric_cols, perf)
        if perf and numeric_cols
        else None
    )
    features = (
        _default_feature_columns(numeric_cols, perf, stab)
        if perf and stab
        else []
    )
    return {
        "ok": True,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "default_performance_target": perf,
        "default_stability_target": stab,
        "default_feature_columns": features,
        "can_compute_instability": can_compute_instability_from_csv(df),
        "preview_rows": df.head(5).replace({np.nan: None}).to_dict(orient="records"),
    }


def find_latest_curve_fitting_results(results_dir: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Find the most recent curve fitting JSON/CSV pair (searches recursively)."""
    from app.services.curve_fitting_exports import find_latest_curve_fitting_pair

    pair = find_latest_curve_fitting_pair(results_dir)
    if not pair:
        return None
    json_path, csv_path = pair
    return json_path, csv_path or ""


def find_composition_csv(
    results_dir: Optional[str] = None,
    data_dir: str = "data",
    extra_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Find composition CSV under results (recursive), data/, or explicit paths."""
    from app.services.curve_fitting_exports import resolve_results_path
    from app.tools.paths import get_results_dir

    if results_dir is None:
        results_dir = get_results_dir()

    for p in extra_paths or []:
        resolved = resolve_results_path(p) if p else None
        if resolved and resolved.is_file():
            return str(resolved)

    if os.path.isdir(results_dir):
        for dirpath, _, filenames in os.walk(results_dir):
            for name in filenames:
                if "composition" in name.lower() and name.endswith(".csv"):
                    return os.path.join(dirpath, name)

    if os.path.isdir(data_dir):
        for dirpath, _, filenames in os.walk(data_dir):
            for name in filenames:
                if "composition" in name.lower() and name.endswith(".csv"):
                    return os.path.join(dirpath, name)

    return None


def run_automated_ml_model(
    model_choice: str,
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    composition_csv: Optional[str] = None,
    auto_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Automatically run the selected ML model on curve fitting results.
    
    Args:
        model_choice: Selected ML model (MODEL_SINGLE_GP or MODEL_DUAL_TORCH_GP)
        json_path: Path to curve fitting JSON results (if None, finds latest)
        csv_path: Path to curve fitting CSV export (if None, finds latest)
        composition_csv: Path to composition CSV (if None, tries to find)
        auto_config: Optional configuration dict for model hyperparameters
    
    Returns:
        Dictionary with model results and recommendations
    """
    # Find files if not provided
    if json_path is None or csv_path is None:
        found_files = find_latest_curve_fitting_results()
        if found_files:
            json_path, csv_path = found_files
        else:
            return {
                "success": False,
                "error": "No curve fitting results found. Please run curve fitting first.",
            }
    
    if composition_csv is None:
        composition_csv = find_composition_csv()
    elif composition_csv and not os.path.isfile(composition_csv):
        from app.services.curve_fitting_exports import resolve_results_path

        resolved = resolve_results_path(str(composition_csv))
        if resolved:
            composition_csv = str(resolved)
    
    auto_config = auto_config or {}
    
    if model_choice == MODEL_SINGLE_GP:
        return _run_single_gp_automated(json_path, composition_csv, auto_config)
    elif model_choice == MODEL_DUAL_TORCH_GP:
        if not csv_path or not os.path.isfile(csv_path):
            return {
                "success": False,
                "error": "Peak export CSV required for dual-objective GP. "
                "Run curve fitting or select a CSV on the ML Models page.",
            }
        return _run_dual_gp_automated(csv_path, _config_for_dual_gp(auto_config))
    elif model_choice == MODEL_MONTE_CARLO_TREE:
        return _run_monte_carlo_tree_automated(_config_for_monte_carlo(auto_config))
    else:
        return {
            "success": False,
            "error": f"Unknown model choice: {model_choice}",
        }


def _run_single_gp_automated(
    json_path: str,
    composition_csv: Optional[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run single GP model automatically."""
    try:
        from app.tools.ml_core import (
            GaussianProcessModel,
            extract_features_from_results,
            generate_exploration_candidates,
        )
        
        # Load data
        with open(json_path, 'r') as f:
            results_data = json.load(f)
        
        if composition_csv is None:
            return {
                "success": False,
                "error": "Composition CSV required for single GP model. Please provide composition data.",
            }
        
        composition_data = pd.read_csv(composition_csv, index_col=0)
        
        # Extract features
        X_df, y_series = extract_features_from_results(results_data, composition_data)
        
        # Plain ML: prefer peak position (peak_1_wavelength) when available; fallback to r_squared
        target_col = config.get("target")
        if not target_col and "peak_1_wavelength" in X_df.columns:
            target_col = "peak_1_wavelength"
        if not target_col:
            target_col = "r_squared"
        if target_col in X_df.columns:
            y_series = X_df[target_col]
        elif target_col == "r_squared":
            y_series = X_df["r_squared"]
        elif target_col == "rmse":
            y_series = X_df["rmse"]
        elif target_col == "num_peaks":
            y_series = X_df["num_peaks"]
        else:
            y_series = X_df["r_squared"]
        
        exclude_cols = ["well", "r_squared", "rmse", "num_peaks"]
        if target_col and target_col in X_df.columns:
            exclude_cols.append(target_col)
        feature_cols = [col for col in X_df.columns if col not in exclude_cols]
        X = X_df[feature_cols].values
        y = y_series.values
        
        # Train model
        kernel_type = config.get("kernel_type", "RBF")
        alpha = config.get("alpha", 1e-6)
        
        gp_model = GaussianProcessModel(kernel_type=kernel_type, alpha=alpha)
        gp_model.fit(X, y, feature_cols, target_col)

        cv_score_mean: float | None = None
        cv_score_std: float | None = None
        try:
            from sklearn.model_selection import cross_val_score

            X_scaled = gp_model.scaler_X.transform(X)
            y_scaled = gp_model.scaler_y.transform(y.reshape(-1, 1)).ravel()
            n_splits = min(5, len(y))
            if n_splits >= 2:
                cv_scores = cross_val_score(
                    gp_model.gp, X_scaled, y_scaled, cv=n_splits, scoring="r2"
                )
                cv_score_mean = float(cv_scores.mean())
                cv_score_std = float(cv_scores.std())
        except Exception:
            pass

        # Generate predictions
        y_pred, y_std = gp_model.predict(X, return_std=True)
        
        # Generate exploration candidates
        n_candidates = config.get("n_candidates", 20)
        candidates_df = generate_exploration_candidates(composition_data, n_candidates)
        X_candidates = candidates_df[feature_cols].values
        
        y_pred_candidates, y_std_candidates = gp_model.predict(X_candidates, return_std=True)
        
        beta = config.get("beta", 2.0)
        acquisition_values = gp_model.acquisition_function(X_candidates, method="UCB", beta=beta)
        
        top_indices = np.argsort(acquisition_values)[::-1][:10]
        
        top_candidates = []
        suggested_compositions = []
        for idx in top_indices:
            candidate = candidates_df.iloc[idx]
            top_candidates.append({
                "rank": len(top_candidates) + 1,
                "candidate": candidate["well"],
                "predicted_value": float(y_pred_candidates[idx]),
                "uncertainty": float(y_std_candidates[idx]),
                "acquisition_score": float(acquisition_values[idx]),
            })
            comp_row = {col.replace("composition_", ""): float(candidate[col])
                       for col in feature_cols if col.startswith("composition_")}
            suggested_compositions.append({
                "rank": len(suggested_compositions) + 1,
                "compositions": comp_row,
                "predicted_value": float(y_pred_candidates[idx]),
                "uncertainty": float(y_std_candidates[idx]),
                "acquisition_score": float(acquisition_values[idx]),
            })
        
        # Generate 2D plot: GP mean + 95% CI + training points + acquisition batch (reference style)
        plot_path = None
        comp_cols_all = [c for c in feature_cols if c.startswith("composition_")]
        if len(comp_cols_all) >= 1 and target_col in X_df.columns:
            try:
                import plotly.graph_objects as go
                comp_col = comp_cols_all[0]
                x_min, x_max = float(X_df[comp_col].min()), float(X_df[comp_col].max())
                x_grid = np.linspace(x_min, x_max, 100)
                grid_X = np.tile(candidates_df[feature_cols].median().values, (len(x_grid), 1))
                col_idx = feature_cols.index(comp_col)
                grid_X[:, col_idx] = x_grid
                mu_grid, std_grid = gp_model.predict(grid_X, return_std=True)
                ci95 = 1.96  # 95% CI
                upper = mu_grid + ci95 * std_grid
                lower = mu_grid - ci95 * std_grid
                fig = go.Figure()
                # Training data (blue dots)
                fig.add_trace(go.Scatter(
                    x=X_df[comp_col], y=X_df[target_col], mode="markers",
                    name="Training data", marker=dict(color="rgb(31, 119, 180)", size=10, symbol="circle")))
                # 95% CI shaded band (light orange)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_grid, x_grid[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself", fillcolor="rgba(255, 165, 0, 0.25)", line=dict(width=0),
                    name="95% CI"))
                # GP mean (solid orange line)
                fig.add_trace(go.Scatter(
                    x=x_grid, y=mu_grid, mode="lines", name="GP mean",
                    line=dict(color="rgb(255, 127, 14)", width=2)))
                # Acquisition batch (red stars)
                top_data = candidates_df.iloc[top_indices]
                fig.add_trace(go.Scatter(
                    x=top_data[comp_col], y=y_pred_candidates[top_indices],
                    mode="markers+text", name="Acquisition batch",
                    marker=dict(color="red", size=14, symbol="star"),
                    text=[f"#{i+1}" for i in range(len(top_indices))], textposition="top center"))
                fig.update_layout(
                    title=f"GP Regression — {comp_col.replace('composition_','')} vs {target_col}",
                    xaxis_title=comp_col.replace("composition_", ""), yaxis_title=target_col,
                    height=450, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
                os.makedirs(results_dir, exist_ok=True)
                plot_path = os.path.join(results_dir, "gp_ucb_acquisition_2d.html")
                fig.write_html(plot_path)
            except Exception as plot_err:
                plot_path = None  # Non-fatal
        
        out = {
            "success": True,
            "model_type": "Single-objective GP (scikit-learn)",
            "kernel": kernel_type,
            "kernel_type": kernel_type,
            "target": target_col,
            "target_name": target_col,
            "acquisition_method": "UCB",
            "beta": float(beta),
            "top_candidates": top_candidates,
            "suggested_compositions": suggested_compositions,
            "predictions": {
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred)),
            },
            "uncertainty_stats": {
                "mean": float(y_std_candidates.mean()),
                "max": float(y_std_candidates.max()),
            },
        }
        if cv_score_mean is not None:
            out["cv_score"] = cv_score_mean
            out["cv_std"] = cv_score_std if cv_score_std is not None else 0.0
        if plot_path:
            out["gp_ucb_acquisition_plot"] = plot_path
        return out
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _run_dual_gp_automated(
    csv_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run dual GP model automatically."""
    try:
        from app.tools.ml_core import TorchGaussianProcess, instability_score, _HAS_TORCH
        from app.utils.json_safe import to_jsonable

        if not _HAS_TORCH:
            return {
                "success": False,
                "error": "PyTorch is not installed. Install backend optional deps: pip install '.[ml]'",
            }

        # Load CSV
        df_dual = pd.read_csv(csv_path)
        
        if df_dual.empty:
            return {
                "success": False,
                "error": "CSV file is empty.",
            }
        
        numeric_cols = _numeric_feature_columns(df_dual)
        
        if len(numeric_cols) < 2:
            return {
                "success": False,
                "error": "Need at least 2 numeric columns (performance target + features) for dual GP.",
            }
        
        perf_col = config.get("performance_target") or _default_performance_column(
            df_dual, numeric_cols
        )
        compute_instability = bool(config.get("compute_instability", False))
        
        if compute_instability:
            # Compute instability score
            instability_params = config.get("instability_params", {})
            compositions_with_multiple_peaks = []
            
            if 'Total_Quality_Peaks' in df_dual.columns:
                multi_peak_mask = df_dual['Total_Quality_Peaks'] > 1
                if 'composition_number' in df_dual.columns and 'iteration' in df_dual.columns:
                    multi_peak_comps = df_dual[multi_peak_mask][['composition_number', 'iteration']].values
                    compositions_with_multiple_peaks = [tuple(row) for row in multi_peak_comps]
            
            y_stab_raw = instability_score(
                df_dual,
                compositions_with_multiple_peaks=compositions_with_multiple_peaks if compositions_with_multiple_peaks else None,
                target_wavelength=instability_params.get('target_wavelength', 700),
                multiple_peak_penalty=instability_params.get('multiple_peak_penalty', 0.5),
                wavelength_tolerance=instability_params.get('wavelength_tolerance', 10),
                degradation_weight=instability_params.get('degradation_weight', 0.4),
                position_weight=instability_params.get('position_weight', 0.6),
            )
            
            # Normalize
            y_stab_min = np.min(y_stab_raw)
            y_stab_max = np.max(y_stab_raw)
            if y_stab_max > y_stab_min:
                y_stab = (y_stab_raw - y_stab_min) / (y_stab_max - y_stab_min)
            else:
                y_stab = y_stab_raw
            
            df_dual["computed_instability_score"] = y_stab
            stab_col = "computed_instability_score"
        else:
            stab_col = config.get("stability_target") or _default_stability_column(
                df_dual, numeric_cols, perf_col
            )
            if stab_col not in df_dual.columns:
                return {
                    "success": False,
                    "error": f"Stability column not found: {stab_col}",
                }
            y_stab = df_dual[stab_col].values.astype(np.float32)
        
        # Select features
        feature_cols = config.get("feature_columns")
        if not feature_cols:
            feature_cols = _default_feature_columns(numeric_cols, perf_col, stab_col)
        feature_cols = [c for c in feature_cols if c in df_dual.columns]
        if not feature_cols:
            return {
                "success": False,
                "error": "No feature columns available. Select feature columns in ML config.",
            }
        
        X_dual = df_dual[feature_cols].values.astype(np.float32)
        y_perf = df_dual[perf_col].values.astype(np.float32)
        
        # Model hyperparameters
        lengthscale = config.get("lengthscale", 1.0)
        noise_level = config.get("noise_level", 1e-4)
        beta_dual = config.get("beta", 2.0)
        instability_threshold_percentile = config.get("instability_threshold_percentile", 0.7)
        use_multiplicative = config.get("use_multiplicative_adjustment", True)
        
        # Train GPs
        gp_perf = TorchGaussianProcess(lengthscale=lengthscale, variance=1.0, noise=noise_level)
        gp_stab = TorchGaussianProcess(lengthscale=lengthscale, variance=1.0, noise=noise_level)
        
        gp_perf.fit(X_dual, y_perf)
        gp_stab.fit(X_dual, y_stab)
        
        # Predictions
        mu_perf, std_perf = gp_perf.predict(X_dual, return_std=True)
        mu_stab, std_stab = gp_stab.predict(X_dual, return_std=True)
        
        # Acquisition
        acq_base = mu_perf + beta_dual * std_perf
        init_tune_score = mu_stab
        threshold_value = np.quantile(init_tune_score, instability_threshold_percentile)
        
        adjust_tune_score = np.where(init_tune_score > threshold_value, 0, init_tune_score)
        if np.max(adjust_tune_score) > 0:
            adjust_tune_score = adjust_tune_score / np.max(adjust_tune_score)
        
        if use_multiplicative:
            acquisition_score = acq_base * adjust_tune_score
            acquisition_score = np.where(init_tune_score > threshold_value, 0, acquisition_score)
        else:
            stability_weight = config.get("stability_weight", 1.0)
            acquisition_score = acq_base - stability_weight * mu_stab
        
        # Rank results
        df_results = df_dual.copy()
        df_results["mu_perf"] = mu_perf
        df_results["std_perf"] = std_perf
        df_results["mu_stab"] = mu_stab
        df_results["std_stab"] = std_stab
        df_results["init_tune_score"] = init_tune_score
        df_results["adjust_tune_score"] = adjust_tune_score
        df_results["acquisition_score_base"] = acq_base
        df_results["acquisition_score"] = acquisition_score
        df_ranked = df_results.sort_values(by="acquisition_score", ascending=False).reset_index(
            drop=True
        )

        display_cols = [
            "acquisition_score",
            "acquisition_score_base",
            "mu_perf",
            "std_perf",
            "init_tune_score",
            "adjust_tune_score",
            perf_col,
            stab_col,
            "Well",
            "Composition",
            "Read",
        ] + feature_cols[:5]
        keep_cols = list(dict.fromkeys(c for c in display_cols if c in df_ranked.columns))
        top_candidates = df_ranked.loc[:, keep_cols].head(10).to_dict(orient="records")

        return to_jsonable(
            {
                "success": True,
                "model_type": "DualTorchGP",
                "target": perf_col,
                "performance_target": perf_col,
                "stability_target": stab_col,
                "feature_columns": feature_cols,
                "acquisition_method": "UCB + stability adjustment",
                "compute_instability": compute_instability,
                "top_candidates": top_candidates,
                "predictions": {
                    "performance_mean": float(np.mean(mu_perf)),
                    "stability_mean": float(np.mean(mu_stab)),
                },
                "uncertainty_stats": {
                    "performance_std_mean": float(np.mean(std_perf)),
                    "stability_std_mean": float(np.mean(std_stab)),
                    "mean": float(np.mean(std_perf)),
                    "max": float(np.max(std_perf)),
                },
                "acquisition_stats": {
                    "mean": float(np.mean(acquisition_score)),
                    "max": float(np.max(acquisition_score)),
                },
            }
        )
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _run_monte_carlo_tree_automated(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the external Monte Carlo Decision Tree project in headless mode.

    This assumes:
      - The repo path points to the folder containing `main.py`
      - The project manages its own data/history under its `data/` folder
    """
    import subprocess
    from pathlib import Path

    repo_path = (
        config.get("repo_path")
        or os.environ.get("MONTE_CARLO_REPO_PATH")
        or os.environ.get("MC_DECISION_TREE_REPO")
    )
    if not repo_path:
        return {
            "success": False,
            "error": "Monte Carlo repo path not configured. Set ml_model_config.monte_carlo_tree.repo_path "
            "or MONTE_CARLO_REPO_PATH on the API server.",
        }

    repo = Path(repo_path)
    if not repo.exists():
        return {
            "success": False,
            "error": f"Monte Carlo repo path does not exist: {repo}",
        }

    n_attempts = int(config.get("n_attempts", 500))
    with_agent = bool(config.get("with_agent", False))

    # Build command
    python_exe = os.environ.get("PYTHON_EXECUTABLE", "python")
    cmd = [python_exe, "main.py"]
    env = os.environ.copy()
    env["MC_N_ATTEMPTS"] = str(n_attempts)
    if with_agent:
        cmd.extend(["--with-agent", "--auto-apply"])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=None,
        )

        success = result.returncode == 0
        
        # Parse results from stdout and output files
        monte_carlo_results = {
            "success": success,
            "model_type": "MonteCarloDecisionTree",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
        
        # Try to parse optimization stats from stdout
        import re
        stdout_lines = result.stdout.split('\n')
        stats = {}
        for line in stdout_lines:
            if "Total Cycles:" in line:
                match = re.search(r'Total Cycles:\s*(\d+)', line)
                if match:
                    stats["total_cycles"] = int(match.group(1))
            elif "Best Quality Achieved:" in line:
                match = re.search(r'Best Quality Achieved:\s*([\d.]+)', line)
                if match:
                    stats["best_quality"] = float(match.group(1))
            elif "Total Improvement:" in line:
                match = re.search(r'Total Improvement:\s*([+-]?[\d.]+)%', line)
                if match:
                    stats["total_improvement_pct"] = float(match.group(1))
            elif "Average Improvement/Cycle:" in line:
                match = re.search(r'Average Improvement/Cycle:\s*([+-]?[\d.]+)%', line)
                if match:
                    stats["avg_improvement_per_cycle_pct"] = float(match.group(1))
        
        if stats:
            monte_carlo_results["optimization_stats"] = stats
        
        # Try to parse LLM agent restriction/constraint information from stdout
        # Look for LLM Agent sections in stdout
        llm_agent_results = {}
        in_llm_section = False
        llm_section_lines = []
        
        for i, line in enumerate(stdout_lines):
            if "LLM Agent" in line or "Running LLM Agent" in line:
                in_llm_section = True
                llm_section_lines = []
            elif in_llm_section:
                if line.strip().startswith("=" * 60) or (line.strip() and not line.startswith(" ")):
                    # End of LLM section or new major section
                    if "RECOMMENDATION SUMMARY" in "\n".join(llm_section_lines):
                        break
                llm_section_lines.append(line)
        
        # Try to extract LLM agent information
        if llm_section_lines:
            llm_text = "\n".join(llm_section_lines)
            
            # Extract patterns (parameter restrictions/observations)
            patterns = []
            for line in llm_section_lines:
                if "Found" in line and "patterns" in line.lower():
                    match = re.search(r'Found\s+(\d+)\s+patterns?', line, re.IGNORECASE)
                    if match:
                        llm_agent_results["num_patterns"] = int(match.group(1))
                elif re.match(r'\s*-\s+[A-Za-z_]+:', line):
                    # Pattern line: "  - parameter: observation... (conf: X.XX)"
                    pattern_match = re.match(r'\s*-\s+([A-Za-z_]+):\s*(.+?)\s*\(conf:\s*([\d.]+)\)', line)
                    if pattern_match:
                        patterns.append({
                            "parameter": pattern_match.group(1),
                            "observation": pattern_match.group(2),
                            "confidence": float(pattern_match.group(3))
                        })
            
            if patterns:
                llm_agent_results["patterns"] = patterns
            
            # Extract config suggestions (parameter restrictions/changes)
            suggestions = []
            for line in llm_section_lines:
                if "Generated" in line and "suggestions" in line.lower():
                    match = re.search(r'Generated\s+(\d+)\s+suggestions?', line, re.IGNORECASE)
                    if match:
                        llm_agent_results["num_suggestions"] = int(match.group(1))
                elif "->" in line and "conf:" in line:
                    # Suggestion line: "  - parameter_path: current -> suggested (conf: X.XX)"
                    sugg_match = re.match(r'\s*-\s+([^:]+):\s*([^\s]+)\s*->\s*([^\s]+)\s*\(conf:\s*([\d.]+)\)', line)
                    if sugg_match:
                        suggestions.append({
                            "parameter_path": sugg_match.group(1).strip(),
                            "current_value": sugg_match.group(2),
                            "suggested_value": sugg_match.group(3),
                            "confidence": float(sugg_match.group(4))
                        })
            
            if suggestions:
                llm_agent_results["suggestions"] = suggestions
            
            # Extract recommendation summary
            summary_start = None
            summary_end = None
            for i, line in enumerate(llm_section_lines):
                if "RECOMMENDATION SUMMARY" in line:
                    summary_start = i + 1
                elif summary_start is not None and line.strip().startswith("-" * 60):
                    summary_end = i
                    break
            
            if summary_start is not None:
                summary_lines = llm_section_lines[summary_start:summary_end] if summary_end else llm_section_lines[summary_start:]
                summary_text = "\n".join(summary_lines).strip()
                if summary_text:
                    llm_agent_results["recommendation_summary"] = summary_text
        
        # Also try to load LLM agent results from a JSON file if it exists
        # (in case the Monte Carlo project saves agent results to a file)
        llm_results_file = repo / "data" / "llm_agent_results.json"
        if llm_results_file.exists():
            try:
                import json as json_lib
                with open(llm_results_file, 'r') as f:
                    saved_llm_results = json_lib.load(f)
                    # Merge saved results with parsed results (saved takes precedence)
                    if saved_llm_results:
                        llm_agent_results.update(saved_llm_results)
            except Exception as e:
                monte_carlo_results["llm_agent_parse_error"] = str(e)
        
        if llm_agent_results:
            monte_carlo_results["llm_agent_results"] = llm_agent_results
        
        # Try to load candidates_analysis.csv if it exists
        candidates_csv = repo / "data" / "candidates_analysis.csv"
        if candidates_csv.exists():
            try:
                import pandas as pd
                candidates_df = pd.read_csv(candidates_csv)
                if not candidates_df.empty:
                    # Extract top candidates
                    top_candidates = []
                    # Look for columns that might indicate ranking/score
                    score_cols = [c for c in candidates_df.columns if 'score' in c.lower() or 'rank' in c.lower() or 'quality' in c.lower()]
                    if score_cols:
                        # Sort by score if available
                        sort_col = score_cols[0]
                        top_df = candidates_df.nlargest(10, sort_col) if candidates_df[sort_col].dtype in ['float64', 'int64'] else candidates_df.head(10)
                    else:
                        top_df = candidates_df.head(10)
                    
                    for idx, row in top_df.iterrows():
                        candidate_info = {
                            "rank": idx + 1,
                            "candidate": str(row.get("Material_Composition", f"Candidate {idx + 1}")),
                        }
                        # Add any numeric columns as metrics
                        for col in candidates_df.columns:
                            if candidates_df[col].dtype in ['float64', 'int64']:
                                candidate_info[col] = float(row[col])
                        top_candidates.append(candidate_info)
                    
                    monte_carlo_results["top_candidates"] = top_candidates
                    monte_carlo_results["candidates_file"] = str(candidates_csv)
            except Exception as e:
                # If we can't parse the CSV, continue without it
                monte_carlo_results["candidates_parse_error"] = str(e)
        
        # Try to load optimization_history.csv if it exists
        history_csv = repo / "data" / "optimization_history.csv"
        if history_csv.exists():
            try:
                import pandas as pd
                history_df = pd.read_csv(history_csv)
                if not history_df.empty:
                    monte_carlo_results["optimization_history"] = {
                        "total_experiments": len(history_df),
                        "file_path": str(history_csv),
                    }
                    # Get latest cycle info if available
                    if "Cycle" in history_df.columns:
                        monte_carlo_results["optimization_history"]["latest_cycle"] = int(history_df["Cycle"].max())
            except Exception as e:
                monte_carlo_results["history_parse_error"] = str(e)
        
        return monte_carlo_results
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
