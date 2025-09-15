from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wasserstein_distance
from scores.continuous import mae, mse, rmse
from scores.continuous.correlation import pearsonr
from scores.spatial import fss_2d


def _window_size(ds: xr.Dataset) -> tuple[int, int]:
    max_spatial_dim = max(ds.longitude.size, ds.latitude.size)
    # Choose a small window relative to grid size, but at least 1
    ws = max(1, max_spatial_dim // 10)
    return (ws, ws)


def _calculate_all_metrics(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    calc_relative: bool,
    n_points: int,
    include: list[str] | None,
) -> pd.DataFrame:
    variables = list(ds.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}
    window_size = _window_size(ds)

    # Determine which metrics to compute. If include is None, compute all.
    all_metric_names = {
        "MAE",
        "RMSE",
        "MSE",
        "Relative MAE",
        "Pearson R",
        "FSS",
        "Wasserstein",
        "Relative L1",
        "Relative L2",
    }
    metrics_to_compute = all_metric_names if include is None else set(include)

    for var in variables:
        y_true = ds[var]
        y_pred = ds_ml[var]
        # Precompute stats only if required
        row: dict[str, float] = {}

        # Base error metrics
        if (include is None) or ("MAE" in metrics_to_compute):
            mae_val = float(mae(y_true, y_pred))
            row["MAE"] = mae_val
        else:
            mae_val = np.nan

        if (include is None) or ("RMSE" in metrics_to_compute):
            rmse_val = float(rmse(y_true, y_pred))
            row["RMSE"] = rmse_val
        else:
            rmse_val = np.nan

        if (include is None) or ("MSE" in metrics_to_compute):
            mse_val = float(mse(y_true, y_pred))
            row["MSE"] = mse_val

        # Correlation
        if (include is None) or ("Pearson R" in metrics_to_compute):
            row["Pearson R"] = float(pearsonr(y_true, y_pred))

        # FSS (expensive); only compute if requested
        if (include is None) or ("FSS" in metrics_to_compute):
            # Compute threshold based on a random sample of y_true
            sample = np.random.default_rng(42).choice(
                y_true.values.ravel(), min(n_points, y_true.size), replace=False
            )
            quantile_90 = float(np.quantile(sample, 0.90))
            try:
                fss_val = float(
                    fss_2d(
                        y_pred.compute(),
                        y_true.compute(),
                        event_threshold=quantile_90,
                        window_size=window_size,
                        spatial_dims=["latitude", "longitude"],
                    )
                )
            except Exception:
                fss_val = float("nan")
            row["FSS"] = fss_val

        # Wasserstein distance
        if (include is None) or ("Wasserstein" in metrics_to_compute):
            row["Wasserstein"] = float(
                wasserstein_distance(
                    y_true.values.flatten(), y_pred.values.flatten()
                )
            )

        # Relative metrics, only if requested and calc_relative True
        if calc_relative:
            l1_norm = float(np.sum(np.abs(y_true.values)))
            l2_norm = float(np.sqrt(np.sum(y_true.values**2)))
            mean_abs = float(np.mean(np.abs(y_true.values)))

            if (include is None) or ("Relative MAE" in metrics_to_compute):
                row["Relative MAE"] = (
                    (mae_val / mean_abs) if mean_abs else float("nan")
                )
            if (include is None) or ("Relative L1" in metrics_to_compute):
                row["Relative L1"] = (
                    (mae_val / l1_norm) if l1_norm else float("nan")
                )
            if (include is None) or ("Relative L2" in metrics_to_compute):
                row["Relative L2"] = (
                    (rmse_val / l2_norm) if l2_norm else float("nan")
                )

        # Finally, if include list is provided, filter to those keys
        if include is not None:
            row = {k: v for k, v in row.items() if k in include}

        metrics_dict[var] = row

    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    return df


def run(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    ds_std: xr.Dataset,
    ds_ml_std: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    metrics_cfg: dict[str, Any] | None = None,
) -> None:
    """Compute and store deterministic (formerly objective) metrics.

    Saves CSVs:
    - deterministic/metrics.csv
    - deterministic/metrics_standardized.csv
    """
    section_output = out_root / "deterministic"
    n_points = (
        min(10_000_000, ds[list(ds.data_vars)[0]].size) if ds.data_vars else 0
    )

    include = None
    std_include = None
    if metrics_cfg and isinstance(metrics_cfg.get("deterministic"), dict):
        include = metrics_cfg["deterministic"].get("include")
        std_include = metrics_cfg["deterministic"].get("standardized_include")

    regular_metrics = _calculate_all_metrics(
        ds, ds_ml, calc_relative=True, n_points=n_points, include=include
    )
    standardized_metrics = _calculate_all_metrics(
        ds_std,
        ds_ml_std,
        calc_relative=False,
        n_points=n_points,
        include=std_include,
    )
    standardized_metrics.columns = [
        f"{c} (Stdized)" for c in standardized_metrics.columns
    ]

    section_output.mkdir(parents=True, exist_ok=True)
    regular_metrics.to_csv(section_output / "metrics.csv")
    standardized_metrics.to_csv(section_output / "metrics_standardized.csv")

    # Console summary similar to ETS
    try:
        print("Deterministic metrics (first 5 rows):")
        print(regular_metrics.head())
    except Exception:
        pass
    try:
        print("Deterministic standardized metrics (first 5 rows):")
        print(standardized_metrics.head())
    except Exception:
        pass
