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
    return (max_spatial_dim // 100,) * 2


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

    for var in variables:
        y_true = ds[var]
        y_pred = ds_ml[var]
        sample = np.random.default_rng(42).choice(
            y_true.values.ravel(), min(n_points, y_true.size), replace=False
        )
        quantile_90 = float(np.quantile(sample, 0.90))

        l1_norm = float(np.sum(np.abs(y_true.values)))
        l2_norm = float(np.sqrt(np.sum(y_true.values**2)))
        mean_abs = float(np.mean(np.abs(y_true.values)))

        mae_val = float(mae(y_true, y_pred))
        rmse_val = float(rmse(y_true, y_pred))
        mse_val = float(mse(y_true, y_pred))
        pearson_val = float(pearsonr(y_true, y_pred))
        fss_val = float(
            fss_2d(
                y_pred.compute(),
                y_true.compute(),
                event_threshold=quantile_90,
                window_size=window_size,
                spatial_dims=["latitude", "longitude"],
            )
        )
        wasserstein_val = float(
            wasserstein_distance(
                y_true.values.flatten(), y_pred.values.flatten()
            )
        )

        row = {
            "MAE": mae_val,
            "RMSE": rmse_val,
            "MSE": mse_val,
            "Relative MAE": mae_val / mean_abs if mean_abs else np.nan,
            "Pearson R": pearson_val,
            "FSS": fss_val,
            "Wasserstein": wasserstein_val,
        }
        if calc_relative:
            row.update({
                "Relative L1": mae_val / l1_norm if l1_norm else np.nan,
                "Relative L2": rmse_val / l2_norm if l2_norm else np.nan,
            })
        if include:
            filtered = {k: v for k, v in row.items() if k in include}
            metrics_dict[var] = filtered
        else:
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
