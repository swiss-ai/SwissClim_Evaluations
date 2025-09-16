from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Tuple

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
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    calc_relative: bool,
    n_points: int,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    # ds_target = ground truth, ds_prediction = model predictions
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}
    # FSS configuration
    # Typical defaults used in global NWP verification:
    # - quantile = 0.90 (90th percentile) when no absolute threshold is provided
    # - window_size = 9x9 grid cells
    auto_window_size = _window_size(ds_target)
    fss_quantile = 0.90
    fss_window_size: Tuple[int, int] = (9, 9)
    # Optional absolute thresholds (either a single float for all vars or per-variable mapping)
    fss_threshold_global: float | None = None
    fss_thresholds_per_var: dict[str, float] | None = None
    if isinstance(fss_cfg, dict):
        # quantile can be [0,1] or percentile in [0,100]
        try:
            if "quantile" in fss_cfg and fss_cfg["quantile"] is not None:
                q_val = float(fss_cfg["quantile"])  # type: ignore[assignment]
                if q_val > 1.0:
                    q_val = q_val / 100.0
                # clamp into (0,1)
                fss_quantile = min(max(q_val, 0.0), 1.0)
        except Exception:
            fss_quantile = 0.90
        # window_size as int or [h, w]
        ws = fss_cfg.get("window_size")
        if ws is not None:
            try:
                if isinstance(ws, int):
                    fss_window_size = (max(1, int(ws)), max(1, int(ws)))
                elif isinstance(ws, Iterable) and not isinstance(
                    ws, (str, bytes)
                ):
                    ws_list = list(ws)  # type: ignore[arg-type]
                    if len(ws_list) >= 2:
                        fss_window_size = (
                            max(1, int(ws_list[0])),
                            max(1, int(ws_list[1])),
                        )
                # else: keep default (9,9)
            except Exception:
                # fallback to an automatic heuristic if provided value is invalid
                fss_window_size = auto_window_size
        # thresholds: a single float or per-variable dict
        th = fss_cfg.get("threshold")
        if th is not None:
            try:
                fss_threshold_global = float(th)
            except Exception:
                fss_threshold_global = None
        th_map = fss_cfg.get("thresholds")
        if isinstance(th_map, dict):
            try:
                fss_thresholds_per_var = {
                    str(k): float(v) for k, v in th_map.items()
                }
            except Exception:
                fss_thresholds_per_var = None

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
        y_true = ds_target[var]
        y_pred = ds_prediction[var]
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
            # Determine event threshold preference order:
            # 1) per-variable absolute threshold (if provided)
            # 2) global absolute threshold (if provided)
            # 3) quantile of observed sample (default)
            if fss_thresholds_per_var and (var in fss_thresholds_per_var):
                event_threshold = float(fss_thresholds_per_var[var])
            elif fss_threshold_global is not None:
                event_threshold = float(fss_threshold_global)
            else:
                sample = np.random.default_rng(42).choice(
                    y_true.values.ravel(),
                    min(n_points, y_true.size),
                    replace=False,
                )
                event_threshold = float(np.quantile(sample, fss_quantile))
            try:
                fss_val = float(
                    fss_2d(
                        y_pred.compute(),
                        y_true.compute(),
                        event_threshold=event_threshold,
                        window_size=fss_window_size,
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
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    ds_target_std: xr.Dataset,
    ds_prediction_std: xr.Dataset,
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
        min(10_000_000, ds_target[list(ds_target.data_vars)[0]].size)
        if ds_target.data_vars
        else 0
    )

    include = None
    std_include = None
    fss_cfg: dict[str, Any] | None = None
    if metrics_cfg and isinstance(metrics_cfg.get("deterministic"), dict):
        include = metrics_cfg["deterministic"].get("include")
        std_include = metrics_cfg["deterministic"].get("standardized_include")
        fss_cfg = metrics_cfg["deterministic"].get("fss")

    regular_metrics = _calculate_all_metrics(
        ds_target,
        ds_prediction,
        calc_relative=True,
        n_points=n_points,
        include=include,
        fss_cfg=fss_cfg,
    )
    standardized_metrics = _calculate_all_metrics(
        ds_target_std,
        ds_prediction_std,
        calc_relative=False,
        n_points=n_points,
        include=std_include,
        fss_cfg=fss_cfg,
    )
    standardized_metrics.columns = [
        f"{c} (Stdized)" for c in standardized_metrics.columns
    ]

    section_output.mkdir(parents=True, exist_ok=True)
    out_csv = section_output / "metrics.csv"
    out_csv_std = section_output / "metrics_standardized.csv"
    regular_metrics.to_csv(out_csv)
    standardized_metrics.to_csv(out_csv_std)
    print(f"[deterministic] saved {out_csv}")
    print(f"[deterministic] saved {out_csv_std}")

    # Console summary similar to ETS
    try:
        print("Deterministic metrics (targets vs predictions) — first 5 rows:")
        print(regular_metrics.head())
    except Exception:
        pass
    try:
        print(
            "Deterministic standardized metrics (targets vs predictions) — first 5 rows:"
        )
        print(standardized_metrics.head())
    except Exception:
        pass
