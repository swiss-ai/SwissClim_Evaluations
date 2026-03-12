from __future__ import annotations

import contextlib
import functools
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import dask
import numpy as np
import pandas as pd
import xarray as xr
from scores.categorical import seeps
from scores.continuous import additive_bias, mae, mse, rmse
from scores.continuous.correlation import pearsonr
from scores.functions import create_latitude_weights
from scores.spatial import fss_2d_single_field

from ... import console as c
from ...dask_utils import (
    compute_global_quantile,
    compute_quantile_preserving,
)


@functools.lru_cache(maxsize=1)
def _open_climatology(path: str) -> xr.Dataset:
    return xr.open_zarr(path)


def _prepare_seeps(y_true: xr.DataArray, path_climatology: str):
    ds_clim = _open_climatology(path_climatology)
    ds_clim = ds_clim.sel(latitude=y_true.latitude, longitude=y_true.longitude)

    prob_dry = ds_clim.total_precipitation_6hr_seeps_dry_fraction
    prob_dry = prob_dry.sel(
        dayofyear=y_true.valid_time.dt.dayofyear, hour=y_true.valid_time.dt.hour
    )

    seeps_threshold = ds_clim.total_precipitation_6hr_seeps_threshold
    seeps_threshold = seeps_threshold.sel(
        dayofyear=y_true.valid_time.dt.dayofyear, hour=y_true.valid_time.dt.hour
    )
    return prob_dry, seeps_threshold


def _window_size(ds: xr.Dataset) -> tuple[int, int]:
    """Heuristic FSS window size based on grid shape."""
    max_spatial_dim = max(int(ds.longitude.size), int(ds.latitude.size))
    ws = max(1, max_spatial_dim // 10)
    return (ws, ws)


def finalize_metrics(
    metrics_dict: dict[str, dict[str, Any]],
    lazy_list: list[tuple[str, str, Any]],
    computed_results: list[Any],
    preserve_dims: list[str] | None,
) -> pd.DataFrame:
    """Public wrapper around the internal metric-finalisation logic.

    Used by callers that collect multiple lazy graphs and run a single
    ``dask.compute`` across them, then dispatch the results here.
    """
    return _finalize_metrics(metrics_dict, lazy_list, computed_results, preserve_dims)


def _finalize_metrics(
    metrics_dict: dict[str, dict[str, Any]],
    lazy_list: list[tuple[str, str, Any]],
    computed_results: list[Any],
    preserve_dims: list[str] | None,
) -> pd.DataFrame:
    """Populate metrics_dict with computed results and convert to DataFrame."""
    for (var, metric, _), res in zip(lazy_list, computed_results, strict=False):
        if preserve_dims and isinstance(res, (xr.DataArray | xr.Dataset)):
            metrics_dict[var][metric] = res
        else:
            try:
                val = float(res)
            except Exception:
                val = float(np.array(res).item())
            metrics_dict[var][metric] = val

    if preserve_dims:
        dfs = []
        for var, metrics in metrics_dict.items():
            try:
                ds_var = xr.Dataset(metrics)
                df_var = ds_var.to_dataframe().reset_index()
                df_var["variable"] = var
                dfs.append(df_var)
            except Exception as e:
                c.print(f"[deterministic] Failed to convert metrics for {var} to DataFrame: {e}")
                continue

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def calculate_all_metrics(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    calc_relative: bool,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None = None,
    seeps_climatology_path: str | None = None,
    preserve_dims: list[str] | None = None,
    compute: bool = True,
    weights: xr.DataArray | None = None,
    performance_cfg: dict[str, Any] | None = None,
    log_variable_progress: bool = True,
) -> pd.DataFrame | tuple[dict[str, dict[str, Any]], list[tuple[str, str, Any]]]:
    """Compute scalar deterministic metrics per variable."""
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, Any]] = {}

    # ---- FSS configuration
    auto_window_size = _window_size(ds_target)
    fss_quantile = 0.90
    fss_window_size: tuple[int, int] | None = None
    fss_thresholds_per_var: dict[str, list] | None = None
    no_event_value: float = 1.0

    if isinstance(fss_cfg, dict):
        q_raw = fss_cfg.get("quantile")
        if q_raw is not None:
            try:
                q_val = float(q_raw)
                if q_val > 1.0:
                    q_val /= 100.0
                if 0.0 < q_val < 1.0:
                    fss_quantile = q_val
            except Exception:
                pass
        ws = fss_cfg.get("window_size")
        if ws is not None:
            try:
                if isinstance(ws, int):
                    fss_window_size = (max(1, int(ws)), max(1, int(ws)))
                elif isinstance(ws, Iterable) and not isinstance(ws, (str | bytes)):
                    ws_list = list(ws)
                    if len(ws_list) >= 2:
                        fss_window_size = (max(1, int(ws_list[0])), max(1, int(ws_list[1])))
            except Exception:
                fss_window_size = None
        th_map = fss_cfg.get("thresholds")
        if isinstance(th_map, dict):
            tmp: dict[str, list] = {}
            for k, v in th_map.items():
                try:
                    tmp[str(k)] = [float(vi) for vi in v]
                except Exception:
                    continue
            if tmp:
                fss_thresholds_per_var = tmp

    if fss_window_size is None:
        fss_window_size = auto_window_size

    metrics_to_compute = (
        set(include)
        if include is not None
        else {
            "MAE",
            "RMSE",
            "MSE",
            "Relative MAE",
            "Bias",
            "Pearson R",
            "FSS",
            "Relative L1",
            "Relative L2",
            "SEEPS",
        }
    )

    if weights is None and "latitude" in ds_target.dims:
        weights = create_latitude_weights(ds_target.latitude)
        weights = weights.clip(min=0.0)

    # --- Pre-compute FSS quantiles if needed (Batch Optimization) ---
    fss_thresholds_map: dict[str, list[Any]] = {}
    if (include is None) or ("FSS" in metrics_to_compute):
        lazy_quantiles = []
        quantile_meta: list[tuple[str, Any]] = []

        for var in variables:
            if fss_thresholds_per_var and (var in fss_thresholds_per_var):
                fss_thresholds_map[var] = [float(f) for f in fss_thresholds_per_var[var]]
                continue

            y_true = ds_target[var]
            should_split_levels = False
            if preserve_dims and "level" in preserve_dims and "level" in y_true.dims:
                should_split_levels = True

            if should_split_levels:
                levels = y_true["level"].values
                for lvl in levels:
                    y_slice = y_true.sel(level=lvl)
                    q_lazy = compute_global_quantile(y_slice, fss_quantile, skipna=True)
                    lazy_quantiles.append(q_lazy)
                    quantile_meta.append((var, lvl))
            elif preserve_dims:
                q_da = compute_quantile_preserving(y_true, [fss_quantile], preserve_dims)
                lazy_quantiles.append(q_da)
                quantile_meta.append((var, None))
            else:
                q_da = compute_global_quantile(y_true, fss_quantile, skipna=True)
                lazy_quantiles.append(q_da)
                quantile_meta.append((var, None))

        if lazy_quantiles:
            if log_variable_progress:
                c.info(
                    "[deterministic] Preparing FSS quantile thresholds "
                    f"for {len(quantile_meta)} variable/level slice(s)."
                )
            computed_quantiles = list(dask.compute(*lazy_quantiles))

            reconstruction = defaultdict(list)
            for (var, lvl), res in zip(quantile_meta, computed_quantiles, strict=False):
                if lvl is not None:
                    reconstruction[var].append((lvl, res))
                else:
                    reconstruction[var] = res

            for var, data in reconstruction.items():
                if isinstance(data, list):
                    try:
                        levels_vals = [x[0] for x in data]
                        q_vals = [
                            float(x[1].item() if hasattr(x[1], "item") else x[1]) for x in data
                        ]
                        da = xr.DataArray(q_vals, coords={"level": levels_vals}, dims="level")
                        fss_thresholds_map[var] = [da]
                    except Exception as e:
                        c.print(f"[deterministic] Failed to reassemble quantiles for {var}: {e}")
                        fss_thresholds_map[var] = []
                else:
                    val = data
                    if isinstance(val, xr.DataArray | np.ndarray) and val.ndim > 0:
                        fss_thresholds_map[var] = [val]
                    else:
                        fss_thresholds_map[var] = [
                            float(val.item() if hasattr(val, "item") else val)
                        ]

    lazy_metrics_to_compute: list[tuple[str, str, Any]] = []
    intermediate_lazy: dict[tuple[str, str], Any] = {}

    for var in variables:
        y_true = ds_target[var]
        y_pred = ds_prediction[var]
        if var not in metrics_dict:
            metrics_dict[var] = {}

        curr_preserve = None
        if preserve_dims:
            curr_preserve = [d for d in preserve_dims if (d in y_true.dims) or (d in y_pred.dims)]
            if not curr_preserve:
                curr_preserve = None

        if (include is None) or ("MAE" in metrics_to_compute):
            val = mae(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "MAE", val))
            intermediate_lazy[(var, "MAE")] = val

        if (include is None) or ("RMSE" in metrics_to_compute):
            val = rmse(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "RMSE", val))

        if (include is None) or ("MSE" in metrics_to_compute):
            val = mse(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "MSE", val))

        if (include is None) or ("Bias" in metrics_to_compute):
            val = additive_bias(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "Bias", val))

        if (include is None) or ("Pearson R" in metrics_to_compute):
            val = pearsonr(y_pred, y_true, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "Pearson R", val))

        if (include is None) or ("FSS" in metrics_to_compute):
            list_event_threshold = fss_thresholds_map.get(var, [])
            if fss_thresholds_per_var and (var in fss_thresholds_per_var):
                list_fss_label = [f"FSS_{et}" for et in list_event_threshold]
            else:
                list_fss_label = [f"FSS_{100 * fss_quantile}%"]

            for event_threshold, fss_label in zip(
                list_event_threshold, list_fss_label, strict=False
            ):
                try:
                    spatial_dims = ["latitude", "longitude"]
                    if isinstance(event_threshold, xr.DataArray | xr.Variable):

                        def _fss_wrapper(p, t, th, **kwargs):
                            return fss_2d_single_field(p, t, event_threshold=th, **kwargs)

                        out = xr.apply_ufunc(
                            _fss_wrapper,
                            y_pred,
                            y_true,
                            event_threshold,
                            input_core_dims=[spatial_dims, spatial_dims, []],
                            kwargs={"window_size": fss_window_size},
                            vectorize=True,
                            dask="parallelized",
                            output_dtypes=[float],
                        )
                    else:
                        out = xr.apply_ufunc(
                            fss_2d_single_field,
                            y_pred,
                            y_true,
                            input_core_dims=[spatial_dims, spatial_dims],
                            kwargs={
                                "event_threshold": event_threshold,
                                "window_size": fss_window_size,
                            },
                            vectorize=True,
                            dask="parallelized",
                            output_dtypes=[float],
                        )

                    if isinstance(out, xr.DataArray):
                        with contextlib.suppress(Exception):
                            evt_t = (y_true >= event_threshold).any(dim=spatial_dims)
                            evt_p = (y_pred >= event_threshold).any(dim=spatial_dims)
                            no_evt = (~evt_t) & (~evt_p)
                            out = out.where(~no_evt, other=1.0)
                        if curr_preserve:
                            reduce_dims = [d for d in out.dims if d not in curr_preserve]
                            fss_scalar = out.mean(dim=reduce_dims, skipna=True)
                        else:
                            fss_scalar = out.mean(skipna=True)
                        lazy_metrics_to_compute.append((var, fss_label, fss_scalar))
                    else:
                        yt_evt = bool((y_true >= event_threshold).any())
                        yp_evt = bool((y_pred >= event_threshold).any())
                        if (not yt_evt) and (not yp_evt):
                            metrics_dict[var][fss_label] = no_event_value
                        else:
                            metrics_dict[var][fss_label] = float(out)
                except Exception as e:
                    with contextlib.suppress(Exception):
                        c.print(f"[deterministic:FSS] fss_2d failed for var='{var}': {e!r}")
                    metrics_dict[var][fss_label] = float("nan")

        if "total_precipitation" in var and "SEEPS" in metrics_to_compute:
            if seeps_climatology_path is None:
                raise ValueError(
                    "SEEPS metric requested but 'seeps_climatology_path' not provided."
                )
            prob_dry, seeps_threshold = _prepare_seeps(y_true, seeps_climatology_path)
            seeps_val = seeps(
                y_pred * 1000, y_true * 1000, prob_dry, seeps_threshold, dry_light_threshold=0.25
            )
            if curr_preserve:
                reduce_dims = [d for d in seeps_val.dims if d not in curr_preserve]
                seeps_val = seeps_val.mean(dim=reduce_dims, skipna=True)
            else:
                seeps_val = seeps_val.mean(skipna=True)
            lazy_metrics_to_compute.append((var, "SEEPS", seeps_val))

        if calc_relative:
            if curr_preserve:
                reduce_dims = [d for d in y_true.dims if d not in curr_preserve]
                l1_norm = np.abs(y_true).sum(dim=reduce_dims, skipna=True)
                l2_norm = (y_true**2).sum(dim=reduce_dims, skipna=True) ** 0.5
                if weights is not None:
                    mean_abs = np.abs(y_true).weighted(weights).mean(dim=reduce_dims, skipna=True)
                else:
                    mean_abs = np.abs(y_true).mean(dim=reduce_dims, skipna=True)
            else:
                l1_norm = np.abs(y_true).sum(skipna=True)
                l2_norm = (y_true**2).sum(skipna=True) ** 0.5
                if weights is not None:
                    mean_abs = np.abs(y_true).weighted(weights).mean(skipna=True)
                else:
                    mean_abs = np.abs(y_true).mean(skipna=True)

            if (include is None) or ("Relative MAE" in metrics_to_compute):
                mae_val_lazy = intermediate_lazy.get((var, "MAE"))
                if mae_val_lazy is None:
                    mae_val_lazy = mae(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
                rel_mae = mae_val_lazy / mean_abs
                lazy_metrics_to_compute.append((var, "Relative MAE", rel_mae))

            err = y_pred - y_true
            if curr_preserve:
                reduce_dims = [d for d in err.dims if d not in curr_preserve]
                l1_err = np.abs(err).sum(dim=reduce_dims, skipna=True)
                l2_err = (err**2).sum(dim=reduce_dims, skipna=True) ** 0.5
            else:
                l1_err = np.abs(err).sum(skipna=True)
                l2_err = (err**2).sum(skipna=True) ** 0.5

            if (include is None) or ("Relative L1" in metrics_to_compute):
                rel_l1 = l1_err / l1_norm
                lazy_metrics_to_compute.append((var, "Relative L1", rel_l1))

            if (include is None) or ("Relative L2" in metrics_to_compute):
                rel_l2 = l2_err / l2_norm
                lazy_metrics_to_compute.append((var, "Relative L2", rel_l2))

    if not compute:
        return metrics_dict, lazy_metrics_to_compute

    if lazy_metrics_to_compute:
        if log_variable_progress:
            c.info(
                "[deterministic] Metric graph built: "
                f"{len(lazy_metrics_to_compute)} task(s) across {len(variables)} variable(s)."
            )
        lazy_objects = [obj for _, _, obj in lazy_metrics_to_compute]
        if log_variable_progress:
            c.info("[deterministic] Submitting metric graph to dask scheduler...")
        computed_results = list(dask.compute(*lazy_objects))
        if log_variable_progress:
            c.info("[deterministic] Dask metric graph completed.")

        return _finalize_metrics(
            metrics_dict, lazy_metrics_to_compute, computed_results, preserve_dims
        )

    return _finalize_metrics(metrics_dict, [], [], preserve_dims)


def calculate_per_level_metrics(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    calc_relative: bool,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None,
    compute: bool = True,
    weights: xr.DataArray | None = None,
    performance_cfg: dict[str, Any] | None = None,
    log_variable_progress: bool = True,
) -> pd.DataFrame | tuple[dict, list] | None:
    """Compute metrics per pressure level for 3D variables."""
    variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
    if not variables_3d or "level" not in ds_target.dims:
        return None

    ds_t_3d = ds_target[variables_3d]
    ds_p_3d = ds_prediction[variables_3d]

    return calculate_all_metrics(
        ds_t_3d,
        ds_p_3d,
        calc_relative,
        include,
        fss_cfg,
        preserve_dims=["level"],
        compute=compute,
        weights=weights,
        performance_cfg=performance_cfg,
        log_variable_progress=log_variable_progress,
    )


def calculate_multi_lead_metrics_split(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None,
    seeps_climatology_path: str | None,
    weights: xr.DataArray | None,
    split_3d_by_level: bool,
    performance_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute multi-lead deterministic metrics using standard dask execution."""
    # Simplified implementation that avoids manual batching
    preserve_dims = ["lead_time"]
    if split_3d_by_level:
        preserve_dims.append("level")

    return calculate_all_metrics(
        ds_target,
        ds_prediction,
        calc_relative=True,
        include=include,
        fss_cfg=fss_cfg,
        seeps_climatology_path=seeps_climatology_path,
        preserve_dims=preserve_dims,
        compute=True,
        weights=weights,
        performance_cfg=performance_cfg,
        log_variable_progress=False,
    )
