from __future__ import annotations

import contextlib
import functools
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scores.categorical import seeps
from scores.continuous import additive_bias, mae, mse, rmse
from scores.continuous.correlation import pearsonr
from scores.functions import create_latitude_weights
from scores.spatial import fss_2d_single_field

from .. import console as c
from ..dask_utils import (
    apply_split_to_dataarray,
    build_variable_level_lead_splits,
    compute_global_quantile,
    compute_jobs,
    compute_quantile_preserving,
    resolve_dynamic_batch_size,
    resolve_module_batching_options,
)
from ..helpers import (
    format_variable_name,
    get_variable_units,
    save_data,
    save_dataframe,
    save_figure,
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
    """Heuristic FSS window size based on grid shape.

    Uses latitude/longitude dims; pick ~10% of the larger dimension, min 1.
    """
    max_spatial_dim = max(int(ds.longitude.size), int(ds.latitude.size))
    ws = max(1, max_spatial_dim // 10)
    return (ws, ws)


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


def _compute_lazy_values_batched(
    lazy_objects: list[Any],
    batch_size: int,
    desc: str,
) -> list[Any]:
    """Compute lazy objects via shared dask batch helper and preserve order."""
    if not lazy_objects:
        return []

    jobs = [{"lazy_obj": obj} for obj in lazy_objects]
    compute_jobs(
        jobs,
        key_map={"lazy_obj": "res"},
        batch_size=max(1, int(batch_size)),
        desc=desc,
    )
    return [job["res"] for job in jobs]


def _calculate_multi_lead_metrics_split(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None,
    seeps_climatology_path: str | None,
    weights: xr.DataArray | None,
    dynamic_batch: int,
    split_3d_by_level: bool,
    split_lead_time: bool,
    split_init_time: bool,
    lead_time_block_size: int | None,
    init_time_block_size: int | None,
    performance_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute multi-lead deterministic metrics with finer-grained splitting.

    Splits work by:
    - variable
    - optional level (for 3D vars)
    - optional lead_time blocks
    - optional init_time blocks
    """
    jobs_meta: list[dict[str, Any]] = []
    split_specs = build_variable_level_lead_splits(
        ds_target,
        split_level=split_3d_by_level,
        split_lead_time=split_lead_time,
        lead_time_block_size=lead_time_block_size,
        split_init_time=split_init_time,
        init_time_block_size=init_time_block_size,
    )
    prepared_groups: list[str] = []
    prepared_seen: set[str] = set()

    for spec in split_specs:
        variable = spec["variable"]
        level_val = spec["level"]
        lead_slice = spec["lead_slice"]
        init_slice = spec.get("init_slice", slice(None))

        group_label = f"{variable}" if level_val is None else f"{variable} level={level_val}"
        if group_label not in prepared_seen:
            prepared_seen.add(group_label)
            prepared_groups.append(group_label)

        da_t_blk = apply_split_to_dataarray(
            ds_target[variable],
            level=level_val,
            lead_slice=lead_slice,
            init_slice=init_slice,
        )
        da_p_blk = apply_split_to_dataarray(
            ds_prediction[variable],
            level=level_val,
            lead_slice=lead_slice,
            init_slice=init_slice,
        )

        ds_t_blk = xr.Dataset({str(variable): da_t_blk})
        ds_p_blk = xr.Dataset({str(variable): da_p_blk})
        _, lead_lazy = _calculate_all_metrics(
            ds_t_blk,
            ds_p_blk,
            calc_relative=True,
            include=include,
            fss_cfg=fss_cfg,
            seeps_climatology_path=seeps_climatology_path,
            preserve_dims=["lead_time"],
            compute=False,
            weights=weights,
            performance_cfg=performance_cfg,
            log_variable_progress=False,
        )

        for _, metric_name, lazy_obj in lead_lazy:
            jobs_meta.append(
                {
                    "variable": str(variable),
                    "level": level_val,
                    "lead_start": int(spec["lead_start"]),
                    "init_start": int(spec.get("init_start", 0)),
                    "init_len": int(spec.get("init_len", 1)),
                    "metric": metric_name,
                    "lazy": lazy_obj,
                }
            )

    if not jobs_meta:
        return pd.DataFrame()

    results: list[Any] = [None] * len(jobs_meta)
    jobs_by_variable_level: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for job_idx, job in enumerate(jobs_meta):
        jobs_by_variable_level[(str(job["variable"]), job.get("level"))].append(job_idx)

    for (var_name, level_val), job_indices in jobs_by_variable_level.items():
        level_suffix = "" if level_val is None else f" level={level_val}"
        desc = f"Computing deterministic metrics variable={var_name}{level_suffix}"
        block_results = _compute_lazy_values_batched(
            [jobs_meta[idx]["lazy"] for idx in job_indices],
            batch_size=dynamic_batch,
            desc=desc,
        )
        for idx, result in zip(job_indices, block_results, strict=False):
            results[idx] = result

    by_var_metric_level: dict[tuple[str, str, Any], list[tuple[int, int, int, Any]]] = defaultdict(
        list
    )
    for job, result in zip(jobs_meta, results, strict=False):
        key = (job["variable"], job["metric"], job["level"])
        by_var_metric_level[key].append(
            (
                int(job["lead_start"]),
                int(job.get("init_start", 0)),
                int(job.get("init_len", 1)),
                result,
            )
        )

    per_var_level_metric: dict[tuple[str, Any], dict[str, xr.DataArray]] = defaultdict(dict)
    for (variable, metric_name, level_val), parts in by_var_metric_level.items():
        by_lead_start: dict[int, list[tuple[int, int, Any]]] = defaultdict(list)
        for lead_start, init_start, init_len, result in parts:
            by_lead_start[int(lead_start)].append((int(init_start), int(init_len), result))

        data_parts: list[Any] = []
        for lead_start in sorted(by_lead_start.keys()):
            init_parts = sorted(by_lead_start[lead_start], key=lambda item: item[0])
            if len(init_parts) == 1:
                data_parts.append(init_parts[0][2])
                continue

            weighted_sum = None
            total_weight = 0
            for _init_start, init_len, result in init_parts:
                weight = max(1, int(init_len))
                weighted_res = result * weight
                weighted_sum = (
                    weighted_res if weighted_sum is None else (weighted_sum + weighted_res)
                )
                total_weight += weight

            if weighted_sum is None:
                continue
            data_parts.append(weighted_sum / max(1, total_weight))

        merged = data_parts[0] if len(data_parts) == 1 else xr.concat(data_parts, dim="lead_time")
        per_var_level_metric[(variable, level_val)][metric_name] = merged

    frames: list[pd.DataFrame] = []
    for (variable, level_val), metric_map in per_var_level_metric.items():
        if metric_map:
            df_var = xr.Dataset(metric_map).to_dataframe().reset_index()
            df_var["variable"] = variable
            if level_val is not None and "level" not in df_var.columns:
                level_py = int(level_val) if hasattr(level_val, "item") else level_val
                df_var["level"] = level_py
            frames.append(df_var)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _calculate_all_metrics(
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
    """Compute scalar deterministic metrics per variable.

    Metrics supported:
      - MAE, RMSE, MSE, Bias, Pearson R, FSS, SEEPS (for precipitation)
      - Relative MAE, Relative L1, Relative L2 (when calc_relative=True)
    """

    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, Any]] = {}

    # ---- FSS configuration
    auto_window_size = _window_size(ds_target)
    fss_quantile = 0.90
    fss_window_size: tuple[int, int] | None = None
    fss_thresholds_per_var: dict[str, list] | None = None
    no_event_value: float = 1.0  # perfect score if both fields have no events

    if isinstance(fss_cfg, dict):
        # quantile: allow 0-100 as percent or 0-1 as fraction
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
        # window size: int -> square; iterable -> first two
        ws = fss_cfg.get("window_size")
        if ws is not None:
            try:
                if isinstance(ws, int):
                    fss_window_size = (max(1, int(ws)), max(1, int(ws)))
                elif isinstance(ws, Iterable) and not isinstance(ws, (str | bytes)):
                    ws_list = list(ws)
                    if len(ws_list) >= 2:
                        fss_window_size = (
                            max(1, int(ws_list[0])),
                            max(1, int(ws_list[1])),
                        )
            except Exception:
                fss_window_size = None
        # thresholds: per-variable mapping only
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

    all_metric_names = {
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
    metrics_to_compute = all_metric_names if include is None else set(include)

    if weights is None and "latitude" in ds_target.dims:
        weights = create_latitude_weights(ds_target.latitude)
        # Fix for floating point errors giving slightly (~-10^-8) negative weights at poles
        weights = weights.clip(min=0.0)

    # --- Pre-compute FSS quantiles if needed (Batch Optimization) ---
    fss_thresholds_map: dict[str, list[Any]] = {}
    if (include is None) or ("FSS" in metrics_to_compute):
        lazy_quantiles = []
        # Store metadata for reconstruction: (variable_name, level_value_or_None)
        quantile_meta: list[tuple[str, Any]] = []

        for var in variables:
            # Check if explicit thresholds exist
            if fss_thresholds_per_var and (var in fss_thresholds_per_var):
                fss_thresholds_map[var] = [float(f) for f in fss_thresholds_per_var[var]]
                continue

            # Need quantile
            y_true = ds_target[var]

            # Check for per-level splitting optimization
            # Only if preserve_dims is active and contains "level"
            should_split_levels = False
            if preserve_dims and "level" in preserve_dims and "level" in y_true.dims:
                should_split_levels = True

            if should_split_levels:
                # Split by level to avoid large graph/memory usage (OOM protection)
                # Compute quantile per-level as separate jobs
                levels = y_true["level"].values
                for lvl in levels:
                    # Select level; result handles Time/Lat/Lon
                    y_slice = y_true.sel(level=lvl)
                    # Compute global quantile for this slice
                    q_lazy = compute_global_quantile(y_slice, fss_quantile, skipna=True)
                    lazy_quantiles.append(q_lazy)
                    quantile_meta.append((var, lvl))
            elif preserve_dims:
                # Fallback for other preserve dims or if level not present
                q_da = compute_quantile_preserving(y_true, [fss_quantile], preserve_dims)
                lazy_quantiles.append(q_da)
                quantile_meta.append((var, None))
            else:
                # Standard global case
                q_da = compute_global_quantile(y_true, fss_quantile, skipna=True)
                lazy_quantiles.append(q_da)
                quantile_meta.append((var, None))

        if lazy_quantiles:
            # Compute all quantiles using batched compute_jobs to avoid OOM
            # Map lazy quantiles to a job structure
            quantile_jobs = [{"q_lazy": q} for q in lazy_quantiles]

            # Determine batch size via shared resolver so manual batch_size is always respected
            dyn_batch = resolve_dynamic_batch_size(performance_cfg, ds=ds_target)

            compute_jobs(
                quantile_jobs,
                key_map={"q_lazy": "q_res"},
                batch_size=dyn_batch,
                desc="Computing FSS quantiles",
            )

            computed_quantiles = [j["q_res"] for j in quantile_jobs]

            # Reconstruct results from flat list back to variable mapping
            from collections import defaultdict

            reconstruction = defaultdict(list)
            for (var, lvl), res in zip(quantile_meta, computed_quantiles, strict=False):
                if lvl is not None:
                    reconstruction[var].append((lvl, res))
                else:
                    reconstruction[var] = res

            for var, data in reconstruction.items():
                if isinstance(data, list):
                    # Reassemble level-split data into a DataArray
                    try:
                        # each item is (level_val, quantile_val)
                        levels_vals = [x[0] for x in data]
                        # extraction heuristic for scalar-like dask/numpy results
                        q_vals = [
                            float(x[1].item() if hasattr(x[1], "item") else x[1]) for x in data
                        ]
                        da = xr.DataArray(q_vals, coords={"level": levels_vals}, dims="level")
                        fss_thresholds_map[var] = [da]
                    except Exception as e:
                        c.print(f"[deterministic] Failed to reassemble quantiles for {var}: {e}")
                        fss_thresholds_map[var] = []
                else:
                    # Single object (DataArray or scalar) from non-split path
                    val = data
                    if isinstance(val, xr.DataArray | np.ndarray) and val.ndim > 0:
                        fss_thresholds_map[var] = [val]
                    else:
                        fss_thresholds_map[var] = [
                            float(val.item() if hasattr(val, "item") else val)
                        ]

    # Store lazy objects to compute in one go
    # List of (variable_name, metric_name, lazy_object)
    lazy_metrics_to_compute: list[tuple[str, str, Any]] = []

    # We also need to store intermediate lazy objects that are used for other metrics
    # e.g. MAE is used for Relative MAE.
    # (variable_name, metric_name) -> lazy_object
    intermediate_lazy: dict[tuple[str, str], Any] = {}

    for var in variables:
        y_true = ds_target[var]
        y_pred = ds_prediction[var]

        # Initialize row in metrics_dict
        if var not in metrics_dict:
            metrics_dict[var] = {}

        # Filter preserve_dims for this variable
        curr_preserve = None
        if preserve_dims:
            curr_preserve = [d for d in preserve_dims if (d in y_true.dims) or (d in y_pred.dims)]
            if not curr_preserve:
                curr_preserve = None

        # Base error metrics
        if (include is None) or ("MAE" in metrics_to_compute):
            if weights is not None:
                val = mae(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
            else:
                val = mae(y_pred, y_true, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "MAE", val))
            intermediate_lazy[(var, "MAE")] = val

        if (include is None) or ("RMSE" in metrics_to_compute):
            if weights is not None:
                val = rmse(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
            else:
                val = rmse(y_pred, y_true, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "RMSE", val))

        if (include is None) or ("MSE" in metrics_to_compute):
            if weights is not None:
                val = mse(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
            else:
                val = mse(y_pred, y_true, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "MSE", val))

        if (include is None) or ("Bias" in metrics_to_compute):
            if weights is not None:
                val = additive_bias(y_pred, y_true, weights=weights, preserve_dims=curr_preserve)
            else:
                val = additive_bias(y_pred, y_true, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "Bias", val))

        # Correlation
        if (include is None) or ("Pearson R" in metrics_to_compute):
            val = pearsonr(y_pred, y_true, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "Pearson R", val))

        # FSS
        if (include is None) or ("FSS" in metrics_to_compute):
            # Use pre-computed thresholds
            list_event_threshold = fss_thresholds_map.get(var, [])
            # If we used quantile, we have one threshold. If explicit, we might have multiple.
            # We need labels.
            if fss_thresholds_per_var and (var in fss_thresholds_per_var):
                list_fss_label = [f"FSS_{et}" for et in list_event_threshold]
            else:
                # Quantile case
                list_fss_label = [f"FSS_{100 * fss_quantile}%"]

            for event_threshold, fss_label in zip(
                list_event_threshold, list_fss_label, strict=False
            ):
                try:
                    spatial_dims = ["latitude", "longitude"]

                    if isinstance(event_threshold, xr.DataArray | xr.Variable):
                        # Wrapper to adapt positional threshold to keyword-only argument
                        def _fss_wrapper(p, t, th, **kwargs):
                            return fss_2d_single_field(p, t, event_threshold=th, **kwargs)

                        out = xr.apply_ufunc(
                            _fss_wrapper,
                            y_pred,
                            y_true,
                            event_threshold,
                            input_core_dims=[
                                spatial_dims,
                                spatial_dims,
                                [],
                            ],
                            kwargs={
                                "window_size": fss_window_size,
                            },
                            vectorize=True,
                            dask="parallelized",
                            output_dtypes=[float],
                        )
                    else:
                        # Pass as kwarg (scalar)
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

                    # Reduce to scalar
                    if isinstance(out, xr.DataArray):
                        try:
                            evt_t = (y_true >= event_threshold).any(dim=spatial_dims)
                            evt_p = (y_pred >= event_threshold).any(dim=spatial_dims)
                            no_evt = (~evt_t) & (~evt_p)
                            out = out.where(~no_evt, other=1.0)
                        except Exception:
                            pass
                        if curr_preserve:
                            reduce_dims = [d for d in out.dims if d not in curr_preserve]
                            fss_scalar = out.mean(dim=reduce_dims, skipna=True)
                        else:
                            fss_scalar = out.mean(skipna=True)
                        lazy_metrics_to_compute.append((var, fss_label, fss_scalar))
                    else:
                        # Eager check for in-memory data (numpy)
                        yt_evt = bool((y_true >= event_threshold).any())
                        yp_evt = bool((y_pred >= event_threshold).any())
                        if (not yt_evt) and (not yp_evt):
                            metrics_dict[var][fss_label] = no_event_value
                        else:
                            metrics_dict[var][fss_label] = float(out)
                except Exception as e:
                    # Surface the error context once while keeping pipeline resilient
                    with contextlib.suppress(Exception):
                        c.print(f"[deterministic:FSS] fss_2d failed for var='{var}': {e!r}")
                    metrics_dict[var][fss_label] = float("nan")

        if "total_precipitation" in var and "SEEPS" in metrics_to_compute:
            if seeps_climatology_path is None:
                raise ValueError(
                    "SEEPS metric requested but 'seeps_climatology_path' not provided in config."
                )
            prob_dry, seeps_threshold = _prepare_seeps(y_true, seeps_climatology_path)
            seeps_val = seeps(
                y_pred * 1000, y_true * 1000, prob_dry, seeps_threshold, dry_light_threshold=0.25
            )  # convert to mm with *1000
            # SEEPS returns a lazy object usually
            if curr_preserve:
                reduce_dims = [d for d in seeps_val.dims if d not in curr_preserve]
                seeps_val = seeps_val.mean(dim=reduce_dims, skipna=True)
            else:
                seeps_val = seeps_val.mean(skipna=True)
            lazy_metrics_to_compute.append((var, "SEEPS", seeps_val))

        # Relative metrics
        if calc_relative:
            # Denominator norms on the target
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

            # Mean-based relative MAE (keep for continuity and interpretability)
            if (include is None) or ("Relative MAE" in metrics_to_compute):
                # We need MAE val. If we computed it lazily, we use the lazy object.
                # If not (e.g. not requested), we compute it now lazily.
                if (var, "MAE") in intermediate_lazy:
                    mae_val_lazy = intermediate_lazy[(var, "MAE")]
                else:
                    if weights is not None:
                        mae_val_lazy = mae(
                            y_pred, y_true, weights=weights, preserve_dims=curr_preserve
                        )
                    else:
                        mae_val_lazy = mae(y_pred, y_true, preserve_dims=curr_preserve)

                rel_mae = mae_val_lazy / mean_abs
                lazy_metrics_to_compute.append((var, "Relative MAE", rel_mae))

            # True norm-based relative errors: ||e||1 / ||y||1 and ||e||2 / ||y||2
            err = y_pred - y_true
            if curr_preserve:
                reduce_dims = [d for d in err.dims if d not in curr_preserve]
                l1_err = np.abs(err).sum(dim=reduce_dims, skipna=True)
                l2_err = (err**2).sum(dim=reduce_dims, skipna=True) ** 0.5
            else:
                l1_err = np.abs(err).sum(skipna=True)
                l2_err = (err**2).sum(skipna=True) ** 0.5

            # We can't easily check for zero division lazily without map_blocks or just letting
            # it be inf/nan. But we can compute the ratio and handle nan/inf later or rely on
            # xarray handling.
            if (include is None) or ("Relative L1" in metrics_to_compute):
                rel_l1 = l1_err / l1_norm
                lazy_metrics_to_compute.append((var, "Relative L1", rel_l1))

            if (include is None) or ("Relative L2" in metrics_to_compute):
                rel_l2 = l2_err / l2_norm
                lazy_metrics_to_compute.append((var, "Relative L2", rel_l2))

    # Compute all lazy metrics at once
    if not compute:
        return metrics_dict, lazy_metrics_to_compute

    if lazy_metrics_to_compute:
        # Use compute_jobs to batch computation and avoid OOM
        metrics_jobs = [{"lazy": obj} for _, _, obj in lazy_metrics_to_compute]

        dyn_batch = resolve_dynamic_batch_size(performance_cfg, ds=ds_target)
        var_list = ",".join(str(v) for v in variables)
        desc = f"Computing deterministic metrics variables={var_list}"

        compute_jobs(
            metrics_jobs,
            key_map={"lazy": "res"},
            batch_size=dyn_batch,
            desc=desc,
        )

        computed_results = [j["res"] for j in metrics_jobs]

        return _finalize_metrics(
            metrics_dict, lazy_metrics_to_compute, computed_results, preserve_dims
        )

    return _finalize_metrics(metrics_dict, [], [], preserve_dims)


def _calculate_per_level_metrics(
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
    if not variables_3d:
        return None

    if "level" not in ds_target.dims:
        return None

    # Optimized implementation using preserve_dims
    ds_t_3d = ds_target[variables_3d]
    ds_p_3d = ds_prediction[variables_3d]

    return _calculate_all_metrics(
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


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    ds_target_std: xr.Dataset,
    ds_prediction_std: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any] | None,
    metrics_cfg: dict[str, Any] | None,
    ensemble_mode: str | None = None,
    lead_policy: Any | None = None,
    performance_cfg: dict[str, Any] | None = None,
) -> None:
    """Compute and write deterministic metrics CSVs.

    Produces two CSV files by default under out_root/deterministic:
      - deterministic_metrics_[...].csv (relative metrics included)
      - deterministic_metrics_standardized_[...].csv (on standardized fields)

    When ensemble_mode='members', writes per-member CSVs (ens0..ensN) plus an
    aggregated 'members_mean' CSV if aggregate_members_mean=True.
    """
    cfg = (metrics_cfg or {}).get("deterministic", {})
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    include = cfg.get("include")
    std_include = cfg.get("standardized_include")
    fss_cfg = cfg.get("fss", {})
    seeps_climatology_path = cfg.get("seeps_climatology_path")
    report_per_level = bool(cfg.get("report_per_level", True))
    perf_cfg = performance_cfg or {}
    batch_opts = resolve_module_batching_options(
        performance_cfg=perf_cfg,
        default_split_level=True,
        default_split_lead_time=True,
        default_split_init_time=True,
        default_lead_time_block_size=6,
        default_init_time_block_size=6,
    )
    split_3d_by_level = bool(batch_opts["split_level"])
    split_lead_time = bool(batch_opts["split_lead_time"])
    split_init_time = bool(batch_opts["split_init_time"])
    lead_time_block_size = int(batch_opts["lead_time_block_size"])
    init_time_block_size = int(batch_opts["init_time_block_size"])
    reduce_ens_mean = True
    try:
        rem = cfg.get("reduce_ensemble_mean")
        if rem is not None:
            reduce_ens_mean = bool(rem)
    except Exception:
        reduce_ens_mean = True
    aggregate_members_mean = bool(cfg.get("aggregate_members_mean", True))

    # Track whether an ensemble dimension was present originally
    had_ensemble_dim = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)

    from ..helpers import (
        build_output_filename,
        ensemble_mode_to_token,
        format_init_time_range,
        resolve_ensemble_mode,
    )

    resolved_mode = resolve_ensemble_mode("deterministic", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for deterministic metrics")
    members_indices: list[int] | None = None
    if resolved_mode == "members" and has_ens:
        members_indices = list(range(int(ds_prediction.sizes["ensemble"])))

    # Reduce ensemble by mean if requested
    if resolved_mode == "mean" and has_ens and reduce_ens_mean:
        if "ensemble" in ds_prediction.dims:
            ds_prediction = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_prediction_std.dims:
            ds_prediction_std = ds_prediction_std.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target_std.dims:
            ds_target_std = ds_target_std.mean(dim="ensemble", keep_attrs=True)

    # Compute metrics
    section_output = out_root / "deterministic"

    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds.coords and "init_time" not in ds.dims:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            return format_init_time_range(vals)
        except Exception:
            return None

    def _extract_lead_range(ds: xr.Dataset):
        if "lead_time" not in ds.coords and "lead_time" not in ds.dims:
            return None
        try:
            vals = ds["lead_time"].values
            if vals.size == 0:
                return None
            hours = (vals / np.timedelta64(1, "h")).astype(int)
            start_h = int(hours.min())
            end_h = int(hours.max())

            def _fmt_lead(h: int) -> str:
                return f"{h:03d}h"

            return (_fmt_lead(start_h), _fmt_lead(end_h))
        except Exception:
            return None

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    section_output.mkdir(parents=True, exist_ok=True)

    # Choose ensemble token for aggregate files
    ens_token: str | None = None
    if resolved_mode == "mean" and had_ensemble_dim and reduce_ens_mean:
        ens_token = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled" and had_ensemble_dim:
        ens_token = ensemble_mode_to_token("pooled")
    elif resolved_mode == "members" and had_ensemble_dim:
        ens_token = None  # per-member handled below

    df_all_lead = pd.DataFrame()

    # Compute weights once if needed
    weights = None
    if "latitude" in ds_target.dims:
        weights = create_latitude_weights(ds_target.latitude)
        # Fix for floating point errors giving slightly (~-10^-8) negative weights at poles
        weights = weights.clip(min=0.0)

    if members_indices is None:
        # Check multi_lead early
        try:
            multi_lead = (
                (lead_policy is not None)
                and ("lead_time" in ds_prediction.dims)
                and int(ds_prediction.sizes.get("lead_time", 0)) > 1
                and getattr(lead_policy, "mode", "first") != "first"
            )
        except Exception:
            multi_lead = False

        # Heuristic batch size used by split-based multi-lead path
        dynamic_batch = resolve_dynamic_batch_size(
            perf_cfg,
            ds=ds_target,
        )

        regular_metrics = pd.DataFrame()
        standardized_metrics = pd.DataFrame()
        per_level_metrics = pd.DataFrame()
        per_level_std = pd.DataFrame()

        if not multi_lead:
            regular_metrics = _calculate_all_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=True,
                weights=weights,
                performance_cfg=perf_cfg,
                log_variable_progress=True,
            )
            standardized_metrics = _calculate_all_metrics(
                ds_target_std,
                ds_prediction_std,
                calc_relative=False,
                include=std_include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=True,
                weights=weights,
                performance_cfg=perf_cfg,
                log_variable_progress=False,
            )

            if report_per_level:
                res_lvl = _calculate_per_level_metrics(
                    ds_target,
                    ds_prediction,
                    calc_relative=True,
                    include=include,
                    fss_cfg=fss_cfg,
                    compute=True,
                    weights=weights,
                    performance_cfg=perf_cfg,
                    log_variable_progress=False,
                )
                if isinstance(res_lvl, pd.DataFrame):
                    per_level_metrics = res_lvl

                res_lvl_std = _calculate_per_level_metrics(
                    ds_target_std,
                    ds_prediction_std,
                    calc_relative=False,
                    include=std_include,
                    fss_cfg=fss_cfg,
                    compute=True,
                    weights=weights,
                    performance_cfg=perf_cfg,
                    log_variable_progress=False,
                )
                if isinstance(res_lvl_std, pd.DataFrame):
                    per_level_std = res_lvl_std

        df_all_lead = pd.DataFrame()
        if multi_lead:
            df_all_lead = _calculate_multi_lead_metrics_split(
                ds_target=ds_target,
                ds_prediction=ds_prediction,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                weights=weights,
                dynamic_batch=dynamic_batch,
                split_3d_by_level=split_3d_by_level,
                split_lead_time=split_lead_time,
                split_init_time=split_init_time,
                lead_time_block_size=lead_time_block_size,
                init_time_block_size=init_time_block_size,
                performance_cfg=perf_cfg,
            )

        # Ensure level column is int if present
        if not per_level_metrics.empty and "level" in per_level_metrics.columns:
            per_level_metrics["level"] = per_level_metrics["level"].astype(int)
        if not per_level_std.empty and "level" in per_level_std.columns:
            per_level_std["level"] = per_level_std["level"].astype(int)

        # Save outputs
        out_csv = section_output / build_output_filename(
            metric="deterministic_metrics",
            variable=None,
            level=None,
            qualifier="averaged" if (init_range or lead_range) else None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        out_csv_std = section_output / build_output_filename(
            metric="deterministic_metrics",
            variable=None,
            level=None,
            qualifier="standardized",
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        if not regular_metrics.empty:
            save_dataframe(regular_metrics, out_csv, index_label="variable", module="deterministic")
        if not standardized_metrics.empty:
            save_dataframe(
                standardized_metrics, out_csv_std, index_label="variable", module="deterministic"
            )

        if per_level_metrics is not None and not per_level_metrics.empty:
            out_csv_lvl = section_output / build_output_filename(
                metric="deterministic_metrics",
                variable=None,
                level=None,
                qualifier="per_level",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(per_level_metrics, out_csv_lvl, index=False, module="deterministic")

        if per_level_std is not None and not per_level_std.empty:
            out_csv_lvl_std = section_output / build_output_filename(
                metric="deterministic_metrics",
                variable=None,
                level=None,
                qualifier="standardized_per_level",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(per_level_std, out_csv_lvl_std, index=False, module="deterministic")

        # Console summary
        try:
            if not regular_metrics.empty:
                c.print("Deterministic metrics (targets vs predictions) — first 5 rows:")
                c.print(regular_metrics.head())
        except Exception:
            pass
        try:
            if not standardized_metrics.empty:
                c.print(
                    "Deterministic standardized metrics (targets vs predictions) — first 5 rows:"
                )
                c.print(standardized_metrics.head())
        except Exception:
            pass
    else:
        # Per-member metrics
        from ..helpers import aggregate_member_dfs, build_output_filename, ensemble_mode_to_token

        pooled_metrics: list[pd.DataFrame] = []
        first_reg_df: pd.DataFrame | None = None
        first_std_df: pd.DataFrame | None = None
        # Heuristic for batch size also in members mode
        dynamic_batch = resolve_dynamic_batch_size(
            perf_cfg,
            ds=ds_target,
        )

        for mi in members_indices:
            ds_pred_m = ds_prediction
            if "ensemble" in ds_prediction.dims:
                ds_pred_m = ds_prediction.isel(ensemble=mi)
            ds_pred_m_std = (
                ds_prediction_std.isel(ensemble=mi)
                if "ensemble" in ds_prediction_std.dims
                else ds_prediction_std
            )

            if "ensemble" in ds_target.dims:
                if ds_target.sizes["ensemble"] == 1:
                    ds_tgt_m = ds_target.isel(ensemble=0)
                else:
                    ds_tgt_m = ds_target.isel(ensemble=mi)
            else:
                ds_tgt_m = ds_target

            if "ensemble" in ds_target_std.dims:
                if ds_target_std.sizes["ensemble"] == 1:
                    ds_tgt_m_std = ds_target_std.isel(ensemble=0)
                else:
                    ds_tgt_m_std = ds_target_std.isel(ensemble=mi)
            else:
                ds_tgt_m_std = ds_target_std

            # 1. Regular metrics
            reg_m = _calculate_all_metrics(
                ds_tgt_m,
                ds_pred_m,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=True,
                weights=weights,
                performance_cfg=perf_cfg,
                log_variable_progress=True,
            )

            # 2. Standardized metrics
            std_m = _calculate_all_metrics(
                ds_tgt_m_std,
                ds_pred_m_std,
                calc_relative=False,
                include=std_include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=True,
                weights=weights,
                performance_cfg=perf_cfg,
                log_variable_progress=False,
            )

            if not isinstance(reg_m, pd.DataFrame) or not isinstance(std_m, pd.DataFrame):
                raise TypeError("Expected DataFrame metrics when compute=True")

            # 3. Per-level metrics
            per_level_m = None
            per_level_m_std = None
            if report_per_level:
                res = _calculate_per_level_metrics(
                    ds_tgt_m,
                    ds_pred_m,
                    calc_relative=True,
                    include=include,
                    fss_cfg=fss_cfg,
                    compute=True,
                    weights=weights,
                    performance_cfg=perf_cfg,
                    log_variable_progress=False,
                )
                if isinstance(res, pd.DataFrame):
                    per_level_m = res

                res_std = _calculate_per_level_metrics(
                    ds_tgt_m_std,
                    ds_pred_m_std,
                    calc_relative=False,
                    include=std_include,
                    fss_cfg=fss_cfg,
                    compute=True,
                    weights=weights,
                    performance_cfg=perf_cfg,
                    log_variable_progress=False,
                )
                if isinstance(res_std, pd.DataFrame):
                    per_level_m_std = res_std

            if first_reg_df is None:
                first_reg_df = reg_m.copy()
            if first_std_df is None:
                first_std_df = std_m.copy()
            token_m = ensemble_mode_to_token("members", mi)
            out_csv_m = section_output / build_output_filename(
                metric="deterministic_metrics",
                variable=None,
                level=None,
                qualifier="averaged" if (init_range or lead_range) else None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=token_m,
                ext="csv",
            )
            out_csv_m_std = section_output / build_output_filename(
                metric="deterministic_metrics",
                variable=None,
                level=None,
                qualifier="standardized",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=token_m,
                ext="csv",
            )
            save_dataframe(reg_m, out_csv_m, index_label="variable", module="deterministic")
            save_dataframe(std_m, out_csv_m_std, index_label="variable", module="deterministic")
            pooled_metrics.append(reg_m)

            if per_level_m is not None and not per_level_m.empty:
                out_csv_m_lvl = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    variable=None,
                    level=None,
                    qualifier="per_level",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=token_m,
                    ext="csv",
                )
                save_dataframe(per_level_m, out_csv_m_lvl, index=False, module="deterministic")

            if per_level_m_std is not None and not per_level_m_std.empty:
                out_csv_m_lvl_std = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    variable=None,
                    level=None,
                    qualifier="standardized_per_level",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=token_m,
                    ext="csv",
                )
                save_dataframe(
                    per_level_m_std, out_csv_m_lvl_std, index=False, module="deterministic"
                )

        # Aggregate pooled metrics across members if requested
        if pooled_metrics and aggregate_members_mean:
            pooled_df = aggregate_member_dfs(pooled_metrics)
            if not pooled_df.empty:
                out_csv_pool = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    variable=None,
                    level=None,
                    qualifier="members_mean",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble="enspooled",
                    ext="csv",
                )
                save_dataframe(
                    pooled_df, out_csv_pool, index_label="variable", module="deterministic"
                )

        # Console previews for members mode
        try:
            df_show = (
                pooled_df
                if ("pooled_df" in locals() and pooled_df is not None and not pooled_df.empty)
                else first_reg_df
            )
            if df_show is not None and not df_show.empty:
                c.print("Deterministic metrics (targets vs predictions) — first 5 rows:")
                c.print(df_show.head())
        except Exception:
            pass
        try:
            if first_std_df is not None and not first_std_df.empty:
                c.print(
                    "Deterministic standardized metrics (targets vs predictions) — first 5 rows:"
                )
                c.print(first_std_df.head())
        except Exception:
            pass

    # Optional per-lead artifacts when a multi-lead policy is provided
    # Note: multi_lead check was already done above for members_indices is None case.
    # We re-evaluate it here for safety or reuse the variable if in scope.
    try:
        multi_lead_check = ("lead_time" in ds_prediction.dims) and int(
            ds_prediction.sizes.get("lead_time", 0)
        ) > 1
    except Exception:
        multi_lead_check = False

    if members_indices is None and multi_lead_check:
        # Use pre-computed df_all_lead if available, otherwise compute (fallback)
        if "df_all_lead" in locals() and not df_all_lead.empty:
            df_all = df_all_lead
        else:
            dynamic_batch_fallback = resolve_dynamic_batch_size(
                perf_cfg,
                ds=ds_target,
            )
            df_all = _calculate_multi_lead_metrics_split(
                ds_target=ds_target,
                ds_prediction=ds_prediction,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                weights=weights,
                dynamic_batch=dynamic_batch_fallback,
                split_3d_by_level=split_3d_by_level,
                split_lead_time=split_lead_time,
                split_init_time=split_init_time,
                lead_time_block_size=lead_time_block_size,
                init_time_block_size=init_time_block_size,
                performance_cfg=perf_cfg,
            )

        wide_df = pd.DataFrame()

        if not df_all.empty and "lead_time" in df_all.columns:
            # Convert lead_time to hours
            def _to_hours(val):
                try:
                    return int(val / np.timedelta64(1, "h"))
                except Exception:
                    try:
                        return int(val)
                    except Exception:
                        return val

            df_all["lead_time_hours"] = df_all["lead_time"].apply(_to_hours)
            df_all = df_all.drop(columns=["lead_time"])

            # Prepare long_df
            id_cols = ["lead_time_hours", "variable"]
            if "level" in df_all.columns:
                id_cols.append("level")
            cols = id_cols + [c for c in df_all.columns if c not in id_cols]
            long_df = df_all[cols]

            # Prepare wide_df
            melted = long_df.melt(id_vars=id_cols, var_name="metric", value_name="value")
            if "level" in melted.columns:
                level_token = (
                    pd.to_numeric(melted["level"], errors="coerce")
                    .astype("Int64")
                    .astype(str)
                    .replace("<NA>", "")
                )
                level_suffix = np.where(level_token != "", "_" + level_token, "")
                melted["key"] = (
                    melted["variable"].astype(str)
                    + level_suffix
                    + "_"
                    + melted["metric"].astype(str)
                )
            else:
                melted["key"] = melted["variable"].astype(str) + "_" + melted["metric"].astype(str)
            wide_df = melted.pivot(
                index="lead_time_hours", columns="key", values="value"
            ).reset_index()
            wide_df.columns.name = None

            # Save long_df
            try:
                out_long = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    variable=None,
                    level=None,
                    qualifier="by_lead_long",
                    init_time_range=init_range,
                    lead_time_range=_extract_lead_range(ds_prediction),
                    ensemble=ens_token,
                    ext="csv",
                )
                save_dataframe(long_df, out_long, index=False)
            except Exception as e:
                c.print(f"[deterministic] Failed to save {out_long}: {e}")

            # Save wide_df
            try:
                out_wide = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    variable=None,
                    level=None,
                    qualifier="by_lead_wide",
                    init_time_range=init_range,
                    lead_time_range=_extract_lead_range(ds_prediction),
                    ensemble=ens_token,
                    ext="csv",
                )
                save_dataframe(wide_df, out_wide, index=False, module="deterministic")

                # Print wide_df head to console if available
                if not wide_df.empty:
                    c.print("Deterministic metrics by lead time (first 5 rows):")
                    c.print(wide_df.head())
            except Exception as e:
                c.print(f"[deterministic] Failed to save {out_wide}: {e}")

        # Optional quick line plots / NPZ over lead_time
        # CSV-by-lead remains enabled for all modes; figure/NPZ follow output_mode.
        try:
            import matplotlib.pyplot as _plt

            if "long_df" in locals() and not long_df.empty:
                # Melt to get (variable, metric, value) triplets
                plot_id_vars = ["lead_time_hours", "variable"]
                if "level" in long_df.columns:
                    plot_id_vars.append("level")
                plot_df = long_df.melt(id_vars=plot_id_vars, var_name="metric", value_name="value")

                for v in plot_df["variable"].unique():
                    v_df = plot_df[plot_df["variable"] == v]
                    level_values: list[int | None] = [None]
                    if "level" in v_df.columns:
                        level_values = [
                            int(level_item)
                            for level_item in sorted(v_df["level"].dropna().unique().tolist())
                        ]
                        if v_df["level"].isna().any():
                            level_values.insert(0, None)
                        if not level_values:
                            level_values = [None]

                    for level_val in level_values:
                        v_lvl = v_df if level_val is None else v_df[v_df["level"] == level_val]
                        for m in v_lvl["metric"].unique():
                            subset = v_lvl[v_lvl["metric"] == m].sort_values("lead_time_hours")
                            if subset.empty:
                                continue

                            x = subset["lead_time_hours"].values
                            y = subset["value"].values
                            display_metric = str(m).replace("_", " ")
                            if save_fig:
                                fig, ax = _plt.subplots(figsize=(10, 6))
                                ax.plot(x, y, marker="o")
                                ax.set_xlabel("Lead Time [h]")

                                units = get_variable_units(ds_target, str(v))
                                ylabel = display_metric
                                if units:
                                    if m in ["MAE", "RMSE", "Bias"]:
                                        ylabel += f" [{units}]"
                                    elif m == "MSE":
                                        ylabel += f" [{units}$^2$]"

                                ax.set_ylabel(ylabel)
                                display_var = str(v).split(".", 1)[1] if "." in str(v) else str(v)
                                lvl_str = f" @ {level_val}" if level_val is not None else ""
                                ax.set_title(
                                    f"{format_variable_name(display_var)}{lvl_str} — "
                                    f"{display_metric} vs Lead Time",
                                    fontsize=10,
                                )
                                out_png = section_output / build_output_filename(
                                    metric="det_line",
                                    variable=str(v),
                                    level=level_val,
                                    qualifier=str(m).replace(" ", "_"),
                                    init_time_range=init_range,
                                    lead_time_range=_extract_lead_range(ds_prediction),
                                    ensemble=ens_token,
                                    ext="png",
                                )
                                _plt.tight_layout()
                                save_figure(fig, out_png, module="deterministic")
                                _plt.close(fig)

                            if save_npz:
                                out_npz = section_output / build_output_filename(
                                    metric="det_line",
                                    variable=str(v),
                                    level=level_val,
                                    qualifier=str(m).replace(" ", "_") + "_data",
                                    init_time_range=init_range,
                                    lead_time_range=_extract_lead_range(ds_prediction),
                                    ensemble=ens_token,
                                    ext="npz",
                                )
                                save_data(
                                    out_npz,
                                    lead_hours=x.astype(float),
                                    values=y.astype(float),
                                    metric=str(m),
                                    variable=str(v),
                                    level=level_val,
                                    module="deterministic",
                                )
                            out_csv = section_output / build_output_filename(
                                metric="det_line",
                                variable=str(v),
                                level=level_val,
                                qualifier=str(m).replace(" ", "_") + "_by_lead",
                                init_time_range=init_range,
                                lead_time_range=_extract_lead_range(ds_prediction),
                                ensemble=ens_token,
                                ext="csv",
                            )
                            df_out = pd.DataFrame({"lead_time_hours": x, m: y})
                            if level_val is not None:
                                df_out.insert(1, "level", int(level_val))
                            save_dataframe(df_out, out_csv, index=False, module="deterministic")
        except Exception:
            pass
