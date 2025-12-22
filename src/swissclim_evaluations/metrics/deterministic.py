from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterable
from pathlib import Path
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

from .. import console as c
from ..dask_utils import (
    calculate_dynamic_chunk_size,
    compute_global_quantile,
    compute_quantile_preserving,
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
        var_needs_quantile = []

        for var in variables:
            # Check if explicit thresholds exist
            if fss_thresholds_per_var and (var in fss_thresholds_per_var):
                fss_thresholds_map[var] = [float(f) for f in fss_thresholds_per_var[var]]
            else:
                # Need quantile
                y_true = ds_target[var]
                # Use custom quantile to avoid memory issues with large arrays
                # If preserve_dims is set (e.g. for per-level metrics), we must compute quantile
                # per slice to avoid aggregating across levels.
                if preserve_dims:
                    q_da = compute_quantile_preserving(y_true, [fss_quantile], preserve_dims)
                else:
                    q_da = compute_global_quantile(y_true, fss_quantile, skipna=True)

                lazy_quantiles.append(q_da)
                var_needs_quantile.append(var)

        if lazy_quantiles:
            # Compute all quantiles in one parallel pass
            computed_quantiles = dask.compute(*lazy_quantiles)
            for var, val in zip(var_needs_quantile, computed_quantiles, strict=False):
                if isinstance(val, xr.DataArray | np.ndarray) and val.ndim > 0:
                    fss_thresholds_map[var] = [val]
                else:
                    fss_thresholds_map[var] = [float(val.item() if hasattr(val, "item") else val)]

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
        # Unzip
        _, _, lazy_objs = zip(*lazy_metrics_to_compute, strict=False)

        # Compute
        computed_results = dask.compute(*lazy_objs)

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
    include = cfg.get("include")
    std_include = cfg.get("standardized_include")
    fss_cfg = cfg.get("fss", {})
    seeps_climatology_path = cfg.get("seeps_climatology_path")
    report_per_level = bool(cfg.get("report_per_level", True))
    chunk_size_cfg = (performance_cfg or {}).get("chunk_size")
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

        # Build all lazy objects on full datasets (not per-variable) for better dask graph sharing
        # This allows dask to optimize across variables and share intermediate computations
        # IMPORTANT: All lazy objects are collected into ONE list and computed in ONE call
        # to allow dask to optimize the entire graph together

        # 1. Regular metrics (all variables at once)
        reg_dict, reg_lazy = ({}, [])
        if not multi_lead:
            reg_dict, reg_lazy = _calculate_all_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=False,
                weights=weights,
            )

        # 2. Standardized metrics (all variables at once)
        std_dict, std_lazy = ({}, [])
        if not multi_lead:
            std_dict, std_lazy = _calculate_all_metrics(
                ds_target_std,
                ds_prediction_std,
                calc_relative=False,
                include=std_include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=False,
                weights=weights,
            )

        # 3. Per-level metrics (all 3D variables at once)
        lvl_dict, lvl_lazy = ({}, [])
        lvl_std_dict, lvl_std_lazy = ({}, [])
        if report_per_level and not multi_lead:
            res = _calculate_per_level_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                compute=False,
                weights=weights,
            )
            if res is not None:
                lvl_dict, lvl_lazy = res

            res_std = _calculate_per_level_metrics(
                ds_target_std,
                ds_prediction_std,
                calc_relative=False,
                include=std_include,
                fss_cfg=fss_cfg,
                compute=False,
                weights=weights,
            )
            if res_std is not None:
                lvl_std_dict, lvl_std_lazy = res_std

        # 4. Multi-lead metrics (all variables at once)
        lead_dict, lead_lazy = ({}, [])
        if multi_lead:
            lead_dict, lead_lazy = _calculate_all_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                preserve_dims=["lead_time"],
                compute=False,
                weights=weights,
            )

        # Heuristic for chunk size: target ~1B points per batch to avoid OOM
        dynamic_chunk = calculate_dynamic_chunk_size(
            config_chunk_size=chunk_size_cfg,
            ds=ds_target,
        )

        # Collect ALL lazy objects into one list for unified dask graph optimization
        # Track the boundaries so we can dispatch results back to categories
        all_lazy_objs: list[Any] = []
        category_boundaries: list[tuple[str, int, int]] = []  # (name, start_idx, end_idx)

        if reg_lazy:
            _, _, lazy_objs = zip(*reg_lazy, strict=False)
            start = len(all_lazy_objs)
            all_lazy_objs.extend(lazy_objs)
            category_boundaries.append(("reg", start, len(all_lazy_objs)))

        if std_lazy:
            _, _, lazy_objs = zip(*std_lazy, strict=False)
            start = len(all_lazy_objs)
            all_lazy_objs.extend(lazy_objs)
            category_boundaries.append(("std", start, len(all_lazy_objs)))

        if lvl_lazy:
            _, _, lazy_objs = zip(*lvl_lazy, strict=False)
            start = len(all_lazy_objs)
            all_lazy_objs.extend(lazy_objs)
            category_boundaries.append(("lvl", start, len(all_lazy_objs)))

        if lvl_std_lazy:
            _, _, lazy_objs = zip(*lvl_std_lazy, strict=False)
            start = len(all_lazy_objs)
            all_lazy_objs.extend(lazy_objs)
            category_boundaries.append(("lvl_std", start, len(all_lazy_objs)))

        if lead_lazy:
            _, _, lazy_objs = zip(*lead_lazy, strict=False)
            start = len(all_lazy_objs)
            all_lazy_objs.extend(lazy_objs)
            category_boundaries.append(("lead", start, len(all_lazy_objs)))

        # Compute ALL lazy objects in ONE call (or chunked for OOM protection)
        # This allows dask to optimize the entire graph together
        all_results: list[Any] = []
        if all_lazy_objs:
            # Use chunk_size for OOM protection, None means compute all at once (fastest)
            chunk_size = dynamic_chunk if dynamic_chunk < 100 else None
            if chunk_size is None or chunk_size >= len(all_lazy_objs):
                all_results = list(dask.compute(*all_lazy_objs))
            else:
                # Chunked computation for OOM protection
                for i in range(0, len(all_lazy_objs), chunk_size):
                    chunk = all_lazy_objs[i : i + chunk_size]
                    all_results.extend(dask.compute(*chunk))

        # Dispatch results back to categories using boundaries
        reg_results: list[Any] = []
        std_results: list[Any] = []
        lvl_results: list[Any] = []
        lvl_std_results: list[Any] = []
        lead_results: list[Any] = []

        for name, start, end in category_boundaries:
            if name == "reg":
                reg_results = all_results[start:end]
            elif name == "std":
                std_results = all_results[start:end]
            elif name == "lvl":
                lvl_results = all_results[start:end]
            elif name == "lvl_std":
                lvl_std_results = all_results[start:end]
            elif name == "lead":
                lead_results = all_results[start:end]

        # Finalize metrics
        regular_metrics = _finalize_metrics(reg_dict, reg_lazy, reg_results, None)
        standardized_metrics = _finalize_metrics(std_dict, std_lazy, std_results, None)

        per_level_metrics = pd.DataFrame()
        if lvl_lazy or lvl_dict:
            per_level_metrics = _finalize_metrics(lvl_dict, lvl_lazy, lvl_results, ["level"])

        per_level_std = pd.DataFrame()
        if lvl_std_lazy or lvl_std_dict:
            per_level_std = _finalize_metrics(
                lvl_std_dict, lvl_std_lazy, lvl_std_results, ["level"]
            )

        df_all_lead = pd.DataFrame()
        if lead_lazy or lead_dict:
            df_all_lead = _finalize_metrics(lead_dict, lead_lazy, lead_results, ["lead_time"])

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
        # Heuristic for chunk size also in members mode
        dynamic_chunk = calculate_dynamic_chunk_size(
            config_chunk_size=chunk_size_cfg,
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
            ds_tgt_m = ds_target.isel(ensemble=mi) if "ensemble" in ds_target.dims else ds_target
            ds_tgt_m_std = (
                ds_target_std.isel(ensemble=mi)
                if "ensemble" in ds_target_std.dims
                else ds_target_std
            )

            # Build all lazy objects on full member datasets (not per-variable)
            # for better dask graph sharing - ALL categories computed in ONE call

            # 1. Regular metrics (all variables at once)
            reg_dict, reg_lazy = _calculate_all_metrics(
                ds_tgt_m,
                ds_pred_m,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=False,
                weights=weights,
            )

            # 2. Standardized metrics (all variables at once)
            std_dict, std_lazy = _calculate_all_metrics(
                ds_tgt_m_std,
                ds_pred_m_std,
                calc_relative=False,
                include=std_include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=False,
                weights=weights,
            )

            # 3. Per-level metrics (all 3D variables at once)
            lvl_dict, lvl_lazy = ({}, [])
            lvl_std_dict, lvl_std_lazy = ({}, [])
            if report_per_level:
                res = _calculate_per_level_metrics(
                    ds_tgt_m,
                    ds_pred_m,
                    calc_relative=True,
                    include=include,
                    fss_cfg=fss_cfg,
                    compute=False,
                    weights=weights,
                )
                if res is not None:
                    lvl_dict, lvl_lazy = res

                res_std = _calculate_per_level_metrics(
                    ds_tgt_m_std,
                    ds_pred_m_std,
                    calc_relative=False,
                    include=std_include,
                    fss_cfg=fss_cfg,
                    compute=False,
                    weights=weights,
                )
                if res_std is not None:
                    lvl_std_dict, lvl_std_lazy = res_std

            # Collect ALL lazy objects into one list for unified dask graph optimization
            m_all_lazy_objs: list = []
            m_category_boundaries = []

            if reg_lazy:
                _, _, lazy_objs = zip(*reg_lazy, strict=False)
                start = len(m_all_lazy_objs)
                m_all_lazy_objs.extend(lazy_objs)
                m_category_boundaries.append(("reg", start, len(m_all_lazy_objs)))

            if std_lazy:
                _, _, lazy_objs = zip(*std_lazy, strict=False)
                start = len(m_all_lazy_objs)
                m_all_lazy_objs.extend(lazy_objs)
                m_category_boundaries.append(("std", start, len(m_all_lazy_objs)))

            if lvl_lazy:
                _, _, lazy_objs = zip(*lvl_lazy, strict=False)
                start = len(m_all_lazy_objs)
                m_all_lazy_objs.extend(lazy_objs)
                m_category_boundaries.append(("lvl", start, len(m_all_lazy_objs)))

            if lvl_std_lazy:
                _, _, lazy_objs = zip(*lvl_std_lazy, strict=False)
                start = len(m_all_lazy_objs)
                m_all_lazy_objs.extend(lazy_objs)
                m_category_boundaries.append(("lvl_std", start, len(m_all_lazy_objs)))

            # Compute ALL lazy objects in ONE call (or chunked for OOM protection)
            m_all_results = []
            if m_all_lazy_objs:
                chunk_size = dynamic_chunk if dynamic_chunk < 100 else None
                if chunk_size is None or chunk_size >= len(m_all_lazy_objs):
                    m_all_results = list(dask.compute(*m_all_lazy_objs))
                else:
                    for i in range(0, len(m_all_lazy_objs), chunk_size):
                        chunk = m_all_lazy_objs[i : i + chunk_size]
                        m_all_results.extend(dask.compute(*chunk))

            # Dispatch results back to categories
            reg_results = []
            std_results = []
            lvl_results = []
            lvl_std_results = []

            for name, start, end in m_category_boundaries:
                if name == "reg":
                    reg_results = m_all_results[start:end]
                elif name == "std":
                    std_results = m_all_results[start:end]
                elif name == "lvl":
                    lvl_results = m_all_results[start:end]
                elif name == "lvl_std":
                    lvl_std_results = m_all_results[start:end]

            # Finalize metrics
            reg_m = _finalize_metrics(reg_dict, reg_lazy, reg_results, None)
            std_m = _finalize_metrics(std_dict, std_lazy, std_results, None)

            per_level_m = None
            if lvl_lazy or lvl_dict:
                per_level_m = _finalize_metrics(lvl_dict, lvl_lazy, lvl_results, ["level"])

            per_level_m_std = None
            if lvl_std_lazy or lvl_std_dict:
                per_level_m_std = _finalize_metrics(
                    lvl_std_dict, lvl_std_lazy, lvl_std_results, ["level"]
                )

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
            df_all = _calculate_all_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                preserve_dims=["lead_time"],
                weights=weights,
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
            cols = ["lead_time_hours", "variable"] + [
                c for c in df_all.columns if c not in ["lead_time_hours", "variable"]
            ]
            long_df = df_all[cols]

            # Prepare wide_df
            melted = long_df.melt(
                id_vars=["lead_time_hours", "variable"], var_name="metric", value_name="value"
            )
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

        # Optional quick line plots over lead_time
        # Always generate one plot per (variable, metric): x=lead_time, y=value
        try:
            import matplotlib.pyplot as _plt

            if "long_df" in locals() and not long_df.empty:
                # Melt to get (variable, metric, value) triplets
                plot_df = long_df.melt(
                    id_vars=["lead_time_hours", "variable"],
                    var_name="metric",
                    value_name="value",
                )

                for v in plot_df["variable"].unique():
                    v_df = plot_df[plot_df["variable"] == v]
                    for m in v_df["metric"].unique():
                        subset = v_df[v_df["metric"] == m].sort_values("lead_time_hours")
                        if subset.empty:
                            continue

                        fig, ax = _plt.subplots(figsize=(10, 6))
                        x = subset["lead_time_hours"].values
                        y = subset["value"].values
                        ax.plot(x, y, marker="o")
                        ax.set_xlabel("Lead Time [h]")

                        # Format y-label with units if applicable
                        units = get_variable_units(ds_target, str(v))
                        display_metric = str(m).replace("_", " ")
                        ylabel = display_metric
                        if units:
                            if m in ["MAE", "RMSE", "Bias"]:
                                ylabel += f" [{units}]"
                            elif m == "MSE":
                                ylabel += f" [{units}$^2$]"

                        ax.set_ylabel(ylabel)
                        display_var = str(v).split(".", 1)[1] if "." in str(v) else str(v)
                        ax.set_title(
                            f"{format_variable_name(display_var)} — {display_metric} vs Lead Time",
                            fontsize=10,
                        )
                        out_png = section_output / build_output_filename(
                            metric="det_line",
                            variable=str(v),
                            level=None,
                            qualifier=str(m).replace(" ", "_"),
                            init_time_range=init_range,
                            lead_time_range=_extract_lead_range(ds_prediction),
                            ensemble=ens_token,
                            ext="png",
                        )
                        _plt.tight_layout()
                        save_figure(fig, out_png, module="deterministic")
                        # Save NPZ and CSV for the line plot
                        out_npz = section_output / build_output_filename(
                            metric="det_line",
                            variable=str(v),
                            level=None,
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
                            module="deterministic",
                        )
                        out_csv = section_output / build_output_filename(
                            metric="det_line",
                            variable=str(v),
                            level=None,
                            qualifier=str(m).replace(" ", "_") + "_by_lead",
                            init_time_range=init_range,
                            lead_time_range=_extract_lead_range(ds_prediction),
                            ensemble=ens_token,
                            ext="csv",
                        )
                        save_dataframe(
                            pd.DataFrame({"lead_time_hours": x, m: y}),
                            out_csv,
                            index=False,
                            module="deterministic",
                        )
        except Exception:
            pass
