from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import dask
import numpy as np
import pandas as pd
import xarray as xr
from scores.categorical import seeps
from scores.continuous import additive_bias, mae, mse, rmse
from scores.continuous.correlation import pearsonr
from scores.spatial import fss_2d_single_field
from skimage.metrics import structural_similarity as ssim

from ..aggregations import latitude_weights
from ..helpers import save_data, save_dataframe, save_figure


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


def _calculate_ssim(
    da_target: xr.DataArray,
    da_pred: xr.DataArray,
    compute: bool = True,
    preserve_dims: list[str] | None = None,
) -> float | Any:
    if "latitude" not in da_target.dims or "longitude" not in da_target.dims:
        return np.nan

    def _ssim_2d(p, t):
        dr = t.max() - t.min()
        if dr == 0:
            return 1.0 if np.allclose(p, t) else 0.0
        return ssim(t, p, data_range=dr)

    try:
        res = xr.apply_ufunc(
            _ssim_2d,
            da_pred,
            da_target,
            input_core_dims=[["latitude", "longitude"], ["latitude", "longitude"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        if preserve_dims:
            reduce_dims = [d for d in res.dims if d not in preserve_dims]
            mean_res = res.mean(dim=reduce_dims)
        else:
            mean_res = res.mean()

        if compute:
            if preserve_dims:
                return mean_res.compute()
            return float(mean_res.compute().item())
        return mean_res
    except Exception:
        return np.nan


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
                print(f"[deterministic] Failed to convert metrics for {var} to DataFrame: {e}")
                continue

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def _calculate_all_metrics(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    calc_relative: bool,
    n_points: int,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None = None,
    seeps_climatology_path: str | None = None,
    preserve_dims: list[str] | None = None,
    compute: bool = True,
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
        "SSIM",
    }
    metrics_to_compute = all_metric_names if include is None else set(include)

    weights = None
    weights = latitude_weights(ds_target.latitude)

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

        # SSIM
        if (include is None) or ("SSIM" in metrics_to_compute):
            # SSIM implementation here calls compute() internally, so we can't easily lazy-batch it
            # without refactoring _calculate_ssim. We leave it as is for now.
            val = _calculate_ssim(y_true, y_pred, compute=False, preserve_dims=curr_preserve)
            lazy_metrics_to_compute.append((var, "SSIM", val))

        # FSS
        if (include is None) or ("FSS" in metrics_to_compute):
            # Determine event threshold: explicit per-variable threshold wins, else quantile
            if fss_thresholds_per_var and (var in fss_thresholds_per_var):
                list_event_threshold = [float(f) for f in fss_thresholds_per_var[var]]
                list_fss_label = [f"FSS_{et}" for et in list_event_threshold]
            else:
                q_da = y_true.quantile(fss_quantile, skipna=True)
                list_event_threshold = [float(q_da.compute().item())]
                list_fss_label = [f"FSS_{100 * fss_quantile}%"]
            for event_threshold, fss_label in zip(
                list_event_threshold, list_fss_label, strict=False
            ):
                try:
                    # Early exit: if both fields have no events anywhere → perfect score
                    yt_evt = bool((y_true >= event_threshold).any().compute().item())
                    yp_evt = bool((y_pred >= event_threshold).any().compute().item())
                    if (not yt_evt) and (not yp_evt):
                        metrics_dict[var][fss_label] = no_event_value
                    else:
                        spatial_dims = ["latitude", "longitude"]

                        # Proper call to fss_2d with config-driven arguments only
                        # We use fss_2d_single_field via apply_ufunc directly to control dask
                        # behavior and avoid inference issues with small chunks by providing
                        # output_dtypes.
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
                                evt_t = (y_true >= event_threshold).any(
                                    dim=spatial_dims, skipna=True
                                )
                                evt_p = (y_pred >= event_threshold).any(
                                    dim=spatial_dims, skipna=True
                                )
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
                            metrics_dict[var][fss_label] = float(out)
                except Exception as e:
                    # Surface the error context once while keeping pipeline resilient
                    with contextlib.suppress(Exception):
                        print(f"[deterministic:FSS] fss_2d failed for var='{var}': {e!r}")
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
    n_points: int,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None,
    compute: bool = True,
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
        n_points,
        include,
        fss_cfg,
        preserve_dims=["level"],
        compute=compute,
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
    n_points = int(sum(ds_target[v].size for v in ds_target.data_vars))
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

    if members_indices is None:
        # Check multi_lead early to include in batch compute
        try:
            multi_lead = (
                (lead_policy is not None)
                and ("lead_time" in ds_prediction.dims)
                and int(ds_prediction.sizes.get("lead_time", 0)) > 1
                and getattr(lead_policy, "mode", "first") != "first"
            )
        except Exception:
            multi_lead = False

        # 1. Regular metrics
        reg_dict, reg_lazy = _calculate_all_metrics(
            ds_target,
            ds_prediction,
            calc_relative=True,
            n_points=n_points,
            include=include,
            fss_cfg=fss_cfg,
            seeps_climatology_path=seeps_climatology_path,
            compute=False,
        )

        # 2. Standardized metrics
        std_dict, std_lazy = _calculate_all_metrics(
            ds_target_std,
            ds_prediction_std,
            calc_relative=False,
            n_points=n_points,
            include=std_include,
            fss_cfg=fss_cfg,
            seeps_climatology_path=seeps_climatology_path,
            compute=False,
        )

        # 3. Per-level metrics
        lvl_dict, lvl_lazy = ({}, [])
        lvl_std_dict, lvl_std_lazy = ({}, [])

        if report_per_level:
            res = _calculate_per_level_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                n_points=n_points,
                include=include,
                fss_cfg=fss_cfg,
                compute=False,
            )
            if res is not None:
                lvl_dict, lvl_lazy = res

            res_std = _calculate_per_level_metrics(
                ds_target_std,
                ds_prediction_std,
                calc_relative=False,
                n_points=n_points,
                include=std_include,
                fss_cfg=fss_cfg,
                compute=False,
            )
            if res_std is not None:
                lvl_std_dict, lvl_std_lazy = res_std

        # 4. Multi-lead metrics
        lead_dict, lead_lazy = ({}, [])
        if multi_lead:
            lead_dict, lead_lazy = _calculate_all_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                n_points=n_points,
                include=include,
                fss_cfg=fss_cfg,
                preserve_dims=["lead_time"],
                compute=False,
            )

        # Combine and compute all
        all_lazy = reg_lazy + std_lazy + lvl_lazy + lvl_std_lazy + lead_lazy

        if all_lazy:
            _, _, lazy_objs = zip(*all_lazy, strict=False)
            results = dask.compute(*lazy_objs)

            # Distribute results
            idx_reg = len(reg_lazy)
            idx_std = idx_reg + len(std_lazy)
            idx_lvl = idx_std + len(lvl_lazy)
            idx_lvl_std = idx_lvl + len(lvl_std_lazy)

            res_reg = results[:idx_reg]
            res_std = results[idx_reg:idx_std]
            res_lvl = results[idx_std:idx_lvl]
            res_lvl_std = results[idx_lvl:idx_lvl_std]
            res_lead = results[idx_lvl_std:]

            regular_metrics = _finalize_metrics(reg_dict, reg_lazy, res_reg, None)
            standardized_metrics = _finalize_metrics(std_dict, std_lazy, res_std, None)

            per_level_metrics = None
            if lvl_lazy or lvl_dict:
                per_level_metrics = _finalize_metrics(lvl_dict, lvl_lazy, res_lvl, ["level"])
                if (
                    per_level_metrics is not None
                    and not per_level_metrics.empty
                    and "level" in per_level_metrics.columns
                ):
                    per_level_metrics["level"] = per_level_metrics["level"].astype(int)

            per_level_std = None
            if lvl_std_lazy or lvl_std_dict:
                per_level_std = _finalize_metrics(
                    lvl_std_dict, lvl_std_lazy, res_lvl_std, ["level"]
                )
                if (
                    per_level_std is not None
                    and not per_level_std.empty
                    and "level" in per_level_std.columns
                ):
                    per_level_std["level"] = per_level_std["level"].astype(int)

            if lead_lazy or lead_dict:
                df_all_lead = _finalize_metrics(lead_dict, lead_lazy, res_lead, ["lead_time"])
        else:
            regular_metrics = pd.DataFrame()
            standardized_metrics = pd.DataFrame()
            per_level_metrics = None
            per_level_std = None

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
        save_dataframe(regular_metrics, out_csv, index_label="variable")
        save_dataframe(standardized_metrics, out_csv_std, index_label="variable")

        if per_level_metrics is not None:
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
            save_dataframe(per_level_metrics, out_csv_lvl, index=False)

        if per_level_std is not None:
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
            save_dataframe(per_level_std, out_csv_lvl_std, index=False)

        # Console summary
        try:
            print("Deterministic metrics (targets vs predictions) — first 5 rows:")
            print(regular_metrics.head())
        except Exception:
            pass
        try:
            print("Deterministic standardized metrics (targets vs predictions) — first 5 rows:")
            print(standardized_metrics.head())
        except Exception:
            pass
    else:
        # Per-member metrics
        from ..helpers import aggregate_member_dfs, build_output_filename, ensemble_mode_to_token

        pooled_metrics: list[pd.DataFrame] = []
        first_reg_df: pd.DataFrame | None = None
        first_std_df: pd.DataFrame | None = None
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

            reg_m = cast(
                pd.DataFrame,
                _calculate_all_metrics(
                    ds_tgt_m,
                    ds_pred_m,
                    calc_relative=True,
                    n_points=n_points,
                    include=include,
                    fss_cfg=fss_cfg,
                    seeps_climatology_path=seeps_climatology_path,
                ),
            )
            std_m = cast(
                pd.DataFrame,
                _calculate_all_metrics(
                    ds_tgt_m_std,
                    ds_pred_m_std,
                    calc_relative=False,
                    n_points=n_points,
                    include=std_include,
                    fss_cfg=fss_cfg,
                    seeps_climatology_path=seeps_climatology_path,
                ),
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
            save_dataframe(reg_m, out_csv_m, index_label="variable")
            save_dataframe(std_m, out_csv_m_std, index_label="variable")
            pooled_metrics.append(reg_m)

            if report_per_level:
                per_level_m = cast(
                    pd.DataFrame | None,
                    _calculate_per_level_metrics(
                        ds_tgt_m,
                        ds_pred_m,
                        calc_relative=True,
                        n_points=n_points,
                        include=include,
                        fss_cfg=fss_cfg,
                    ),
                )
                if per_level_m is not None:
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
                    save_dataframe(per_level_m, out_csv_m_lvl, index=False)

                per_level_m_std = cast(
                    pd.DataFrame | None,
                    _calculate_per_level_metrics(
                        ds_tgt_m_std,
                        ds_pred_m_std,
                        calc_relative=False,
                        n_points=n_points,
                        include=std_include,
                        fss_cfg=fss_cfg,
                    ),
                )
                if per_level_m_std is not None:
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
                    save_dataframe(per_level_m_std, out_csv_m_lvl_std, index=False)

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
                save_dataframe(pooled_df, out_csv_pool, index_label="variable")

        # Console previews for members mode
        try:
            print("Deterministic metrics (targets vs predictions) — first 5 rows:")
            df_show = (
                pooled_df
                if ("pooled_df" in locals() and pooled_df is not None and not pooled_df.empty)
                else first_reg_df
            )
            if df_show is not None:
                print(df_show.head())
        except Exception:
            pass
        try:
            print("Deterministic standardized metrics (targets vs predictions) — first 5 rows:")
            if first_std_df is not None:
                print(first_std_df.head())
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
                n_points=n_points,
                include=include,
                fss_cfg=fss_cfg,
                preserve_dims=["lead_time"],
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
                print(f"[deterministic] Failed to save {out_long}: {e}")

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
                save_dataframe(wide_df, out_wide, index=False)
            except Exception as e:
                print(f"[deterministic] Failed to save {out_wide}: {e}")

        # Optional quick line plots over lead_time
        # Always generate one plot per (variable, metric): x=lead_time, y=value
        try:
            import matplotlib.pyplot as _plt

            if not wide_df.empty:
                var_cols = [c for c in wide_df.columns if c != "lead_time_hours"]
                split = [c.split("_", 1) for c in var_cols if "_" in c]
                by_var: dict[str, list[str]] = {}
                for v, m in split:
                    by_var.setdefault(v, []).append(m)
                for v, metrics in by_var.items():
                    for m in sorted(set(metrics)):
                        col = f"{v}_{m}"
                        if col not in wide_df:
                            continue
                        fig, ax = _plt.subplots(figsize=(6.5, 3.2))
                        x = wide_df["lead_time_hours"].values
                        y = wide_df[col].values
                        ax.plot(x, y, marker="o")
                        ax.set_xlabel("lead_time (h)")
                        ax.set_ylabel(m)
                        ax.set_title(f"{v} — {m} vs lead_time")
                        out_png = section_output / build_output_filename(
                            metric="det_line",
                            variable=str(v),
                            level=None,
                            qualifier=m.replace(" ", "_"),
                            init_time_range=init_range,
                            lead_time_range=_extract_lead_range(ds_prediction),
                            ensemble=ens_token,
                            ext="png",
                        )
                        _plt.tight_layout()
                        save_figure(fig, out_png)
                        # Save NPZ and CSV for the line plot
                        out_npz = section_output / build_output_filename(
                            metric="det_line",
                            variable=str(v),
                            level=None,
                            qualifier=m.replace(" ", "_") + "_data",
                            init_time_range=init_range,
                            lead_time_range=_extract_lead_range(ds_prediction),
                            ensemble=ens_token,
                            ext="npz",
                        )
                        save_data(
                            out_npz,
                            lead_hours=x.astype(float),
                            values=y.astype(float),
                            metric=m,
                            variable=str(v),
                        )
                        out_csv = section_output / build_output_filename(
                            metric="det_line",
                            variable=str(v),
                            level=None,
                            qualifier=m.replace(" ", "_") + "_by_lead",
                            init_time_range=init_range,
                            lead_time_range=_extract_lead_range(ds_prediction),
                            ensemble=ens_token,
                            ext="csv",
                        )
                        save_dataframe(
                            pd.DataFrame({"lead_time_hours": x, m: y}), out_csv, index=False
                        )
        except Exception:
            pass
        # (Removed duplicate standardized metrics console summary to reduce noise)
