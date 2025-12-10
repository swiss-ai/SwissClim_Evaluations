from __future__ import annotations

import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scores.categorical import seeps
from scores.continuous import additive_bias, mae, mse, rmse
from scores.continuous.correlation import pearsonr
from scores.spatial import fss_2d
from skimage.metrics import structural_similarity as ssim


def _prepare_seeps(y_true: xr.DataArray, path_climatology: str):
    ds_clim = xr.open_zarr(path_climatology)
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


def _calculate_ssim(da_target: xr.DataArray, da_pred: xr.DataArray) -> float:
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
        return float(res.mean().compute().item())
    except Exception:
        return np.nan


def _calculate_all_metrics(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    calc_relative: bool,
    n_points: int,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None = None,
    seeps_climatology_path: str | None = None,
) -> pd.DataFrame:
    """Compute scalar deterministic metrics per variable.

    Metrics supported:
      - MAE, RMSE, MSE, Bias, Pearson R, FSS, SEEPS (for precipitation)
      - Relative MAE, Relative L1, Relative L2 (when calc_relative=True)
    """

    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}

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

    for var in variables:
        y_true = ds_target[var]
        y_pred = ds_prediction[var]
        row: dict[str, float] = {}

        # Base error metrics
        if (include is None) or ("MAE" in metrics_to_compute):
            mae_val = float(mae(y_pred, y_true))
            row["MAE"] = mae_val
        else:
            mae_val = np.nan

        if (include is None) or ("RMSE" in metrics_to_compute):
            rmse_val = float(rmse(y_pred, y_true))
            row["RMSE"] = rmse_val
        else:
            rmse_val = np.nan

        if (include is None) or ("MSE" in metrics_to_compute):
            mse_val = float(mse(y_pred, y_true))
            row["MSE"] = mse_val

        if (include is None) or ("Bias" in metrics_to_compute):
            bias_val = float(additive_bias(y_pred, y_true))
            row["Bias"] = bias_val
        else:
            bias_val = np.nan

        # Correlation
        if (include is None) or ("Pearson R" in metrics_to_compute):
            row["Pearson R"] = float(pearsonr(y_pred, y_true))

        # SSIM
        if (include is None) or ("SSIM" in metrics_to_compute):
            row["SSIM"] = _calculate_ssim(y_true, y_pred)

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
                        row[fss_label] = no_event_value
                    else:
                        spatial_dims = ["latitude", "longitude"]

                        # Proper call to fss_2d with config-driven arguments only
                        out = fss_2d(
                            y_pred.compute(),
                            y_true.compute(),
                            event_threshold=event_threshold,
                            window_size=fss_window_size,
                            spatial_dims=spatial_dims,
                            # this has no effect here, unfortunately we need to compute() above
                            dask="parallelized",
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
                            fss_scalar = float(out.mean(skipna=True).compute().item())
                        else:
                            fss_scalar = float(out)
                        row[fss_label] = fss_scalar
                except Exception as e:
                    # Surface the error context once while keeping pipeline resilient
                    with contextlib.suppress(Exception):
                        print(f"[deterministic:FSS] fss_2d failed for var='{var}': {e!r}")
                    row[fss_label] = float("nan")

        if "total_precipitation" in var and "SEEPS" in metrics_to_compute:
            if seeps_climatology_path is None:
                raise ValueError(
                    "SEEPS metric requested but 'seeps_climatology_path' not provided in config."
                )
            prob_dry, seeps_threshold = _prepare_seeps(y_true, seeps_climatology_path)
            seeps_val = seeps(
                y_pred * 1000, y_true * 1000, prob_dry, seeps_threshold, dry_light_threshold=0.25
            )  # convert to mm with *1000
            row["SEEPS"] = float(seeps_val)

        # Relative metrics
        if calc_relative:
            # Denominator norms on the target
            l1_norm = float(np.abs(y_true).sum(skipna=True).compute())
            l2_norm = float(((y_true**2).sum(skipna=True).compute()) ** 0.5)
            mean_abs = float(np.abs(y_true).mean(skipna=True).compute())

            # Mean-based relative MAE (keep for continuity and interpretability)
            if (include is None) or ("Relative MAE" in metrics_to_compute):
                row["Relative MAE"] = (mae_val / mean_abs) if mean_abs else float("nan")

            # True norm-based relative errors: ||e||1 / ||y||1 and ||e||2 / ||y||2
            err = y_pred - y_true
            l1_err = float(np.abs(err).sum(skipna=True).compute())
            l2_err = float(((err**2).sum(skipna=True).compute()) ** 0.5)

            eps = 1e-12
            if (include is None) or ("Relative L1" in metrics_to_compute):
                row["Relative L1"] = (l1_err / l1_norm) if (abs(l1_norm) > eps) else float("nan")
            if (include is None) or ("Relative L2" in metrics_to_compute):
                row["Relative L2"] = (l2_err / l2_norm) if (abs(l2_norm) > eps) else float("nan")

        # If include provided, trim extra keys

        metrics_dict[var] = row

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def _calculate_per_level_metrics(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    calc_relative: bool,
    n_points: int,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None,
) -> pd.DataFrame | None:
    """Compute metrics per pressure level for 3D variables."""
    variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
    if not variables_3d:
        return None

    if "level" not in ds_target.dims:
        return None

    levels = ds_target.level.values
    dfs = []
    for level in levels:
        # Select level for all 3D variables
        ds_t_lvl = ds_target[variables_3d].sel(level=level)
        ds_p_lvl = ds_prediction[variables_3d].sel(level=level)

        df = _calculate_all_metrics(ds_t_lvl, ds_p_lvl, calc_relative, n_points, include, fss_cfg)
        df["level"] = int(level)
        df["variable"] = df.index
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs).reset_index(drop=True)


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

    if members_indices is None:
        # Aggregate (non per-member) outputs
        regular_metrics = _calculate_all_metrics(
            ds_target,
            ds_prediction,
            calc_relative=True,
            n_points=n_points,
            include=include,
            fss_cfg=fss_cfg,
            seeps_climatology_path=seeps_climatology_path,
        )
        standardized_metrics = _calculate_all_metrics(
            ds_target_std,
            ds_prediction_std,
            calc_relative=False,
            n_points=n_points,
            include=std_include,
            fss_cfg=fss_cfg,
            seeps_climatology_path=seeps_climatology_path,
        )

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
        regular_metrics.to_csv(out_csv, index_label="variable")
        standardized_metrics.to_csv(out_csv_std, index_label="variable")
        print(f"[deterministic] saved {out_csv}")
        print(f"[deterministic] saved {out_csv_std}")

        if report_per_level:
            per_level_metrics = _calculate_per_level_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                n_points=n_points,
                include=include,
                fss_cfg=fss_cfg,
            )
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
                per_level_metrics.to_csv(out_csv_lvl, index=False)
                print(f"[deterministic] saved {out_csv_lvl}")

            per_level_std = _calculate_per_level_metrics(
                ds_target_std,
                ds_prediction_std,
                calc_relative=False,
                n_points=n_points,
                include=std_include,
                fss_cfg=fss_cfg,
            )
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
                per_level_std.to_csv(out_csv_lvl_std, index=False)
                print(f"[deterministic] saved {out_csv_lvl_std}")

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

            reg_m = _calculate_all_metrics(
                ds_tgt_m,
                ds_pred_m,
                calc_relative=True,
                n_points=n_points,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
            )
            std_m = _calculate_all_metrics(
                ds_tgt_m_std,
                ds_pred_m_std,
                calc_relative=False,
                n_points=n_points,
                include=std_include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
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
            reg_m.to_csv(out_csv_m, index_label="variable")
            std_m.to_csv(out_csv_m_std, index_label="variable")
            print(f"[deterministic] saved {out_csv_m}")
            print(f"[deterministic] saved {out_csv_m_std}")
            pooled_metrics.append(reg_m)

            if report_per_level:
                per_level_m = _calculate_per_level_metrics(
                    ds_tgt_m,
                    ds_pred_m,
                    calc_relative=True,
                    n_points=n_points,
                    include=include,
                    fss_cfg=fss_cfg,
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
                    per_level_m.to_csv(out_csv_m_lvl, index=False)
                    print(f"[deterministic] saved {out_csv_m_lvl}")

                per_level_m_std = _calculate_per_level_metrics(
                    ds_tgt_m_std,
                    ds_pred_m_std,
                    calc_relative=False,
                    n_points=n_points,
                    include=std_include,
                    fss_cfg=fss_cfg,
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
                    per_level_m_std.to_csv(out_csv_m_lvl_std, index=False)
                    print(f"[deterministic] saved {out_csv_m_lvl_std}")

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
                pooled_df.to_csv(out_csv_pool, index_label="variable")
                print(f"[deterministic] saved {out_csv_pool}")

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
    try:
        multi_lead = (
            (lead_policy is not None)
            and ("lead_time" in ds_prediction.dims)
            and int(ds_prediction.sizes.get("lead_time", 0)) > 1
            and getattr(lead_policy, "mode", "first") != "first"
        )
    except Exception:
        multi_lead = False
    if members_indices is None and multi_lead:
        rows = []
        wide_rows = []
        leads = list(ds_prediction["lead_time"].values)
        for i, lt in enumerate(leads):
            try:
                hours = int(np.timedelta64(lt) / np.timedelta64(1, "h"))
            except Exception:
                hours = int(i)
            ds_t_i = (
                ds_target.isel(lead_time=i, drop=False)
                if "lead_time" in ds_target.dims
                else ds_target
            )
            ds_p_i = ds_prediction.isel(lead_time=i, drop=False)
            df_i = _calculate_all_metrics(
                ds_t_i,
                ds_p_i,
                calc_relative=True,
                n_points=n_points,
                include=include,
                fss_cfg=fss_cfg,
            )
            df_i = df_i.reset_index().rename(columns={"index": "variable"})
            df_i.insert(0, "lead_time_hours", hours)
            rows.append(df_i)
            # Build a wide row with columns like var_metric
            flat: dict[str, float] = {"lead_time_hours": float(hours)}
            for _idx, row in df_i.iterrows():
                var = str(row["variable"]) if "variable" in row else None
                for col, val in row.items():
                    if col in {"lead_time_hours", "variable"}:
                        continue
                    key = f"{var}_{col}" if var is not None else col
                    with contextlib.suppress(Exception):
                        flat[key] = float(val)
            wide_rows.append(flat)
        if rows:
            long_df = pd.concat(rows, ignore_index=True)
            # Write standardized filename with tokens for downstream tooling
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
                long_df.to_csv(out_long, index=False)
                print(f"[deterministic] saved {out_long}")
            except Exception:
                pass
        if wide_rows:
            wide_df = pd.DataFrame(wide_rows)
            # Write standardized filename variant
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
                wide_df.to_csv(out_wide, index=False)
                print(f"[deterministic] saved {out_wide}")
            except Exception:
                pass

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
                        _plt.savefig(out_png, bbox_inches="tight", dpi=150)
                        _plt.close(fig)
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
                        np.savez(
                            out_npz,
                            lead_hours=x.astype(float),
                            values=y.astype(float),
                            metric=m,
                            variable=str(v),
                        )  # line wrapped to satisfy E501
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
                        pd.DataFrame({"lead_time_hours": x, m: y}).to_csv(out_csv, index=False)
        except Exception:
            pass
        # (Removed duplicate standardized metrics console summary to reduce noise)
