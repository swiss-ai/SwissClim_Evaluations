from __future__ import annotations

from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import xarray as xr
from skimage.metrics import structural_similarity as ssim

from swissclim_evaluations.helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    resolve_ensemble_mode,
)


def calculate_ssim(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    gaussian_weights: bool = True,
    use_sample_covariance: bool = True,
    report_per_level: bool = False,
    report_per_lead_time: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Calculate SSIM for each variable.

    SSIM is calculated for each 2D spatial slice (lat/lon) and averaged over other dimensions.
    Returns:
        (df_global, df_per_level, df_per_lead_time)
    """
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}
    per_level_data: list[dict[str, Any]] = []
    per_lead_time_data: list[dict[str, Any]] = []

    for var in variables:
        if var not in ds_prediction:
            continue

        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

        # Identify spatial dimensions
        dims = list(da_target.dims)
        spatial_dims = [d for d in dims if d in ["latitude", "longitude", "lat", "lon"]]

        if len(spatial_dims) != 2:
            # Skip variables that don't have exactly 2 spatial dimensions
            continue

        # Calculate global data range for the variable to ensure consistent SSIM calculation
        # We use the union of target and prediction ranges
        t_min_lazy = da_target.min(skipna=True)
        t_max_lazy = da_target.max(skipna=True)
        p_min_lazy = da_prediction.min(skipna=True)
        p_max_lazy = da_prediction.max(skipna=True)

        t_min, t_max, p_min, p_max = dask.compute(t_min_lazy, t_max_lazy, p_min_lazy, p_max_lazy)

        t_min = float(t_min)
        t_max = float(t_max)
        p_min = float(p_min)
        p_max = float(p_max)

        data_range = max(t_max, p_max) - min(t_min, p_min)
        if data_range == 0:
            data_range = 1.0

        def _ssim_wrapper(t, p, data_range=data_range):
            # Use Gaussian weights to match the standard SSIM definition (Wang et al. 2004)
            # and the MATLAB implementation used in Baker et al. 2019.
            return ssim(
                t,
                p,
                data_range=data_range,
                gaussian_weights=gaussian_weights,
                sigma=sigma,
                use_sample_covariance=use_sample_covariance,
                K1=K1,
                K2=K2,
            )

        # Apply SSIM over spatial dimensions
        ssim_da = xr.apply_ufunc(
            _ssim_wrapper,
            da_target,
            da_prediction,
            input_core_dims=[spatial_dims, spatial_dims],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Average SSIM over all non-spatial dimensions (e.g., init_time, lead_time, level, ensemble)
        mean_ssim = float(ssim_da.mean(skipna=True).compute())
        metrics_dict[var] = {"SSIM": mean_ssim}

        # Per-level breakdown
        if report_per_level and "level" in ssim_da.dims:
            dims_to_mean = [d for d in ssim_da.dims if d != "level"]
            ssim_level = ssim_da.mean(dim=dims_to_mean, skipna=True).compute()
            # ssim_level is 1D array along 'level'
            for lvl_val, score in zip(ssim_level["level"].values, ssim_level.values, strict=False):
                per_level_data.append(
                    {
                        "variable": var,
                        "level": float(lvl_val),
                        "SSIM": float(score),
                    }
                )

        # Per-lead-time breakdown
        if report_per_lead_time and "lead_time" in ssim_da.dims:
            dims_to_mean = [d for d in ssim_da.dims if d != "lead_time"]
            ssim_lead = ssim_da.mean(dim=dims_to_mean, skipna=True).compute()
            # ssim_lead is 1D array along 'lead_time'
            for lt_val, score in zip(ssim_lead["lead_time"].values, ssim_lead.values, strict=False):
                if isinstance(lt_val, np.timedelta64):
                    lt_val_hours = lt_val / np.timedelta64(1, "h")
                else:
                    lt_val_hours = lt_val
                per_lead_time_data.append(
                    {
                        "variable": var,
                        "lead_time": float(lt_val_hours),
                        "SSIM": float(score),
                    }
                )

    df_global = pd.DataFrame.from_dict(metrics_dict, orient="index")

    df_per_level = None
    if per_level_data:
        df_per_level = pd.DataFrame(per_level_data)
        # Sort for tidiness
        df_per_level = df_per_level.sort_values(by=["variable", "level"])

    df_per_lead_time = None
    if per_lead_time_data:
        df_per_lead_time = pd.DataFrame(per_lead_time_data)
        # Sort for tidiness
        df_per_lead_time = df_per_lead_time.sort_values(by=["variable", "lead_time"])

    return df_global, df_per_level, df_per_lead_time


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    metrics_cfg: dict[str, Any] | None,
    ensemble_mode: str | None = None,
) -> None:
    """
    Compute and write SSIM metrics CSVs.
    """
    cfg = (metrics_cfg or {}).get("ssim", {})

    # Extract SSIM parameters
    sigma = float(cfg.get("sigma", 1.5))
    K1 = float(cfg.get("K1", 0.01))
    K2 = float(cfg.get("K2", 0.03))
    gaussian_weights = bool(cfg.get("gaussian_weights", True))
    use_sample_covariance = bool(cfg.get("use_sample_covariance", True))
    report_per_level = bool(cfg.get("report_per_level", True))
    report_per_lead_time = bool(cfg.get("report_per_lead_time", True))

    # Resolve ensemble mode
    mode = resolve_ensemble_mode("ssim", ensemble_mode, ds_target, ds_prediction)

    # Helper to save output
    def _save_output(df: pd.DataFrame, ens_token: str, qualifier: str = "ssim"):
        if df.empty:
            return

        # Calculate average SSIM across variables if it's the main table
        if "level" not in df.columns and "lead_time" not in df.columns:
            avg_score = df["SSIM"].mean()
            summary_row = pd.DataFrame({"SSIM": [avg_score]}, index=["AVERAGE_SSIM"])
            df_final = pd.concat([df, summary_row])
            index_arg = True
            index_label = "variable"
        else:
            df_final = df
            index_arg = False
            index_label = None

        filename = build_output_filename(
            metric="ssim", qualifier=qualifier, ensemble=ens_token, ext="csv"
        )
        out_path = out_root / "ssim" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(out_path, index=index_arg, index_label=index_label)
        print(f"[ssim] saved {out_path}")

    # Handle ensemble dimension
    if "ensemble" in ds_prediction.dims and mode == "mean":
        ds_prediction = ds_prediction.mean(dim="ensemble")
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble")

    if mode == "members" and "ensemble" in ds_prediction.dims:
        # Per-member outputs
        n_members = ds_prediction.sizes["ensemble"]
        for m in range(n_members):
            ds_p_mem = ds_prediction.isel(ensemble=m, drop=True)
            if "ensemble" in ds_target.dims:
                ds_t_mem = ds_target.isel(ensemble=m, drop=True)
            else:
                ds_t_mem = ds_target

            df, df_lvl, df_lead = calculate_ssim(
                ds_t_mem,
                ds_p_mem,
                sigma=sigma,
                K1=K1,
                K2=K2,
                gaussian_weights=gaussian_weights,
                use_sample_covariance=use_sample_covariance,
                report_per_level=report_per_level,
                report_per_lead_time=report_per_lead_time,
            )
            ens_token = ensemble_mode_to_token(mode, member_index=m)
            _save_output(df, ens_token)
            if df_lvl is not None:
                _save_output(df_lvl, ens_token, qualifier="ssim_per_level")
            if df_lead is not None:
                _save_output(df_lead, ens_token, qualifier="ssim_per_lead_time")

    elif mode == "pooled" and "ensemble" in ds_prediction.dims:
        # Stack ensemble into the sample dimension (e.g., "time")
        sample_dim = (
            "init_time"
            if "init_time" in ds_prediction.dims
            else (
                "lead_time"
                if "lead_time" in ds_prediction.dims
                else ("time" if "time" in ds_prediction.dims else list(ds_prediction.dims)[0])
            )
        )
        ds_prediction_stacked = ds_prediction.stack(pooled_sample=("ensemble", sample_dim))
        ds_target_stacked = ds_target
        if "ensemble" in ds_target.dims:
            ds_target_stacked = ds_target.stack(pooled_sample=("ensemble", sample_dim))

        df, df_lvl, df_lead = calculate_ssim(
            ds_target_stacked,
            ds_prediction_stacked,
            sigma=sigma,
            K1=K1,
            K2=K2,
            gaussian_weights=gaussian_weights,
            use_sample_covariance=use_sample_covariance,
            report_per_level=report_per_level,
            report_per_lead_time=report_per_lead_time,
        )
        ens_token = ensemble_mode_to_token(mode)
        _save_output(df, ens_token)
        if df_lvl is not None:
            _save_output(df_lvl, ens_token, qualifier="ssim_per_level")
        if df_lead is not None:
            _save_output(df_lead, ens_token, qualifier="ssim_per_lead_time")

    else:
        # Single output (mean, none, or pooled if supported)
        df, df_lvl, df_lead = calculate_ssim(
            ds_target,
            ds_prediction,
            sigma=sigma,
            K1=K1,
            K2=K2,
            gaussian_weights=gaussian_weights,
            use_sample_covariance=use_sample_covariance,
            report_per_level=report_per_level,
            report_per_lead_time=report_per_lead_time,
        )
        ens_token = ensemble_mode_to_token(mode)
        _save_output(df, ens_token)
        if df_lvl is not None:
            _save_output(df_lvl, ens_token, qualifier="ssim_per_level")
        if df_lead is not None:
            _save_output(df_lead, ens_token, qualifier="ssim_per_lead_time")
