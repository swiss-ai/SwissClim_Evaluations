from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from skimage.metrics import structural_similarity as ssim

from ..dask_utils import compute_jobs
from ..helpers import (
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
) -> pd.DataFrame:
    """
    Calculate SSIM for each variable.

    SSIM is calculated for each 2D spatial slice (lat/lon) and averaged over other dimensions.
    """
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}

    # 1. First pass: Collect lazy min/max computations for all variables
    range_jobs = []
    valid_vars = []

    for var in variables:
        if var not in ds_prediction:
            continue

        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

        # Identify spatial dimensions
        dims = list(da_target.dims)
        spatial_dims = [d for d in dims if d in ["latitude", "longitude", "lat", "lon"]]

        if len(spatial_dims) != 2:
            continue

        valid_vars.append(var)

        # Calculate global data range for the variable
        t_min_lazy = da_target.min(skipna=True)
        t_max_lazy = da_target.max(skipna=True)
        p_min_lazy = da_prediction.min(skipna=True)
        p_max_lazy = da_prediction.max(skipna=True)

        range_jobs.append(
            {
                "t_min": t_min_lazy,
                "t_max": t_max_lazy,
                "p_min": p_min_lazy,
                "p_max": p_max_lazy,
                "var": var,
            }
        )

    if not range_jobs:
        return pd.DataFrame()

    # Compute ranges in batch
    compute_jobs(
        range_jobs,
        key_map={
            "t_min": "t_min_res",
            "t_max": "t_max_res",
            "p_min": "p_min_res",
            "p_max": "p_max_res",
        },
    )

    # 2. Second pass: Create SSIM lazy objects using computed ranges
    ssim_jobs = []

    for job in range_jobs:
        var = job["var"]
        t_min = float(job["t_min_res"])
        t_max = float(job["t_max_res"])
        p_min = float(job["p_min_res"])
        p_max = float(job["p_max_res"])

        data_range = max(t_max, p_max) - min(t_min, p_min)
        if data_range == 0:
            data_range = 1.0

        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

        # Re-identify spatial dims (safe as we filtered already)
        dims = list(da_target.dims)
        spatial_dims = [d for d in dims if d in ["latitude", "longitude", "lat", "lon"]]

        def _ssim_wrapper(t, p, data_range=data_range):
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

        # Average SSIM over all non-spatial dimensions
        mean_ssim_lazy = ssim_da.mean(skipna=True)

        ssim_jobs.append({"ssim_mean": mean_ssim_lazy, "var": var})

    # Compute SSIM means in batch
    compute_jobs(
        ssim_jobs,
        key_map={"ssim_mean": "ssim_res"},
    )

    # 3. Populate results
    for job in ssim_jobs:
        var = job["var"]
        val = float(job["ssim_res"])
        metrics_dict[var] = {"SSIM": val}

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


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

    # Resolve ensemble mode
    mode = resolve_ensemble_mode("ssim", ensemble_mode, ds_target, ds_prediction)

    # Helper to save output
    def _save_output(df: pd.DataFrame, ens_token: str | None):
        if df.empty:
            return

        # Calculate average SSIM across variables
        avg_score = df["SSIM"].mean()
        summary_row = pd.DataFrame({"SSIM": [avg_score]}, index=["AVERAGE_SSIM"])
        df_final = pd.concat([df, summary_row])

        filename = build_output_filename(
            metric="ssim", qualifier="ssim", ensemble=ens_token, ext="csv"
        )
        out_path = out_root / "ssim" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(out_path, index_label="variable")
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

            df = calculate_ssim(
                ds_t_mem,
                ds_p_mem,
                sigma=sigma,
                K1=K1,
                K2=K2,
                gaussian_weights=gaussian_weights,
                use_sample_covariance=use_sample_covariance,
            )
            ens_token = ensemble_mode_to_token(mode, member_index=m)
            _save_output(df, ens_token)

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

        df = calculate_ssim(
            ds_target_stacked,
            ds_prediction_stacked,
            sigma=sigma,
            K1=K1,
            K2=K2,
            gaussian_weights=gaussian_weights,
            use_sample_covariance=use_sample_covariance,
        )
        ens_token = ensemble_mode_to_token(mode)
        _save_output(df, ens_token)

    else:
        # Single output (mean, none, or pooled if supported)
        df = calculate_ssim(
            ds_target,
            ds_prediction,
            sigma=sigma,
            K1=K1,
            K2=K2,
            gaussian_weights=gaussian_weights,
            use_sample_covariance=use_sample_covariance,
        )
        ens_token = ensemble_mode_to_token(mode)
        _save_output(df, ens_token)
