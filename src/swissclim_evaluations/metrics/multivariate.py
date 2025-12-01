from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from skimage.metrics import structural_similarity

from ..helpers import build_output_filename, ensemble_mode_to_token, resolve_ensemble_mode


def calculate_ssim(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    variables: list[str] | None = None,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    gaussian_weights: bool = True,
    use_sample_covariance: bool = False,
) -> pd.DataFrame:
    """
    Calculate SSIM for specified variables in the datasets using skimage.

    Args:
        ds_target: Target dataset (reference).
        ds_prediction: Prediction dataset.
        variables: List of variables to compute SSIM for. If None, use all common variables.
        sigma: Standard deviation for the Gaussian filter. Default 1.5 (Wang et al. 2004).
        k1: Constant 1. Default 0.01 (Wang et al. 2004).
        k2: Constant 2. Default 0.03 (Wang et al. 2004).
        gaussian_weights: Whether to use Gaussian weights. Default True (Wang et al. 2004).
        use_sample_covariance: Whether to use sample covariance (N-1) or population covariance (N).
                               Default False (matches Wang et al. 2004 implementation).

    Returns:
        DataFrame with SSIM scores.
    """
    if variables is None:
        variables = list(set(ds_target.data_vars) & set(ds_prediction.data_vars))

    results = {}

    for var in variables:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

        # Ensure we have lat/lon dimensions
        if "latitude" not in da_target.dims or "longitude" not in da_target.dims:
            continue

        # Define a wrapper for apply_ufunc
        def _wrapper(a, b):
            # a and b are 2D arrays (lat, lon)
            # Determine data_range from the slice
            dr = max(a.max() - a.min(), b.max() - b.min())
            if dr == 0:
                dr = 1.0

            return structural_similarity(
                a,
                b,
                data_range=dr,
                sigma=sigma,
                K1=k1,
                K2=k2,
                gaussian_weights=gaussian_weights,
                use_sample_covariance=use_sample_covariance,
            )

        # Determine input core dims
        input_core_dims = [["latitude", "longitude"], ["latitude", "longitude"]]
        output_core_dims: list[list[str]] = [[]]

        ssim_da = xr.apply_ufunc(
            _wrapper,
            da_target,
            da_prediction,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Average over all other dimensions to get a single scalar score per variable
        results[var] = float(ssim_da.mean().values)

    # Add average across all variables
    if results:
        results["average"] = sum(results.values()) / len(results)

    df = pd.DataFrame([results], index=["SSIM"])
    df.index.name = "variable"
    return df


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    metrics_cfg: dict[str, Any] | None,
    ensemble_mode: str | None = None,
) -> None:
    """
    Compute and write multivariate metrics (SSIM) CSVs.
    """
    cfg = (metrics_cfg or {}).get("multivariate", {})

    # SSIM configuration
    ssim_cfg = cfg.get("ssim", {})

    # Extract SSIM parameters
    sigma = ssim_cfg.get("sigma", 1.5)
    k1 = ssim_cfg.get("k1", 0.01)
    k2 = ssim_cfg.get("k2", 0.03)
    gaussian_weights = ssim_cfg.get("gaussian_weights", True)
    use_sample_covariance = ssim_cfg.get("use_sample_covariance", False)

    # Resolve ensemble mode
    # Default for multivariate/SSIM is usually 'mean' (compare ensemble mean to target)
    # But we support 'members' too.
    mode = resolve_ensemble_mode("multivariate", ensemble_mode, ds_target, ds_prediction)

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
                k1=k1,
                k2=k2,
                gaussian_weights=gaussian_weights,
                use_sample_covariance=use_sample_covariance,
            )

            ens_token = ensemble_mode_to_token(mode, member_index=m)
            filename = build_output_filename(
                metric="multivariate", qualifier="ssim", ensemble=ens_token, ext="csv"
            )
            out_path = out_root / "multivariate" / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path)
    else:
        # Single output (mean, none, or pooled if supported)
        df = calculate_ssim(
            ds_target,
            ds_prediction,
            sigma=sigma,
            k1=k1,
            k2=k2,
            gaussian_weights=gaussian_weights,
            use_sample_covariance=use_sample_covariance,
        )

        ens_token = ensemble_mode_to_token(mode)
        filename = build_output_filename(
            metric="multivariate", qualifier="ssim", ensemble=ens_token, ext="csv"
        )
        out_path = out_root / "multivariate" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_path)
