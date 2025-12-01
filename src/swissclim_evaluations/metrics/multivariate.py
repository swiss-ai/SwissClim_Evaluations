from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from skimage.metrics import structural_similarity

from ..helpers import build_output_filename, ensemble_mode_to_token, resolve_ensemble_mode
from ..plots.bivariate_histograms import calculate_and_plot_bivariate_histograms


def calculate_ssim(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    variables: list[str] | None = None,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
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
        K1: Constant 1. Default 0.01 (Wang et al. 2004).
        K2: Constant 2. Default 0.03 (Wang et al. 2004).
        gaussian_weights: Whether to use Gaussian weights. Default True (Wang et al. 2004).
        use_sample_covariance: Whether to use sample covariance (N-1) or population covariance (N).
                               Default False (matches Wang et al. 2004 implementation).

    Returns:
        DataFrame with SSIM scores.
    """
    if variables is None:
        variables = list(set(ds_target.data_vars) & set(ds_prediction.data_vars))

    results = {}

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
            K1=K1,
            K2=K2,
            gaussian_weights=gaussian_weights,
            use_sample_covariance=use_sample_covariance,
        )

    for var in variables:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

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


def calculate_bivariate_histograms(
    ds_prediction: xr.Dataset,
    ds_target: xr.Dataset | None,
    pairs: list[list[str]],
    out_root: Path,
    bins: int = 100,
) -> None:
    """
    Calculate and save bivariate histograms for specified pairs.
    Delegates to calculate_and_plot_bivariate_histograms to avoid code duplication.
    """
    calculate_and_plot_bivariate_histograms(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        pairs=pairs,
        bins=bins,
        out_root=out_root,
    )


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

    # Handle ensemble dimension according to mode
    if "ensemble" in ds_prediction.dims and mode == "mean":
        ds_prediction = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble", keep_attrs=True)

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
                K1=k1,
                K2=k2,
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
        # Single output (mean or pooled)
        # For pooled mode, we pass the full dataset (with ensemble dim).
        # calculate_ssim uses apply_ufunc which broadcasts over ensemble,
        # and the final .mean() averages over all dimensions (including ensemble),
        # effectively pooling all samples.
        df = calculate_ssim(
            ds_target,
            ds_prediction,
            sigma=sigma,
            K1=k1,
            K2=k2,
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

    # Bivariate histograms
    bivariate_pairs = cfg.get("bivariate_pairs")
    if bivariate_pairs:
        # If mode is members, we want per-member plots
        if mode == "members" and "ensemble" in ds_prediction.dims:
            n_members = ds_prediction.sizes["ensemble"]
            for m in range(n_members):
                ds_p_mem = ds_prediction.isel(ensemble=m, drop=True)
                if "ensemble" in ds_target.dims:
                    ds_t_mem = ds_target.isel(ensemble=m, drop=True)
                else:
                    ds_t_mem = ds_target

                ens_token = ensemble_mode_to_token(mode, member_index=m)
                calculate_and_plot_bivariate_histograms(
                    ds_p_mem,
                    ds_t_mem,
                    bivariate_pairs,
                    out_root,
                    bins=100,
                    ensemble_token=ens_token,
                )
        else:
            # If mode is mean, ds_prediction is already reduced to mean above
            # If mode is pooled, we pass the full dataset and the plotter flattens it (pooling)
            ens_token = ensemble_mode_to_token(mode) if mode != "none" else None
            calculate_and_plot_bivariate_histograms(
                ds_prediction,
                ds_target,
                bivariate_pairs,
                out_root,
                bins=100,
                ensemble_token=ens_token,
            )
