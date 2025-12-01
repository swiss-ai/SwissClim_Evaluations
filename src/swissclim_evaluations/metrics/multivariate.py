from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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
    """
    for pair in pairs:
        if len(pair) != 2:
            continue
        var_x, var_y = pair

        # Compute for Prediction
        hist_pred, xedges, yedges = None, None, None
        if var_x in ds_prediction and var_y in ds_prediction:
            x_data = ds_prediction[var_x].values.flatten()
            y_data = ds_prediction[var_y].values.flatten()
            mask = np.isfinite(x_data) & np.isfinite(y_data)
            x_data = x_data[mask]
            y_data = y_data[mask]

            if len(x_data) > 0:
                hist_pred, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)

        # Compute for Target
        hist_target = None
        if ds_target is not None and var_x in ds_target and var_y in ds_target:
            x_data_t = ds_target[var_x].values.flatten()
            y_data_t = ds_target[var_y].values.flatten()
            mask_t = np.isfinite(x_data_t) & np.isfinite(y_data_t)
            x_data_t = x_data_t[mask_t]
            y_data_t = y_data_t[mask_t]

            if len(x_data_t) > 0:
                # Use same bins as prediction if available
                if xedges is not None and yedges is not None:
                    hist_target, _, _ = np.histogram2d(x_data_t, y_data_t, bins=[xedges, yedges])
                else:
                    hist_target, xedges, yedges = np.histogram2d(x_data_t, y_data_t, bins=bins)

        if hist_pred is not None:
            out_file = out_root / "multivariate" / f"bivariate_hist_{var_x}_{var_y}.npz"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                "hist": hist_pred,
                "bins_x": xedges,
                "bins_y": yedges,
            }
            if hist_target is not None:
                save_dict["hist_target"] = hist_target

            np.savez(out_file, **save_dict)


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
    elif mode == "pooled" and "ensemble" in ds_prediction.dims:
        # Stack ensemble into the sample dimension (e.g., "time")
        sample_dim = "time" if "time" in ds_prediction.dims else list(ds_prediction.dims)[0]
        ds_prediction_stacked = ds_prediction.stack(pooled_sample=("ensemble", sample_dim))
        ds_target_stacked = ds_target
        if "ensemble" in ds_target.dims:
            ds_target_stacked = ds_target.stack(pooled_sample=("ensemble", sample_dim))
        # Drop the original dimensions to avoid confusion
        # Note: stack() already removes the original dimensions from the variables,
        # but they might remain as coordinates. We don't need to explicitly drop_dims
        # if we just use the stacked dataset.

        df = calculate_ssim(
            ds_target_stacked,
            ds_prediction_stacked,
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
    else:
        # Single output (mean, none, or pooled if supported)
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
        # Use the full dataset (effectively pooling ensemble members if present)
        calculate_and_plot_bivariate_histograms(ds_prediction, ds_target, bivariate_pairs, out_root)
