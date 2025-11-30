from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from skimage.metrics import structural_similarity as ssim

from swissclim_evaluations.helpers import build_output_filename, ensemble_mode_to_token


def _calculate_multivariate_ssim(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    sigma: float,
) -> pd.DataFrame:
    """
    Calculate SSIM for each variable.

    SSIM is calculated for each 2D spatial slice (lat/lon) and averaged over other dimensions.
    """
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}

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
        t_min = float(da_target.min(skipna=True))
        t_max = float(da_target.max(skipna=True))
        p_min = float(da_prediction.min(skipna=True))
        p_max = float(da_prediction.max(skipna=True))

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
                gaussian_weights=True,
                sigma=sigma,
                use_sample_covariance=True,  # Match MATLAB default (N-1 normalization)
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

        # Average SSIM over init_time/level/ensemble/
        mean_ssim = float(ssim_da.mean(skipna=True).compute())
        metrics_dict[var] = {"SSIM": mean_ssim}

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    ds_target_std: xr.Dataset,  # Unused but kept for signature compatibility
    ds_prediction_std: xr.Dataset,  # Unused but kept for signature compatibility
    out_root: Path,
    plotting_cfg: dict[str, Any] | None,
    metrics_cfg: dict[str, Any] | None,
    ensemble_mode: str | None = None,
) -> None:
    """Compute and write multivariate SSIM metrics."""

    # Determine output directory
    section_output = out_root / "multivariate"
    section_output.mkdir(parents=True, exist_ok=True)

    # Resolve ensemble token
    token = ensemble_mode_to_token(ensemble_mode) if ensemble_mode else "det"

    # Helper functions to extract time ranges (copied from deterministic.py)
    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds.coords and "init_time" not in ds.dims:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            start = np.datetime64(vals.min()).astype("datetime64[h]")
            end = np.datetime64(vals.max()).astype("datetime64[h]")

            def _fmt_init(x):
                return (
                    np.datetime_as_string(x, unit="h")
                    .replace("-", "")
                    .replace(":", "")
                    .replace("T", "")
                )

            return (_fmt_init(start), _fmt_init(end))
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

    # Calculate SSIM
    print("[multivariate] Calculating SSIM metrics...")
    multi_cfg = (metrics_cfg or {}).get("multivariate", {})
    sigma = float(multi_cfg.get("ssim_sigma", 1.5))

    df_ssim = _calculate_multivariate_ssim(ds_target, ds_prediction, sigma=sigma)

    if not df_ssim.empty:
        # Calculate multivariate average (average of SSIM across all variables)
        multivariate_score = df_ssim["SSIM"].mean()

        # Add a summary row
        summary_row = pd.DataFrame({"SSIM": [multivariate_score]}, index=["MULTIVARIATE_AVERAGE"])
        df_final = pd.concat([df_ssim, summary_row])

        # Build output filename
        out_csv = section_output / build_output_filename(
            metric="multivariate_ssim",
            variable=None,
            level=None,
            qualifier="summary",
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=token,
            ext="csv",
        )

        df_final.to_csv(out_csv, index_label="variable")
        print(f"[multivariate] saved {out_csv}")

        # Print preview
        print("Multivariate SSIM metrics — first 5 rows:")
        print(df_final.head())
