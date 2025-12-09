from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from ..helpers import ensemble_mode_to_token, resolve_ensemble_mode
from ..plots.bivariate_histograms import calculate_and_plot_bivariate_histograms


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
        ds_prediction=ds_prediction,
        ds_target=ds_target,
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
    Compute and write multivariate metrics (Bivariate Histograms).
    """
    cfg = (metrics_cfg or {}).get("multivariate", {})

    # Resolve ensemble mode
    # Default for multivariate is usually 'mean' (compare ensemble mean to target)
    # But we support 'members' too.
    mode = resolve_ensemble_mode("multivariate", ensemble_mode, ds_target, ds_prediction)

    # Handle ensemble dimension according to mode
    # Use separate variables to avoid confusion and side effects
    ds_p_run = ds_prediction
    ds_t_run = ds_target

    if "ensemble" in ds_prediction.dims and mode == "mean":
        ds_p_run = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_t_run = ds_target.mean(dim="ensemble", keep_attrs=True)

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
            # If mode is mean, ds_p_run is already reduced to mean above
            # If mode is pooled, we pass the full dataset and the plotter flattens it (pooling)
            ens_token = ensemble_mode_to_token(mode) if mode != "none" else None
            # Normalize 'mean' to 'ensmean' for consistency with build_output_filename
            if ens_token == "mean":
                ens_token = "ensmean"

            calculate_and_plot_bivariate_histograms(
                ds_p_run,
                ds_t_run,
                bivariate_pairs,
                out_root,
                bins=100,
                ensemble_token=ens_token,
            )
