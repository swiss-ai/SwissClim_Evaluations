from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from ..helpers import ensemble_mode_to_token, resolve_ensemble_mode
from ..plots.bivariate_histograms import calculate_and_plot_bivariate_histograms


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    metrics_cfg: dict[str, Any] | None,
    ensemble_mode: str | None = None,
) -> None:
    """Compute and write multivariate metrics (Bivariate Histograms).

    Args:
        ds_target: Target/reference dataset.
        ds_prediction: Prediction dataset.
        out_root: Root output directory.
        metrics_cfg: Full ``metrics`` config dict (``metrics_cfg["multivariate"]``
            is read for ``bivariate_pairs`` and ``bins``).
        ensemble_mode: Resolved ensemble handling mode (``mean`` / ``pooled`` /
            ``members``). Falls back to module default (``mean``) if ``None``.
    """
    cfg = (metrics_cfg or {}).get("multivariate", {})

    # Resolve ensemble mode
    mode = resolve_ensemble_mode("multivariate", ensemble_mode, ds_target, ds_prediction)

    # Handle ensemble dimension according to mode
    ds_p_run = ds_prediction
    ds_t_run = ds_target

    if "ensemble" in ds_prediction.dims and mode == "mean":
        ds_p_run = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_t_run = ds_target.mean(dim="ensemble", keep_attrs=True)

    elif mode == "pooled" and "ensemble" in ds_prediction.dims:
        # Stack ensemble with a sample dimension so downstream flatten() treats
        # all ensemble members as independent samples (explicit pooling).
        non_ens_dims = [d for d in ds_prediction.dims if d != "ensemble"]
        sample_dim = next(
            (d for d in ("init_time", "time", "lead_time") if d in non_ens_dims),
            non_ens_dims[0] if non_ens_dims else None,
        )
        if sample_dim is None:
            raise ValueError(
                "Pooled ensemble mode requires at least one non-ensemble dimension to stack "
                f"over, but only found: {tuple(ds_prediction.dims)!r}"
            )
        ds_p_run = ds_prediction.stack(pooled_sample=("ensemble", sample_dim))
        ds_t_run = (
            ds_target.stack(pooled_sample=("ensemble", sample_dim))
            if "ensemble" in ds_target.dims
            else ds_target
        )

    # Bivariate histograms
    bivariate_pairs = cfg.get("bivariate_pairs")
    if not bivariate_pairs:
        return

    bins = int(cfg.get("bins", 100))

    # If mode is members, produce per-member plots
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
                bins=bins,
                ensemble_token=ens_token,
            )
    else:
        ens_token = ensemble_mode_to_token(mode)
        calculate_and_plot_bivariate_histograms(
            ds_p_run,
            ds_t_run,
            bivariate_pairs,
            out_root,
            bins=bins,
            ensemble_token=ens_token,
        )
