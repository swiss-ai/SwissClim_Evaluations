from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..metrics.probabilistic import (
    crps_ensemble,
    probability_integral_transform,
)


def _select_base_variable(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    plotting_cfg: dict[str, Any],
) -> str:
    cfg_var = (
        (plotting_cfg or {}).get("map_variable")
        if isinstance(plotting_cfg, dict)
        else None
    )
    if cfg_var and isinstance(cfg_var, str):
        if cfg_var.startswith("CRPS."):
            return cfg_var.split(".", 1)[1]
        return cfg_var
    common = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not common:
        raise ValueError(
            "No common variables between targets and predictions for probabilistic plots."
        )
    return common[0]


def _time_reduce_dims(da: xr.DataArray) -> list[str]:
    return [
        d
        for d in ["time", "init_time", "lead_time", "ensemble"]
        if d in da.dims
    ]


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    """Generate probabilistic plots (CRPS map and PIT histogram) and save to disk.

    Saves under out_root/probabilistic as PNGs and optionally NPZ with data if output_mode is 'npz' or 'both'.
    """
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int((plotting_cfg or {}).get("dpi", 48))
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    base_var = _select_base_variable(ds_target, ds_prediction, plotting_cfg)

    # --- CRPS map (reduce over time-like dims, keep lat/lon) ---
    crps = crps_ensemble(
        ds_target[base_var], ds_prediction[base_var], ensemble_dim="ensemble"
    )
    reduce_dims = _time_reduce_dims(crps)
    crps_map = crps.mean(dim=reduce_dims, skipna=True) if reduce_dims else crps

    # Detect lat/lon and sort latitude ascending for pcolormesh compatibility
    lat_name = next(
        (n for n in crps_map.dims if n in ("latitude", "lat", "y")), None
    )
    lon_name = next(
        (n for n in crps_map.dims if n in ("longitude", "lon", "x")), None
    )
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot find lat/lon dims in CRPS map dims: {crps_map.dims}"
        )
    lat_vals = crps_map[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        crps_map = crps_map.sortby(lat_name)

    # Plot CRPS map
    fig = plt.figure(figsize=(10, 6), dpi=dpi * 2)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    mesh = ax.pcolormesh(
        crps_map[lon_name],
        crps_map[lat_name],
        crps_map.values,
        cmap="viridis",
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(
        mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8
    )
    cbar.set_label(f"CRPS — {base_var}")
    ax.set_title(f"CRPS map (mean over time): {base_var}")

    if save_fig:
        out_png = section / f"crps_map_{base_var}.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"[probabilistic-plots] saved {out_png}")
    if save_npz:
        out_npz = section / f"crps_map_{base_var}.npz"
        np.savez(
            out_npz,
            crps=crps_map.values,
            latitude=crps_map[lat_name].values,
            longitude=crps_map[lon_name].values,
            variable=base_var,
            metric="CRPS",
        )
        print(f"[probabilistic-plots] saved {out_npz}")
    plt.close(fig)

    # --- PIT histogram (global) ---
    pit = probability_integral_transform(
        ds_target[base_var],
        ds_prediction[base_var],
        ensemble_dim="ensemble",
        name_prefix=None,
    )
    pit_flat = pit.values.ravel()
    pit_flat = pit_flat[np.isfinite(pit_flat)]

    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi * 2)
    counts, edges, _ = ax.hist(
        pit_flat,
        bins=20,
        range=(0.0, 1.0),
        density=True,
        color="#4C78A8",
        edgecolor="white",
    )
    ax.set_title(f"PIT histogram — {base_var}")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.axhline(1.0, color="brown", linestyle="--", linewidth=1, label="Uniform")
    ax.legend()

    if save_fig:
        out_png = section / f"pit_hist_{base_var}.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"[probabilistic-plots] saved {out_png}")
    if save_npz:
        out_npz = section / f"pit_hist_{base_var}.npz"
        np.savez(out_npz, counts=counts, edges=edges, variable=base_var)
        print(f"[probabilistic-plots] saved {out_npz}")
    plt.close(fig)
