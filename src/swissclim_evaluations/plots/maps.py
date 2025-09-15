from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def run(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    seed = int(plotting_cfg.get("random_seed", 42))

    section_output = out_root / "maps"

    random.seed(seed)
    time_index = 0
    lead_index = 0
    time_selected = None
    if "init_time" in ds.dims and ds.init_time.size > 0:
        time_index = random.randint(0, ds.init_time.size - 1)
        time_selected = ds.init_time[time_index]
    if "lead_time" in ds.dims and ds.lead_time.size > 0:
        lead_index = 0
    time_fmt = (
        time_selected.dt.strftime("%Y%m%d%H%M").item()
        if (time_selected is not None and hasattr(time_selected, "dt"))
        else (
            str(time_selected.values) if time_selected is not None else "notime"
        )
    )

    # Determine variables
    variables_2d = [v for v in ds.data_vars if "level" not in ds[v].dims]
    variables_3d = [v for v in ds.data_vars if "level" in ds[v].dims]

    # 2D maps
    for i, var in enumerate(variables_2d):
        print(f"[maps] 2D variable: {var}")
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 4),
            dpi=dpi * 2,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        ds_var = ds[var]
        ds_ml_var = ds_ml[var]
        if "init_time" in ds_var.dims:
            ds_var = ds_var.isel(init_time=time_index)
        if "lead_time" in ds_var.dims:
            ds_var = ds_var.isel(lead_time=lead_index)
        if "init_time" in ds_ml_var.dims:
            ds_ml_var = ds_ml_var.isel(init_time=time_index)
        if "lead_time" in ds_ml_var.dims:
            ds_ml_var = ds_ml_var.isel(lead_time=lead_index)

        vmin = min(float(ds_var.min()), float(ds_ml_var.min()))
        vmax = max(float(ds_var.max()), float(ds_ml_var.max()))

        im0 = ds_var.plot(
            ax=axes[0],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )
        axes[0].add_feature(cfeature.BORDERS, linewidth=0.5)
        axes[0].coastlines(linewidth=0.5)
        axes[0].set_title("Ground Truth")

        ds_ml_var.plot(
            ax=axes[1],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )
        axes[1].add_feature(cfeature.BORDERS, linewidth=0.5)
        axes[1].coastlines(linewidth=0.5)
        axes[1].set_title("Model Prediction")

        cbar_ax = plt.gcf().add_axes([0.15, 0.1, 0.7, 0.02])
        plt.colorbar(
            im0,
            cax=cbar_ax,
            orientation="horizontal",
            label=ds[var].attrs.get("units", ""),
        )

        if time_selected is not None:
            plt.suptitle(
                f"{var} at {str(time_selected.dt.date.values)} - {time_selected.dt.hour.values} UTC"
            )

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                section_output / f"{time_fmt}_{var}_sfc.png",
                bbox_inches="tight",
                dpi=200,
            )
        if save_npz:
            # Save the exact data used for this plot as NPZ for portability
            section_output.mkdir(parents=True, exist_ok=True)
            npz_path = section_output / f"{time_fmt}_{var}_sfc.npz"
            # store arrays and minimal coords
            np.savez(
                npz_path,
                nwp=ds_var.values,
                ml=ds_ml_var.values,
                latitude=ds_var.coords.get("latitude", None).values
                if "latitude" in ds_var.coords
                else None,
                longitude=ds_var.coords.get("longitude", None).values
                if "longitude" in ds_var.coords
                else None,
            )

        plt.close(fig)

    # 3D maps per level
    for i, var in enumerate(variables_3d):
        print(f"[maps] 3D variable: {var}")
        levels = list(ds[var].coords.get("level", []))
        if not levels:
            continue
        num_rows = len(levels)
        fig, axes = plt.subplots(
            num_rows,
            2,
            figsize=(14, 4 * num_rows),
            dpi=dpi * 2,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        ds_var = ds[var]
        ds_ml_var = ds_ml[var]
        if "init_time" in ds_var.dims:
            ds_var = ds_var.isel(init_time=time_index)
        if "lead_time" in ds_var.dims:
            ds_var = ds_var.isel(lead_time=lead_index)
        if "init_time" in ds_ml_var.dims:
            ds_ml_var = ds_ml_var.isel(init_time=time_index)
        if "lead_time" in ds_ml_var.dims:
            ds_ml_var = ds_ml_var.isel(lead_time=lead_index)
        vmin = min(float(ds_var.min()), float(ds_ml_var.min()))
        vmax = max(float(ds_var.max()), float(ds_ml_var.max()))

        for idx, level in enumerate(levels):
            ax_ds = axes[idx, 0]
            ds_var_lev = ds_var.sel(level=level)
            ds_ml_var_lev = ds_ml_var.sel(level=level)

            im_ds = ds_var_lev.plot(
                ax=ax_ds,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                transform=ccrs.PlateCarree(),
            )
            ax_ds.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax_ds.coastlines(linewidth=0.5)
            ax_ds.set_title(f"Ground Truth - Level {level}")

            ax_ds_ml = axes[idx, 1]
            ds_ml_var_lev.plot(
                ax=ax_ds_ml,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                transform=ccrs.PlateCarree(),
            )
            ax_ds_ml.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax_ds_ml.coastlines(linewidth=0.5)
            ax_ds_ml.set_title(f"Model - Level {level}")

        cbar_ax = plt.gcf().add_axes([0.15, 0.05, 0.7, 0.02])
        plt.colorbar(
            im_ds,
            cax=cbar_ax,
            orientation="horizontal",
            label=ds[var].attrs.get("units", ""),
        )

        if time_selected is not None:
            plt.suptitle(
                f"{var} at {str(time_selected.dt.date.values)} - {time_selected.dt.hour.values} UTC"
            )
        plt.subplots_adjust(bottom=0.1, top=0.95, hspace=0.05, wspace=0.05)

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                section_output / f"{time_fmt}_{var}_pl.png",
                bbox_inches="tight",
                dpi=200,
            )
        if save_npz:
            section_output.mkdir(parents=True, exist_ok=True)
            npz_path = section_output / f"{time_fmt}_{var}_pl.npz"
            np.savez(
                npz_path,
                nwp=ds_var.values,
                ml=ds_ml_var.values,
                latitude=ds_var.coords.get("latitude", None).values
                if "latitude" in ds_var.coords
                else None,
                longitude=ds_var.coords.get("longitude", None).values
                if "longitude" in ds_var.coords
                else None,
                level=ds_var.coords.get("level", None).values
                if "level" in ds_var.coords
                else None,
            )
        plt.close(fig)
