from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr


def run(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    outputs = plotting_cfg.get(
        "outputs", ["file"]
    )  # ["file", "cell", "cell-first"]
    dpi = int(plotting_cfg.get("dpi", 48))
    seed = int(plotting_cfg.get("random_seed", 42))
    save_plot_data = bool(plotting_cfg.get("save_plot_data", False))

    section_output = out_root / "maps"

    random.seed(seed)
    time_index = 0
    if "time" in ds.dims and ds.time.size > 0:
        time_index = random.randint(0, ds.time.size - 1)
        time_selected = ds.time[time_index]
        time_fmt = (
            time_selected.dt.strftime("%Y%m%d%H%M").item()
            if hasattr(time_selected, "dt")
            else str(time_selected.values)
        )
    else:
        time_selected = None
        time_fmt = "notime"

    # Determine variables
    variables_2d = [v for v in ds.data_vars if "level" not in ds[v].dims]
    variables_3d = [v for v in ds.data_vars if "level" in ds[v].dims]

    # 2D maps
    for i, var in enumerate(variables_2d):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 4),
            dpi=dpi * 2,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        ds_var = (
            ds[var].isel(time=time_index) if "time" in ds[var].dims else ds[var]
        )
        ds_ml_var = (
            ds_ml[var].isel(time=time_index)
            if "time" in ds_ml[var].dims
            else ds_ml[var]
        )

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
        axes[0].set_title("Ground Truth - ERA5")

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
        axes[1].set_title("Model Prediction - SwissAI")

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

        if "file" in outputs:
            section_output.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                section_output / f"{time_fmt}_{var}_sfc.png",
                bbox_inches="tight",
                dpi=200,
            )
        if save_plot_data:
            # Save the exact data used for this plot
            ds_save = xr.Dataset({
                "nwp": ds_var,
                "ml": ds_ml_var,
            })
            section_output.mkdir(parents=True, exist_ok=True)
            ds_save.to_netcdf(section_output / f"{time_fmt}_{var}_sfc.nc")

        if i == 0 and "cell-first" in outputs:
            plt.show()
        if i > 0 and "cell" not in outputs:
            plt.close(fig)

    # 3D maps per level
    for i, var in enumerate(variables_3d):
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

        ds_var = (
            ds[var].isel(time=time_index) if "time" in ds[var].dims else ds[var]
        )
        ds_ml_var = (
            ds_ml[var].isel(time=time_index)
            if "time" in ds_ml[var].dims
            else ds_ml[var]
        )
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
            ax_ds_ml.set_title(f"SwissAI - Level {level}")

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

        if "file" in outputs:
            section_output.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                section_output / f"{time_fmt}_{var}_pl.png",
                bbox_inches="tight",
                dpi=200,
            )
        if save_plot_data:
            ds_save = xr.Dataset({
                "nwp": ds_var,
                "ml": ds_ml_var,
            })
            section_output.mkdir(parents=True, exist_ok=True)
            ds_save.to_netcdf(section_output / f"{time_fmt}_{var}_pl.nc")
        if i == 0 and "cell-first" in outputs:
            plt.show()
        if i > 0 and "cell" not in outputs:
            plt.close(fig)
