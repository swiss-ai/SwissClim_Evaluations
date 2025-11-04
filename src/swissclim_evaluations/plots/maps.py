from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    resolve_ensemble_mode,
)


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    ensemble_mode: str | None = None,
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    seed = int(plotting_cfg.get("random_seed", 42))

    section_output = out_root / "maps"

    rng = np.random.default_rng(seed)
    time_index = 0
    lead_index = 0
    time_selected = None
    if "init_time" in ds_target.dims and ds_target.init_time.size > 0:
        time_index = int(rng.integers(0, ds_target.init_time.size))
        time_selected = ds_target.init_time[time_index]
    if "lead_time" in ds_target.dims and ds_target.lead_time.size > 0:
        lead_index = 0
    if "time" in ds_target.dims and ds_target.time.size > 0:
        time_index = 0

    # Extract time/lead ranges for naming helper
    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            start = np.datetime64(vals.min()).astype("datetime64[h]")
            end = np.datetime64(vals.max()).astype("datetime64[h]")

            def _fmt(x):
                return (
                    np.datetime_as_string(x, unit="h")
                    .replace("-", "")
                    .replace(":", "")
                    .replace("T", "")
                )

            return (_fmt(start), _fmt(end))
        except Exception:
            return None

    def _extract_lead_range(ds: xr.Dataset):
        if "lead_time" not in ds:
            return None
        try:
            vals = ds["lead_time"].values
            if vals.size == 0:
                return None
            hours = (vals / np.timedelta64(1, "h")).astype(int)
            sh = int(hours.min())
            eh = int(hours.max())

            def _fmt(h: int) -> str:
                return f"{h:03d}h"

            return (_fmt(sh), _fmt(eh))
        except Exception:
            return None

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    # Determine variables
    if "level" in ds_target.dims and int(ds_target.level.size) > 1:
        variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
        variables_2d = [v for v in ds_target.data_vars if v not in variables_3d]
    else:
        variables_3d = []
        variables_2d = list(ds_target.data_vars)

    # Assume no missing data per project requirement; use direct min/max.

    # Resolve ensemble handling (maps: mean/pooled/members/none). Prob invalid.
    resolved_mode = resolve_ensemble_mode("maps", ensemble_mode, ds_target, ds_prediction)
    has_ens = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for maps")
    if resolved_mode == "none" and has_ens:
        resolved_mode = "mean"  # degrade to historical behaviour

    if resolved_mode == "mean" and has_ens:
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble")
        if "ensemble" in ds_prediction.dims:
            ds_prediction = ds_prediction.mean(dim="ensemble")
        ensemble_members: list[int | None] = [None]
        ens_token_global = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled" and has_ens:
        ensemble_members = [None]
        ens_token_global = ensemble_mode_to_token("pooled")
    elif resolved_mode == "members" and has_ens:
        # enumerate members; token set per iteration
        if "ensemble" in ds_prediction.dims:
            ensemble_members = [int(i) for i in range(int(ds_prediction.sizes["ensemble"]))]
        else:
            ensemble_members = [int(i) for i in range(int(ds_target.sizes["ensemble"]))]
        ens_token_global = None
    else:
        ensemble_members = [None]
        ens_token_global = None

    # 2D maps (one figure per ensemble member if present)
    for _i, var in enumerate(variables_2d):  # _i unused (ruff B007)
        print(f"[maps] 2D variable: {var}")
        for ens in ensemble_members:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(14, 4),
                dpi=dpi * 2,
                subplot_kw={"projection": ccrs.PlateCarree()},
                constrained_layout=True,
            )

            ds_var = ds_target[var]
            ds_ml_var = ds_prediction[var]
            if ens is not None:
                if "ensemble" in ds_var.dims:
                    ds_var = ds_var.isel(ensemble=ens)
                if "ensemble" in ds_ml_var.dims:
                    ds_ml_var = ds_ml_var.isel(ensemble=ens)
            if "init_time" in ds_var.dims:
                ds_var = ds_var.isel(init_time=time_index)
            if "lead_time" in ds_var.dims:
                ds_var = ds_var.isel(lead_time=lead_index)
            if "time" in ds_var.dims:
                ds_var = ds_var.isel(time=time_index)
            if "init_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(init_time=time_index)
            if "lead_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(lead_time=lead_index)
            if "time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(time=time_index)
            # Drop any remaining singleton temporal dims
            for dim_drop in ("time", "init_time", "lead_time"):
                if dim_drop in ds_var.dims and ds_var.sizes[dim_drop] == 1:
                    ds_var = ds_var.isel({dim_drop: 0})
                if dim_drop in ds_ml_var.dims and ds_ml_var.sizes[dim_drop] == 1:
                    ds_ml_var = ds_ml_var.isel({dim_drop: 0})
            ds_var = ds_var.squeeze()
            ds_ml_var = ds_ml_var.squeeze()
            vmin = min(float(ds_var.min()), float(ds_ml_var.min()))
            vmax = max(float(ds_var.max()), float(ds_ml_var.max()))

            lon = ds_var.coords.get("longitude", None)
            lat = ds_var.coords.get("latitude", None)
            im0 = axes[0].pcolormesh(
                lon if lon is not None else ds_var.longitude,
                lat if lat is not None else ds_var.latitude,
                ds_var.values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )
            axes[0].add_feature(cfeature.BORDERS, linewidth=0.5)
            axes[0].coastlines(linewidth=0.5)
            axes[0].set_title("Ground Truth")

            lon_ml = ds_ml_var.coords.get("longitude", None)
            lat_ml = ds_ml_var.coords.get("latitude", None)
            axes[1].pcolormesh(
                lon_ml if lon_ml is not None else ds_ml_var.longitude,
                lat_ml if lat_ml is not None else ds_ml_var.latitude,
                ds_ml_var.values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )
            axes[1].add_feature(cfeature.BORDERS, linewidth=0.5)
            axes[1].coastlines(linewidth=0.5)
            axes[1].set_title("Model Prediction")

            # Colorbar spanning both axes
            cb = fig.colorbar(
                im0,
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.08,
            )
            try:
                if cb is not None:
                    cb.set_label(ds_target[var].attrs.get("units", ""))
            except Exception:
                pass

            title_extra = "" if ens is None else f" (Ensemble {ens})"
            if time_selected is not None:
                plt.suptitle(
                    f"{var}{title_extra} at {str(time_selected.dt.date.values)} - "
                    f"{time_selected.dt.hour.values} UTC"
                )

            ens_token = (
                ensemble_mode_to_token("members", ens)
                if (resolved_mode == "members" and ens is not None)
                else ens_token_global
            )
            if save_fig:
                section_output.mkdir(parents=True, exist_ok=True)
                out_png = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level=None,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[maps] saved {out_png}")
            if save_npz:
                section_output.mkdir(parents=True, exist_ok=True)
                npz_path = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level=None,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="npz",
                )
                np.savez(
                    npz_path,
                    nwp=ds_var.values,
                    ml=ds_ml_var.values,
                    latitude=(
                        ds_var.latitude.values if "latitude" in ds_var.coords else np.array([])
                    ),
                    longitude=(
                        ds_var.longitude.values if "longitude" in ds_var.coords else np.array([])
                    ),
                    ensemble=int(ens) if ens is not None else -1,
                )
                print(f"[maps] saved {npz_path}")
            plt.close(fig)

        # Optional grid of subplots for multiple lead_times
        per_lead_grid = bool(plotting_cfg.get("maps_per_lead_grid", False)) and (
            "lead_time" in ds_prediction.dims and int(ds_prediction.lead_time.size) > 1
        )
        if per_lead_grid:
            # Select panel hours if provided; else first N
            panel_hours = plotting_cfg.get("lead_panel_hours")
            all_hours = None
            try:
                vals = ds_prediction["lead_time"].values
                all_hours = [int(v / np.timedelta64(1, "h")) for v in vals]
            except Exception:
                pass
            if not panel_hours and all_hours:
                max_panels = int(plotting_cfg.get("maps_max_panels", 4))
                panel_hours = all_hours[:max_panels]
            if panel_hours:
                rows = len(panel_hours)
                fig, axes = plt.subplots(
                    rows,
                    2,
                    figsize=(14, 4 * rows),
                    dpi=dpi * 2,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    squeeze=False,
                    constrained_layout=True,
                )
                for r, h in enumerate(panel_hours):
                    # locate index of this hour
                    try:
                        idx = all_hours.index(int(h)) if all_hours is not None else r
                    except Exception:
                        idx = r
                    ds_var = ds_target[var]
                    ds_ml_var = ds_prediction[var]
                    if ens is not None:
                        if "ensemble" in ds_var.dims:
                            ds_var = ds_var.isel(ensemble=ens)
                        if "ensemble" in ds_ml_var.dims:
                            ds_ml_var = ds_ml_var.isel(ensemble=ens)
                    if "lead_time" in ds_var.dims:
                        ds_var = ds_var.isel(lead_time=idx)
                    if "lead_time" in ds_ml_var.dims:
                        ds_ml_var = ds_ml_var.isel(lead_time=idx)
                    for dim_drop in ("time", "init_time", "lead_time"):
                        if dim_drop in ds_var.dims and ds_var.sizes[dim_drop] == 1:
                            ds_var = ds_var.isel({dim_drop: 0})
                        if dim_drop in ds_ml_var.dims and ds_ml_var.sizes[dim_drop] == 1:
                            ds_ml_var = ds_ml_var.isel({dim_drop: 0})
                    ds_var = ds_var.squeeze()
                    ds_ml_var = ds_ml_var.squeeze()
                    vmin = min(float(ds_var.min()), float(ds_ml_var.min()))
                    vmax = max(float(ds_var.max()), float(ds_ml_var.max()))
                    lon = ds_var.coords.get("longitude", None)
                    lat = ds_var.coords.get("latitude", None)
                    axes[r, 0].pcolormesh(
                        lon if lon is not None else ds_var.longitude,
                        lat if lat is not None else ds_var.latitude,
                        ds_var.values,
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                    )
                    axes[r, 0].add_feature(cfeature.BORDERS, linewidth=0.5)
                    axes[r, 0].coastlines(linewidth=0.5)
                    axes[r, 0].set_title(f"Ground Truth — lead {int(h)}h")
                    lon_ml = ds_ml_var.coords.get("longitude", None)
                    lat_ml = ds_ml_var.coords.get("latitude", None)
                    axes[r, 1].pcolormesh(
                        lon_ml if lon_ml is not None else ds_ml_var.longitude,
                        lat_ml if lat_ml is not None else ds_ml_var.latitude,
                        ds_ml_var.values,
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                    )
                    axes[r, 1].add_feature(cfeature.BORDERS, linewidth=0.5)
                    axes[r, 1].coastlines(linewidth=0.5)
                    axes[r, 1].set_title(f"Model Prediction — lead {int(h)}h")
                # save combined grid
                ens_token = (
                    ensemble_mode_to_token("members", ens)
                    if (resolved_mode == "members" and ens is not None)
                    else ens_token_global
                )
                out_png = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level=None,
                    qualifier="grid",
                    init_time_range=init_range,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="png",
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[maps] saved {out_png}")
                plt.close(fig)

    # 3D maps per level (one figure with rows per level)
    for _i, var in enumerate(variables_3d):
        print(f"[maps] 3D variable: {var}")
        levels = list(ds_target[var].coords.get("level", []))
        if not levels:
            continue
        num_rows = len(levels)
        for ens in ensemble_members:
            fig, axes = plt.subplots(
                num_rows,
                2,
                figsize=(14, 4 * num_rows),
                dpi=dpi * 2,
                subplot_kw={"projection": ccrs.PlateCarree()},
                squeeze=False,
                constrained_layout=True,
            )

            for r, lvl in enumerate(levels):
                ds_var = ds_target[var].sel(level=lvl)
                ds_ml_var = ds_prediction[var].sel(level=lvl)
                if ens is not None:
                    if "ensemble" in ds_var.dims:
                        ds_var = ds_var.isel(ensemble=ens)
                    if "ensemble" in ds_ml_var.dims:
                        ds_ml_var = ds_ml_var.isel(ensemble=ens)
                if "init_time" in ds_var.dims:
                    ds_var = ds_var.isel(init_time=time_index)
                if "lead_time" in ds_var.dims:
                    ds_var = ds_var.isel(lead_time=lead_index)
                if "time" in ds_var.dims:
                    ds_var = ds_var.isel(time=time_index)
                if "init_time" in ds_ml_var.dims:
                    ds_ml_var = ds_ml_var.isel(init_time=time_index)
                if "lead_time" in ds_ml_var.dims:
                    ds_ml_var = ds_ml_var.isel(lead_time=lead_index)
                if "time" in ds_ml_var.dims:
                    ds_ml_var = ds_ml_var.isel(time=time_index)
                for dim_drop in ("time", "init_time", "lead_time"):
                    if dim_drop in ds_var.dims and ds_var.sizes[dim_drop] == 1:
                        ds_var = ds_var.isel({dim_drop: 0})
                    if dim_drop in ds_ml_var.dims and ds_ml_var.sizes[dim_drop] == 1:
                        ds_ml_var = ds_ml_var.isel({dim_drop: 0})
                ds_var = ds_var.squeeze()
                ds_ml_var = ds_ml_var.squeeze()
                vmin = min(float(ds_var.min()), float(ds_ml_var.min()))
                vmax = max(float(ds_var.max()), float(ds_ml_var.max()))

                lon = ds_var.coords.get("longitude", None)
                lat = ds_var.coords.get("latitude", None)
                im0 = axes[r, 0].pcolormesh(
                    lon if lon is not None else ds_var.longitude,
                    lat if lat is not None else ds_var.latitude,
                    ds_var.values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                axes[r, 0].add_feature(cfeature.BORDERS, linewidth=0.5)
                axes[r, 0].coastlines(linewidth=0.5)
                axes[r, 0].set_title(f"Ground Truth — level {lvl}")

                lon_ml = ds_ml_var.coords.get("longitude", None)
                lat_ml = ds_ml_var.coords.get("latitude", None)
                axes[r, 1].pcolormesh(
                    lon_ml if lon_ml is not None else ds_ml_var.longitude,
                    lat_ml if lat_ml is not None else ds_ml_var.latitude,
                    ds_ml_var.values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                axes[r, 1].add_feature(cfeature.BORDERS, linewidth=0.5)
                axes[r, 1].coastlines(linewidth=0.5)
                axes[r, 1].set_title(f"Model Prediction — level {lvl}")

            ens_token = (
                ensemble_mode_to_token("members", ens)
                if (resolved_mode == "members" and ens is not None)
                else ens_token_global
            )
            if save_fig:
                section_output.mkdir(parents=True, exist_ok=True)
                out_png = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level=None,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[maps] saved {out_png}")
            if save_npz:
                section_output.mkdir(parents=True, exist_ok=True)
                npz_path = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level=None,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="npz",
                )
                # For 3D, store concatenated arrays along a new first axis per level
                nwp_stack = []
                ml_stack = []
                lat_arr = None
                lon_arr = None
                for lvl in levels:
                    dv = ds_target[var].sel(level=lvl).squeeze()
                    dm = ds_prediction[var].sel(level=lvl).squeeze()
                    nwp_stack.append(dv.values)
                    ml_stack.append(dm.values)
                    if lat_arr is None and "latitude" in dv.coords:
                        lat_arr = dv.latitude.values
                    if lon_arr is None and "longitude" in dv.coords:
                        lon_arr = dv.longitude.values
                np.savez(
                    npz_path,
                    nwp=np.array(nwp_stack, dtype=object),
                    ml=np.array(ml_stack, dtype=object),
                    latitude=(lat_arr if lat_arr is not None else np.array([])),
                    longitude=(lon_arr if lon_arr is not None else np.array([])),
                    level=np.array(levels),
                    ensemble=int(ens) if ens is not None else -1,
                    allow_pickle=True,
                )
                print(f"[maps] saved {npz_path}")
            plt.close(fig)
