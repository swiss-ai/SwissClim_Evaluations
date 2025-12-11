from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_init_time_range,
    format_level_label,
    format_level_token,
    format_variable_name,
    get_colormap_for_variable,
    get_variable_units,
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
    if "init_time" in ds_target.dims and ds_target.init_time.size > 0:
        time_index = int(rng.integers(0, ds_target.init_time.size))
    if "lead_time" in ds_target.dims and ds_target.lead_time.size > 0:
        lead_index = 0
    if "time" in ds_target.dims and ds_target.time.size > 0:
        time_index = 0

    time_selected = None
    if "init_time" in ds_target.dims and ds_target.init_time.size > time_index:
        time_selected = ds_target.init_time.isel(init_time=time_index)

    # Extract time/lead ranges for naming helper
    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            return format_init_time_range(vals)
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

    def _plot_maps_grid(
        da_target: xr.DataArray,
        da_pred: xr.DataArray,
        variable_name: str,
        ens_token: str | None,
        init_time_idx: int,
    ) -> None:
        if "lead_time" not in da_pred.dims:
            return
        leads = da_pred["lead_time"].values
        if len(leads) < 2:
            return

        # Select a subset of leads to plot (max 6 leads -> 12 panels)
        n_leads = len(leads)
        if n_leads > 6:
            indices = np.linspace(0, n_leads - 1, 6, dtype=int)
            selected_leads = leads[indices]
        else:
            selected_leads = leads

        n_sel = len(selected_leads)
        rows = n_sel
        cols = 2  # Target, Prediction

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(12, 4 * rows),
            dpi=dpi,
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)

        # Helper: unwrap wrapped longitudes for plotting
        def _unwrap_lon_for_plot(da: xr.DataArray) -> xr.DataArray:
            if "longitude" not in da.coords:
                return da
            lons = np.asarray(da.longitude.values)
            if lons.size == 0:
                return da
            lmin, lmax = float(np.min(lons)), float(np.max(lons))
            if (lmax - lmin) > 180 and np.any(lons < 90) and np.any(lons > 270):
                new = lons.copy()
                new[new > 180] -= 360
                order = np.argsort(new)
                da = da.isel(longitude=order).assign_coords(longitude=("longitude", new[order]))
            return da

        # Compute global vmin/vmax from first and last selected lead to save time
        # (approximate but usually sufficient)
        def _get_min_max(da, lt):
            if np.issubdtype(np.asarray(leads).dtype, np.timedelta64):
                d = da.sel(lead_time=lt)
            else:
                d = da.sel(lead_time=lt)
            return float(d.min()), float(d.max())

        vmin, vmax = float("inf"), float("-inf")
        for lt in [selected_leads[0], selected_leads[-1]]:
            if "lead_time" in da_target.dims:
                mn, mx = _get_min_max(da_target, lt)
                vmin = min(vmin, mn)
                vmax = max(vmax, mx)
            mn, mx = _get_min_max(da_pred, lt)
            vmin = min(vmin, mn)
            vmax = max(vmax, mx)

        for i, lt in enumerate(selected_leads):
            if np.issubdtype(np.asarray(leads).dtype, np.timedelta64):
                lt_sel = lt
                hours = int(lt / np.timedelta64(1, "h"))
                label = f"{hours}h"
            else:
                lt_sel = lt
                label = str(lt)

            ds_var = da_target.sel(lead_time=lt_sel) if "lead_time" in da_target.dims else da_target
            ds_ml_var = da_pred.sel(lead_time=lt_sel)

            # Handle init_time selection
            if "init_time" in ds_var.dims:
                ds_var = ds_var.isel(init_time=init_time_idx)
            if "init_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(init_time=init_time_idx)

            ds_var = ds_var.squeeze()
            ds_ml_var = ds_ml_var.squeeze()
            ds_var = _unwrap_lon_for_plot(ds_var)
            ds_ml_var = _unwrap_lon_for_plot(ds_ml_var)

            lon = ds_var.coords.get("longitude", None)
            lat = ds_var.coords.get("latitude", None)

            # Target
            im = axes[i, 0].pcolormesh(
                lon if lon is not None else ds_var.longitude,
                lat if lat is not None else ds_var.latitude,
                ds_var.values,
                cmap=get_colormap_for_variable(variable_name),
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )
            axes[i, 0].coastlines(linewidth=0.5)
            axes[i, 0].set_title(f"Target — Lead {label}")

            # Prediction
            lon_ml = ds_ml_var.coords.get("longitude", None)
            lat_ml = ds_ml_var.coords.get("latitude", None)
            axes[i, 1].pcolormesh(
                lon_ml if lon_ml is not None else ds_ml_var.longitude,
                lat_ml if lat_ml is not None else ds_ml_var.latitude,
                ds_ml_var.values,
                cmap=get_colormap_for_variable(variable_name),
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )
            axes[i, 1].coastlines(linewidth=0.5)
            axes[i, 1].set_title(f"Prediction — Lead {label}")

        # Colorbar
        cb = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
        try:
            if cb is not None:
                cb.set_label(get_variable_units(ds_target, variable_name))
        except Exception:
            pass

        plt.suptitle(f"Maps Evolution — {format_variable_name(variable_name)}")

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png = section_output / build_output_filename(
                metric="map_grid",
                variable=variable_name,
                level="surface",
                qualifier=None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="png",
            )
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"[maps] saved {out_png}")
        plt.close(fig)

    # Determine variables
    if "level" in ds_target.dims and int(ds_target.level.size) > 1:
        variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
        variables_2d = [v for v in ds_target.data_vars if v not in variables_3d]
    else:
        variables_3d = []
        variables_2d = list(ds_target.data_vars)

    # Assume no missing data per project requirement; use direct min/max.

    # Resolve ensemble handling (maps: mean/pooled/members). Prob invalid.
    resolved_mode = resolve_ensemble_mode("maps", ensemble_mode, ds_target, ds_prediction)
    has_ens = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for maps")

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
            # Helper: unwrap wrapped longitudes for plotting (e.g. 335..360 U 0..45 -> -25..45)
            def _unwrap_lon_for_plot(da: xr.DataArray) -> xr.DataArray:
                if "longitude" not in da.coords:
                    return da
                lons = np.asarray(da.longitude.values)
                if lons.size == 0:
                    return da
                lmin, lmax = float(np.min(lons)), float(np.max(lons))
                # Heuristic: large span plus presence of values on both sides of Greenwich
                if (lmax - lmin) > 180 and np.any(lons < 90) and np.any(lons > 270):
                    new = lons.copy()
                    new[new > 180] -= 360  # shift western segment to negative degrees
                    order = np.argsort(new)
                    da = da.isel(longitude=order).assign_coords(longitude=("longitude", new[order]))
                return da

            ds_var_full = ds_target[var]
            ds_ml_var_full = ds_prediction[var]

            # Check if original variable has multiple init times (before slicing)
            is_single_init = True
            if "init_time" in ds_var_full.dims and ds_var_full.sizes["init_time"] > 1:
                is_single_init = False

            if ens is not None:
                if "ensemble" in ds_var_full.dims:
                    ds_var_full = ds_var_full.isel(ensemble=ens)
                if "ensemble" in ds_ml_var_full.dims:
                    ds_ml_var_full = ds_ml_var_full.isel(ensemble=ens)
            if "init_time" in ds_var_full.dims:
                ds_var_full = ds_var_full.isel(init_time=time_index)
            if "init_time" in ds_ml_var_full.dims:
                ds_ml_var_full = ds_ml_var_full.isel(init_time=time_index)

            # Determine lead times to plot
            lead_indices = [0]
            lead_coords = [None]
            if "lead_time" in ds_ml_var_full.dims:
                lead_indices = list(range(ds_ml_var_full.sizes["lead_time"]))
                lead_coords = ds_ml_var_full["lead_time"].values
            elif "lead_time" in ds_var_full.dims:
                lead_indices = list(range(ds_var_full.sizes["lead_time"]))
                lead_coords = ds_var_full["lead_time"].values

            n_leads = len(lead_indices)

            # Compute global vmin/vmax for consistent color scale
            vmin = min(float(ds_var_full.min()), float(ds_ml_var_full.min()))
            vmax = max(float(ds_var_full.max()), float(ds_ml_var_full.max()))

            fig, axes = plt.subplots(
                n_leads,
                2,
                figsize=(14, 6 * n_leads),
                dpi=dpi * 2,
                subplot_kw={"projection": ccrs.PlateCarree()},
                constrained_layout=True,
            )
            if n_leads == 1:
                axes = np.array([axes])

            im0 = None
            for i, lead_idx in enumerate(lead_indices):
                ax_tgt = axes[i, 0]
                ax_pred = axes[i, 1]

                ds_var = ds_var_full
                ds_ml_var = ds_ml_var_full

                if "lead_time" in ds_var.dims:
                    ds_var = ds_var.isel(lead_time=lead_idx)
                if "lead_time" in ds_ml_var.dims:
                    ds_ml_var = ds_ml_var.isel(lead_time=lead_idx)

                # Ensure we also select a single time slice if a free 'time' dimension exists
                if "time" in ds_var.dims:
                    ds_var = ds_var.isel(time=time_index)
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
                ds_var = _unwrap_lon_for_plot(ds_var)
                ds_ml_var = _unwrap_lon_for_plot(ds_ml_var)

                lon = ds_var.coords.get("longitude", None)
                lat = ds_var.coords.get("latitude", None)
                im0 = ax_tgt.pcolormesh(
                    lon if lon is not None else ds_var.longitude,
                    lat if lat is not None else ds_var.latitude,
                    ds_var.values,
                    cmap=get_colormap_for_variable(str(var)),
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                ax_tgt.coastlines(linewidth=0.5)
                # Zoom to data extent
                _lon = lon.values if lon is not None else ds_var.longitude.values
                _lat = lat.values if lat is not None else ds_var.latitude.values
                if hasattr(ax_tgt, "set_extent"):
                    ax_tgt.set_extent(
                        [
                            float(np.min(_lon)),
                            float(np.max(_lon)),
                            float(np.min(_lat)),
                            float(np.max(_lat)),
                        ],
                        crs=ccrs.PlateCarree(),
                    )

                # Title
                lead_str = ""
                if lead_coords[i] is not None:
                    val = lead_coords[i]
                    if np.issubdtype(type(val), np.timedelta64):
                        h = int(val / np.timedelta64(1, "h"))
                        lead_str = f" (+{h}h)"
                    else:
                        lead_str = f" (lead={val})"

                date_str = extract_date_from_dataset(ds_var) if is_single_init else ""
                ax_tgt.set_title(f"{format_variable_name(str(var))} — Target{date_str}{lead_str}")

                lon_ml = ds_ml_var.coords.get("longitude", None)
                lat_ml = ds_ml_var.coords.get("latitude", None)
                ax_pred.pcolormesh(
                    lon_ml if lon_ml is not None else ds_ml_var.longitude,
                    lat_ml if lat_ml is not None else ds_ml_var.latitude,
                    ds_ml_var.values,
                    cmap=get_colormap_for_variable(str(var)),
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                ax_pred.coastlines(linewidth=0.5)
                _lon = lon_ml.values if lon_ml is not None else ds_ml_var.longitude.values
                _lat = lat_ml.values if lat_ml is not None else ds_ml_var.latitude.values
                if hasattr(ax_pred, "set_extent"):
                    ax_pred.set_extent(
                        [
                            float(np.min(_lon)),
                            float(np.max(_lon)),
                            float(np.min(_lat)),
                            float(np.max(_lat)),
                        ],
                        crs=ccrs.PlateCarree(),
                    )
                ax_pred.set_title(f"Prediction{lead_str}")

            # Colorbar spanning both axes — vertical to save vertical space
            cb = fig.colorbar(
                im0,
                ax=axes,
                orientation="vertical",
                fraction=0.025,
                pad=0.02,
            )
            # In test mode, colorbar may be a dummy (None); guard the label call
            try:
                if cb is not None:
                    cb.set_label(get_variable_units(ds_target, str(var)))
            except Exception:
                # Non-fatal: continue without setting label
                pass

            title_extra = "" if ens is None else f" (Ensemble {ens})"
            if time_selected is not None:
                # Robust formatting without relying on .dt accessor
                ts_val = np.asarray(time_selected.values, dtype="datetime64[h]")
                init_label = str(np.datetime_as_string(ts_val)).replace("T", " ") + "Z"
                ensemble_label = ens_token_global or (
                    "member " + str(ens) if ens is not None else "none"
                )
                title_text = (
                    f"Maps — {var}{title_extra} | ensemble={ensemble_label} | "
                    f"init_time={init_label} | lead_range={lead_range}"
                )
                plt.suptitle(title_text, fontsize=14)

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
                    level="surface",
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
                    level="surface",
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="npz",
                )
                np.savez(
                    npz_path,
                    target=ds_var.values,
                    prediction=ds_ml_var.values,
                    latitude=(
                        ds_var.latitude.values if "latitude" in ds_var.coords else np.array([])
                    ),
                    longitude=(
                        ds_var.longitude.values if "longitude" in ds_var.coords else np.array([])
                    ),
                    ensemble=int(ens) if ens is not None else -1,
                    variable=str(var),
                    units=get_variable_units(ds_target, str(var)),
                )
                print(f"[maps] saved {npz_path}")
            plt.close(fig)

            # Optional grid of subplots for multiple lead_times PER ENSEMBLE MEMBER
            # Force grid generation whenever lead_time is present (>=1)
            do_grid = ("lead_time" in ds_prediction.dims) and int(ds_prediction.lead_time.size) > 1
            if do_grid:
                ds_var_full = ds_target[var]
                ds_ml_var_full = ds_prediction[var]
                if ens is not None:
                    if "ensemble" in ds_var_full.dims:
                        ds_var_full = ds_var_full.isel(ensemble=ens)
                    if "ensemble" in ds_ml_var_full.dims:
                        ds_ml_var_full = ds_ml_var_full.isel(ensemble=ens)

                _plot_maps_grid(
                    ds_var_full,
                    ds_ml_var_full,
                    str(var),
                    ens_token,
                    time_index,
                )

    # 3D maps per level (one figure with rows per level)
    for _i, var in enumerate(variables_3d):
        print(f"[maps] 3D variable: {var}")
        levels = list(ds_target[var].coords.get("level", []))
        if not levels:
            continue

        level_tokens = [format_level_token(lvl) for lvl in levels]
        if not level_tokens:
            level_label = "levels"
        elif len(level_tokens) == 1:
            level_label = level_tokens[0]
        else:
            level_label = f"{level_tokens[0]}_to_{level_tokens[-1]}"
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

            ds_var = ds_target[var]
            ds_ml_var = ds_prediction[var]

            # Check if original variable has multiple init times (before slicing)
            is_single_init = True
            if "init_time" in ds_var.dims and ds_var.sizes["init_time"] > 1:
                is_single_init = False

            if ens is not None:
                if "ensemble" in ds_var.dims:
                    ds_var = ds_var.isel(ensemble=ens)
                if "ensemble" in ds_ml_var.dims:
                    ds_ml_var = ds_ml_var.isel(ensemble=ens)
            if "init_time" in ds_var.dims:
                ds_var = ds_var.isel(init_time=time_index)
            if "lead_time" in ds_var.dims:
                ds_var = ds_var.isel(lead_time=lead_index)
            if "init_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(init_time=time_index)
            if "lead_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(lead_time=lead_index)
            # Ensure we also select a single time slice if a free 'time' dimension exists
            if "time" in ds_var.dims:
                ds_var = ds_var.isel(time=time_index)
            if "time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(time=time_index)

            for idx, level in enumerate(levels):
                level_val = int(level.values) if hasattr(level, "values") else int(level)
                ax_ds, ax_ds_ml = axes[idx]
                ds_var_lev = ds_var.sel(level=level)
                ds_ml_var_lev = ds_ml_var.sel(level=level)
                # After selecting level we expect dims (latitude, longitude). If an extra
                # singleton dimension remains (e.g., time/init_time/lead_time), squeeze it.
                ds_var_lev = ds_var_lev.squeeze()
                ds_ml_var_lev = ds_ml_var_lev.squeeze()

                vmin = min(float(ds_var_lev.min()), float(ds_ml_var_lev.min()))
                vmax = max(float(ds_var_lev.max()), float(ds_ml_var_lev.max()))

                im_ds = ax_ds.pcolormesh(
                    ds_var_lev.coords.get("longitude"),
                    ds_var_lev.coords.get("latitude"),
                    ds_var_lev.values,
                    cmap=get_colormap_for_variable(str(var)),
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                ax_ds.coastlines(linewidth=0.5)

                # HEAD's set_extent logic for target
                lon = ds_var_lev.coords.get("longitude", None)
                lat = ds_var_lev.coords.get("latitude", None)
                _lon = lon.values if lon is not None else ds_var_lev.longitude.values
                _lat = lat.values if lat is not None else ds_var_lev.latitude.values
                if hasattr(ax_ds, "set_extent"):
                    ax_ds.set_extent(
                        [
                            float(np.min(_lon)),
                            float(np.max(_lon)),
                            float(np.min(_lat)),
                            float(np.max(_lat)),
                        ],
                        crs=ccrs.PlateCarree(),
                    )

                date_str = extract_date_from_dataset(ds_var) if is_single_init else ""
                ax_ds.set_title(
                    f"{format_variable_name(str(var))} — Target"
                    f"{format_level_label(level_val)}{date_str}"
                )

                ax_ds_ml.pcolormesh(
                    ds_ml_var_lev.coords.get("longitude"),
                    ds_ml_var_lev.coords.get("latitude"),
                    ds_ml_var_lev.values,
                    cmap=get_colormap_for_variable(str(var)),
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                ax_ds_ml.coastlines(linewidth=0.5)

                # HEAD's set_extent logic for prediction
                lon_ml = ds_ml_var_lev.coords.get("longitude", None)
                lat_ml = ds_ml_var_lev.coords.get("latitude", None)
                _lon = lon_ml.values if lon_ml is not None else ds_ml_var_lev.longitude.values
                _lat = lat_ml.values if lat_ml is not None else ds_ml_var_lev.latitude.values
                if hasattr(ax_ds_ml, "set_extent"):
                    ax_ds_ml.set_extent(
                        [
                            float(np.min(_lon)),
                            float(np.max(_lon)),
                            float(np.min(_lat)),
                            float(np.max(_lat)),
                        ],
                        crs=ccrs.PlateCarree(),
                    )

                ax_ds_ml.set_title(f"Model{format_level_label(level_val)}")

                fig.colorbar(
                    im_ds,
                    ax=[ax_ds, ax_ds_ml],
                    orientation="horizontal",
                    fraction=0.05,
                    pad=0.07,
                    label=f"{get_variable_units(ds_target, str(var))} (level {level_val})",
                )

            # HEAD's suptitle logic
            title_extra = "" if ens is None else f" (Ensemble {ens})"
            if time_selected is not None:
                # Robust formatting without relying on .dt accessor
                ts_val = np.asarray(time_selected.values, dtype="datetime64[h]")
                init_label = str(np.datetime_as_string(ts_val)).replace("T", " ") + "Z"
                ensemble_label = ens_token_global or (
                    "member " + str(ens) if ens is not None else "none"
                )
                title_text = (
                    f"Maps — {var}{title_extra} | ensemble={ensemble_label} | "
                    f"init_time={init_label} | lead_range={lead_range}"
                )
                plt.suptitle(title_text, fontsize=14)

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
                    level=level_label,
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
                    level=level_label,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="npz",
                )
                np.savez(
                    npz_path,
                    target=ds_var.values,
                    prediction=ds_ml_var.values,
                    latitude=(
                        ds_var.latitude.values if "latitude" in ds_var.coords else np.array([])
                    ),
                    longitude=(
                        ds_var.longitude.values if "longitude" in ds_var.coords else np.array([])
                    ),
                    level=(ds_var.level.values if "level" in ds_var.coords else np.array([])),
                    ensemble=int(ens) if ens is not None else -1,
                    variable=str(var),
                    units=get_variable_units(ds_target, str(var)),
                    allow_pickle=True,
                )
                print(f"[maps] saved {npz_path}")
