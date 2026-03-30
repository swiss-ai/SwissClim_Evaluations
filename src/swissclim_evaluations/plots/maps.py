from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .. import console as c
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
    save_data,
    save_figure,
    unwrap_longitude_for_plot,
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
    if not save_fig and not save_npz:
        c.print("[maps] Skipping module: output_mode=none (no PNG/NPZ outputs requested).")
        return
    dpi = int(plotting_cfg.get("dpi", 48))
    single_map_mode = str(plotting_cfg.get("maps_lead_layout", "stacked")).lower() == "per_lead"
    level_grid_mode = str(plotting_cfg.get("maps_level_layout", "per_level")).lower() == "stacked"

    section_output = out_root / "maps"

    # Determine time index to plot
    time_index = 0
    plot_dt = plotting_cfg.get("plot_datetime")

    if "init_time" in ds_target.dims and ds_target.init_time.size > 0:
        if plot_dt is not None:
            try:
                target_dt = np.datetime64(plot_dt)
                matches = np.where(ds_target.init_time.values == target_dt)[0]
                if matches.size > 0:
                    time_index = int(matches[0])
                else:
                    c.print(
                        f"[maps] Warning: plot_datetime {plot_dt} not found. Using first init_time."
                    )
                    time_index = 0
            except Exception as e:
                c.print(
                    f"[maps] Warning: Error selecting plot_datetime {plot_dt}: {e}. "
                    "Using first init_time."
                )
                time_index = 0
        else:
            time_index = 0

    lead_index = 0
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

    # Determine variables
    variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
    variables_2d = [v for v in ds_target.data_vars if "level" not in ds_target[v].dims]

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
        c.print(f"[maps] 2D variable: {var}")
        for ens in ensemble_members:
            ds_var_full = ds_target[var]
            ds_prediction_var_full = ds_prediction[var]

            # Check if original variable has multiple init times (before slicing)
            is_single_init = True
            if "init_time" in ds_var_full.dims and ds_var_full.sizes["init_time"] > 1:
                is_single_init = False

            if ens is not None:
                if "ensemble" in ds_var_full.dims:
                    # If target has only 1 member (e.g. ERA5), reuse it for all prediction members
                    if ds_var_full.sizes["ensemble"] == 1:
                        ds_var_full = ds_var_full.isel(ensemble=0)
                    else:
                        ds_var_full = ds_var_full.isel(ensemble=ens)
                if "ensemble" in ds_prediction_var_full.dims:
                    ds_prediction_var_full = ds_prediction_var_full.isel(ensemble=ens)
            if "init_time" in ds_var_full.dims:
                ds_var_full = ds_var_full.isel(init_time=time_index)
            if "init_time" in ds_prediction_var_full.dims:
                ds_prediction_var_full = ds_prediction_var_full.isel(init_time=time_index)

            # Determine lead times to plot
            lead_indices = [0]
            lead_coords = [None]
            if "lead_time" in ds_prediction_var_full.dims:
                lead_indices = list(range(ds_prediction_var_full.sizes["lead_time"]))
                lead_coords = ds_prediction_var_full["lead_time"].values
            elif "lead_time" in ds_var_full.dims:
                lead_indices = list(range(ds_var_full.sizes["lead_time"]))
                lead_coords = ds_var_full["lead_time"].values

            n_leads = len(lead_indices)

            # Compute global vmin/vmax for consistent color scale
            # Batch all 4 reductions into a single dask.compute() to avoid
            # 4 separate scheduler round-trips and redundant data traversals.
            _vmin_t, _vmin_p, _vmax_t, _vmax_p = dask.compute(
                ds_var_full.min(),
                ds_prediction_var_full.min(),
                ds_var_full.max(),
                ds_prediction_var_full.max(),
            )
            vmin = min(float(_vmin_t), float(_vmin_p))
            vmax = max(float(_vmax_t), float(_vmax_p))
            if get_colormap_for_variable(str(var)) == "RdBu_r":
                abs_max = max(abs(vmin), abs(vmax))
                vmin, vmax = -abs_max, abs_max

            ens_token = (
                ensemble_mode_to_token("members", ens)
                if (resolved_mode == "members" and ens is not None)
                else ens_token_global
            )

            if not single_map_mode:
                fig, axes = plt.subplots(
                    n_leads,
                    2,
                    figsize=(14, 4 * n_leads),
                    dpi=dpi * 2,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    constrained_layout=True,
                )
                if n_leads > 1:
                    fig.get_layout_engine().set(h_pad=0.15)
                if n_leads == 1:
                    axes = np.array([axes])
                im0 = None

            for i, lead_idx in enumerate(lead_indices):
                if single_map_mode:
                    fig, _ax_pair = plt.subplots(
                        1,
                        2,
                        figsize=(14, 4),
                        dpi=dpi * 2,
                        subplot_kw={"projection": ccrs.PlateCarree()},
                        constrained_layout=True,
                    )
                    axes = np.array([_ax_pair])
                    im0 = None

                row = 0 if single_map_mode else i
                ax_tgt = axes[row, 0]
                ax_pred = axes[row, 1]

                ds_var = ds_var_full
                ds_prediction_var = ds_prediction_var_full

                if "lead_time" in ds_var.dims:
                    ds_var = ds_var.isel(lead_time=lead_idx)
                if "lead_time" in ds_prediction_var.dims:
                    ds_prediction_var = ds_prediction_var.isel(lead_time=lead_idx)

                # Ensure we also select a single time slice if a free 'time' dimension exists
                if "time" in ds_var.dims:
                    ds_var = ds_var.isel(time=time_index)
                if "time" in ds_prediction_var.dims:
                    ds_prediction_var = ds_prediction_var.isel(time=time_index)

                # Drop any remaining singleton temporal dims
                for dim_drop in ("time", "init_time", "lead_time"):
                    if dim_drop in ds_var.dims and ds_var.sizes[dim_drop] == 1:
                        ds_var = ds_var.isel({dim_drop: 0})
                    if (
                        dim_drop in ds_prediction_var.dims
                        and ds_prediction_var.sizes[dim_drop] == 1
                    ):
                        ds_prediction_var = ds_prediction_var.isel({dim_drop: 0})
                ds_var = ds_var.squeeze()
                ds_prediction_var = ds_prediction_var.squeeze()
                ds_var = unwrap_longitude_for_plot(ds_var)
                ds_prediction_var = unwrap_longitude_for_plot(ds_prediction_var)

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

                date_str = (
                    extract_date_from_dataset(ds_var) if (is_single_init or single_map_mode) else ""
                )
                if i == 0 or single_map_mode:
                    ax_tgt.set_title(f"{format_variable_name(str(var))} — Target{date_str}")
                else:
                    ax_tgt.set_title(f"Target{lead_str}")

                ens_str = f" (member {ens})" if ens is not None else ""
                lon_prediction = ds_prediction_var.coords.get("longitude", None)
                lat_prediction = ds_prediction_var.coords.get("latitude", None)
                ax_pred.pcolormesh(
                    lon_prediction if lon_prediction is not None else ds_prediction_var.longitude,
                    lat_prediction if lat_prediction is not None else ds_prediction_var.latitude,
                    ds_prediction_var.values,
                    cmap=get_colormap_for_variable(str(var)),
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                ax_pred.coastlines(linewidth=0.5)
                _lon = (
                    lon_prediction.values
                    if lon_prediction is not None
                    else ds_prediction_var.longitude.values
                )
                _lat = (
                    lat_prediction.values
                    if lat_prediction is not None
                    else ds_prediction_var.latitude.values
                )
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
                ax_pred.set_title(f"Prediction{ens_str}{lead_str if single_map_mode else ''}")

                if single_map_mode:
                    cb = fig.colorbar(
                        im0,
                        ax=axes,
                        orientation="horizontal",
                        fraction=0.05,
                        pad=0.02,
                    )
                    try:
                        if cb is not None:
                            cb.set_label(get_variable_units(ds_target, str(var)))
                    except Exception:
                        pass
                    lead_val = lead_coords[i] if lead_coords[i] is not None else None
                    lead_qualifier = None
                    if lead_val is not None and np.issubdtype(type(lead_val), np.timedelta64):
                        h = int(lead_val / np.timedelta64(1, "h"))
                        lead_qualifier = f"lead{h:03d}h"
                    if save_fig:
                        out_png = section_output / build_output_filename(
                            metric="map",
                            variable=str(var),
                            level="surface",
                            qualifier=lead_qualifier,
                            init_time_range=init_range,
                            lead_time_range=None,
                            ensemble=ens_token,
                            ext="png",
                        )
                        save_figure(fig, out_png, module="maps")
                    else:
                        plt.close(fig)

            if not single_map_mode:
                cb = fig.colorbar(
                    im0,
                    ax=axes,
                    orientation="horizontal",
                    fraction=0.05,
                    pad=0.02,
                )
                try:
                    if cb is not None:
                        cb.set_label(get_variable_units(ds_target, str(var)))
                except Exception:
                    pass
                if save_fig:
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
                    save_figure(fig, out_png, module="maps")
                else:
                    plt.close(fig)
            if save_npz:
                # Prepare data for saving (full lead time stack)
                ds_save_target = ds_var_full
                ds_save_pred = ds_prediction_var_full

                # Squeeze singleton dims (except lead_time if n_leads > 1)
                for dim_drop in ("time", "init_time"):
                    if dim_drop in ds_save_target.dims and ds_save_target.sizes[dim_drop] == 1:
                        ds_save_target = ds_save_target.isel({dim_drop: 0})
                    if dim_drop in ds_save_pred.dims and ds_save_pred.sizes[dim_drop] == 1:
                        ds_save_pred = ds_save_pred.isel({dim_drop: 0})

                # If lead_time is size 1, we might want to squeeze it to match previous behavior (2D
                # map)
                if "lead_time" in ds_save_target.dims and ds_save_target.sizes["lead_time"] == 1:
                    ds_save_target = ds_save_target.isel(lead_time=0)
                if "lead_time" in ds_save_pred.dims and ds_save_pred.sizes["lead_time"] == 1:
                    ds_save_pred = ds_save_pred.isel(lead_time=0)

                ds_save_target = unwrap_longitude_for_plot(ds_save_target)
                ds_save_pred = unwrap_longitude_for_plot(ds_save_pred)

                # Ensure consistent dimension order: (lead_time, latitude, longitude) or (latitude,
                # longitude) This prevents issues where dimensions might be permuted (e.g.
                # (longitude, lead_time)) which causes pcolormesh errors in intercompare.
                if "latitude" in ds_save_target.dims and "longitude" in ds_save_target.dims:
                    if "lead_time" in ds_save_target.dims:
                        ds_save_target = ds_save_target.transpose(
                            "lead_time", "latitude", "longitude", ...
                        )
                    else:
                        ds_save_target = ds_save_target.transpose("latitude", "longitude", ...)

                if "latitude" in ds_save_pred.dims and "longitude" in ds_save_pred.dims:
                    if "lead_time" in ds_save_pred.dims:
                        ds_save_pred = ds_save_pred.transpose(
                            "lead_time", "latitude", "longitude", ...
                        )
                    else:
                        ds_save_pred = ds_save_pred.transpose("latitude", "longitude", ...)

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
                save_data(
                    npz_path,
                    target=ds_save_target.values,
                    prediction=ds_save_pred.values,
                    latitude=(
                        ds_save_target.latitude.values
                        if "latitude" in ds_save_target.coords
                        else np.array([])
                    ),
                    longitude=(
                        ds_save_target.longitude.values
                        if "longitude" in ds_save_target.coords
                        else np.array([])
                    ),
                    ensemble=int(ens) if ens is not None else -1,
                    variable=str(var),
                    units=get_variable_units(ds_target, str(var)),
                    lead_time=lead_coords if n_leads > 1 else None,
                    module="maps",
                )
            plt.close(fig)

    # 3D maps per level (one figure with rows per level)
    for _i, var in enumerate(variables_3d):
        c.print(f"[maps] 3D variable: {var}")
        levels = list(ds_target[var].coords.get("level", []))
        # Ensure levels are scalars for formatting
        levels = [lvl.item() if hasattr(lvl, "item") else lvl for lvl in levels]
        if not levels:
            continue

        level_tokens = [format_level_token(lvl) for lvl in levels]
        if not level_tokens:
            level_label = "levels"
        elif len(level_tokens) == 1:
            level_label = level_tokens[0]
        else:
            level_label = f"{level_tokens[0]}_to_{level_tokens[-1]}"
        for ens in ensemble_members:
            ds_var_full_3d = ds_target[var]
            ds_prediction_var_full_3d = ds_prediction[var]

            # Check if original variable has multiple init times (before slicing)
            is_single_init = True
            if "init_time" in ds_var_full_3d.dims and ds_var_full_3d.sizes["init_time"] > 1:
                is_single_init = False

            if ens is not None:
                if "ensemble" in ds_var_full_3d.dims:
                    if ds_var_full_3d.sizes["ensemble"] == 1:
                        ds_var_full_3d = ds_var_full_3d.isel(ensemble=0)
                    else:
                        ds_var_full_3d = ds_var_full_3d.isel(ensemble=ens)
                if "ensemble" in ds_prediction_var_full_3d.dims:
                    ds_prediction_var_full_3d = ds_prediction_var_full_3d.isel(ensemble=ens)
            if "init_time" in ds_var_full_3d.dims:
                ds_var_full_3d = ds_var_full_3d.isel(init_time=time_index)
            if "init_time" in ds_prediction_var_full_3d.dims:
                ds_prediction_var_full_3d = ds_prediction_var_full_3d.isel(init_time=time_index)
            if "time" in ds_var_full_3d.dims:
                ds_var_full_3d = ds_var_full_3d.isel(time=time_index)
            if "time" in ds_prediction_var_full_3d.dims:
                ds_prediction_var_full_3d = ds_prediction_var_full_3d.isel(time=time_index)

            # Determine lead times for 3D variables
            lead_indices_3d = [0]
            lead_coords_3d = [None]
            has_multi_lead_3d = False
            if "lead_time" in ds_prediction_var_full_3d.dims:
                lead_indices_3d = list(range(ds_prediction_var_full_3d.sizes["lead_time"]))
                lead_coords_3d = ds_prediction_var_full_3d["lead_time"].values
                has_multi_lead_3d = len(lead_indices_3d) > 1
            elif "lead_time" in ds_var_full_3d.dims:
                lead_indices_3d = list(range(ds_var_full_3d.sizes["lead_time"]))
                lead_coords_3d = ds_var_full_3d["lead_time"].values
                has_multi_lead_3d = len(lead_indices_3d) > 1

            ens_str = f" (member {ens})" if ens is not None else ""

            if has_multi_lead_3d:
                if single_map_mode and level_grid_mode:
                    # ── Grid mode: one figure per lead, all pressure levels as rows ──
                    # Colour scale is fixed per level across all lead frames so that
                    # animating (GIF) gives a consistent visual comparison.
                    n_levels_3d = len(levels)
                    lev_vmin_vmax: dict[int, tuple[float, float]] = {}
                    for _lev in levels:
                        _lev_t = ds_var_full_3d.sel(level=_lev)
                        _lev_p = ds_prediction_var_full_3d.sel(level=_lev)
                        _mn_t, _mn_p, _mx_t, _mx_p = dask.compute(
                            _lev_t.min(), _lev_p.min(), _lev_t.max(), _lev_p.max()
                        )
                        _lk = int(_lev.values) if hasattr(_lev, "values") else int(_lev)
                        lev_vmin_vmax[_lk] = (
                            min(float(_mn_t), float(_mn_p)),
                            max(float(_mx_t), float(_mx_p)),
                        )

                    for i, lead_idx in enumerate(lead_indices_3d):
                        lead_val_3d = lead_coords_3d[i] if lead_coords_3d[i] is not None else None
                        lead_str = ""
                        lead_qualifier = None
                        if lead_val_3d is not None and np.issubdtype(
                            type(lead_val_3d), np.timedelta64
                        ):
                            h_lead = int(lead_val_3d / np.timedelta64(1, "h"))
                            lead_str = f" (+{h_lead}h)"
                            lead_qualifier = f"lead{h_lead:03d}h"

                        if save_fig:
                            fig, axes = plt.subplots(
                                n_levels_3d,
                                2,
                                figsize=(14, 4 * n_levels_3d),
                                dpi=dpi * 2,
                                subplot_kw={"projection": ccrs.PlateCarree()},
                                squeeze=False,
                                constrained_layout=True,
                            )
                            if n_levels_3d > 1:
                                fig.get_layout_engine().set(h_pad=0.15)

                        for idx, level in enumerate(levels):
                            level_val = (
                                int(level.values) if hasattr(level, "values") else int(level)
                            )
                            lev_vmin, lev_vmax = lev_vmin_vmax[level_val]

                            ds_t_ll = ds_var_full_3d.sel(level=level)
                            ds_p_ll = ds_prediction_var_full_3d.sel(level=level)
                            if "lead_time" in ds_t_ll.dims:
                                ds_t_ll = ds_t_ll.isel(lead_time=lead_idx)
                            if "lead_time" in ds_p_ll.dims:
                                ds_p_ll = ds_p_ll.isel(lead_time=lead_idx)
                            ds_t_ll = ds_t_ll.squeeze()
                            ds_p_ll = ds_p_ll.squeeze()
                            ds_t_ll = unwrap_longitude_for_plot(ds_t_ll)
                            ds_p_ll = unwrap_longitude_for_plot(ds_p_ll)

                            if save_fig:
                                ax_tgt = axes[idx, 0]
                                ax_pred = axes[idx, 1]
                                lon = ds_t_ll.coords.get("longitude", None)
                                lat = ds_t_ll.coords.get("latitude", None)
                                im_row = ax_tgt.pcolormesh(
                                    lon if lon is not None else ds_t_ll.longitude,
                                    lat if lat is not None else ds_t_ll.latitude,
                                    ds_t_ll.values,
                                    cmap=get_colormap_for_variable(str(var)),
                                    vmin=lev_vmin,
                                    vmax=lev_vmax,
                                    transform=ccrs.PlateCarree(),
                                    shading="auto",
                                )
                                ax_tgt.coastlines(linewidth=0.5)
                                _lon = lon.values if lon is not None else ds_t_ll.longitude.values
                                _lat = lat.values if lat is not None else ds_t_ll.latitude.values
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
                                # Row 0: variable name + lead + init; other rows: level only
                                date_str = extract_date_from_dataset(ds_t_ll)
                                if idx == 0:
                                    ax_tgt.set_title(
                                        f"{format_variable_name(str(var))} — Target"
                                        f"{format_level_label(level_val)}"
                                        f"{lead_str}{date_str}"
                                    )
                                else:
                                    ax_tgt.set_title(f"Target{format_level_label(level_val)}")
                                lon_p = ds_p_ll.coords.get("longitude", None)
                                lat_p = ds_p_ll.coords.get("latitude", None)
                                ax_pred.pcolormesh(
                                    lon_p if lon_p is not None else ds_p_ll.longitude,
                                    lat_p if lat_p is not None else ds_p_ll.latitude,
                                    ds_p_ll.values,
                                    cmap=get_colormap_for_variable(str(var)),
                                    vmin=lev_vmin,
                                    vmax=lev_vmax,
                                    transform=ccrs.PlateCarree(),
                                    shading="auto",
                                )
                                ax_pred.coastlines(linewidth=0.5)
                                _lon_p = (
                                    lon_p.values if lon_p is not None else ds_p_ll.longitude.values
                                )
                                _lat_p = (
                                    lat_p.values if lat_p is not None else ds_p_ll.latitude.values
                                )
                                if hasattr(ax_pred, "set_extent"):
                                    ax_pred.set_extent(
                                        [
                                            float(np.min(_lon_p)),
                                            float(np.max(_lon_p)),
                                            float(np.min(_lat_p)),
                                            float(np.max(_lat_p)),
                                        ],
                                        crs=ccrs.PlateCarree(),
                                    )
                                pred_lead = lead_str if idx == 0 else ""
                                ax_pred.set_title(
                                    f"Prediction{format_level_label(level_val)}"
                                    f"{ens_str}{pred_lead}"
                                )
                                # Per-row colorbar — level scale stays fixed across leads
                                fig.colorbar(
                                    im_row,
                                    ax=[ax_tgt, ax_pred],
                                    orientation="horizontal",
                                    fraction=0.05,
                                    pad=0.07,
                                    label=get_variable_units(ds_target, str(var)),
                                )

                        ens_token = (
                            ensemble_mode_to_token("members", ens)
                            if (resolved_mode == "members" and ens is not None)
                            else ens_token_global
                        )
                        if save_fig:
                            out_png = section_output / build_output_filename(
                                metric="map",
                                variable=str(var),
                                level=level_label,
                                qualifier=lead_qualifier,
                                init_time_range=init_range,
                                lead_time_range=None,
                                ensemble=ens_token,
                                ext="png",
                            )
                            save_figure(fig, out_png, module="maps")
                        else:
                            plt.close(fig)

                else:
                    # Existing: outer loop over levels (panel or per-level-single modes)
                    for level in levels:
                        level_val = int(level.values) if hasattr(level, "values") else int(level)
                        n_leads_3d = len(lead_indices_3d)
                        if not single_map_mode:
                            fig, axes = plt.subplots(
                                n_leads_3d,
                                2,
                                figsize=(14, 4 * n_leads_3d),
                                dpi=dpi * 2,
                                subplot_kw={"projection": ccrs.PlateCarree()},
                                squeeze=False,
                                constrained_layout=True,
                            )
                            if n_leads_3d > 1:
                                fig.get_layout_engine().set(h_pad=0.15)

                        # Compute global vmin/vmax across all leads for this level
                        ds_lev_t = ds_var_full_3d.sel(level=level)
                        ds_lev_p = ds_prediction_var_full_3d.sel(level=level)
                        _vmin_t, _vmin_p, _vmax_t, _vmax_p = dask.compute(
                            ds_lev_t.min(),
                            ds_lev_p.min(),
                            ds_lev_t.max(),
                            ds_lev_p.max(),
                        )
                        vmin = min(float(_vmin_t), float(_vmin_p))
                        vmax = max(float(_vmax_t), float(_vmax_p))

                        im0 = None
                        for i, lead_idx in enumerate(lead_indices_3d):
                            if single_map_mode:
                                fig, _ax_pair = plt.subplots(
                                    1,
                                    2,
                                    figsize=(14, 4),
                                    dpi=dpi * 2,
                                    subplot_kw={"projection": ccrs.PlateCarree()},
                                    constrained_layout=True,
                                )
                                axes = np.array([_ax_pair])
                                im0 = None
                            row = 0 if single_map_mode else i
                            ax_tgt = axes[row, 0]
                            ax_pred = axes[row, 1]
                            ds_t_lead = ds_lev_t
                            ds_p_lead = ds_lev_p
                            if "lead_time" in ds_t_lead.dims:
                                ds_t_lead = ds_t_lead.isel(lead_time=lead_idx)
                            if "lead_time" in ds_p_lead.dims:
                                ds_p_lead = ds_p_lead.isel(lead_time=lead_idx)
                            ds_t_lead = ds_t_lead.squeeze()
                            ds_p_lead = ds_p_lead.squeeze()
                            ds_t_lead = unwrap_longitude_for_plot(ds_t_lead)
                            ds_p_lead = unwrap_longitude_for_plot(ds_p_lead)

                            lon = ds_t_lead.coords.get("longitude", None)
                            lat = ds_t_lead.coords.get("latitude", None)
                            im0 = ax_tgt.pcolormesh(
                                lon if lon is not None else ds_t_lead.longitude,
                                lat if lat is not None else ds_t_lead.latitude,
                                ds_t_lead.values,
                                cmap=get_colormap_for_variable(str(var)),
                                vmin=vmin,
                                vmax=vmax,
                                transform=ccrs.PlateCarree(),
                                shading="auto",
                            )
                            ax_tgt.coastlines(linewidth=0.5)
                            _lon = lon.values if lon is not None else ds_t_lead.longitude.values
                            _lat = lat.values if lat is not None else ds_t_lead.latitude.values
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

                            lead_str = ""
                            if lead_coords_3d[i] is not None:
                                val = lead_coords_3d[i]
                                if np.issubdtype(type(val), np.timedelta64):
                                    h = int(val / np.timedelta64(1, "h"))
                                    lead_str = f" (+{h}h)"
                                else:
                                    lead_str = f" (lead={val})"

                            date_str = (
                                extract_date_from_dataset(ds_t_lead)
                                if (is_single_init or single_map_mode)
                                else ""
                            )
                            if i == 0 or single_map_mode:
                                ax_tgt.set_title(
                                    f"{format_variable_name(str(var))} — Target"
                                    f"{format_level_label(level_val)}{date_str}"
                                )
                            else:
                                ax_tgt.set_title(f"Target{lead_str}")

                            lon_p = ds_p_lead.coords.get("longitude", None)
                            lat_p = ds_p_lead.coords.get("latitude", None)
                            ax_pred.pcolormesh(
                                lon_p if lon_p is not None else ds_p_lead.longitude,
                                lat_p if lat_p is not None else ds_p_lead.latitude,
                                ds_p_lead.values,
                                cmap=get_colormap_for_variable(str(var)),
                                vmin=vmin,
                                vmax=vmax,
                                transform=ccrs.PlateCarree(),
                                shading="auto",
                            )
                            ax_pred.coastlines(linewidth=0.5)
                            _lon_p = (
                                lon_p.values if lon_p is not None else ds_p_lead.longitude.values
                            )
                            _lat_p = (
                                lat_p.values if lat_p is not None else ds_p_lead.latitude.values
                            )
                            if hasattr(ax_pred, "set_extent"):
                                ax_pred.set_extent(
                                    [
                                        float(np.min(_lon_p)),
                                        float(np.max(_lon_p)),
                                        float(np.min(_lat_p)),
                                        float(np.max(_lat_p)),
                                    ],
                                    crs=ccrs.PlateCarree(),
                                )
                            ax_pred.set_title(f"Prediction{ens_str}{lead_str}")

                            if single_map_mode:
                                cb = fig.colorbar(
                                    im0,
                                    ax=axes,
                                    orientation="horizontal",
                                    fraction=0.05,
                                    pad=0.02,
                                )
                                try:
                                    if cb is not None:
                                        cb.set_label(get_variable_units(ds_target, str(var)))
                                except Exception:
                                    pass
                                lead_val_3d = (
                                    lead_coords_3d[i] if lead_coords_3d[i] is not None else None
                                )
                                lead_qualifier = None
                                if lead_val_3d is not None and np.issubdtype(
                                    type(lead_val_3d), np.timedelta64
                                ):
                                    h = int(lead_val_3d / np.timedelta64(1, "h"))
                                    lead_qualifier = f"lead{h:03d}h"
                                ens_token = (
                                    ensemble_mode_to_token("members", ens)
                                    if (resolved_mode == "members" and ens is not None)
                                    else ens_token_global
                                )
                                lev_token = format_level_token(level_val)
                                if save_fig:
                                    out_png = section_output / build_output_filename(
                                        metric="map",
                                        variable=str(var),
                                        level=lev_token,
                                        qualifier=lead_qualifier,
                                        init_time_range=init_range,
                                        lead_time_range=None,
                                        ensemble=ens_token,
                                        ext="png",
                                    )
                                    save_figure(fig, out_png, module="maps")
                                else:
                                    plt.close(fig)

                        if not single_map_mode:
                            cb = fig.colorbar(
                                im0,
                                ax=axes,
                                orientation="horizontal",
                                fraction=0.05,
                                pad=0.02,
                            )
                            try:
                                if cb is not None:
                                    cb.set_label(get_variable_units(ds_target, str(var)))
                            except Exception:
                                pass

                            ens_token = (
                                ensemble_mode_to_token("members", ens)
                                if (resolved_mode == "members" and ens is not None)
                                else ens_token_global
                            )
                            lev_token = format_level_token(level_val)
                            if save_fig:
                                out_png = section_output / build_output_filename(
                                    metric="map",
                                    variable=str(var),
                                    level=lev_token,
                                    qualifier=None,
                                    init_time_range=init_range,
                                    lead_time_range=lead_range,
                                    ensemble=ens_token,
                                    ext="png",
                                )
                                save_figure(fig, out_png, module="maps")
                            else:
                                plt.close(fig)
                            plt.close(fig)
            else:
                # Single-lead: original all-levels-in-one-figure layout
                num_rows = len(levels)
                fig, axes = plt.subplots(
                    num_rows,
                    2,
                    figsize=(14, 4 * num_rows),
                    dpi=dpi * 2,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    squeeze=False,
                    constrained_layout=True,
                )

                ds_var = ds_var_full_3d
                ds_prediction_var = ds_prediction_var_full_3d
                if "lead_time" in ds_var.dims:
                    ds_var = ds_var.isel(lead_time=lead_index)
                if "lead_time" in ds_prediction_var.dims:
                    ds_prediction_var = ds_prediction_var.isel(lead_time=lead_index)

                for idx, level in enumerate(levels):
                    level_val = int(level.values) if hasattr(level, "values") else int(level)
                    ax_ds, ax_pred = axes[idx]
                    ds_var_lev = ds_var.sel(level=level)
                    ds_prediction_var_lev = ds_prediction_var.sel(level=level)
                    ds_var_lev = ds_var_lev.squeeze()
                    ds_prediction_var_lev = ds_prediction_var_lev.squeeze()

                    arr_t = ds_var_lev.values
                    arr_p = ds_prediction_var_lev.values
                    vmin = min(float(np.nanmin(arr_t)), float(np.nanmin(arr_p)))
                    vmax = max(float(np.nanmax(arr_t)), float(np.nanmax(arr_p)))

                    im_ds = ax_ds.pcolormesh(
                        ds_var_lev.coords.get("longitude"),
                        ds_var_lev.coords.get("latitude"),
                        arr_t,
                        cmap=get_colormap_for_variable(str(var)),
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                    )
                    ax_ds.coastlines(linewidth=0.5)
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

                    ax_pred.pcolormesh(
                        ds_prediction_var_lev.coords.get("longitude"),
                        ds_prediction_var_lev.coords.get("latitude"),
                        arr_p,
                        cmap=get_colormap_for_variable(str(var)),
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                    )
                    ax_pred.coastlines(linewidth=0.5)
                    lon_prediction = ds_prediction_var_lev.coords.get("longitude", None)
                    lat_prediction = ds_prediction_var_lev.coords.get("latitude", None)
                    _lon = (
                        lon_prediction.values
                        if lon_prediction is not None
                        else ds_prediction_var_lev.longitude.values
                    )
                    _lat = (
                        lat_prediction.values
                        if lat_prediction is not None
                        else ds_prediction_var_lev.latitude.values
                    )
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
                    ax_pred.set_title(f"Prediction{format_level_label(level_val)}{ens_str}")
                    fig.colorbar(
                        im_ds,
                        ax=[ax_ds, ax_pred],
                        orientation="horizontal",
                        fraction=0.05,
                        pad=0.07,
                        label=f"{get_variable_units(ds_target, str(var))}",
                    )

                ens_token = (
                    ensemble_mode_to_token("members", ens)
                    if (resolved_mode == "members" and ens is not None)
                    else ens_token_global
                )
                if save_fig:
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
                    save_figure(fig, out_png, module="maps")
                else:
                    plt.close(fig)
            if save_npz:
                ens_token = (
                    ensemble_mode_to_token("members", ens)
                    if (resolved_mode == "members" and ens is not None)
                    else ens_token_global
                )
                if has_multi_lead_3d:
                    # Save one NPZ per level with all lead times preserved,
                    # so that intercompare can render per-lead gridded plots.
                    for level in levels:
                        level_val_npz = (
                            int(level.values) if hasattr(level, "values") else int(level)
                        )
                        lev_token_npz = format_level_token(level_val_npz)
                        ds_lev_t_npz = ds_var_full_3d.sel(level=level)
                        ds_lev_p_npz = ds_prediction_var_full_3d.sel(level=level)
                        # Squeeze singleton dims except lead_time
                        for dim_drop in ("time", "init_time"):
                            if dim_drop in ds_lev_t_npz.dims and ds_lev_t_npz.sizes[dim_drop] == 1:
                                ds_lev_t_npz = ds_lev_t_npz.isel({dim_drop: 0})
                            if dim_drop in ds_lev_p_npz.dims and ds_lev_p_npz.sizes[dim_drop] == 1:
                                ds_lev_p_npz = ds_lev_p_npz.isel({dim_drop: 0})
                        ds_lev_t_npz = unwrap_longitude_for_plot(ds_lev_t_npz)
                        ds_lev_p_npz = unwrap_longitude_for_plot(ds_lev_p_npz)
                        # Transpose to (lead_time, latitude, longitude)
                        if (
                            "lead_time" in ds_lev_t_npz.dims
                            and "latitude" in ds_lev_t_npz.dims
                            and "longitude" in ds_lev_t_npz.dims
                        ):
                            ds_lev_t_npz = ds_lev_t_npz.transpose(
                                "lead_time", "latitude", "longitude", ...
                            )
                        if (
                            "lead_time" in ds_lev_p_npz.dims
                            and "latitude" in ds_lev_p_npz.dims
                            and "longitude" in ds_lev_p_npz.dims
                        ):
                            ds_lev_p_npz = ds_lev_p_npz.transpose(
                                "lead_time", "latitude", "longitude", ...
                            )
                        npz_path = section_output / build_output_filename(
                            metric="map",
                            variable=str(var),
                            level=lev_token_npz,
                            qualifier=None,
                            init_time_range=init_range,
                            lead_time_range=lead_range,
                            ensemble=ens_token,
                            ext="npz",
                        )
                        save_data(
                            npz_path,
                            target=ds_lev_t_npz.values,
                            prediction=ds_lev_p_npz.values,
                            latitude=(
                                ds_lev_t_npz.latitude.values
                                if "latitude" in ds_lev_t_npz.coords
                                else np.array([])
                            ),
                            longitude=(
                                ds_lev_t_npz.longitude.values
                                if "longitude" in ds_lev_t_npz.coords
                                else np.array([])
                            ),
                            ensemble=int(ens) if ens is not None else -1,
                            variable=str(var),
                            units=get_variable_units(ds_target, str(var)),
                            lead_time=lead_coords_3d if len(lead_coords_3d) > 1 else None,
                            module="maps",
                        )
                else:
                    # Single-lead: save one NPZ with all levels (original behavior)
                    ds_var_npz = ds_var_full_3d
                    ds_pred_npz = ds_prediction_var_full_3d
                    if "lead_time" in ds_var_npz.dims:
                        ds_var_npz = ds_var_npz.isel(lead_time=lead_index)
                    if "lead_time" in ds_pred_npz.dims:
                        ds_pred_npz = ds_pred_npz.isel(lead_time=lead_index)
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
                    save_data(
                        npz_path,
                        target=ds_var_npz.values,
                        prediction=ds_pred_npz.values,
                        latitude=(
                            ds_var_npz.latitude.values
                            if "latitude" in ds_var_npz.coords
                            else np.array([])
                        ),
                        longitude=(
                            ds_var_npz.longitude.values
                            if "longitude" in ds_var_npz.coords
                            else np.array([])
                        ),
                        level=(
                            ds_var_npz.level.values
                            if "level" in ds_var_npz.coords
                            else np.array([])
                        ),
                        ensemble=int(ens) if ens is not None else -1,
                        variable=str(var),
                        units=get_variable_units(ds_target, str(var)),
                        allow_pickle=True,
                        module="maps",
                    )
