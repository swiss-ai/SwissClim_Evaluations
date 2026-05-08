from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ... import console as c
from ...helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_level_label,
    get_variable_units,
    save_data,
)
from .plots import plot_lead_time_evolution, plot_single_map, plot_spaghetti_timeseries


def _extract_date_str(ds: xr.Dataset | None) -> str:
    if ds is None:
        return ""
    from ...helpers import extract_date_from_dataset

    return extract_date_from_dataset(ds)


def visualize_probabilistic_metrics(
    results_spatial: xr.Dataset,
    results_temporal: xr.Dataset,  # Temporal might be same as spatial if reduced
    out_root: Path,
    plotting_cfg: dict[str, Any],
    init_range: tuple | None,
    lead_range: tuple | None,
    ens_token: str | None,
    ds_target: xr.Dataset | None = None,  # For date extraction
):
    """Generate plots for CRPS and SSR metrics (Maps, Line Plots)."""

    dpi = int((plotting_cfg or {}).get("dpi", 48))
    date_str = _extract_date_str(ds_target)
    section = out_root / "probabilistic"

    # Use spatial results as primary source for plotting maps
    # If a metric is only in temporal (fully reduced), we can only plot lines

    # Identify metrics to plot
    metrics_to_plot = [
        v
        for v in results_spatial.data_vars
        if str(v).startswith("CRPS") or str(v).startswith("SSR")
    ]

    for var_name in metrics_to_plot:
        da = results_spatial[var_name]

        parts = str(var_name).split(".", 1)
        metric_type = parts[0]  # CRPS or SSR
        display_var = parts[1] if len(parts) > 1 else str(var_name)

        levels = [None]
        if "level" in da.dims:
            levels = list(da["level"].values)

        for lvl in levels:
            da_lvl = da.sel(level=lvl, drop=True) if lvl is not None else da

            # --- Map Plot ---
            # Check for lat/lon
            lat_name = next((n for n in da_lvl.dims if n in ("latitude", "lat", "y")), None)
            lon_name = next((n for n in da_lvl.dims if n in ("longitude", "lon", "x")), None)

            # If we have lead_time > 1, we might want to average it for the map
            # OR produce a grid. For now, let's average over lead_time for the main map.
            da_map = da_lvl
            if "lead_time" in da_map.dims:
                da_map = da_map.mean(dim="lead_time", skipna=True)

            if lat_name and lon_name:
                # Configuration for colorbars
                if metric_type == "SSR":
                    # Spread/Skill Ratio: Ideal is 1.
                    # SSR < 1 means under-dispersive/over-confident.
                    # SSR > 1 means over-dispersive/under-confident.
                    # SSR > 1 means spread > error (over-dispersive/under-confident)
                    cmap = "RdBu"
                    vmin = 0.0
                    vmax = 2.0
                    extend = "both"
                else:
                    # CRPS: Lower is better
                    cmap = "viridis"
                    vmin = 0.0
                    vmax = None  # Auto
                    extend = "max"

                out_png = section / build_output_filename(
                    metric=f"{metric_type.lower()}_map",
                    variable=display_var,
                    level=lvl,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )

                plot_single_map(
                    da=da_map,
                    out_path=out_png,
                    variable=display_var,
                    metric_label=metric_type,
                    level=lvl,
                    time_label=date_str,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extend=extend,
                    dpi=dpi,
                )

                # --- Per-lead-time grid map ---
                if "lead_time" in da_lvl.dims and da_lvl.sizes["lead_time"] > 1:
                    lead_vals = da_lvl["lead_time"].values
                    if np.issubdtype(np.asarray(lead_vals).dtype, np.timedelta64):
                        lead_hours = (lead_vals / np.timedelta64(1, "h")).astype(int)
                    else:
                        lead_hours = np.asarray(lead_vals).astype(int)

                    n_leads = len(lead_hours)
                    fig_g, axes_g = plt.subplots(
                        1,
                        n_leads,
                        figsize=(5 * n_leads, 4),
                        dpi=dpi * 2,
                        subplot_kw={"projection": ccrs.PlateCarree()},
                        constrained_layout=True,
                    )
                    if n_leads == 1:
                        axes_g = [axes_g]

                    # Common colour range across all leads
                    all_vals = np.asarray(da_lvl.values)
                    grid_vmin = vmin if vmin is not None else float(np.nanmin(all_vals))
                    grid_vmax = vmax if vmax is not None else float(np.nanmax(all_vals))

                    im_last = None
                    for li, (ax_l, lh) in enumerate(zip(axes_g, lead_hours, strict=True)):
                        da_lead = da_lvl.isel(lead_time=li)
                        z = np.asarray(da_lead.values)
                        ax_l.coastlines(linewidth=0.5)
                        im_last = ax_l.pcolormesh(
                            da_lead[lon_name],
                            da_lead[lat_name],
                            z,
                            cmap=cmap,
                            vmin=grid_vmin,
                            vmax=grid_vmax,
                            shading="auto",
                            transform=ccrs.PlateCarree(),
                        )
                        ax_l.set_title(f"Lead {lh}h", fontsize=9)

                    if im_last is not None:
                        fig_g.colorbar(
                            im_last,
                            ax=list(axes_g),
                            orientation="horizontal",
                            fraction=0.04,
                            pad=0.06,
                            label=metric_type,
                            extend=extend,
                        )
                    lvl_str = f" @ {lvl}" if lvl is not None else ""
                    fig_g.suptitle(
                        f"{metric_type} Map per Lead — " f"{display_var}{lvl_str}",
                        fontsize=11,
                        y=1.02,
                    )
                    out_grid = section / build_output_filename(
                        metric=f"{metric_type.lower()}_map_per_lead",
                        variable=display_var,
                        level=lvl,
                        qualifier=None,
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token,
                        ext="png",
                    )
                    plt.savefig(out_grid, bbox_inches="tight", dpi=200)
                    plt.close(fig_g)

            # --- Line Plot (Evolution) ---
            # We need the version WITH lead_time preserved.
            # results_temporal usually retains lead_time if it existed?
            # Check results_temporal vs results_spatial
            # Usually users pass the same dataset if spatial dimensions are present.

            da_line_source = results_temporal.get(var_name, da_lvl)
            if lvl is not None and "level" in da_line_source.dims:
                da_line_source = da_line_source.sel(level=lvl, drop=True)

            if "lead_time" in da_line_source.dims and da_line_source.sizes["lead_time"] > 1:
                # Reduce spatial dims if present
                reduce_dims = [d for d in da_line_source.dims if d not in ("lead_time",)]
                da_line = da_line_source.mean(dim=reduce_dims, skipna=True)

                leads = da_line["lead_time"].values
                if np.issubdtype(np.asarray(leads).dtype, np.timedelta64):
                    lead_hours = (leads / np.timedelta64(1, "h")).astype(int)
                else:
                    lead_hours = np.asarray(leads).astype(int)

                df_line = pd.DataFrame({"lead_time_hours": lead_hours, metric_type: da_line.values})

                out_png_line = section / build_output_filename(
                    metric=f"{metric_type.lower()}_line",
                    variable=display_var,
                    level=lvl,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )

                plot_lead_time_evolution(
                    df=df_line,
                    out_path=out_png_line,
                    variable=display_var,
                    metric_label=metric_type,
                    level=lvl,
                    time_label=date_str,
                    dpi=dpi,
                )


def generate_spaghetti_plots(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> None:
    """Generate ensemble spaghetti time-series plots for all common variables.

    For each variable (and pressure level for 3D variables), produces a line plot
    where each ensemble member is drawn as a thin coloured line and the target
    (ground truth) is overlaid as a thicker black line.  The x-axis shows lead
    time (hours) and the y-axis shows the spatially averaged physical quantity.

    By default the **first** available ``init_time`` is selected.  This can be
    overridden via ``plotting.plot_datetime`` in the config, which is resolved
    through the standard ``select_plot_datetime`` mechanism.

    The function honours ``plotting.output_mode``: plots are skipped when the
    mode is ``"none"``.

    Args:
        ds_target: Target / ground-truth dataset (may contain ``init_time``,
            ``lead_time``, ``latitude``/``longitude``, and optionally ``level``).
        ds_prediction: Model prediction dataset.  Must contain an ``ensemble``
            dimension with >=2 members.
        out_root: Root output directory; spaghetti plots are saved under
            ``<out_root>/probabilistic/``.
        plotting_cfg: Plotting configuration dict (parsed from
            ``config.plotting``).
        cfg: Full configuration dict (used to resolve ``plot_datetime``).
    """
    import dask as _dask_mod

    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    if not save_fig and not save_npz:
        c.print("[probabilistic] Skipping spaghetti plots: output_mode=none.")
        return

    # --- Ensemble guard ---
    if "ensemble" not in ds_prediction.dims or ds_prediction.sizes["ensemble"] < 2:
        c.print("[probabilistic] Skipping spaghetti plots: requires ensemble size >= 2.")
        return

    # --- Lead-time guard: need lead_time for x-axis ---
    if "lead_time" not in ds_prediction.dims or ds_prediction.sizes["lead_time"] < 2:
        c.print("[probabilistic] Skipping spaghetti plots: requires multiple lead times.")
        return

    dpi = int((plotting_cfg or {}).get("dpi", 48))
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    # --- Use all init_times (averaged after spatial reduction) ---
    ds_tgt_plot = ds_target
    ds_pred_plot = ds_prediction

    date_str = "all init times"
    ens_token = ensemble_mode_to_token("prob")

    # --- Time-range tokens for filename ---
    init_range = None
    if "init_time" in ds_pred_plot.coords:
        it_vals = ds_pred_plot["init_time"].values
        if it_vals.size:
            s = np.datetime64(np.min(it_vals)).astype("datetime64[h]")
            e = np.datetime64(np.max(it_vals)).astype("datetime64[h]")
            _fmt = lambda x: (  # noqa: E731
                np.datetime_as_string(x, unit="h")
                .replace("-", "")
                .replace(":", "")
                .replace("T", "")
            )
            init_range = (_fmt(s), _fmt(e))

    lead_range = None
    if "lead_time" in ds_pred_plot.coords:
        lt_vals = ds_pred_plot["lead_time"].values
        if getattr(lt_vals, "size", 0):
            hours = (lt_vals / np.timedelta64(1, "h")).astype(int)
            lead_range = (f"{int(np.min(hours)):03d}h", f"{int(np.max(hours)):03d}h")

    # --- Lead hours array for x-axis ---
    lead_vals = ds_pred_plot["lead_time"].values
    lead_hours = (lead_vals / np.timedelta64(1, "h")).astype(int)

    # --- Identify spatial dims to average over ---
    spatial_dims_names = {"latitude", "lat", "y", "longitude", "lon", "x"}

    common_vars = [v for v in ds_pred_plot.data_vars if v in ds_tgt_plot.data_vars]
    if not common_vars:
        c.print("[probabilistic] Spaghetti: no common variables found.")
        return

    c.print(f"[probabilistic] Generating spaghetti plots for {len(common_vars)} variable(s).")

    for var_name in common_vars:
        da_pred = ds_pred_plot[var_name]
        da_tgt = ds_tgt_plot[var_name]

        # Strip ensemble from target if present
        if "ensemble" in da_tgt.dims:
            da_tgt = da_tgt.isel(ensemble=0, drop=True)
        if "ensemble" in da_tgt.coords:
            da_tgt = da_tgt.drop_vars("ensemble")

        units = get_variable_units(ds_tgt_plot, str(var_name))

        # --- Per-level iteration for 3D variables ---
        has_level = "level" in da_tgt.dims
        level_iter: list[Any] = list(da_tgt["level"].values) if has_level else [None]

        for lvl in level_iter:
            if lvl is not None:
                da_tgt_lvl = da_tgt.sel(level=lvl, drop=True)
                da_pred_lvl = da_pred.sel(level=lvl, drop=True)
            else:
                da_tgt_lvl = da_tgt
                da_pred_lvl = da_pred

            # Average over init_time (skipna handles missing lead times)
            if "init_time" in da_tgt_lvl.dims:
                da_tgt_lvl = da_tgt_lvl.mean(dim="init_time", skipna=True)
            if "init_time" in da_pred_lvl.dims:
                da_pred_lvl = da_pred_lvl.mean(dim="init_time", skipna=True)

            # Identify spatial dims present in this DataArray
            spatial_reduce = [d for d in da_tgt_lvl.dims if str(d) in spatial_dims_names]

            # --- Spatial mean per ensemble member ---
            # prediction: (ensemble, lead_time, lat, lon, ...) → (ensemble, lead_time)
            da_pred_mean = da_pred_lvl.mean(dim=spatial_reduce, skipna=True)
            # target: (lead_time, lat, lon, ...) → (lead_time,)
            da_tgt_mean = da_tgt_lvl.mean(dim=spatial_reduce, skipna=True)

            # Compute via dask
            member_vals, target_vals = _dask_mod.compute(
                da_pred_mean.data, da_tgt_mean.data, optimize_graph=False
            )
            member_vals = np.asarray(member_vals)  # (n_members, n_leads)
            target_vals = np.asarray(target_vals)  # (n_leads,)

            # Ensure correct shape
            if member_vals.ndim == 1:
                member_vals = member_vals.reshape(1, -1)

            lvl_label = format_level_label(lvl)
            c.print(
                f"[probabilistic] Spaghetti: {var_name}{lvl_label} "
                f"({member_vals.shape[0]} members, {member_vals.shape[1]} leads)"
            )

            # --- Save NPZ data artifact (for intercomparison) ---
            if save_npz:
                out_npz = section / build_output_filename(
                    metric="spaghetti",
                    variable=str(var_name),
                    level=lvl,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="npz",
                )
                npz_kwargs: dict[str, Any] = {
                    "lead_hours": np.asarray(lead_hours),
                    "member_values": member_vals,
                    "target_values": target_vals,
                    "variable": str(var_name),
                    "units": units,
                }
                if lvl is not None:
                    npz_kwargs["level"] = lvl
                save_data(out_npz, module="probabilistic", **npz_kwargs)

            # --- Save PNG figure ---
            if save_fig:
                out_png = section / build_output_filename(
                    metric="spaghetti",
                    variable=str(var_name),
                    level=lvl,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )

                plot_spaghetti_timeseries(
                    lead_hours=lead_hours,
                    member_values=member_vals,
                    target_values=target_vals,
                    out_path=out_png,
                    variable=str(var_name),
                    level=lvl,
                    units=units,
                    time_label=date_str,
                    dpi=dpi,
                )
