from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ...helpers import (
    build_output_filename,
)
from .plots import plot_lead_time_evolution, plot_single_map


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
