from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ...helpers import (
    format_level_token,
    format_variable_name,
    save_figure,
    unwrap_longitude_for_plot,
)


def plot_single_map(
    da: xr.DataArray,
    out_path: Path,
    variable: str,
    metric_label: str,
    level: Any = None,
    time_label: str = "",
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    norm=None,
    extend: str = "neither",
    dpi: int = 96,
):
    """Plot a single 2D map with coastlines."""
    lat_name = next((n for n in da.dims if n in ("latitude", "lat", "y")), None)
    lon_name = next((n for n in da.dims if n in ("longitude", "lon", "x")), None)

    if lat_name is None or lon_name is None:
        return

    # Sort latitude if needed
    lat_vals = da[lat_name].values
    if lat_vals.size > 1 and lat_vals[0] > lat_vals[-1]:
        da = da.sortby(lat_name)
    da = unwrap_longitude_for_plot(da, lon_name)

    fig = plt.figure(figsize=(12, 6), dpi=dpi * 2)
    ax = plt.axes(projection=ccrs.PlateCarree())
    if hasattr(ax, "add_feature"):
        ax.add_feature(cfeature.COASTLINE, lw=0.5)

    z = np.asarray(da.values)

    # Auto-scale if not provided (though passed vmin/vmax usually respected)
    # If using norm, vmin/vmax might be ignored by matplotlib or used for clipping

    mesh = ax.pcolormesh(
        da[lon_name],
        da[lat_name],
        z,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    plt.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.8,
        label=metric_label,
        extend=extend,
    )

    lvl_str = f" @ {format_level_token(level)}" if level is not None else ""
    ax.set_title(
        f"{metric_label} Map — {format_variable_name(str(variable))}{lvl_str}",
        loc="left",
        fontsize=10,
    )
    ax.set_title(time_label, loc="right", fontsize=10)

    save_figure(fig, out_path, module="probabilistic")
    plt.close(fig)


def plot_lead_time_evolution(
    df: pd.DataFrame,
    out_path: Path,
    variable: str,
    metric_label: str,
    level: Any = None,
    time_label: str = "",
    dpi: int = 96,
):
    """Plot metric evolution over lead time."""
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi * 2)

    # Ensure sorted by lead time
    df = df.sort_values("lead_time_hours")

    ax.plot(df["lead_time_hours"], df[metric_label], marker="o")

    lvl_str = f" @ {format_level_token(level)}" if level is not None else ""
    ax.set_title(
        f"{metric_label} Evolution — {format_variable_name(str(variable))}{lvl_str}",
        loc="left",
        fontsize=10,
    )
    ax.set_title(time_label, loc="right", fontsize=10)
    ax.set_xlabel("Lead Time [h]")
    ax.set_ylabel(metric_label)
    ax.grid(True, linestyle="--", alpha=0.6)

    save_figure(fig, out_path, module="probabilistic")
    plt.close(fig)


def plot_regional_bar_chart(
    s_spatial: pd.Series,
    out_path: Path,
    variable: str,
    metric_label: str,
    level: Any = None,
    time_label: str = "",
    ref_line: float | None = None,
    dpi: int = 96,
):
    """Plot bar chart for regional metrics."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi * 2)
    x_pos = np.arange(len(s_spatial), dtype=float)

    ax.bar(x_pos, s_spatial.values)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in s_spatial.index])

    lvl_str = f" @ {format_level_token(level)}" if level is not None else ""
    ax.set_title(
        f"{metric_label} by Region — {format_variable_name(str(variable))}{lvl_str}",
        loc="left",
        fontsize=10,
    )
    ax.set_title(time_label, loc="right", fontsize=10)
    ax.set_ylabel(metric_label)

    if ref_line is not None:
        ax.axhline(ref_line, color="k", linestyle="--", alpha=0.5, label=f"Ref ({ref_line})")
        ax.legend()

    if hasattr(ax, "tick_params"):
        ax.tick_params(axis="x", labelrotation=30)

    plt.tight_layout()
    save_figure(fig, out_path, module="probabilistic")
    plt.close(fig)
