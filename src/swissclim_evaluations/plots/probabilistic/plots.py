from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

from ...helpers import (
    COLOR_GROUND_TRUTH,
    COLOR_MODEL_PREDICTION,
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
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
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


def plot_spaghetti_timeseries(
    lead_hours: np.ndarray,
    member_values: np.ndarray,
    target_values: np.ndarray,
    out_path: Path,
    variable: str,
    level: Any = None,
    units: str = "",
    time_label: str = "",
    dpi: int = 96,
    show_kde: bool = True,
):
    """Plot spatially-averaged timeseries with one line per ensemble member + target.

    Produces a "spaghetti plot" showing ensemble spread alongside the ground truth.
    Each ensemble member is drawn as a thin coloured line, the ensemble mean as a
    dashed line, and the target/observation as a thicker black line.  When
    ``show_kde=True`` and at least two members are available, a kernel density
    panel is appended to the right showing the ensemble distribution at the final
    lead time.

    Args:
        lead_hours: 1-D array of lead-time hours (x-axis values).
        member_values: 2-D array of shape ``(n_members, n_lead_times)`` with spatially
            averaged values per member.
        target_values: 1-D array of shape ``(n_lead_times,)`` with spatially averaged
            target/observation values.
        out_path: Destination file path for the saved figure.
        variable: Variable name (used in the title).
        level: Optional pressure level (included in the title when not ``None``).
        units: Physical units string for the y-axis label.
        time_label: Extra label shown in the upper-right corner of the plot
            (typically the init-time stamp).
        dpi: Base dots-per-inch for the figure.
        show_kde: Append a KDE density panel (ensemble at final lead time) to the
            right of the spaghetti axes.  Requires ``n_members >= 2``.
    """
    n_members = member_values.shape[0]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=dpi * 2)

    member_alpha = max(0.15, min(0.6, 3.0 / n_members))

    # --- Ensemble members: thin coloured lines ---
    for m in range(n_members):
        ax.plot(
            lead_hours,
            member_values[m],
            color=COLOR_MODEL_PREDICTION,
            alpha=member_alpha,
            linewidth=0.8,
            label="Ensemble members" if m == 0 else None,
        )

    # --- Ensemble mean: dashed line ---
    ens_mean = member_values.mean(axis=0)
    ax.plot(
        lead_hours,
        ens_mean,
        color=COLOR_MODEL_PREDICTION,
        linewidth=2.0,
        linestyle="--",
        label="Ensemble mean",
        zorder=9,
    )

    # --- Target / ground truth: thick black line ---
    ax.plot(
        lead_hours,
        target_values,
        color=COLOR_GROUND_TRUTH,
        linewidth=2.0,
        label="Target",
        zorder=10,
    )

    lvl_str = f" @ {format_level_token(level)}" if level is not None else ""
    ax.set_title(
        f"Ensemble Spaghetti — {format_variable_name(str(variable))}{lvl_str}",
        loc="left",
        fontsize=10,
    )
    ax.set_title(time_label, loc="right", fontsize=10)
    ax.set_xlabel("Lead Time [h]")
    y_label = f"Spatial Mean [{units}]" if units else "Spatial Mean"
    ax.set_ylabel(y_label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)

    # --- KDE panel at final lead time ---
    if show_kde and n_members >= 2:
        final_vals = member_values[:, -1]
        if np.isfinite(final_vals).sum() >= 2 and np.std(final_vals) > 0:
            kde = gaussian_kde(final_vals[np.isfinite(final_vals)])
            y_lim = ax.get_ylim()
            y_eval = np.linspace(y_lim[0], y_lim[1], 300)
            density = kde(y_eval)

            divider = make_axes_locatable(ax)
            ax_kde = divider.append_axes("right", size="20%", pad=0.06)
            ax_kde.plot(density, y_eval, color=COLOR_MODEL_PREDICTION, linewidth=1.2)
            ax_kde.fill_betweenx(y_eval, 0, density, alpha=0.2, color=COLOR_MODEL_PREDICTION)
            ax_kde.set_ylim(y_lim)
            ax_kde.yaxis.set_visible(False)
            ax_kde.set_xlabel("Density", fontsize=8)
            ax_kde.tick_params(axis="x", labelsize=7)
            ax_kde.grid(True, linestyle="--", alpha=0.4)

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
