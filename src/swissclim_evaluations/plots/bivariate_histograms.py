from __future__ import annotations

import warnings
from pathlib import Path

import dask.array as da
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from .. import console as c
from ..dask_utils import compute_jobs
from ..helpers import format_variable_name, get_variable_units


def _get_label(da: xr.DataArray, var_name: str) -> str:
    """Get a formatted label with unit for a variable from DataArray attributes."""
    name = format_variable_name(var_name)
    unit = get_variable_units(da, var_name)
    if unit:
        return f"{name} [{unit}]"
    return name


def calculate_and_plot_bivariate_histograms(
    ds_prediction: xr.Dataset,
    ds_target: xr.Dataset | None,
    pairs: list[list[str]],
    out_root: Path,
    bins: int = 100,
    ensemble_token: str | None = None,
) -> None:
    """
    Calculate and save bivariate histograms for specified pairs.
    Also generates the plot immediately if target is available.
    """
    plotted_pairs = []
    skipped_pairs = []

    # 1. First pass: Collect lazy min/max computations for all pairs
    range_jobs = []
    valid_pairs_indices = []

    for i, pair in enumerate(pairs):
        if len(pair) != 2:
            continue
        var_x, var_y = pair

        # Check if variables exist in prediction dataset
        if var_x not in ds_prediction or var_y not in ds_prediction:
            skipped_pairs.append(f"{var_x} vs {var_y} (missing in prediction)")
            continue

        # Check if variables exist in target dataset
        if ds_target is None:
            skipped_pairs.append(f"{var_x} vs {var_y} (no target dataset)")
            continue

        if var_x not in ds_target or var_y not in ds_target:
            skipped_pairs.append(f"{var_x} vs {var_y} (missing in target)")
            continue

        # Compute for Prediction
        da_x = ds_prediction[var_x].data.flatten()
        da_y = ds_prediction[var_y].data.flatten()

        # Compute min/max lazily to determine range and handle NaNs
        min_x_lazy = da.nanmin(da_x)
        max_x_lazy = da.nanmax(da_x)
        min_y_lazy = da.nanmin(da_y)
        max_y_lazy = da.nanmax(da_y)

        range_jobs.append(
            {
                "min_x": min_x_lazy,
                "max_x": max_x_lazy,
                "min_y": min_y_lazy,
                "max_y": max_y_lazy,
                "pair_idx": i,
                "var_x": var_x,
                "var_y": var_y,
            }
        )
        valid_pairs_indices.append(i)

    if not range_jobs:
        if skipped_pairs:
            unique_skips = sorted(set(skipped_pairs))
            c.warn("[multivariate] Skipped pairs:\n  • " + "\n  • ".join(unique_skips))
        return

    # Compute ranges in batch
    compute_jobs(
        range_jobs,
        key_map={
            "min_x": "min_x_res",
            "max_x": "max_x_res",
            "min_y": "min_y_res",
            "max_y": "max_y_res",
        },
        desc="Computing ranges",
    )

    # 2. Second pass: Create histogram lazy objects using computed ranges
    hist_jobs = []

    for job in range_jobs:
        var_x = job["var_x"]
        var_y = job["var_y"]

        try:
            min_x = float(job["min_x_res"])
            max_x = float(job["max_x_res"])
            min_y = float(job["min_y_res"])
            max_y = float(job["max_y_res"])
        except Exception:
            skipped_pairs.append(f"{var_x} vs {var_y} (computation failed)")
            continue

        if np.isnan(min_x) or np.isnan(max_x) or np.isnan(min_y) or np.isnan(max_y):
            skipped_pairs.append(f"{var_x} vs {var_y} (all NaNs)")
            continue

        range_x = [min_x, max_x]
        range_y = [min_y, max_y]

        # Replace NaNs with out-of-range value to filter them during histogram2d
        fill_x = min_x - 1.0
        fill_y = min_y - 1.0

        # Re-access data (dask arrays)
        da_x = ds_prediction[var_x].data.flatten()
        da_y = ds_prediction[var_y].data.flatten()

        da_x = da.where(da.isnan(da_x), fill_x, da_x)
        da_y = da.where(da.isnan(da_y), fill_y, da_y)

        # Calculate edges manually to avoid dask array edges which cannot be passed to bins=
        xedges = np.linspace(min_x, max_x, bins + 1)
        yedges = np.linspace(min_y, max_y, bins + 1)

        h_pred_lazy, _, _ = da.histogram2d(
            da_x, da_y, bins=[xedges, yedges], range=[range_x, range_y]
        )

        # Target data
        if ds_target is not None:
            da_x_t = ds_target[var_x].data.flatten()
            da_y_t = ds_target[var_y].data.flatten()

            da_x_t = da.where(da.isnan(da_x_t), fill_x, da_x_t)
            da_y_t = da.where(da.isnan(da_y_t), fill_y, da_y_t)

            h_target_lazy, _, _ = da.histogram2d(
                da_x_t, da_y_t, bins=[xedges, yedges], range=[range_x, range_y]
            )
        else:
            # Should not happen given checks above, but for safety
            h_target_lazy = None

        hist_jobs.append(
            {
                "h_pred": h_pred_lazy,
                "h_target": h_target_lazy,
                "xedges": xedges,
                "yedges": yedges,
                "var_x": var_x,
                "var_y": var_y,
                "range_x": range_x,
                "range_y": range_y,
            }
        )

    # Compute histograms in batch
    compute_jobs(
        hist_jobs,
        key_map={
            "h_pred": "hist_pred",
            "h_target": "hist_target",
        },
        desc="Computing histograms",
    )

    # 3. Plotting
    for job in hist_jobs:
        if "hist_pred" not in job or "hist_target" not in job:
            continue

        hist_pred = job["hist_pred"]
        hist_target = job["hist_target"]
        xedges = job["xedges"]
        yedges = job["yedges"]
        var_x = job["var_x"]
        var_y = job["var_y"]

        # Save and Plot
        suffix = f"_{ensemble_token}" if ensemble_token else ""
        out_file = out_root / "multivariate" / f"bivariate_hist_{var_x}_{var_y}{suffix}.npz"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_file,
            hist=hist_pred,
            bins_x=xedges,
            bins_y=yedges,
            hist_target=hist_target,
        )

        # Generate plot immediately
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

        # Suppress log scale warnings for zero values
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Log scale: values of z <= 0 have been masked"
            )
            plot_bivariate_histogram(
                hist_1=hist_pred,
                hist_2=hist_target,
                bins_x=xedges,
                bins_y=yedges,
                label_1="Model Prediction",
                label_2="Ground Truth",
                var_x=var_x,
                var_y=var_y,
                ax=ax,
                xlabel=_get_label(ds_prediction[var_x], var_x),
                ylabel=_get_label(ds_prediction[var_y], var_y),
            )

        plot_out = out_root / "multivariate" / f"bivariate_{var_x}_{var_y}{suffix}.png"
        fig.savefig(plot_out, bbox_inches="tight")
        plt.close(fig)

        print(f"[multivariate] Saved bivariate plot: {plot_out.name}")
        plotted_pairs.append(f"{var_x} vs {var_y}")

    # Summary output
    if plotted_pairs:
        print(f"[multivariate] Plotted {len(plotted_pairs)} pairs: {', '.join(plotted_pairs)}")

    if skipped_pairs:
        # Only show unique reasons
        unique_skips = sorted(set(skipped_pairs))
        c.warn("[multivariate] Skipped pairs:\n  • " + "\n  • ".join(unique_skips))


def plot_bivariate_histogram(
    hist_1: np.ndarray,
    hist_2: np.ndarray,
    bins_x: np.ndarray,
    bins_y: np.ndarray,
    label_1: str,
    label_2: str,
    var_x: str,
    var_y: str,
    ax: plt.Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> plt.Axes:
    """
    Plot bivariate histograms for two models/datasets.

    Model 1 is plotted with greyscale contour lines.
    Model 2 is plotted with filled color contours.

    Args:
        hist_1: 2D histogram counts for model 1. Shape (nx, ny).
        hist_2: 2D histogram counts for model 2. Shape (nx, ny).
        bins_x: Bin edges for x variable.
        bins_y: Bin edges for y variable.
        label_1: Label for model 1 (contours).
        label_2: Label for model 2 (filled).
        var_x: Name of x variable.
        var_y: Name of y variable.
        ax: Axes to plot on. If None, creates new figure.
        xlabel: Label for x-axis. If None, uses var_x.
        ylabel: Label for y-axis. If None, uses var_y.

    Returns:
        The axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Calculate centers
    x_centers = (bins_x[:-1] + bins_x[1:]) / 2
    y_centers = (bins_y[:-1] + bins_y[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    # Normalize histograms to density
    # We divide by the sum to get probability mass, then divide by bin area to get density.
    # Assuming uniform bins for simplicity in area calculation.
    # Use average bin width for area calculation to support non-uniform bins.
    dx = np.diff(bins_x).mean()
    dy = np.diff(bins_y).mean()

    if dx <= 0 or dy <= 0:
        # Avoid division by zero or invalid bin sizes
        if ax:
            ax.text(
                0.5,
                0.5,
                "Invalid bin sizes (dx/dy <= 0)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        return ax

    bin_area = dx * dy

    sum_1 = hist_1.sum()
    sum_2 = hist_2.sum()

    dens_1 = hist_1 / (sum_1 * bin_area) if sum_1 > 0 else hist_1
    dens_2 = hist_2 / (sum_2 * bin_area) if sum_2 > 0 else hist_2

    # Logarithmic scale
    # Define levels based on the filled distribution (Model 2)
    valid_2 = dens_2[dens_2 > 0]
    if len(valid_2) == 0:
        ax.text(
            0.5,
            0.5,
            "No data in reference distribution",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    vmin = valid_2.min()
    vmax = valid_2.max()

    # Ensure vmin is positive for log scale
    if vmin <= 0:
        vmin = 1e-10

    # Create log-spaced levels
    # We generate N+2 levels and take the inner N to ensure the contour lines are visible
    # and not at the absolute min/max (which are hard to see).
    # This aligns the number of visible lines on the plot with the lines on the colorbar.
    # Using 5 internal levels gives a clean look similar to the reference paper.
    n_levels = 5
    full_levels = np.logspace(np.log10(vmin), np.log10(vmax), n_levels + 2)
    levels = full_levels[1:-1]

    # Plot Model 2 (Filled, Color) - Reference
    cs2 = ax.contourf(
        X,
        Y,
        dens_2,
        levels=levels,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="plasma",
        extend="both",
    )

    # Plot Model 1 (Lines, Greyscale) - Prediction
    # Use a truncated Greys colormap so low density is light grey (not white) and high is black
    cmap_base = plt.get_cmap("Greys")
    colors_sampled = cmap_base(np.linspace(0.3, 1.0, 256))
    cmap_greys = mcolors.LinearSegmentedColormap.from_list("truncated_greys", colors_sampled)

    cs1 = ax.contour(
        X,
        Y,
        dens_1,
        levels=levels,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap_greys,
        linewidths=1.5,
    )

    fig = ax.get_figure()
    if fig:
        cbar = fig.colorbar(cs2, ax=ax, format="%.2e")
        cbar.set_label("Density (log scale)")
        cbar.add_lines(cs1)

    ax.set_xlabel(xlabel if xlabel else var_x)
    ax.set_ylabel(ylabel if ylabel else var_y)
    ax.set_title(f"{format_variable_name(var_x)} vs {format_variable_name(var_y)}")

    # Zoom out by 25%
    x_min, x_max = bins_x.min(), bins_x.max()
    y_min, y_max = bins_y.min(), bins_y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    # Expand by 25% (factor 1.25)
    ax.set_xlim(x_center - 0.625 * x_range, x_center + 0.625 * x_range)
    ax.set_ylim(y_center - 0.625 * y_range, y_center + 0.625 * y_range)

    # Add legend for the contours
    # Use 3 representative colors for the filled contours (low, mid, high)
    cmap = plt.get_cmap("plasma")
    patch1 = mpatches.Patch(color=cmap(0.2))
    patch2 = mpatches.Patch(color=cmap(0.5))
    patch3 = mpatches.Patch(color=cmap(0.8))

    handles = [
        (patch1, patch2, patch3),
        Line2D([0], [0], color="grey", lw=1.5),
    ]
    labels = [label_2, label_1]

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
    )

    return ax
