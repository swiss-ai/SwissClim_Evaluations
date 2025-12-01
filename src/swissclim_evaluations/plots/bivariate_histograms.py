from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D


def _get_label(da: xr.DataArray, var_name: str) -> str:
    """Get a formatted label with unit for a variable from DataArray attributes."""
    name = var_name
    unit = da.attrs.get("units", "")
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
    for pair in pairs:
        if len(pair) != 2:
            continue
        var_x, var_y = pair

        # Compute for Prediction
        hist_pred, xedges, yedges = None, None, None
        if var_x in ds_prediction and var_y in ds_prediction:
            x_data = ds_prediction[var_x].values.flatten()
            y_data = ds_prediction[var_y].values.flatten()
            mask = np.isfinite(x_data) & np.isfinite(y_data)
            x_data = x_data[mask]
            y_data = y_data[mask]

            if len(x_data) > 0:
                hist_pred, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)

        # Compute for Target
        hist_target = None
        if ds_target is not None and var_x in ds_target and var_y in ds_target:
            x_data_t = ds_target[var_x].values.flatten()
            y_data_t = ds_target[var_y].values.flatten()
            mask_t = np.isfinite(x_data_t) & np.isfinite(y_data_t)
            x_data_t = x_data_t[mask_t]
            y_data_t = y_data_t[mask_t]

            if len(x_data_t) > 0:
                # Use same bins as prediction if available
                if xedges is not None and yedges is not None:
                    hist_target, _, _ = np.histogram2d(x_data_t, y_data_t, bins=[xedges, yedges])
                else:
                    hist_target, xedges, yedges = np.histogram2d(x_data_t, y_data_t, bins=bins)

        if hist_pred is not None:
            suffix = f"_{ensemble_token}" if ensemble_token else ""
            out_file = out_root / "multivariate" / f"bivariate_hist_{var_x}_{var_y}{suffix}.npz"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                "hist": hist_pred,
                "bins_x": xedges,
                "bins_y": yedges,
            }
            if hist_target is not None:
                save_dict["hist_target"] = hist_target

                # Generate plot immediately
                fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
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

            np.savez(out_file, **save_dict)


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
    dx = bins_x[1] - bins_x[0]
    dy = bins_y[1] - bins_y[0]

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
    ax.set_title(f"{var_x} vs {var_y}")

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
