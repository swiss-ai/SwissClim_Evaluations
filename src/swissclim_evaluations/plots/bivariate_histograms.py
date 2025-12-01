from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D


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
    bin_area = dx * dy

    sum_1 = hist_1.sum()
    sum_2 = hist_2.sum()

    dens_1 = hist_1 / (sum_1 * bin_area) if sum_1 > 0 else hist_1
    dens_2 = hist_2 / (sum_2 * bin_area) if sum_2 > 0 else hist_2

    # Logarithmic scale
    # Define levels based on the filled distribution (Model 2)
    valid_2 = dens_2[dens_2 > 0]
    if len(valid_2) == 0:
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
    )  # Plot Model 1 (Lines, Greyscale) - Prediction
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
