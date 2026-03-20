from __future__ import annotations

import warnings
from pathlib import Path

import dask.array as da
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def _is_geopotential_height(name: str) -> bool:
    return "geopotential_height" in str(name).lower() and "gradient" not in str(name).lower()


def _is_geopotential_height_gradient(name: str) -> bool:
    return "geopotential_height_gradient" in str(name).lower()


def _is_wind_speed(name: str) -> bool:
    return "wind_speed" in str(name).lower()


def _is_geostrophic_gradient_pair(var_x: str, var_y: str) -> bool:
    lx = str(var_x).lower()
    ly = str(var_y).lower()
    return ("geopotential_height_gradient" in lx and "wind_speed" in ly) or (
        "geopotential_height_gradient" in ly and "wind_speed" in lx
    )


def _format_level_suffix(level_hpa: float | None) -> str:
    if level_hpa is None:
        return ""
    if float(level_hpa).is_integer():
        return f"_level{int(level_hpa)}"
    return f"_level{level_hpa:g}"


# ─────────────────────────────────────────────────────────────────────────────
# Physical-constraint overlay helpers
# ─────────────────────────────────────────────────────────────────────────────


def _q_sat_at_pressure(T_K: np.ndarray, P_Pa: float = 50000.0) -> np.ndarray:
    """Saturation specific humidity [kg kg⁻¹] via Bolton (1980).

    Uses the standard Magnus/Tetens saturation-vapour-pressure formula valid
    for temperatures in the range roughly −40 °C to +60 °C.

    Parameters
    ----------
    T_K : array_like
        Temperature in Kelvin.
    P_Pa : float
        Ambient pressure in Pa (default: 50 000 Pa = 500 hPa).
    """
    epsilon = 0.6219  # Rd / Rv ratio
    T_C = np.asarray(T_K, dtype=float) - 273.15
    # Bolton (1980) saturation vapour pressure [Pa]
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    e_s = np.clip(e_s, 0.0, 0.99 * P_Pa)  # prevent q_sat → ∞
    return epsilon * e_s / (P_Pa - (1.0 - epsilon) * e_s)


def _get_physical_constraints(
    var_x: str,
    var_y: str,
    bins_x: np.ndarray,
    bins_y: np.ndarray,
    level_hpa: float | None = None,
) -> list[dict]:
    """Return a list of physical-constraint specs for the given variable pair.

    Each spec is a plain dict that drives :func:`_draw_physical_constraints`.
    Currently handled pairs:

    * **temperature × specific_humidity** — Clausius–Clapeyron saturation curve
      at 500 hPa (Bolton 1980).  The supersaturated region and q < 0 are shaded.
        * **geopotential_height_gradient × wind_speed** — geostrophic reference
            line plus hard lower bound wind speed >= 0.

    Parameters
    ----------
    var_x, var_y : str
        Variable names as they appear in the dataset.
    bins_x, bins_y : np.ndarray
        Bin-edge arrays produced by the histogram step.
    """
    constraints: list[dict] = []

    # Draw physical overlays only for 500 hPa views.
    if level_hpa is None or not np.isclose(float(level_hpa), 500.0):
        return constraints
    lx, ly = var_x.lower(), var_y.lower()

    is_temp_x = "temperature" in lx
    is_temp_y = "temperature" in ly
    is_q_x = "specific_humidity" in lx
    is_q_y = "specific_humidity" in ly
    is_zgrad_x = "geopotential_height_gradient" in lx
    is_zgrad_y = "geopotential_height_gradient" in ly
    is_ws_x = "wind_speed" in lx
    is_ws_y = "wind_speed" in ly

    # ── Temperature vs Specific Humidity ────────────────────────────────────
    # Physical upper bound: q ≤ q_sat(T) — Clausius–Clapeyron.
    # Physical lower bound: q ≥ 0.
    if (is_temp_x and is_q_y) or (is_q_x and is_temp_y):
        if is_temp_x:  # temperature on x-axis, q on y-axis
            T_arr = np.linspace(bins_x[0], bins_x[-1], 400)
            q_sat = _q_sat_at_pressure(T_arr, P_Pa=50000.0)
            constraints.append(
                {
                    "type": "curve",
                    "value_x": T_arr,
                    "value_y": q_sat,
                    "fill_x": T_arr,
                    "fill_y": q_sat,
                    "fill": "above",
                    "color": "#d62728",
                    "lw": 2.0,
                    "ls": "--",
                    "label": r"Saturation $q_\mathrm{sat}(T)$",
                    "fill_alpha": 0.13,
                    "fill_color": "#d62728",
                    "fill_hatch": "///",
                    "fill_label": "Supersaturated (unphysical)",
                }
            )
            constraints.append(
                {
                    "type": "hline",
                    "value": 0.0,
                    "fill": "below",
                    "color": "#8c1515",
                    "lw": 1.5,
                    "ls": ":",
                    "label": "$q = 0$",
                    "fill_alpha": 0.10,
                    "fill_color": "#8c1515",
                    "fill_hatch": "\\\\\\\\",
                    "fill_label": "$q < 0$ (unphysical)",
                }
            )
        else:  # q on x-axis, temperature on y-axis
            T_arr = np.linspace(bins_y[0], bins_y[-1], 400)
            q_sat = _q_sat_at_pressure(T_arr, P_Pa=50000.0)
            constraints.append(
                {
                    "type": "curve",
                    "value_x": q_sat,
                    "value_y": T_arr,
                    "fill_x": q_sat,
                    "fill_y": T_arr,
                    "fill": "right",
                    "color": "#d62728",
                    "lw": 2.0,
                    "ls": "--",
                    "label": r"Saturation $q_\mathrm{sat}(T)$",
                    "fill_alpha": 0.13,
                    "fill_color": "#d62728",
                    "fill_hatch": "///",
                    "fill_label": "Supersaturated (unphysical)",
                }
            )
            constraints.append(
                {
                    "type": "vline",
                    "value": 0.0,
                    "fill": "left",
                    "color": "#8c1515",
                    "lw": 1.5,
                    "ls": ":",
                    "label": "$q = 0$",
                    "fill_alpha": 0.10,
                    "fill_color": "#8c1515",
                    "fill_hatch": "\\\\\\\\",
                    "fill_label": "$q < 0$ (unphysical)",
                }
            )

    # ── Geopotential Height Gradient vs Wind Speed ──────────────────────────
    # Geostrophic balance: U_g = (g / f) * |∇Z|
    # Plotted as a diagonal reference line through the origin.
    # Using representative mid-latitude |f| = 1e-4 s⁻¹ and g = 9.81 m s⁻².
    if (is_zgrad_x or is_zgrad_y) and (is_ws_x or is_ws_y):
        g = 9.81
        f_abs = 1.13e-4  # mid-latitude Coriolis parameter for Europe (~50°N), s⁻¹
        slope = g / f_abs  # ≈ 98 100  (m s⁻¹) / (m m⁻¹)

        if is_zgrad_x and is_ws_y:
            # x = |∇Z|  [m m⁻¹],  y = wind speed  [m s⁻¹]
            x_arr = np.linspace(max(bins_x[0], 0.0), bins_x[-1], 400)
            y_arr = slope * x_arr
            constraints.append(
                {
                    "type": "curve",
                    "value_x": x_arr,
                    "value_y": y_arr,
                    "color": "#ff7f0e",
                    "lw": 2.0,
                    "ls": "--",
                    "label": r"Geostrophic wind $U_g$",
                }
            )
        elif is_zgrad_y and is_ws_x:
            # x = wind speed  [m s⁻¹],  y = |∇Z|  [m m⁻¹]
            x_arr = np.linspace(max(bins_x[0], 0.0), bins_x[-1], 400)
            y_arr = x_arr / slope
            constraints.append(
                {
                    "type": "curve",
                    "value_x": x_arr,
                    "value_y": y_arr,
                    "color": "#ff7f0e",
                    "lw": 2.0,
                    "ls": "--",
                    "label": r"Geostrophic wind $U_g$",
                }
            )

        # Wind speed ≥ 0 hard bound
        if is_ws_y:
            constraints.append(
                {
                    "type": "hline",
                    "value": 0.0,
                    "fill": "below",
                    "color": "#d62728",
                    "lw": 1.5,
                    "ls": ":",
                    "label": "Wind speed $= 0$",
                    "fill_alpha": 0.10,
                    "fill_color": "#d62728",
                    "fill_hatch": "\\\\\\\\",
                    "fill_label": "Wind speed $< 0$ (unphysical)",
                }
            )
        else:
            constraints.append(
                {
                    "type": "vline",
                    "value": 0.0,
                    "fill": "left",
                    "color": "#d62728",
                    "lw": 1.5,
                    "ls": ":",
                    "label": "Wind speed $= 0$",
                    "fill_alpha": 0.10,
                    "fill_color": "#d62728",
                    "fill_hatch": "\\\\\\\\",
                    "fill_label": "Wind speed $< 0$ (unphysical)",
                }
            )

    return constraints


def _draw_physical_constraints(
    ax: plt.Axes,
    constraints: list[dict],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> list[tuple]:
    """Draw physical-constraint lines and shaded regions on *ax*.

    Returns a list of ``(artist, label)`` pairs to be appended to the legend.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    legend_entries: list[tuple] = []
    fill_labels_seen: set[str] = set()

    for spec in constraints:
        ctype = spec["type"]
        color = spec.get("color", "#d62728")
        lw = spec.get("lw", 1.5)
        ls = spec.get("ls", "--")
        label = spec.get("label", "")
        fill_side = spec.get("fill", None)
        fill_alpha = spec.get("fill_alpha", 0.12)
        fill_color = spec.get("fill_color", color)
        fill_hatch = spec.get("fill_hatch", None)
        fill_label = spec.get("fill_label", None)

        if ctype == "hline":
            val = float(spec["value"])
            (line,) = ax.plot(
                [x_min, x_max],
                [val, val],
                color=color,
                lw=lw,
                ls=ls,
                zorder=6,
                label=label,
            )
            legend_entries.append((line, label))
            if fill_side == "above":
                ax.fill_between(
                    [x_min, x_max],
                    val,
                    y_max,
                    color=fill_color,
                    alpha=fill_alpha,
                    hatch=fill_hatch,
                    zorder=4,
                    linewidth=0,
                )
            elif fill_side == "below":
                ax.fill_between(
                    [x_min, x_max],
                    y_min,
                    val,
                    color=fill_color,
                    alpha=fill_alpha,
                    hatch=fill_hatch,
                    zorder=4,
                    linewidth=0,
                )

        elif ctype == "vline":
            val = float(spec["value"])
            (line,) = ax.plot(
                [val, val],
                [y_min, y_max],
                color=color,
                lw=lw,
                ls=ls,
                zorder=6,
                label=label,
            )
            legend_entries.append((line, label))
            if fill_side == "right":
                ax.fill_betweenx(
                    [y_min, y_max],
                    val,
                    x_max,
                    color=fill_color,
                    alpha=fill_alpha,
                    hatch=fill_hatch,
                    zorder=4,
                    linewidth=0,
                )
            elif fill_side == "left":
                ax.fill_betweenx(
                    [y_min, y_max],
                    x_min,
                    val,
                    color=fill_color,
                    alpha=fill_alpha,
                    hatch=fill_hatch,
                    zorder=4,
                    linewidth=0,
                )

        elif ctype == "curve":
            xs = np.asarray(spec["value_x"])
            ys = np.asarray(spec["value_y"])
            (line,) = ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=6, label=label)
            legend_entries.append((line, label))
            fill_xs = np.asarray(spec.get("fill_x", xs))
            fill_ys = np.asarray(spec.get("fill_y", ys))
            if fill_side == "above":
                ax.fill_between(
                    fill_xs,
                    fill_ys,
                    y_max,
                    color=fill_color,
                    alpha=fill_alpha,
                    hatch=fill_hatch,
                    zorder=4,
                    linewidth=0,
                )
            elif fill_side == "below":
                ax.fill_between(
                    fill_xs,
                    y_min,
                    fill_ys,
                    color=fill_color,
                    alpha=fill_alpha,
                    hatch=fill_hatch,
                    zorder=4,
                    linewidth=0,
                )
            elif fill_side == "right":
                ax.fill_betweenx(
                    fill_ys,
                    fill_xs,
                    x_max,
                    color=fill_color,
                    alpha=fill_alpha,
                    hatch=fill_hatch,
                    zorder=4,
                    linewidth=0,
                )
            elif fill_side == "left":
                ax.fill_betweenx(
                    fill_ys,
                    x_min,
                    fill_xs,
                    color=fill_color,
                    alpha=fill_alpha,
                    hatch=fill_hatch,
                    zorder=4,
                    linewidth=0,
                )

        # Add shaded-region label (deduplicated)
        if fill_label and fill_label not in fill_labels_seen:
            fill_labels_seen.add(fill_label)
            patch = mpatches.Patch(
                color=fill_color,
                alpha=fill_alpha + 0.1,
                hatch=fill_hatch,
                label=fill_label,
                linewidth=0,
            )
            legend_entries.append((patch, fill_label))

    return legend_entries


def _plot_bivariate_per_lead_grid(
    ds_prediction: xr.Dataset,
    ds_target: xr.Dataset | None,
    var_x: str,
    var_y: str,
    xedges: np.ndarray,
    yedges: np.ndarray,
    out_root: Path,
    ensemble_token: str | None = None,
    level_hpa: float | None = None,
) -> None:
    """Generate a per-lead-time grid of bivariate histograms for one variable pair.

    Produces a single figure where each subplot corresponds to one lead time,
    arranged in a grid of up to 3 columns.  All subplots share the same bin
    edges (``xedges`` / ``yedges``) so that densities are directly comparable
    across lead times.

    **Dask contract**: All lazy histogram objects for every lead time (both
    prediction and target) are collected into a single :func:`compute_jobs`
    call.  No per-lead ``compute()`` is issued, ensuring a single scheduler
    round-trip.

    Parameters
    ----------
    ds_prediction : xr.Dataset
        Prediction dataset.  **Must** contain a ``lead_time`` dimension with
        at least 2 values for this function to produce output.
    ds_target : xr.Dataset or None
        Reference/observation dataset.  If ``None`` the subplots show only
        the prediction distribution (no filled ground-truth contours).
    var_x : str
        Variable name plotted on the x-axis.
    var_y : str
        Variable name plotted on the y-axis.
    xedges : np.ndarray
        Bin edges for the x-axis, pre-computed from the global data range.
    yedges : np.ndarray
        Bin edges for the y-axis, pre-computed from the global data range.
    out_root : Path
        Root output directory.  A ``multivariate/`` sub-folder is created
        automatically.
    ensemble_token : str or None
        Optional token appended to the output filename
        (e.g. ``ensmean``, ``ens0``).
    level_hpa : float or None
        Pressure-level value in hPa (e.g. ``500.0``) to select from 3-D
        variables before slicing by lead time.  When ``None`` the full array
        is used (2-D variable case).  The caller must pass the same value
        that was used to compute ``xedges``/``yedges``.
    """
    # ── Guard: need multi-lead prediction ────────────────────────────────────
    if "lead_time" not in ds_prediction[var_x].dims:
        return
    leads = ds_prediction["lead_time"].values
    n_leads = len(leads)
    if n_leads < 2:
        return

    # ── Pre-compute fill sentinels (out-of-range substitutes for NaN) ────────
    fill_x = float(xedges[0]) - 1.0
    fill_y = float(yedges[0]) - 1.0
    range_x = [float(xedges[0]), float(xedges[-1])]
    range_y = [float(yedges[0]), float(yedges[-1])]

    # ── Build one lazy histogram job per lead time ────────────────────────────
    # All lazy objects are collected here; a single compute_jobs call resolves
    # the entire graph without repeated scheduler round-trips.
    hist_jobs: list[dict] = []
    for lt in leads:
        # ── Prediction slice ──────────────────────────────────────────────────
        # Apply level selection for 3-D variables before slicing by lead time.
        _px = ds_prediction[var_x]
        if level_hpa is not None and "level" in _px.dims:
            _px = _px.sel(level=level_hpa)
        da_xp = _px.sel(lead_time=lt).data.flatten()
        _py = ds_prediction[var_y]
        if level_hpa is not None and "level" in _py.dims:
            _py = _py.sel(level=level_hpa)
        da_yp = _py.sel(lead_time=lt).data.flatten()
        da_xp = da.where(da.isnan(da_xp), fill_x, da_xp)
        da_yp = da.where(da.isnan(da_yp), fill_y, da_yp)
        h_pred_lazy, _, _ = da.histogram2d(
            da_xp, da_yp, bins=[xedges, yedges], range=[range_x, range_y]
        )

        # ── Target slice (may lack lead_time dim — use full array in that case)
        # Apply level selection first when dealing with 3-D target variables.
        h_target_lazy = None
        if ds_target is not None:
            _tx = ds_target[var_x]
            _ty = ds_target[var_y]
            if level_hpa is not None:
                if "level" in _tx.dims:
                    _tx = _tx.sel(level=level_hpa)
                if "level" in _ty.dims:
                    _ty = _ty.sel(level=level_hpa)
            if "lead_time" in _tx.dims:
                da_xt = _tx.sel(lead_time=lt).data.flatten()
            else:
                da_xt = _tx.data.flatten()
            if "lead_time" in _ty.dims:
                da_yt = _ty.sel(lead_time=lt).data.flatten()
            else:
                da_yt = _ty.data.flatten()
            da_xt = da.where(da.isnan(da_xt), fill_x, da_xt)
            da_yt = da.where(da.isnan(da_yt), fill_y, da_yt)
            h_target_lazy, _, _ = da.histogram2d(
                da_xt, da_yt, bins=[xedges, yedges], range=[range_x, range_y]
            )

        hist_jobs.append(
            {
                "h_pred": h_pred_lazy,
                "h_target": h_target_lazy,
                "lt": lt,
            }
        )

    # ── Single batch compute for all lead-time histograms ────────────────────
    compute_jobs(
        hist_jobs,
        key_map={"h_pred": "hist_pred", "h_target": "hist_target"},
        desc=f"Computing bivariate histograms by lead: {var_x} vs {var_y}",
    )

    # ── Global norm across all lead times ────────────────────────────────────
    dx = np.diff(xedges).mean()
    dy = np.diff(yedges).mean()
    bin_area = dx * dy
    all_valid: list[np.ndarray] = []
    for job in hist_jobs:
        for key in ("hist_pred", "hist_target"):
            h = job.get(key)
            if h is not None:
                total = h.sum()
                if total > 0:
                    dens = h / (total * bin_area)
                    valid = dens[dens > 0]
                    if len(valid):
                        all_valid.append(valid)
    if all_valid:
        all_vals = np.concatenate(all_valid)
        global_vmin = max(all_vals.min(), 1e-10)
        global_vmax = all_vals.max()
        global_norm: LogNorm | None = LogNorm(vmin=global_vmin, vmax=global_vmax)
    else:
        global_norm = None

    # ── Grid layout ───────────────────────────────────────────────────────────
    cols = min(3, n_leads)
    rows = int(np.ceil(n_leads / cols))
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 5 * rows),
        dpi=150,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axs_flat = np.atleast_1d(np.array(axs)).flatten()

    xlabel_str = _get_label(ds_prediction[var_x], var_x)
    ylabel_str = _get_label(ds_prediction[var_y], var_y)

    for i, job in enumerate(hist_jobs):
        lt = job["lt"]
        if np.issubdtype(type(lt), np.timedelta64):
            hours = int(lt / np.timedelta64(1, "h"))
            lead_label = f"+{hours}h"
        else:
            lead_label = str(lt)

        ax = axs_flat[i]
        hist_pred = job.get("hist_pred")
        hist_target = job.get("hist_target")

        if hist_pred is None:
            ax.set_title(lead_label, fontsize=10)
            ax.axis("off")
            continue

        if hist_target is None:
            hist_target = np.zeros_like(hist_pred)

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
                level_hpa=level_hpa,
                ax=ax,
                xlabel=xlabel_str,
                ylabel=ylabel_str,
                norm=global_norm,
                show_colorbar=False,
                full_legend=(i == 0),
            )
        ax.set_title(lead_label, fontsize=10)

    # ── Hide surplus subplots ─────────────────────────────────────────────────
    for j in range(n_leads, len(axs_flat)):
        axs_flat[j].axis("off")

    # ── Shared axis labels — only edges show labels and ticks ─────────────────
    for i in range(n_leads):
        _ax = axs_flat[i]
        col = i % cols
        if col != 0:
            _ax.set_ylabel("")
            _ax.tick_params(labelleft=False)
        # Effective bottom: no active subplot one full row below this one
        if i + cols < n_leads:
            _ax.set_xlabel("")
            _ax.tick_params(labelbottom=False)

    # ── Shared colorbar at the bottom ─────────────────────────────────────────
    if global_norm is not None:
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=global_norm)
        sm.set_array([])
        n_levels = 5
        cbar_levels = np.logspace(
            np.log10(global_norm.vmin), np.log10(global_norm.vmax), n_levels + 2
        )[1:-1]
        cbar = fig.colorbar(
            sm,
            ax=list(axs_flat[:n_leads]),
            orientation="horizontal",
            location="bottom",
            pad=0.04,
            fraction=0.04,
            shrink=1.0,
            ticks=cbar_levels,
        )
        cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
        cbar.set_label("Density (log scale)")

    lev_title = f" @ {level_hpa:g} hPa" if level_hpa is not None else ""
    fig.suptitle(
        f"Bivariate Histograms by Lead Time — "
        f"{format_variable_name(var_x)} vs {format_variable_name(var_y)}{lev_title}",
        fontsize=14,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    suffix = f"_{ensemble_token}" if ensemble_token else ""
    lev_sfx = _format_level_suffix(level_hpa)
    out_dir = out_root / "multivariate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"bivariate_by_lead_{var_x}_{var_y}{lev_sfx}{suffix}.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    c.info(f"[multivariate] Saved per-lead-time bivariate grid: {out_png.name}")


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
    level_hpa: float | None = None,
    ax: plt.Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    norm: LogNorm | None = None,
    show_colorbar: bool = True,
    full_legend: bool = True,
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
        xlim: Optional explicit x-axis limits. When provided with ``ylim``,
            these limits are applied before physical overlays are drawn.
        ylim: Optional explicit y-axis limits. When provided with ``xlim``,
            these limits are applied before physical overlays are drawn.
        norm: Optional pre-computed ``LogNorm`` to use for both contourf and
            contour.  When ``None`` (default) the norm is derived from the
            reference distribution (``hist_2``).
        show_colorbar: Whether to attach a per-axes colorbar.  Set to
            ``False`` when the caller manages a shared colorbar.
        full_legend: When ``True`` (default) the full legend is rendered
            (target patches + model line + physical constraints).  When
            ``False`` only the model name (``label_1``) is shown.

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

    if norm is None:
        vmin = valid_2.min()
        vmax = valid_2.max()
        # Ensure vmin is positive for log scale
        if vmin <= 0:
            vmin = 1e-10
        norm = LogNorm(vmin=vmin, vmax=vmax)

    # Create log-spaced levels
    # We generate N+2 levels and take the inner N to ensure the contour lines are visible
    # and not at the absolute min/max (which are hard to see).
    # This aligns the number of visible lines on the plot with the lines on the colorbar.
    # Using 5 internal levels gives a clean look similar to the reference paper.
    n_levels = 5
    full_levels = np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), n_levels + 2)
    levels = full_levels[1:-1]

    # Plot Model 2 (Filled, Color) - Reference
    cs2 = ax.contourf(
        X,
        Y,
        dens_2,
        levels=levels,
        norm=norm,
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
        norm=norm,
        cmap=cmap_greys,
        linewidths=1.5,
    )

    if show_colorbar:
        fig = ax.get_figure()
        if fig:
            cbar = fig.colorbar(cs2, ax=ax, ticks=levels)
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
            cbar.set_label("Density (log scale)", fontsize=13)
            cbar.ax.tick_params(labelsize=12)
            cbar.add_lines(cs1)

    ax.set_xlabel(xlabel if xlabel else var_x, fontsize=13)
    ax.set_ylabel(ylabel if ylabel else var_y, fontsize=13)
    ax.tick_params(labelsize=12)

    if _is_geostrophic_gradient_pair(var_x, var_y):
        ax.tick_params(axis="x", labelrotation=45, labelsize=12)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

    title = f"{format_variable_name(var_x)} vs {format_variable_name(var_y)}"
    if level_hpa is not None:
        title += f" ({level_hpa:g} hPa)"
    ax.set_title(title, fontsize=13)

    # Expand x by ~12.5% each side, y by ~5% each side (zoomed in more on y).
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        x_min, x_max = bins_x.min(), bins_x.max()
        y_min, y_max = bins_y.min(), bins_y.max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2

        ax.set_xlim(x_center - 0.625 * x_range, x_center + 0.625 * x_range)
        ax.set_ylim(y_center - 0.525 * y_range, y_center + 0.525 * y_range)

    # ── Physical-constraint overlays ─────────────────────────────────────────
    final_xlim = ax.get_xlim()
    final_ylim = ax.get_ylim()
    constraints = _get_physical_constraints(
        var_x,
        var_y,
        bins_x,
        bins_y,
        level_hpa=level_hpa,
    )
    constraint_entries = _draw_physical_constraints(
        ax,
        constraints,
        x_range=(final_xlim[0], final_xlim[1]),
        y_range=(final_ylim[0], final_ylim[1]),
    )
    # Re-apply limits after fill operations (fill_between can expand them)
    ax.set_xlim(final_xlim)
    ax.set_ylim(final_ylim)

    # ── Legend ───────────────────────────────────────────────────────────────
    _lx, _ly = var_x.lower(), var_y.lower()
    if _is_geostrophic_gradient_pair(var_x, var_y):
        _legend_loc = "lower right"
    elif ("temperature" in _lx and "specific_humidity" in _ly) or (
        "specific_humidity" in _lx and "temperature" in _ly
    ):
        _legend_loc = "upper left"
    else:
        _legend_loc = "best"

    if full_legend:
        # Use 3 representative colors for the filled contours (low, mid, high)
        cmap = plt.get_cmap("plasma")
        patch1 = mpatches.Patch(color=cmap(0.2))
        patch2 = mpatches.Patch(color=cmap(0.5))
        patch3 = mpatches.Patch(color=cmap(0.8))

        handles: list = [
            (patch1, patch2, patch3),
            Line2D([0], [0], color="grey", lw=1.5),
        ]
        labels: list = [label_2, label_1]

        # Append physical-constraint artists
        for artist, lbl in constraint_entries:
            handles.append(artist)
            labels.append(lbl)

        ax.legend(
            handles=handles,
            labels=labels,
            loc=_legend_loc,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            fontsize=11,
            framealpha=0.85,
        )
    else:
        ax.legend_ = None  # no legend for non-primary subplots

    return ax
