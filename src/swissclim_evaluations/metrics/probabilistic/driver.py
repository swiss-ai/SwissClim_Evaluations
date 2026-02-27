from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ... import console as c
from ...dask_utils import (
    dask_histogram,
)
from ...helpers import (
    COLOR_DIAGNOSTIC,
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_variable_name,
    save_data,
    save_figure,
)
from . import calc
from .wbx import run_probabilistic_wbx


def run_probabilistic(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    cfg_plot: dict[str, Any],
    cfg_all: dict[str, Any],
    performance_cfg: dict[str, Any] | None = None,
    include_wbx_outputs: bool = True,
    ensemble_mode: str | None = None,
) -> None:
    """Run probabilistic metrics (PIT histograms) and optionally WBX metrics."""
    effective_plot_cfg = dict(cfg_plot or {})
    legacy_save_plot_data = bool(effective_plot_cfg.get("save_plot_data", False))
    if legacy_save_plot_data and "output_mode" not in effective_plot_cfg:
        effective_plot_cfg["output_mode"] = "both"

    plot_probabilistic(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        out_root=out_root,
        plotting_cfg=effective_plot_cfg,
    )

    if include_wbx_outputs:
        run_probabilistic_wbx(
            ds_target=ds_target,
            ds_prediction=ds_prediction,
            out_root=out_root,
            plotting_cfg=effective_plot_cfg,
            all_cfg=cfg_all,
            performance_cfg=performance_cfg,
        )


def plot_probabilistic(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    """Generate probabilistic plots (PIT histogram only).

    Saves under out_root/probabilistic. If output_mode in {'npz','both'} also
    writes NPZ data artifacts.
    """
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    if not save_fig and not save_npz:
        c.print("[probabilistic] Skipping plot_probabilistic: output_mode=none.")
        return
    dpi = int((plotting_cfg or {}).get("dpi", 48))
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    # Identify common variables
    common_vars = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not common_vars:
        c.print("[probabilistic] No common variables found for plotting.")
        return

    # Helper for converting lead hours
    def _to_hour_safe(val, fallback: int) -> int:
        arr = np.asarray(val)
        if np.issubdtype(arr.dtype, np.timedelta64):
            return int(arr / np.timedelta64(1, "h"))
        return int(arr) if np.isfinite(arr).all() else fallback

    # Attempt time range extraction for plots (once for the dataset)
    def _extract_init_range_plot(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        vals = ds["init_time"].values
        if vals.size == 0:
            return None
        start = np.datetime64(np.min(vals)).astype("datetime64[h]")
        end = np.datetime64(np.max(vals)).astype("datetime64[h]")

        def _fmt(x):
            return (
                np.datetime_as_string(x, unit="h")
                .replace("-", "")
                .replace(":", "")
                .replace("T", "")
            )

        return (_fmt(start), _fmt(end))

    def _extract_lead_range_plot(ds: xr.Dataset):
        if "lead_time" not in ds:
            return None
        vals = ds["lead_time"].values
        if getattr(vals, "size", 0) == 0:
            return None
        hours = (vals / np.timedelta64(1, "h")).astype(int)
        if hours.size == 0:
            return None
        sh = int(np.min(hours))
        eh = int(np.max(hours))

        def _fmt(h: int) -> str:
            return f"{h:03d}h"

        return (_fmt(sh), _fmt(eh))

    init_range_plot = _extract_init_range_plot(ds_prediction)
    lead_range_plot = _extract_lead_range_plot(ds_prediction)

    for var_name in common_vars:
        da_t_var = ds_target[var_name]
        da_p_var = ds_prediction[var_name]

        # Ensure target does not have ensemble dimension (even after align)
        if "ensemble" in da_t_var.dims:
            da_t_var = da_t_var.isel(ensemble=0, drop=True)
        if "ensemble" in da_t_var.coords:
            da_t_var = da_t_var.drop_vars("ensemble")

        # Use calc module for PIT calculation
        pit = calc.probability_integral_transform(
            da_t_var,
            da_p_var,
            ensemble_dim="ensemble",
            name_prefix="PIT",
        )

        if pit.size > 0:
            if "lead_time" in pit.dims and pit.sizes["lead_time"] > 1:
                # Define edges for per-lead plots even when global histogram is skipped.
                edges = np.linspace(0.0, 1.0, 21)

                # Compute global (all-leads) histogram for NPZ and summary PNG
                pit_all = np.asarray(pit.values).ravel()
                pit_all = pit_all[np.isfinite(pit_all)]
                counts_global, _ = np.histogram(pit_all, bins=edges)
                width_g = np.diff(edges)
                total_g = counts_global.sum()
                density_global = (
                    counts_global / (total_g * width_g.mean()) if total_g > 0 else counts_global
                )

                ens_token_plot = ensemble_mode_to_token("prob")
                if save_npz:
                    out_npz = section / build_output_filename(
                        metric="pit_hist",
                        variable=str(var_name),
                        level=None,
                        qualifier=None,
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=ens_token_plot,
                        ext="npz",
                    )
                    save_data(
                        out_npz,
                        counts=density_global,
                        edges=edges,
                        variable=str(var_name),
                        module="probabilistic",
                    )

                if save_fig:
                    fig_g, ax_g = plt.subplots(figsize=(7, 3), dpi=dpi * 2)
                    ax_g.bar(
                        edges[:-1],
                        density_global,
                        width=width_g,
                        align="edge",
                        color=COLOR_DIAGNOSTIC,
                        edgecolor="white",
                    )
                    ax_g.axhline(1.0, color="brown", linestyle="--", linewidth=1, label="Uniform")
                    ax_g.legend()
                    ax_g.set_title(
                        f"PIT Histogram (all leads) — {format_variable_name(str(var_name))}",
                        loc="left",
                        fontsize=10,
                    )
                    ax_g.set_title(extract_date_from_dataset(ds_target), loc="right", fontsize=10)
                    ax_g.set_xlabel("PIT value")
                    ax_g.set_ylabel("Density")
                    out_png_g = section / build_output_filename(
                        metric="pit_hist",
                        variable=str(var_name),
                        level=None,
                        qualifier=None,
                        init_time_range=init_range_plot,
                        lead_time_range=lead_range_plot,
                        ensemble=ens_token_plot,
                        ext="png",
                    )
                    save_figure(fig_g, out_png_g)
            else:
                # Use dask_histogram from dask_utils (imported as dask_histogram)
                edges = np.linspace(0.0, 1.0, 21)
                counts = dask_histogram(pit, bins=edges)

                # Post-compute density manually since dask_utils.dask_histogram returns only counts
                count_val = counts.compute() if hasattr(counts, "compute") else counts
                total = count_val.sum()
                width = np.diff(edges)
                density = count_val / (total * width.mean()) if total > 0 else count_val

                # For consistency with downstream 'counts' variable usage
                counts = density

                if save_npz:
                    # Use standardized filename builder for NPZ (with ensprob token)
                    ens_token_plot = ensemble_mode_to_token("prob")
                    out_npz = section / build_output_filename(
                        metric="pit_hist",
                        variable=str(var_name),
                        level=None,
                        qualifier=None,
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=ens_token_plot,
                        ext="npz",
                    )
                    save_data(
                        out_npz,
                        counts=counts,
                        edges=edges,
                        variable=str(var_name),
                        module="probabilistic",
                    )

                if save_fig:
                    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi * 2)
                    widths = np.diff(edges)
                    ax.bar(
                        edges[:-1],
                        counts,
                        width=widths,
                        align="edge",
                        color=COLOR_DIAGNOSTIC,
                        edgecolor="white",
                    )
                    # Check for single date
                    date_str = extract_date_from_dataset(ds_target)

                    ax.set_title(
                        f"PIT Histogram — {format_variable_name(str(var_name))}",
                        loc="left",
                        fontsize=10,
                    )
                    ax.set_title(date_str, loc="right", fontsize=10)
                    ax.set_xlabel("PIT value")
                    ax.set_ylabel("Density")
                    ax.axhline(1.0, color="brown", linestyle="--", linewidth=1, label="Uniform")
                    ax.legend()

                    ens_token_plot = ensemble_mode_to_token("prob")
                    out_png = section / build_output_filename(
                        metric="pit_hist",
                        variable=str(var_name),
                        level=None,
                        qualifier=None,
                        init_time_range=init_range_plot,
                        lead_time_range=lead_range_plot,
                        ensemble=ens_token_plot,
                        ext="png",
                    )
                    save_figure(fig, out_png)

        # PIT per-lead panel plot (multi-row grid) over all retained hours
        if (
            pit.size > 0
            and "lead_time" in pit.dims
            and int(pit.sizes.get("lead_time", 0)) > 1
            and (save_fig or save_npz)
        ):
            full_hours = [_to_hour_safe(x, idx) for idx, x in enumerate(pit["lead_time"].values)]
            if full_hours:
                raw_leads = pit["lead_time"].values
                all_hours: list[int] = [_to_hour_safe(x, idx) for idx, x in enumerate(raw_leads)]
                # Debug visibility
                hour_index_pairs = []
                for h in full_hours:
                    try:
                        idx = all_hours.index(int(h))
                        hour_index_pairs.append((int(h), idx))
                    except Exception:
                        continue
                if hour_index_pairs:
                    n = len(hour_index_pairs)
                    ncols = int((plotting_cfg or {}).get("panel_cols", 2))
                    nrows = (n + ncols - 1) // ncols

                    # Compute per-lead densities (needed for both NPZ and plot)
                    lead_hours_arr = np.array([h for h, _ in hour_index_pairs], dtype=float)
                    local_edges = np.linspace(0.0, 1.0, 21)
                    counts_per_lead: list[np.ndarray] = []
                    for _, li in hour_index_pairs:
                        sub = pit.isel(lead_time=li)
                        data = np.asarray(sub.values).ravel()
                        data = data[np.isfinite(data)]
                        c_loc, _ = np.histogram(data, bins=local_edges)
                        width = np.diff(local_edges)
                        total = c_loc.sum()
                        dens = c_loc / (total * width.mean()) if total > 0 else c_loc
                        counts_per_lead.append(dens)

                    ens_token_plot = ensemble_mode_to_token("prob")

                    if save_npz:
                        out_npz_grid = section / build_output_filename(
                            metric="pit_hist",
                            variable=str(var_name),
                            level=None,
                            qualifier="grid",
                            init_time_range=None,
                            lead_time_range=None,
                            ensemble=ens_token_plot,
                            ext="npz",
                        )
                        save_data(
                            out_npz_grid,
                            counts=np.stack(counts_per_lead, axis=0),
                            edges=local_edges,
                            lead_hours=lead_hours_arr,
                            variable=str(var_name),
                            module="probabilistic",
                        )

                    if save_fig:
                        fig, axes = plt.subplots(
                            nrows,
                            ncols,
                            figsize=(5.4 * ncols, 3.0 * nrows),
                            dpi=dpi * 2,
                            squeeze=False,
                            constrained_layout=True,
                        )
                        axes_flat = axes.flatten()
                        width = np.diff(local_edges)
                        for i, ((h, _), dens) in enumerate(
                            zip(hour_index_pairs, counts_per_lead, strict=False)
                        ):
                            r, col = divmod(i, ncols)
                            ax = axes_flat[i]
                            ax.bar(
                                local_edges[:-1],
                                dens,
                                width=width,
                                align="edge",
                                color="#4C78A8",
                                edgecolor="white",
                            )
                            ax.axhline(1.0, color="brown", linestyle="--", linewidth=1)
                            ax.set_title(f"PIT (+{int(h)}h)", fontsize=10)
                            if r == nrows - 1:
                                ax.set_xlabel("PIT value")
                            if col == 0:
                                ax.set_ylabel("Density")

                        # Hide unused axes if n not multiple of ncols
                        for j in range(n, nrows * ncols):
                            if hasattr(axes_flat[j], "axis"):
                                axes_flat[j].axis("off")

                        date_str = extract_date_from_dataset(ds_target)
                        # Adjust layout to leave space for suptitle
                        layout_engine = (
                            fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
                        )
                        if layout_engine is not None and hasattr(layout_engine, "set"):
                            layout_engine.set(rect=[0, 0, 1, 0.92])
                        plt.suptitle(
                            f"PIT Histograms by Lead Time — "
                            f"{format_variable_name(str(var_name))}{date_str}",
                            fontsize=16,
                            y=0.98,
                        )

                        out_png = section / build_output_filename(
                            metric="pit_hist",
                            variable=str(var_name),
                            level=None,
                            qualifier="grid",
                            init_time_range=init_range_plot,
                            lead_time_range=lead_range_plot,
                            ensemble=ens_token_plot,
                            ext="png",
                        )
                        save_figure(fig, out_png)
