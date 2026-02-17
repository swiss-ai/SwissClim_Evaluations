from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .. import console as c
from ..dask_utils import (
    as_float_array,
    compute_jobs,
    dask_histogram,
    resolve_dynamic_batch_size,
    to_finite_array,
)
from ..helpers import (
    COLOR_GROUND_TRUTH,
    COLOR_MODEL_PREDICTION,
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_level_label,
    format_variable_name,
    get_variable_units,
    resolve_ensemble_mode,
    save_data,
    save_figure,
    subsample_values,
)


def _lat_bands() -> tuple[np.ndarray, int, int]:
    lat_bins = np.arange(-90, 91, 10)
    n_bands = len(lat_bins) - 1
    n_rows = n_bands // 2
    return lat_bins, n_bands, n_rows


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    ensemble_mode: str | None = None,
    performance_cfg: dict[str, Any] | None = None,
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    if not save_fig and not save_npz:
        c.print("[histograms] Skipping module: output_mode=none (no PNG/NPZ outputs requested).")
        return
    dpi = int(plotting_cfg.get("dpi", 48))
    # Optional subsampling to avoid loading full arrays.
    # Behavior:
    # - missing key / "auto" => conservative default
    # - null => disable subsampling (advanced; can be memory intensive)
    raw_hist_samples = plotting_cfg.get("histogram_max_samples", "auto")
    max_samples: int | None
    try:
        if isinstance(raw_hist_samples, str) and raw_hist_samples.strip().lower() == "auto":
            max_samples = 200_000
        elif raw_hist_samples is None:
            max_samples = None
        else:
            max_samples = int(raw_hist_samples)
            if max_samples <= 0:
                max_samples = None
    except Exception:
        max_samples = 200_000
    base_seed = int(plotting_cfg.get("random_seed", 42))
    dynamic_batch = resolve_dynamic_batch_size(
        performance_cfg,
        ds=ds_target,
    )

    perf = performance_cfg or {}
    dask_profile = str(perf.get("dask_profile", "safe")).strip().lower()
    if max_samples is None and dask_profile == "safe":
        c.warn(
            "[histograms] histogram_max_samples=null disables subsampling and can be memory "
            "intensive. Consider histogram_max_samples=200000 for safer execution."
        )

    section_output = out_root / "histograms"

    # Config options for 3D handling
    process_3d = bool(plotting_cfg.get("histograms_include_3d", True))
    max_levels = plotting_cfg.get("histograms_max_levels")
    try:
        max_levels = int(max_levels) if max_levels is not None else None
        if max_levels is not None and max_levels <= 0:
            max_levels = None
    except Exception:
        max_levels = None

    per_lat_band = bool(plotting_cfg.get("histograms_per_lat_band", False))

    # Select only genuine 2D variables (no 'level' dimension)
    variables_2d = [v for v in ds_target.data_vars if "level" not in ds_target[v].dims]
    variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
    if not variables_2d and (not process_3d or not variables_3d):
        c.print("[histograms] No eligible variables found – skipping.")
        return
    if variables_2d:
        c.print(f"[histograms] Processing {len(variables_2d)} 2D variables.")
    if process_3d and variables_3d:
        c.print(f"[histograms] Processing {len(variables_3d)} 3D variables (per-level).")
    lat_bins, n_bands, n_rows = _lat_bands()

    # Helper to choose common bin edges without loading full arrays
    def _choose_edges_lazy(da1: xr.DataArray, da2: xr.DataArray):
        if int(getattr(da1, "size", 0) or 0) == 0 or int(getattr(da2, "size", 0) or 0) == 0:
            return None
        both = xr.concat([da1, da2], dim="_t")
        return both.quantile([0.001, 0.999], skipna=True)

    # Resolve ensemble handling
    resolved_mode = resolve_ensemble_mode("histograms", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for histograms")

    def _plot_lat_bands(
        da_target_var: xr.DataArray,
        da_pred_var: xr.DataArray,
        variable_name: str,
        level_token: str,
        qualifier: str,
        ens_token: str | None,
        lead_time_range: tuple[str, str] | None = None,
        level_val: Any = None,
    ):
        if not per_lat_band:
            return
        level_desc = "" if level_val is None else f" level={level_val}"

        i = 0  # retained for seeding when needed (simplified for 3D reuse)
        c.print(f"[histograms] lat_bands: {variable_name}")

        fig, axs = plt.subplots(
            n_rows,
            2,
            figsize=(16, 3 * n_rows),
            dpi=dpi,
            constrained_layout=True,
        )
        # Collect combined NPZ data across all bands
        combined: dict[str, list[np.ndarray | tuple[np.ndarray, np.ndarray] | float]] = {
            "neg_counts": [],
            "neg_bins": [],
            "pos_counts": [],
            "pos_bins": [],
            "neg_lat_min": [],
            "neg_lat_max": [],
            "pos_lat_min": [],
            "pos_lat_max": [],
        }

        # Prepare jobs for batch computation
        jobs = []

        # Negative latitudes (right column)
        for j in range(n_bands // 2):
            lat_max = lat_bins[j]
            lat_min = lat_bins[j + 1]
            da_true = da_target_var.sel(latitude=slice(lat_min, lat_max))
            da_pred = da_pred_var.sel(latitude=slice(lat_min, lat_max))

            job = {
                "type": "neg",
                "j": j,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "da_true": da_true,
                "da_pred": da_pred,
            }

            if max_samples is not None:
                seed = base_seed + (i + 1) * 1000 + (j + 1) * 10 + 1
                job["sub_true_lazy"] = subsample_values(da_true, max_samples, seed, lazy=True)
                job["sub_pred_lazy"] = subsample_values(da_pred, max_samples, seed, lazy=True)
            else:
                job["quantile_lazy"] = _choose_edges_lazy(da_true, da_pred)

            jobs.append(job)

        # Positive latitudes (left column)
        for j in range(n_bands // 2):
            idx = -(j + 1)
            lat_max = lat_bins[idx - 1]
            lat_min = lat_bins[idx]
            da_true = da_target_var.sel(latitude=slice(lat_min, lat_max))
            da_pred = da_pred_var.sel(latitude=slice(lat_min, lat_max))

            job = {
                "type": "pos",
                "j": j,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "da_true": da_true,
                "da_pred": da_pred,
            }

            if max_samples is not None:
                seed = base_seed + (i + 1) * 1000 + (j + 1) * 10 + 2
                job["sub_true_lazy"] = subsample_values(da_true, max_samples, seed, lazy=True)
                job["sub_pred_lazy"] = subsample_values(da_pred, max_samples, seed, lazy=True)
            else:
                job["quantile_lazy"] = _choose_edges_lazy(da_true, da_pred)

            jobs.append(job)

        # Step 1: Compute subsamples or quantiles
        if max_samples is not None:
            compute_jobs(
                jobs,
                key_map={"sub_true_lazy": "sub_true", "sub_pred_lazy": "sub_pred"},
                post_process={"sub_true": to_finite_array, "sub_pred": to_finite_array},
                batch_size=dynamic_batch,
                desc=f"Computing histogram subsamples variable={variable_name}{level_desc}",
            )
            for job in jobs:
                # Calculate edges immediately (fast in memory)
                if job["sub_true"].size == 0 or job["sub_pred"].size == 0:
                    job["edges"] = np.linspace(-1.0, 1.0, 1001)
                else:
                    try:
                        both = np.concatenate([job["sub_true"], job["sub_pred"]])
                        qlow, qhigh = np.quantile(both, [0.001, 0.999])
                        if not np.isfinite(qlow) or not np.isfinite(qhigh) or qlow == qhigh:
                            qlow, qhigh = -1.0, 1.0
                    except Exception:
                        qlow, qhigh = -1.0, 1.0
                    job["edges"] = np.linspace(qlow, qhigh, 1001)
        else:
            compute_jobs(
                jobs,
                key_map={"quantile_lazy": "quantile_res"},
                batch_size=dynamic_batch,
                desc=f"Computing histogram quantiles variable={variable_name}{level_desc}",
            )
            for job in jobs:
                q = job.get("quantile_res")
                if q is not None:
                    vmin = float(q.isel(quantile=0).item())
                    vmax = float(q.isel(quantile=1).item())
                    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                        vmin, vmax = -1.0, 1.0
                else:
                    vmin, vmax = -1.0, 1.0
                job["edges"] = np.linspace(vmin, vmax, 1001)

        # Step 2: Compute histograms
        if max_samples is None:
            for job in jobs:
                job["hist_true_lazy"] = dask_histogram(job["da_true"], job["edges"])
                job["hist_pred_lazy"] = dask_histogram(job["da_pred"], job["edges"])

            compute_jobs(
                jobs,
                key_map={"hist_true_lazy": "counts_ds", "hist_pred_lazy": "counts_prediction"},
                post_process={"counts_ds": as_float_array, "counts_prediction": as_float_array},
                batch_size=dynamic_batch,
                desc=f"Computing histograms variable={variable_name}{level_desc}",
            )
        else:
            for job in jobs:
                # Compute histogram on in-memory subsamples
                counts_ds, _ = np.histogram(job["sub_true"], bins=job["edges"])
                counts_prediction, _ = np.histogram(job["sub_pred"], bins=job["edges"])
                job["counts_ds"] = counts_ds
                job["counts_prediction"] = counts_prediction

        # Plotting
        for job in jobs:
            j = job["j"]
            col = 1 if job["type"] == "neg" else 0
            ax = axs[j, col]

            counts_ds = job["counts_ds"]
            counts_prediction = job["counts_prediction"]
            edges = job["edges"]
            lat_min = job["lat_min"]
            lat_max = job["lat_max"]

            if counts_ds.sum() == 0 or counts_prediction.sum() == 0:
                ax.set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue

            width = np.diff(edges)
            bin_area = counts_ds.sum() * width.mean() if counts_ds.sum() > 0 else 1.0
            counts_ds = counts_ds / bin_area
            bin_area_prediction = (
                counts_prediction.sum() * width.mean() if counts_prediction.sum() > 0 else 1.0
            )
            counts_prediction = counts_prediction / bin_area_prediction

            ax.bar(
                edges[:-1],
                counts_ds,
                width=width,
                align="edge",
                alpha=0.5,
                color=COLOR_GROUND_TRUTH,
                label="Target",
            )
            ax.bar(
                edges[:-1],
                counts_prediction,
                width=width,
                align="edge",
                alpha=0.5,
                color=COLOR_MODEL_PREDICTION,
                label="Prediction",
            )
            ax.set_title(f"Lat {lat_min}° to {lat_max}°")
            ax.legend(loc="upper right")
            units = get_variable_units(ds_target, variable_name)
            if units:
                ax.set_xlabel(f"{format_variable_name(variable_name)} [{units}]")

            if save_npz:
                key_counts = "neg_counts" if job["type"] == "neg" else "pos_counts"
                key_bins = "neg_bins" if job["type"] == "neg" else "pos_bins"
                key_lat_min = "neg_lat_min" if job["type"] == "neg" else "pos_lat_min"
                key_lat_max = "neg_lat_max" if job["type"] == "neg" else "pos_lat_max"

                combined[key_counts].append((counts_ds, counts_prediction))
                combined[key_bins].append(edges)
                combined[key_lat_min].append(float(lat_min))
                combined[key_lat_max].append(float(lat_max))

        units = get_variable_units(ds_target, variable_name)

        # Check for single date
        date_str = extract_date_from_dataset(da_target_var)
        level_str = format_level_label(level_val if level_val is not None else level_token)

        plt.suptitle(
            f"Distributions by Latitude Bands — {format_variable_name(variable_name)}"
            f"{level_str}{date_str}",
            y=1.02,
        )

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png = section_output / build_output_filename(
                metric="hist",
                variable=variable_name,
                level=level_token,
                qualifier=qualifier,
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token,
                ext="png",
            )
            save_figure(fig, out_png, module="histograms")
        else:
            plt.close(fig)

        if save_npz:
            out_npz = section_output / build_output_filename(
                metric="hist",
                variable=variable_name,
                level=level_token,
                qualifier=qualifier,
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token,
                ext="npz",
            )
            save_data(
                out_npz,
                neg_counts=np.array(combined["neg_counts"], dtype=object),
                neg_bins=np.array(combined["neg_bins"], dtype=object),
                pos_counts=np.array(combined["pos_counts"], dtype=object),
                pos_bins=np.array(combined["pos_bins"], dtype=object),
                neg_lat_min=np.array(combined["neg_lat_min"]),
                neg_lat_max=np.array(combined["neg_lat_max"]),
                pos_lat_min=np.array(combined["pos_lat_min"]),
                pos_lat_max=np.array(combined["pos_lat_max"]),
                units=units,
                allow_pickle=True,
                module="histograms",
            )

    def _plot_global_hist(
        da_target_var: xr.DataArray,
        da_pred_var: xr.DataArray,
        variable_name: str,
        level_token: str,
        ens_token: str | None,
        lead_time_range: tuple[str, str] | None,
        level_val: int | float | None = None,
    ) -> None:
        """Plot a single global histogram comparing target vs prediction.

        - Uses paired subsampling if enabled
        - Chooses common bin edges from combined quantiles
        - Saves PNG and NPZ with counts/edges
        """

        def _choose_edges_arr(arr1: np.ndarray, arr2: np.ndarray, bins: int = 1000):
            if arr1.size == 0 or arr2.size == 0:
                return np.linspace(-1.0, 1.0, bins + 1)
            both = np.concatenate([arr1, arr2])
            try:
                qlow, qhigh = np.quantile(both, [0.001, 0.999])
                if not np.isfinite(qlow) or not np.isfinite(qhigh) or qlow == qhigh:
                    qlow, qhigh = -1.0, 1.0
            except Exception:
                qlow, qhigh = -1.0, 1.0
            return np.linspace(qlow, qhigh, bins + 1)

        # Get paired samples or full arrays
        if max_samples is not None:
            a_true = subsample_values(da_target_var, max_samples, base_seed + 9001)
            a_pred = subsample_values(da_pred_var, max_samples, base_seed + 9001)
        else:
            a_true = np.asarray(da_target_var.compute().values).ravel()
            a_pred = np.asarray(da_pred_var.compute().values).ravel()
            a_true = a_true[np.isfinite(a_true)]
            a_pred = a_pred[np.isfinite(a_pred)]
        edges = _choose_edges_arr(a_true, a_pred, bins=400)
        # Compute histograms
        counts_true, _ = np.histogram(a_true, bins=edges)
        counts_pred, _ = np.histogram(a_pred, bins=edges)
        width = np.diff(edges)
        area_t = counts_true.sum() * width.mean() if counts_true.sum() > 0 else 1.0
        area_p = counts_pred.sum() * width.mean() if counts_pred.sum() > 0 else 1.0
        dens_true = counts_true / area_t
        dens_pred = counts_pred / area_p

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi * 2)
        ax.bar(
            edges[:-1],
            dens_true,
            width=width,
            align="edge",
            alpha=0.5,
            color=COLOR_GROUND_TRUTH,
            label="Target",
        )
        ax.bar(
            edges[:-1],
            dens_pred,
            width=width,
            align="edge",
            alpha=0.5,
            color=COLOR_MODEL_PREDICTION,
            label="Prediction",
        )
        units = get_variable_units(da_target_var, variable_name)
        title_lt = f" — lead {lead_time_range[0]}" if lead_time_range else ""
        level_str = format_level_label(level_val if level_val is not None else level_token)
        ax.set_title(
            f"{format_variable_name(variable_name)} — Global Histogram{level_str}{title_lt}"
        )
        if units:
            ax.set_xlabel(f"{format_variable_name(variable_name)} [{units}]")
        else:
            ax.set_xlabel(f"{format_variable_name(variable_name)}")
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.legend()

        if save_fig:
            out_png = section_output / build_output_filename(
                metric="hist",
                variable=variable_name,
                level=level_token,
                qualifier="global",
                init_time_range=None,
                lead_time_range=lead_time_range,
                ensemble=ens_token,
                ext="png",
            )
            save_figure(fig, out_png, module="histograms")
        else:
            plt.close(fig)

        if save_npz:
            out_npz = section_output / build_output_filename(
                metric="hist",
                variable=variable_name,
                level=level_token,
                qualifier="global_data",
                init_time_range=None,
                lead_time_range=lead_time_range,
                ensemble=ens_token,
                ext="npz",
            )
            save_data(
                out_npz,
                counts_target=dens_true,
                counts_prediction=dens_pred,
                edges=edges,
                module="histograms",
            )

    def _plot_global_hist_gridded(
        da_target_var: xr.DataArray,
        da_pred_var: xr.DataArray,
        variable_name: str,
        level_token: str,
        ens_token: str | None,
        lead_time_range: tuple[str, str] | None,
        level_val: Any = None,
    ) -> None:
        """Plot global histograms gridded by lead_time (one subplot per lead)."""
        if "lead_time" not in da_pred_var.dims:
            return

        leads = da_pred_var["lead_time"].values
        n_leads = len(leads)
        if n_leads < 2:
            return

        # Determine grid layout
        cols = min(4, n_leads)
        rows = int(np.ceil(n_leads / cols))

        fig, axs = plt.subplots(
            rows, cols, figsize=(4 * cols, 3 * rows), dpi=dpi, constrained_layout=True
        )
        axs = np.atleast_1d(axs).flatten()

        # Compute global bin edges once to keep x-axis consistent
        # We use a subsample of the full data to determine edges
        jobs = []

        # Edge determination job
        edge_job: dict[str, Any] = {"type": "edges"}
        if max_samples:
            edge_job["sub_t_lazy"] = subsample_values(
                da_target_var, max_samples // n_leads, base_seed, lazy=True
            )
            edge_job["sub_p_lazy"] = subsample_values(
                da_pred_var, max_samples // n_leads, base_seed, lazy=True
            )
        else:
            edge_job["sub_t_lazy"] = subsample_values(da_target_var, None, base_seed, lazy=True)
            edge_job["sub_p_lazy"] = subsample_values(da_pred_var, None, base_seed, lazy=True)
        jobs.append(edge_job)

        for i, lt in enumerate(leads):
            # Select lead time
            # Handle timedelta vs int lead times
            if np.issubdtype(np.asarray(leads).dtype, np.timedelta64):
                lt_sel = lt
                hours = int(lt / np.timedelta64(1, "h"))
                label = f"{hours}h"
            else:
                lt_sel = lt
                label = str(lt)

            if "lead_time" in da_target_var.dims:
                da_t = da_target_var.sel(lead_time=lt_sel)
            else:
                da_t = da_target_var
            da_p = da_pred_var.sel(lead_time=lt_sel)

            # Subsample per lead
            k = max_samples // n_leads if max_samples else None
            # Use different seed per lead to avoid correlation artifacts if random
            seed = base_seed + i

            job = {
                "type": "lead",
                "i": i,
                "label": label,
                "da_t": da_t,
                "da_p": da_p,
            }
            job["sub_t_lazy"] = subsample_values(da_t, k, seed, lazy=True)
            job["sub_p_lazy"] = subsample_values(da_p, k, seed, lazy=True)
            jobs.append(job)

        # Compute all
        compute_jobs(
            jobs,
            key_map={"sub_t_lazy": "val_t", "sub_p_lazy": "val_p"},
            post_process={"val_t": to_finite_array, "val_p": to_finite_array},
            batch_size=dynamic_batch,
            desc=(
                f"Computing global histograms variable={variable_name}"
                if level_val is None
                else f"Computing global histograms variable={variable_name} level={level_val}"
            ),
        )

        # Calculate edges from edge_job
        edge_job = jobs[0]
        val_t_edge = cast(np.ndarray, edge_job["val_t"])
        val_p_edge = cast(np.ndarray, edge_job["val_p"])
        if val_t_edge.size == 0 or val_p_edge.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            combined = np.concatenate([val_t_edge, val_p_edge])
            vmin = float(np.nanquantile(combined, 0.001))
            vmax = float(np.nanquantile(combined, 0.999))

        edges = np.linspace(vmin, vmax, 100 + 1)

        for job in jobs[1:]:
            i = cast(int, job["i"])
            label = cast(str, job["label"])
            val_t = cast(np.ndarray, job["val_t"])
            val_p = cast(np.ndarray, job["val_p"])
            ax = axs[i]

            if val_t.size > 0:
                ax.hist(
                    val_t,
                    bins=edges,
                    density=True,
                    alpha=0.5,
                    color=COLOR_GROUND_TRUTH,
                    label="Target",
                )
            if val_p.size > 0:
                ax.hist(
                    val_p,
                    bins=edges,
                    density=True,
                    alpha=0.5,
                    color=COLOR_MODEL_PREDICTION,
                    label="Model",
                )

            ax.set_title(f"Lead: {label}", fontsize=14)
            if units := da_target_var.attrs.get("units"):
                ax.set_xlabel(str(units))
            if i == 0:
                ax.legend()

        # Hide unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.suptitle(
            f"Histograms by Lead Time — {format_variable_name(variable_name)}", fontsize=20
        )

        if save_fig:
            out_png = section_output / build_output_filename(
                metric="hist_by_lead",
                variable=variable_name,
                level=level_token,
                qualifier=None,
                init_time_range=None,
                lead_time_range=lead_time_range,
                ensemble=ens_token,
                ext="png",
            )
            save_figure(fig, out_png, module="histograms")
        else:
            plt.close(fig)

    # 2D variables
    def _iter_members():
        if not has_ens:
            yield None, ds_target, ds_prediction
        else:
            for i in range(int(ds_prediction.sizes["ensemble"])):
                if "ensemble" in ds_target.dims:
                    if ds_target.sizes["ensemble"] == 1:
                        tgt_m = ds_target.isel(ensemble=0)
                    else:
                        tgt_m = ds_target.isel(ensemble=i)
                else:
                    tgt_m = ds_target
                pred_m = ds_prediction.isel(ensemble=i)
                yield i, tgt_m, pred_m

    if resolved_mode == "members" and not has_ens:
        resolved_mode = "pooled"  # degrade silently

    if resolved_mode == "mean" and has_ens:
        ds_prediction_mean = ds_prediction.mean(dim="ensemble")
        ds_target_mean = (
            ds_target.mean(dim="ensemble") if "ensemble" in ds_target.dims else ds_target
        )
        # Determine per-lead behavior
        per_lead = bool(plotting_cfg.get("histograms_per_lead", True)) and (
            "lead_time" in ds_prediction_mean.dims and int(ds_prediction_mean.lead_time.size) > 1
        )
        if not per_lead:
            for variable_name in variables_2d:
                _plot_global_hist(
                    ds_target_mean[variable_name],
                    ds_prediction_mean[variable_name],
                    str(variable_name),
                    level_token="",
                    ens_token=ensemble_mode_to_token("mean"),
                    lead_time_range=None,
                )
                _plot_lat_bands(
                    ds_target_mean[variable_name],
                    ds_prediction_mean[variable_name],
                    str(variable_name),
                    level_token="",
                    qualifier="latbands",
                    ens_token=ensemble_mode_to_token("mean"),
                    lead_time_range=None,
                )
        # Optional: combined global histogram grid across selected leads
        # Force grid generation whenever lead_time is present (>=1)
        do_grid = ("lead_time" in ds_prediction_mean.dims) and int(
            ds_prediction_mean.lead_time.size
        ) >= 1
        if do_grid:
            for variable_name in variables_2d:
                _plot_global_hist_gridded(
                    ds_target_mean[variable_name],
                    ds_prediction_mean[variable_name],
                    str(variable_name),
                    level_token="",
                    ens_token=ensemble_mode_to_token("mean"),
                    lead_time_range=None,
                )
                if save_npz:
                    # Persist panel data (edges, densities) for table recreation
                    pass

    elif resolved_mode == "members" and has_ens:
        for member_index, tgt_m, pred_m in _iter_members():
            token = ensemble_mode_to_token("members", member_index)
            per_lead = bool(plotting_cfg.get("histograms_per_lead", True)) and (
                "lead_time" in pred_m.dims and int(pred_m.lead_time.size) > 1
            )
            # removed unused lead_indices (ruff F841)
            # Suppress member per-lead globals; only grid
            if not per_lead:
                for variable_name in variables_2d:
                    _plot_global_hist(
                        tgt_m[variable_name],
                        pred_m[variable_name],
                        str(variable_name),
                        level_token="surface",
                        ens_token=token,
                        lead_time_range=None,
                    )
                    _plot_lat_bands(
                        tgt_m[variable_name],
                        pred_m[variable_name],
                        str(variable_name),
                        level_token="surface",
                        qualifier="latbands",
                        ens_token=token,
                        lead_time_range=None,
                    )
            # Optional global grid per member
            # Force grid generation for members mode whenever lead_time present (>=1)
            do_grid = ("lead_time" in pred_m.dims) and int(pred_m.lead_time.size) >= 1
            if do_grid:
                for variable_name in variables_2d:
                    _plot_global_hist_gridded(
                        tgt_m[variable_name],
                        pred_m[variable_name],
                        str(variable_name),
                        level_token="surface",
                        ens_token=token,
                        lead_time_range=None,
                    )
    else:  # pooled or none
        token = (
            ensemble_mode_to_token("pooled") if (resolved_mode == "pooled" and has_ens) else None
        )
        per_lead = bool(plotting_cfg.get("histograms_per_lead", True)) and (
            "lead_time" in ds_prediction.dims and int(ds_prediction.lead_time.size) > 1
        )
        # removed unused variable lead_indices (ruff F841)
        if not per_lead:
            for variable_name in variables_2d:
                _plot_global_hist(
                    ds_target[variable_name],
                    ds_prediction[variable_name],
                    str(variable_name),
                    level_token="surface",
                    ens_token=token,
                    lead_time_range=None,
                )
                _plot_lat_bands(
                    ds_target[variable_name],
                    ds_prediction[variable_name],
                    str(variable_name),
                    level_token="surface",
                    qualifier="latbands",
                    ens_token=token,
                    lead_time_range=None,
                )
        # Optional combined grid when lead_time present
        # Force grid generation for pooled/none modes whenever lead_time present (>=1)
        do_grid = ("lead_time" in ds_prediction.dims) and int(ds_prediction.lead_time.size) >= 1
        if do_grid:
            all_hours = []
            try:
                vals = ds_prediction["lead_time"].values
                all_hours = [int(v / np.timedelta64(1, "h")) for v in vals]
            except Exception:
                all_hours = list(range(int(ds_prediction.lead_time.size)))
            panel_hours = all_hours
            for variable_name in variables_2d:
                n_panels = len(panel_hours)
                if n_panels == 0:
                    continue
                ncols = 2
                nrows = (n_panels + ncols - 1) // ncols
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(12, max(2.5, 2.2 * nrows)),
                    dpi=dpi * 2,
                    constrained_layout=True,
                )
                axes = np.atleast_1d(axes).ravel()
                all_edges_min = []
                all_edges_max = []
                all_y_max = []
                panel_results = []
                for i, h in enumerate(panel_hours):
                    try:
                        idx = all_hours.index(int(h))
                    except Exception:
                        idx = i
                    da_t = ds_target[variable_name]
                    da_p = ds_prediction[variable_name]
                    if "lead_time" in da_t.dims:
                        da_t = da_t.isel(lead_time=idx, drop=True)
                    if "lead_time" in da_p.dims:
                        da_p = da_p.isel(lead_time=idx, drop=True)
                    if max_samples is not None:
                        a_true = cast(
                            np.ndarray, subsample_values(da_t, max_samples, base_seed + 9001 + i)
                        )
                        a_pred = cast(
                            np.ndarray, subsample_values(da_p, max_samples, base_seed + 9001 + i)
                        )
                    else:
                        a_true = np.asarray(da_t.compute().values).ravel()
                        a_pred = np.asarray(da_p.compute().values).ravel()
                        a_true = a_true[np.isfinite(a_true)]
                        a_pred = a_pred[np.isfinite(a_pred)]
                    both = (
                        np.concatenate([a_true, a_pred])
                        if (a_true.size and a_pred.size)
                        else np.array([-1.0, 1.0])
                    )
                    try:
                        qlow, qhigh = np.quantile(both, [0.001, 0.999])
                        if not np.isfinite(qlow) or not np.isfinite(qhigh) or qlow == qhigh:
                            qlow, qhigh = -1.0, 1.0
                    except Exception:
                        qlow, qhigh = -1.0, 1.0
                    edges = np.linspace(qlow, qhigh, 400)
                    ct, _ = np.histogram(a_true, bins=edges)
                    cp, _ = np.histogram(a_pred, bins=edges)
                    width = np.diff(edges)
                    area_t = ct.sum() * width.mean() if ct.sum() > 0 else 1.0
                    area_p = cp.sum() * width.mean() if cp.sum() > 0 else 1.0
                    dt = ct / area_t
                    dp = cp / area_p
                    panel_results.append((edges, dt, dp, h))
                    all_edges_min.append(edges[0])
                    all_edges_max.append(edges[-1])
                    ymax_i = float(
                        max(np.nanmax(dt) if dt.size else 0.0, np.nanmax(dp) if dp.size else 0.0)
                    )
                    all_y_max.append(ymax_i)
                x_min = float(min(all_edges_min)) if all_edges_min else -1.0
                x_max = float(max(all_edges_max)) if all_edges_max else 1.0
                y_max = float(max(all_y_max)) if all_y_max else 1.0
                for i, (edges, dt, dp, h) in enumerate(panel_results):
                    ax = axes[i]
                    width = np.diff(edges)
                    ax.bar(
                        edges[:-1],
                        dt,
                        width=width,
                        align="edge",
                        alpha=0.5,
                        color=COLOR_GROUND_TRUTH,
                        label="Target" if i == 0 else None,
                    )
                    ax.bar(
                        edges[:-1],
                        dp,
                        width=width,
                        align="edge",
                        alpha=0.5,
                        color=COLOR_MODEL_PREDICTION,
                        label="Prediction" if i == 0 else None,
                    )
                    ax.set_title(f"Lead {int(h)}h", fontsize=11)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(0.0, y_max)
                    if i % ncols != 0:
                        ax.set_ylabel("")
                        ax.tick_params(axis="y", labelleft=False)
                    else:
                        ax.set_ylabel("Density")
                    if i >= (nrows - 1) * ncols:
                        ax.set_xlabel(variable_name)
                    else:
                        ax.tick_params(axis="x", labelbottom=False)
                axes[0].legend(loc="upper right")
                for j in range(n_panels, nrows * ncols):
                    axes[j].axis("off")
                if save_fig:
                    section_output.mkdir(parents=True, exist_ok=True)
                    out_png = section_output / build_output_filename(
                        metric="hist_global",
                        variable=str(variable_name),
                        level="surface",
                        qualifier="grid",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=token,
                        ext="png",
                    )
                    save_figure(fig, out_png, module="histograms")
                else:
                    plt.close(fig)

                if save_npz:
                    out_npz = section_output / build_output_filename(
                        metric="hist_global",
                        variable=str(variable_name),
                        level="surface",
                        qualifier="grid_data",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=token,
                        ext="npz",
                    )
                    lead_hours = np.array(panel_hours, dtype=float)
                    dens_true_list = [pr[1] for pr in panel_results]
                    dens_pred_list = [pr[2] for pr in panel_results]
                    edges_list = [pr[0] for pr in panel_results]
                    save_data(
                        out_npz,
                        lead_hours=lead_hours,
                        densities_true=np.array(dens_true_list, dtype=object),
                        densities_pred=np.array(dens_pred_list, dtype=object),
                        edges=np.array(edges_list, dtype=object),
                        allow_pickle=True,
                        module="histograms",
                    )

    # 3D variables per level
    if process_3d:
        for variable_name in variables_3d:
            da_t = ds_target[variable_name]
            da_p = ds_prediction[variable_name]
            levels = list(da_t["level"].values)
            if max_levels is not None:
                levels = levels[:max_levels]
            for lvl in levels:
                # Select single level slice (dropping level dimension for logic reuse)
                da_t_lvl = da_t.sel(level=lvl)
                da_p_lvl = da_p.sel(level=lvl)
                lvl_clean = str(lvl).replace(".", "_")
                if resolved_mode == "mean" and has_ens:
                    da_p_lvl_mean = da_p_lvl.mean(dim="ensemble")
                    da_t_lvl_mean = (
                        da_t_lvl.mean(dim="ensemble") if "ensemble" in da_t_lvl.dims else da_t_lvl
                    )
                    _plot_global_hist(
                        da_t_lvl_mean,
                        da_p_lvl_mean,
                        str(variable_name),
                        level_token=str(lvl_clean),
                        ens_token=ensemble_mode_to_token("mean"),
                        lead_time_range=None,
                        level_val=lvl,
                    )
                    _plot_lat_bands(
                        da_t_lvl_mean,
                        da_p_lvl_mean,
                        str(variable_name),
                        level_token=str(lvl_clean),
                        qualifier="latbands",
                        ens_token=ensemble_mode_to_token("mean"),
                        lead_time_range=None,
                        level_val=lvl,
                    )
                elif resolved_mode == "members" and has_ens:
                    for member_index, tgt_m, pred_m in _iter_members():
                        token = ensemble_mode_to_token("members", member_index)
                        _plot_global_hist(
                            (
                                tgt_m[variable_name].sel(level=lvl)
                                if "ensemble" in tgt_m[variable_name].dims
                                else tgt_m.sel(level=lvl)
                            ),
                            pred_m[variable_name].sel(level=lvl),
                            str(variable_name),
                            level_token=str(lvl_clean),
                            ens_token=token,
                            lead_time_range=None,
                            level_val=lvl,
                        )
                        _plot_lat_bands(
                            (
                                tgt_m[variable_name].sel(level=lvl)
                                if "ensemble" in tgt_m[variable_name].dims
                                else tgt_m.sel(level=lvl)
                            ),
                            pred_m[variable_name].sel(level=lvl),
                            str(variable_name),
                            level_token=str(lvl_clean),
                            qualifier="latbands",
                            ens_token=token,
                            level_val=lvl,
                        )
                else:  # pooled/none
                    token = (
                        ensemble_mode_to_token("pooled")
                        if (resolved_mode == "pooled" and has_ens)
                        else None
                    )
                    _plot_global_hist(
                        da_t_lvl,
                        da_p_lvl,
                        str(variable_name),
                        level_token=str(lvl_clean),
                        ens_token=token,
                        lead_time_range=None,
                        level_val=lvl,
                    )
                    _plot_lat_bands(
                        da_t_lvl,
                        da_p_lvl,
                        str(variable_name),
                        level_token=str(lvl_clean),
                        qualifier="latbands",
                        ens_token=token,
                        lead_time_range=None,
                        level_val=lvl,
                    )
