from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as dsa
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..helpers import build_output_filename, ensemble_mode_to_token, resolve_ensemble_mode


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
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    # Optional subsampling  to avoid loading full arrays.
    max_samples = plotting_cfg.get("histogram_max_samples")
    try:
        max_samples = int(max_samples) if max_samples is not None else None
        if max_samples is not None and max_samples <= 0:
            max_samples = None
    except Exception:
        max_samples = None
    base_seed = int(plotting_cfg.get("random_seed", 42))
    # Always use identical subsamples for target/prediction (paired subsampling enforced)
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

    # Select only genuine 2D variables (no 'level' dimension)
    variables_2d = [v for v in ds_target.data_vars if "level" not in ds_target[v].dims]
    variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
    if not variables_2d and (not process_3d or not variables_3d):
        print("[histograms] No eligible variables found – skipping.")
        return
    if variables_2d:
        print(f"[histograms] Processing {len(variables_2d)} 2D variables.")
    if process_3d and variables_3d:
        print(f"[histograms] Processing {len(variables_3d)} 3D variables (per-level).")
    lat_bins, n_bands, n_rows = _lat_bands()

    # Run-scope helper for paired subsampling, visible to all inner blocks
    def _subsample_values(da: xr.DataArray, k: int | None, seed: int) -> np.ndarray:
        if k is None:
            arr = np.asarray(da.compute().values).ravel()
            return arr[np.isfinite(arr)]
        size = int(getattr(da, "size", 0) or 0)
        if size == 0:
            return np.array([], dtype=float)
        if size <= k:
            arr = np.asarray(da.compute().values).ravel()
            return arr[np.isfinite(arr)]
        dims = list(da.dims)
        nd = max(1, len(dims))
        frac = (k / float(size)) ** (1.0 / nd)
        rng = np.random.default_rng(seed)
        indexers: dict[str, Any] = {}
        for d in dims:
            n = int(da.sizes.get(str(d), 1))
            take = max(1, int(np.ceil(frac * n)))
            take = min(take, n)
            idx = rng.choice(n, size=take, replace=False)
            idx.sort()
            indexers[str(d)] = np.asarray(idx)
        sub = da.isel(indexers)
        arr = np.asarray(sub.compute().values).ravel()
        return arr[np.isfinite(arr)]

    # Resolve ensemble handling
    resolved_mode = resolve_ensemble_mode("histograms", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "none" and has_ens:
        raise ValueError(
            "ensemble_mode=none requested but 'ensemble' dimension present; "
            "choose mean|pooled|members"
        )
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for histograms")

    def _plot_variable(
        da_target_var: xr.DataArray,
        da_pred_var: xr.DataArray,
        variable_name: str,
        level_token: str,
        qualifier: str,
        ens_token: str | None,
        lead_time_range: tuple[str, str] | None,
    ) -> None:
        i = 0  # retained for seeding when needed (simplified for 3D reuse)
        print(f"[histograms] variable: {variable_name}")

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

        # Helper to choose common bin edges without loading full arrays
        def _choose_edges(da1: xr.DataArray, da2: xr.DataArray, bins: int = 1000):
            # Fast guard: empty selections → default symmetric range
            if int(getattr(da1, "size", 0) or 0) == 0 or int(getattr(da2, "size", 0) or 0) == 0:
                vmin, vmax = -1.0, 1.0
            else:
                both = xr.concat([da1, da2], dim="_t")
                q = both.quantile([0.001, 0.999], skipna=True).compute()
                vmin = float(q.isel(quantile=0).item())
                vmax = float(q.isel(quantile=1).item())
                if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                    vmin, vmax = -1.0, 1.0
            return np.linspace(vmin, vmax, bins + 1)

        def _dask_hist(da: xr.DataArray, edges: np.ndarray):
            data = getattr(da, "data", da)
            darr = dsa.asarray(data)
            darr = darr.ravel()
            darr = darr[~dsa.isnan(darr)]
            counts = dsa.histogram(darr, bins=np.asarray(edges))[0]
            return counts

        # Helper subsampling function
        def _subsample_values(da: xr.DataArray, k: int, seed: int) -> np.ndarray:
            if k is None:
                arr = np.asarray(da.compute().values).ravel()
                return arr[np.isfinite(arr)]
            size = int(getattr(da, "size", 0) or 0)
            if size == 0:
                return np.array([], dtype=float)
            if size <= k:
                arr = np.asarray(da.compute().values).ravel()
                return arr[np.isfinite(arr)]
            dims = list(da.dims)
            nd = max(1, len(dims))
            frac = (k / float(size)) ** (1.0 / nd)
            rng = np.random.default_rng(seed)
            indexers: dict[str, Any] = {}
            for d in dims:
                n = int(da.sizes.get(str(d), 1))
                take = max(1, int(np.ceil(frac * n)))
                take = min(take, n)
                idx = rng.choice(n, size=take, replace=False)
                idx.sort()
                indexers[str(d)] = np.asarray(idx)
            sub = da.isel(indexers)
            arr = np.asarray(sub.compute().values).ravel()
            return arr[np.isfinite(arr)]

        # Track global x/y limits across all latitude bands (both hemispheres)
        global_x_min: float | None = None
        global_x_max: float | None = None
        global_y_max: float = 0.0

        # Negative latitudes (right column)
        for j in range(n_bands // 2):
            lat_max = lat_bins[j]
            lat_min = lat_bins[j + 1]
            da_true = da_target_var.sel(latitude=slice(lat_min, lat_max))
            da_pred = da_pred_var.sel(latitude=slice(lat_min, lat_max))
            # If subsampling enabled, compute edges on subsampled arrays
            if max_samples is not None:
                seed = base_seed + (i + 1) * 1000 + (j + 1) * 10 + 1
                ds_sample = _subsample_values(da_true, max_samples, seed)
                ml_sample = _subsample_values(da_pred, max_samples, seed)
                if ds_sample.size == 0 or ml_sample.size == 0:
                    axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                    continue
                both = np.concatenate([ds_sample, ml_sample])
                if both.size <= 1:
                    qlow, qhigh = -1.0, 1.0
                else:
                    qlow, qhigh = np.quantile(both, [0.001, 0.999])
                    if (not np.isfinite(qlow)) or (not np.isfinite(qhigh)) or (qlow == qhigh):
                        qlow, qhigh = -1.0, 1.0
                edges = np.linspace(qlow, qhigh, 1001)
                dsa_ds = dsa.from_array(ds_sample, chunks=ds_sample.shape[0] // 4 or 1)
                dsa_ml = dsa.from_array(ml_sample, chunks=ml_sample.shape[0] // 4 or 1)
                counts_ds = dsa.histogram(dsa_ds, bins=edges)[0].compute()
                counts_ml = dsa.histogram(dsa_ml, bins=edges)[0].compute()
            else:
                # Use dask-based histogram over explicit edges (full arrays)
                edges = _choose_edges(da_true, da_pred, bins=1000)
                counts_ds = _dask_hist(da_true, edges).compute().astype(float)
                counts_ml = _dask_hist(da_pred, edges).compute().astype(float)
            # Convert to density
            width = np.diff(edges)
            bin_area = counts_ds.sum() * width.mean() if counts_ds.sum() > 0 else 1.0
            counts_ds = counts_ds / bin_area
            bin_area_ml = counts_ml.sum() * width.mean() if counts_ml.sum() > 0 else 1.0
            counts_ml = counts_ml / bin_area_ml
            axs[j, 1].bar(
                edges[:-1],
                counts_ds,
                width=width,
                align="edge",
                alpha=0.5,
                color="skyblue",
                label="Ground Truth",
            )
            axs[j, 1].bar(
                edges[:-1],
                counts_ml,
                width=width,
                align="edge",
                alpha=0.5,
                color="salmon",
                label="Model Prediction",
            )
            axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}°")
            # Update global limits
            if global_x_min is None or float(edges[0]) < global_x_min:
                global_x_min = float(edges[0])
            if global_x_max is None or float(edges[-1]) > global_x_max:
                global_x_max = float(edges[-1])
            local_y_max = float(
                max(
                    np.nanmax(counts_ds) if counts_ds.size else 0.0,
                    np.nanmax(counts_ml) if counts_ml.size else 0.0,
                )
            )
            global_y_max = max(global_y_max, local_y_max)
            axs[j, 1].legend(loc="upper right")
            if save_npz:
                combined["neg_counts"].append((counts_ds, counts_ml))
                combined["neg_bins"].append(edges)
                combined["neg_lat_min"].append(float(lat_min))
                combined["neg_lat_max"].append(float(lat_max))

        # Positive latitudes (left column)
        for j in range(n_bands // 2):
            idx = -(j + 1)
            lat_max = lat_bins[idx - 1]
            lat_min = lat_bins[idx]
            da_true = da_target_var.sel(latitude=slice(lat_min, lat_max))
            da_pred = da_pred_var.sel(latitude=slice(lat_min, lat_max))
            if max_samples is not None:
                seed = base_seed + (i + 1) * 1000 + (j + 1) * 10 + 2
                ds_sample = _subsample_values(da_true, max_samples, seed)
                ml_sample = _subsample_values(da_pred, max_samples, seed)
                if ds_sample.size == 0 or ml_sample.size == 0:
                    axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                    continue
                both = np.concatenate([ds_sample, ml_sample])
                if both.size <= 1:
                    qlow, qhigh = -1.0, 1.0
                else:
                    qlow, qhigh = np.quantile(both, [0.001, 0.999])
                    if (not np.isfinite(qlow)) or (not np.isfinite(qhigh)) or (qlow == qhigh):
                        qlow, qhigh = -1.0, 1.0
                edges = np.linspace(qlow, qhigh, 1001)
                dsa_ds = dsa.from_array(ds_sample, chunks=ds_sample.shape[0] // 4 or 1)
                dsa_ml = dsa.from_array(ml_sample, chunks=ml_sample.shape[0] // 4 or 1)
                counts_ds = dsa.histogram(dsa_ds, bins=edges)[0].compute()
                counts_ml = dsa.histogram(dsa_ml, bins=edges)[0].compute()
            else:
                edges = _choose_edges(da_true, da_pred, bins=1000)
                counts_ds = _dask_hist(da_true, edges).compute().astype(float)
                counts_ml = _dask_hist(da_pred, edges).compute().astype(float)
            width = np.diff(edges)
            bin_area = counts_ds.sum() * width.mean() if counts_ds.sum() > 0 else 1.0
            counts_ds = counts_ds / bin_area
            bin_area_ml = counts_ml.sum() * width.mean() if counts_ml.sum() > 0 else 1.0
            counts_ml = counts_ml / bin_area_ml
            axs[j, 0].bar(
                edges[:-1],
                counts_ds,
                width=width,
                align="edge",
                alpha=0.5,
                color="skyblue",
                label="Ground Truth",
            )
            axs[j, 0].bar(
                edges[:-1],
                counts_ml,
                width=width,
                align="edge",
                alpha=0.5,
                color="salmon",
                label="Model Prediction",
            )
            axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}°")
            # Update global limits
            if global_x_min is None or float(edges[0]) < global_x_min:
                global_x_min = float(edges[0])
            if global_x_max is None or float(edges[-1]) > global_x_max:
                global_x_max = float(edges[-1])
            local_y_max = float(
                max(
                    np.nanmax(counts_ds) if counts_ds.size else 0.0,
                    np.nanmax(counts_ml) if counts_ml.size else 0.0,
                )
            )
            global_y_max = max(global_y_max, local_y_max)
            axs[j, 0].legend(loc="upper right")
            if save_npz:
                combined["pos_counts"].append((counts_ds, counts_ml))
                combined["pos_bins"].append(edges)
                combined["pos_lat_min"].append(float(lat_min))
                combined["pos_lat_max"].append(float(lat_max))

        # Apply unified axes after plotting (post-update of global limits)
        if global_x_min is not None and global_x_max is not None:
            for j in range(n_bands // 2):
                axs[j, 1].set_xlim(global_x_min, global_x_max)
                axs[j, 0].set_xlim(global_x_min, global_x_max)
                axs[j, 1].set_ylim(0.0, global_y_max * 1.05 if global_y_max > 0 else 1.0)
                axs[j, 0].set_ylim(0.0, global_y_max * 1.05 if global_y_max > 0 else 1.0)

        units = da_target_var.attrs.get("units", "")
        plt.suptitle(
            f"Distribution of {variable_name} ({units}) by latitude bands",
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
                lead_time_range=lead_time_range,
                ensemble=ens_token,
                ext="png",
            )
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"[histograms] saved {out_png}")
        if save_npz:
            section_output.mkdir(parents=True, exist_ok=True)
            out_npz = section_output / build_output_filename(
                metric="hist",
                variable=variable_name,
                level=level_token,
                qualifier=f"{qualifier}_combined",
                init_time_range=None,
                lead_time_range=lead_time_range,
                ensemble=ens_token,
                ext="npz",
            )
            np.savez(
                out_npz,
                neg_counts=np.array(combined["neg_counts"], dtype=object),
                neg_bins=np.array(combined["neg_bins"], dtype=object),
                pos_counts=np.array(combined["pos_counts"], dtype=object),
                pos_bins=np.array(combined["pos_bins"], dtype=object),
                neg_lat_min=np.array(combined["neg_lat_min"]),
                neg_lat_max=np.array(combined["neg_lat_max"]),
                pos_lat_min=np.array(combined["pos_lat_min"]),
                pos_lat_max=np.array(combined["pos_lat_max"]),
                allow_pickle=True,
            )
            print(f"[histograms] saved {out_npz}")
        plt.close(fig)

    def _plot_global_hist(
        da_target_var: xr.DataArray,
        da_pred_var: xr.DataArray,
        variable_name: str,
        level_token: str,
        ens_token: str | None,
        lead_time_range: tuple[str, str] | None,
    ) -> None:
        """Plot a single global histogram comparing target vs prediction.

        - Uses paired subsampling if enabled
        - Chooses common bin edges from combined quantiles
        - Saves PNG and NPZ with counts/edges
        """

        # Helper subsampling function (reuse logic inline to avoid nested closures surprises)
        def _subsample_values(da: xr.DataArray, k: int, seed: int) -> np.ndarray:
            if k is None:
                arr = np.asarray(da.compute().values).ravel()
                return arr[np.isfinite(arr)]
            size = int(getattr(da, "size", 0) or 0)
            if size == 0:
                return np.array([], dtype=float)
            if size <= k:
                arr = np.asarray(da.compute().values).ravel()
                return arr[np.isfinite(arr)]
            dims = list(da.dims)
            nd = max(1, len(dims))
            frac = (k / float(size)) ** (1.0 / nd)
            rng = np.random.default_rng(base_seed + 1337)
            indexers: dict[str, Any] = {}
            for d in dims:
                n = int(da.sizes.get(str(d), 1))
                take = max(1, int(np.ceil(frac * n)))
                take = min(take, n)
                idx = rng.choice(n, size=take, replace=False)
                idx.sort()
                indexers[str(d)] = np.asarray(idx)
            sub = da.isel(indexers)
            arr = np.asarray(sub.compute().values).ravel()
            return arr[np.isfinite(arr)]

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
            a_true = _subsample_values(da_target_var, max_samples, base_seed + 9001)
            a_pred = _subsample_values(da_pred_var, max_samples, base_seed + 9001)
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
            color="skyblue",
            label="Ground Truth",
        )
        ax.bar(
            edges[:-1],
            dens_pred,
            width=width,
            align="edge",
            alpha=0.5,
            color="salmon",
            label="Model Prediction",
        )
        units = da_target_var.attrs.get("units", "")
        title_lt = f" — lead {lead_time_range[0]}" if lead_time_range else ""
        ax.set_title(f"Global histogram {variable_name}{title_lt} ({units})")
        ax.set_xlabel(variable_name)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png = section_output / build_output_filename(
                metric="hist_global",
                variable=variable_name,
                level=level_token,
                qualifier=None,
                init_time_range=None,
                lead_time_range=lead_time_range,
                ensemble=ens_token,
                ext="png",
            )
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"[histograms] saved {out_png}")
        if save_npz:
            section_output.mkdir(parents=True, exist_ok=True)
            out_npz = section_output / build_output_filename(
                metric="hist_global",
                variable=variable_name,
                level=level_token,
                qualifier="data",
                init_time_range=None,
                lead_time_range=lead_time_range,
                ensemble=ens_token,
                ext="npz",
            )
            np.savez(out_npz, counts_true=dens_true, counts_pred=dens_pred, edges=edges)
            print(f"[histograms] saved {out_npz}")
        plt.close(fig)

    # 2D variables
    def _iter_members():
        if not has_ens:
            yield None, ds_target, ds_prediction
        else:
            for i in range(int(ds_prediction.sizes["ensemble"])):
                tgt_m = ds_target.isel(ensemble=i) if "ensemble" in ds_target.dims else ds_target
                pred_m = ds_prediction.isel(ensemble=i)
                yield i, tgt_m, pred_m

    if resolved_mode == "members" and not has_ens:
        resolved_mode = "none"  # degrade silently

    if resolved_mode == "mean" and has_ens:
        ds_prediction_mean = ds_prediction.mean(dim="ensemble")
        ds_target_mean = (
            ds_target.mean(dim="ensemble") if "ensemble" in ds_target.dims else ds_target
        )
        # Determine per-lead behavior
        per_lead = bool(plotting_cfg.get("histograms_per_lead", True)) and (
            "lead_time" in ds_prediction_mean.dims and int(ds_prediction_mean.lead_time.size) > 1
        )
        # removed unused lead_indices (ruff F841)
        # Suppress individual per-lead global PNGs: only produce grid later
        # (retain potential single-lead case by emitting one global if only one lead)
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
        # Optional: combined global histogram grid across selected leads
        # Force grid generation whenever lead_time is present (>=1)
        do_grid = ("lead_time" in ds_prediction_mean.dims) and int(
            ds_prediction_mean.lead_time.size
        ) >= 1
        if do_grid:
            # Use all retained lead_time hours (panel concept removed)
            all_hours = []
            try:
                vals = ds_prediction_mean["lead_time"].values
                all_hours = [int(v / np.timedelta64(1, "h")) for v in vals]
            except Exception:
                all_hours = list(range(int(ds_prediction_mean.lead_time.size)))
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
                # Determine global x/y limits (data range across all panels) for shared axes
                all_edges_min = []
                all_edges_max = []
                all_y_max = []
                panel_results = []  # store (edges, dt, dp, h)
                for i, h in enumerate(panel_hours):
                    try:
                        idx = all_hours.index(int(h))
                    except Exception:
                        idx = i
                    da_t = ds_target_mean[variable_name]
                    da_p = ds_prediction_mean[variable_name]
                    if "lead_time" in da_t.dims:
                        da_t = da_t.isel(lead_time=idx, drop=True)
                    if "lead_time" in da_p.dims:
                        da_p = da_p.isel(lead_time=idx, drop=True)
                    # paired subsampling
                    if max_samples is not None:
                        a_true = _subsample_values(da_t, max_samples, base_seed + 9001 + i)
                        a_pred = _subsample_values(da_p, max_samples, base_seed + 9001 + i)
                    else:
                        a_true = np.asarray(da_t.compute().values).ravel()
                        a_pred = np.asarray(da_p.compute().values).ravel()
                        a_true = a_true[np.isfinite(a_true)]
                        a_pred = a_pred[np.isfinite(a_pred)]
                    # edges
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
                    all_edges_min.append(edges[0])
                    all_edges_max.append(edges[-1])
                    # Track y-limits from both series for unified ylim
                    ymax_i = float(
                        max(np.nanmax(dt) if dt.size else 0.0, np.nanmax(dp) if dp.size else 0.0)
                    )
                    all_y_max.append(ymax_i)
                    panel_results.append((edges, dt, dp, h))
                # Unified x/y limits
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
                        color="skyblue",
                        label="Ground Truth" if i == 0 else None,
                    )
                    ax.bar(
                        edges[:-1],
                        dp,
                        width=width,
                        align="edge",
                        alpha=0.5,
                        color="salmon",
                        label="Model Prediction" if i == 0 else None,
                    )
                    ax.set_title(f"Lead {int(h)}h", fontsize=11)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(0.0, y_max)
                    if i % ncols != 0:
                        ax.set_ylabel("")
                        ax.tick_params(axis="y", labelleft=False)
                    else:
                        ax.set_ylabel("Density")
                    # bottom row x-labels only
                    if i >= (nrows - 1) * ncols:
                        ax.set_xlabel(variable_name)
                    else:
                        ax.tick_params(axis="x", labelbottom=False)
                # Single legend
                axes[0].legend(loc="upper right")
                # hide any unused axes
                for j in range(n_panels, nrows * ncols):
                    axes[j].axis("off")
                if save_fig:
                    section_output.mkdir(parents=True, exist_ok=True)
                    out_png = section_output / build_output_filename(
                        metric="hist_global",
                        variable=str(variable_name),
                        level="",
                        qualifier="grid",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=ensemble_mode_to_token("mean"),
                        ext="png",
                    )
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    print(f"[histograms] saved {out_png}")
                if save_npz:
                    # Persist panel data (edges, densities) for table recreation
                    # Store variable length object arrays for flexibility
                    out_npz = section_output / build_output_filename(
                        metric="hist_global",
                        variable=str(variable_name),
                        level="",
                        qualifier="grid_data",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=ensemble_mode_to_token("mean"),
                        ext="npz",
                    )
                    lead_hours = np.array(panel_hours, dtype=float)
                    dens_true_list = [pr[1] for pr in panel_results]  # dt
                    dens_pred_list = [pr[2] for pr in panel_results]  # dp
                    edges_list = [pr[0] for pr in panel_results]
                    np.savez(
                        out_npz,
                        lead_hours=lead_hours,
                        densities_true=np.array(dens_true_list, dtype=object),
                        densities_pred=np.array(dens_pred_list, dtype=object),
                        edges=np.array(edges_list, dtype=object),
                        allow_pickle=True,
                    )
                    print(f"[histograms] saved {out_npz}")
                plt.close(fig)
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
                        level_token="",
                        ens_token=token,
                        lead_time_range=None,
                    )
            # Optional global grid per member
            # Force grid generation for members mode whenever lead_time present (>=1)
            do_grid = ("lead_time" in pred_m.dims) and int(pred_m.lead_time.size) >= 1
            if do_grid:
                all_hours = []
                try:
                    vals = pred_m["lead_time"].values
                    all_hours = [int(v / np.timedelta64(1, "h")) for v in vals]
                except Exception:
                    all_hours = list(range(int(pred_m.lead_time.size)))
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
                        da_t = tgt_m[variable_name]
                        da_p = pred_m[variable_name]
                        if "lead_time" in da_t.dims:
                            da_t = da_t.isel(lead_time=idx, drop=True)
                        if "lead_time" in da_p.dims:
                            da_p = da_p.isel(lead_time=idx, drop=True)
                        if max_samples is not None:
                            a_true = _subsample_values(da_t, max_samples, base_seed + 9001 + i)
                            a_pred = _subsample_values(da_p, max_samples, base_seed + 9001 + i)
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
                            max(
                                np.nanmax(dt) if dt.size else 0.0, np.nanmax(dp) if dp.size else 0.0
                            )
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
                            color="skyblue",
                            label="Ground Truth" if i == 0 else None,
                        )
                        ax.bar(
                            edges[:-1],
                            dp,
                            width=width,
                            align="edge",
                            alpha=0.5,
                            color="salmon",
                            label="Model Prediction" if i == 0 else None,
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
                            level="",
                            qualifier="grid",
                            init_time_range=None,
                            lead_time_range=None,
                            ensemble=token,
                            ext="png",
                        )
                        plt.savefig(out_png, bbox_inches="tight", dpi=200)
                        print(f"[histograms] saved {out_png}")
                    if save_npz:
                        out_npz = section_output / build_output_filename(
                            metric="hist_global",
                            variable=str(variable_name),
                            level="",
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
                        np.savez(
                            out_npz,
                            lead_hours=lead_hours,
                            densities_true=np.array(dens_true_list, dtype=object),
                            densities_pred=np.array(dens_pred_list, dtype=object),
                            edges=np.array(edges_list, dtype=object),
                            allow_pickle=True,
                        )
                        print(f"[histograms] saved {out_npz}")
                    plt.close(fig)
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
                    level_token="",
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
                        a_true = _subsample_values(da_t, max_samples, base_seed + 9001 + i)
                        a_pred = _subsample_values(da_p, max_samples, base_seed + 9001 + i)
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
                        color="skyblue",
                        label="Ground Truth" if i == 0 else None,
                    )
                    ax.bar(
                        edges[:-1],
                        dp,
                        width=width,
                        align="edge",
                        alpha=0.5,
                        color="salmon",
                        label="Model Prediction" if i == 0 else None,
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
                        level="",
                        qualifier="grid",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=token,
                        ext="png",
                    )
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    print(f"[histograms] saved {out_png}")
                if save_npz:
                    out_npz = section_output / build_output_filename(
                        metric="hist_global",
                        variable=str(variable_name),
                        level="",
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
                    np.savez(
                        out_npz,
                        lead_hours=lead_hours,
                        densities_true=np.array(dens_true_list, dtype=object),
                        densities_pred=np.array(dens_pred_list, dtype=object),
                        edges=np.array(edges_list, dtype=object),
                        allow_pickle=True,
                    )
                    print(f"[histograms] saved {out_npz}")
                plt.close(fig)

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
                    _plot_variable(
                        da_t_lvl_mean,
                        da_p_lvl_mean,
                        str(variable_name),
                        level_token=str(lvl_clean),
                        qualifier="latbands",
                        ens_token=ensemble_mode_to_token("mean"),
                        lead_time_range=None,
                    )
                elif resolved_mode == "members" and has_ens:
                    for member_index, tgt_m, pred_m in _iter_members():
                        token = ensemble_mode_to_token("members", member_index)
                        _plot_variable(
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
                            lead_time_range=None,
                        )
                else:  # pooled/none
                    token = (
                        ensemble_mode_to_token("pooled")
                        if (resolved_mode == "pooled" and has_ens)
                        else None
                    )
                    _plot_variable(
                        da_t_lvl,
                        da_p_lvl,
                        str(variable_name),
                        level_token=str(lvl_clean),
                        qualifier="latbands",
                        ens_token=token,
                        lead_time_range=None,
                    )
