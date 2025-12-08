from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as dsa
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_level_label,
    format_variable_name,
    get_variable_units,
    resolve_ensemble_mode,
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

    per_lat_band = bool(plotting_cfg.get("histograms_per_lat_band", False))

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

    # Resolve ensemble handling
    resolved_mode = resolve_ensemble_mode("histograms", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for histograms")

    def _plot_variable(
        da_target_var: xr.DataArray,
        da_pred_var: xr.DataArray,
        variable_name: str,
        level_token: str,
        qualifier: str,
        ens_token: str | None,
        level_val: Any = None,
    ):
        i = 0  # retained for seeding when needed (simplified for 3D reuse)
        print(f"[histograms] variable: {variable_name}")

        # Helper to choose common bin edges without loading full arrays
        def _choose_edges(da1: xr.DataArray, da2: xr.DataArray, bins: int = 1000):
            """Choose common bin edges based on robust quantiles over both arrays.
            Falls back to [-1, 1] if bounds are degenerate.
            """
            try:
                both = xr.concat([da1, da2], dim="_t")
                q = both.quantile([0.001, 0.999], skipna=True).compute()
                vmin = float(q.isel(quantile=0).item())
                vmax = float(q.isel(quantile=1).item())
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = -1.0, 1.0
            except Exception:
                vmin, vmax = -1.0, 1.0
            return np.linspace(vmin, vmax, bins + 1)

        def _dask_hist(da: xr.DataArray, edges: np.ndarray):
            """Compute histogram counts with Dask without materializing the array.
            Filters NaNs and returns a Dask array of counts matching edges-1 length.
            """
            data = getattr(da, "data", da)
            darr = dsa.asarray(data)
            darr = darr.ravel()
            darr = darr[~dsa.isnan(darr)]
            counts = dsa.histogram(darr, bins=np.asarray(edges))[0]
            return counts

        # Helper subsampling function
        def _subsample_values(da: xr.DataArray, k: int, seed: int) -> np.ndarray:
            if k is None:
                # No subsampling requested; compute only the flattened, finite array
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

        # --- Global Histogram ---
        fig_g, ax_g = plt.subplots(figsize=(10, 6), dpi=dpi)

        # Global data
        da_true_g = da_target_var
        da_pred_g = da_pred_var

        # Compute global histogram
        if max_samples is not None:
            seed = base_seed + (i + 1) * 1000
            ds_sample = _subsample_values(da_true_g, max_samples, seed)
            ml_sample = _subsample_values(da_pred_g, max_samples, seed)

            if ds_sample.size > 0 and ml_sample.size > 0:
                try:
                    both = np.concatenate([ds_sample, ml_sample])
                    qlow, qhigh = np.quantile(both, [0.001, 0.999])
                    if not np.isfinite(qlow) or not np.isfinite(qhigh) or qlow == qhigh:
                        qlow, qhigh = -1.0, 1.0
                except Exception:
                    qlow, qhigh = -1.0, 1.0
                edges_g = np.linspace(qlow, qhigh, 1001)
                dsa_ds = dsa.from_array(ds_sample, chunks=ds_sample.shape[0] // 4 or 1)
                dsa_ml = dsa.from_array(ml_sample, chunks=ml_sample.shape[0] // 4 or 1)
                counts_ds_g = dsa.histogram(dsa_ds, bins=edges_g)[0].compute()
                counts_ml_g = dsa.histogram(dsa_ml, bins=edges_g)[0].compute()
            else:
                counts_ds_g = np.array([])
                counts_ml_g = np.array([])
                edges_g = np.array([])
        else:
            edges_g = _choose_edges(da_true_g, da_pred_g, bins=1000)
            counts_ds_g = _dask_hist(da_true_g, edges_g).compute().astype(float)
            counts_ml_g = _dask_hist(da_pred_g, edges_g).compute().astype(float)

        if len(counts_ds_g) > 0:
            width = np.diff(edges_g)
            bin_area = counts_ds_g.sum() * width.mean() if counts_ds_g.sum() > 0 else 1.0
            counts_ds_g = counts_ds_g / bin_area
            bin_area_ml = counts_ml_g.sum() * width.mean() if counts_ml_g.sum() > 0 else 1.0
            counts_ml_g = counts_ml_g / bin_area_ml

            ax_g.bar(
                edges_g[:-1],
                counts_ds_g,
                width=width,
                align="edge",
                alpha=0.5,
                color="skyblue",
                label="Ground Truth",
            )
            ax_g.bar(
                edges_g[:-1],
                counts_ml_g,
                width=width,
                align="edge",
                alpha=0.5,
                color="salmon",
                label="Model Prediction",
            )
            ax_g.legend(loc="upper right")

        units = get_variable_units(ds_target, variable_name)

        # Check for single date
        date_str = extract_date_from_dataset(da_target_var)
        lev_part = format_level_label(level_val if level_val is not None else level_token)

        if len(counts_ds_g) > 0:
            ax_g.set_title(
                f"Global Histogram — {format_variable_name(variable_name)}{lev_part}{date_str}"
            )
            if units:
                ax_g.set_xlabel(units)
        else:
            ax_g.set_title(
                f"Global Histogram — {format_variable_name(variable_name)}{lev_part} "
                f"(No data){date_str}"
            )

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png_g = section_output / build_output_filename(
                metric="hist",
                variable=variable_name,
                level=level_token,
                qualifier="global",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token,
                ext="png",
            )
            fig_g.savefig(out_png_g, bbox_inches="tight", dpi=200)
            print(f"[histograms] saved {out_png_g}")

        if save_npz:
            section_output.mkdir(parents=True, exist_ok=True)
            out_npz_g = section_output / build_output_filename(
                metric="hist",
                variable=variable_name,
                level=level_token,
                qualifier="global",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token,
                ext="npz",
            )
            np.savez(
                out_npz_g,
                counts_ds=counts_ds_g,
                counts_ml=counts_ml_g,
                bins=edges_g,
                units=units,
                allow_pickle=True,
            )
            print(f"[histograms] saved {out_npz_g}")
        plt.close(fig_g)

        if per_lat_band:
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
                    # Determine edges from combined sample quantiles
                    try:
                        both = np.concatenate([ds_sample, ml_sample])
                        qlow, qhigh = np.quantile(both, [0.001, 0.999])
                        if not np.isfinite(qlow) or not np.isfinite(qhigh) or qlow == qhigh:
                            qlow, qhigh = -1.0, 1.0
                    except Exception:
                        qlow, qhigh = -1.0, 1.0
                    edges = np.linspace(qlow, qhigh, 1001)
                    # Histogram on subsamples using dask.histogram for consistency
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
                    try:
                        both = np.concatenate([ds_sample, ml_sample])
                        qlow, qhigh = np.quantile(both, [0.001, 0.999])
                        if not np.isfinite(qlow) or not np.isfinite(qhigh) or qlow == qhigh:
                            qlow, qhigh = -1.0, 1.0
                    except Exception:
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
                axs[j, 0].legend(loc="upper right")
                if save_npz:
                    combined["pos_counts"].append((counts_ds, counts_ml))
                    combined["pos_bins"].append(edges)
                    combined["pos_lat_min"].append(float(lat_min))
                    combined["pos_lat_max"].append(float(lat_max))

            units = get_variable_units(ds_target, variable_name)

            # Check for single date
            date_str = extract_date_from_dataset(da_target_var)

            plt.suptitle(
                f"Distributions by Latitude Bands — "
                f"{format_variable_name(variable_name)}{date_str}",
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
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[histograms] saved {out_png}")
            if save_npz:
                section_output.mkdir(parents=True, exist_ok=True)
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
                    units=units,
                    allow_pickle=True,
                )
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
        resolved_mode = "pooled"  # degrade silently

    if resolved_mode == "mean" and has_ens:
        ds_prediction_mean = ds_prediction.mean(dim="ensemble")
        ds_target_mean = (
            ds_target.mean(dim="ensemble") if "ensemble" in ds_target.dims else ds_target
        )
        for variable_name in variables_2d:
            _plot_variable(
                ds_target_mean[variable_name],
                ds_prediction_mean[variable_name],
                str(variable_name),
                level_token="surface",
                qualifier="latbands",
                ens_token=ensemble_mode_to_token("mean"),
            )
    elif resolved_mode == "members" and has_ens:
        for member_index, tgt_m, pred_m in _iter_members():
            token = ensemble_mode_to_token("members", member_index)
            for variable_name in variables_2d:
                _plot_variable(
                    tgt_m[variable_name],
                    pred_m[variable_name],
                    str(variable_name),
                    level_token="surface",
                    qualifier="latbands",
                    ens_token=token,
                )
    else:  # pooled
        token = ensemble_mode_to_token("pooled") if has_ens else None
        for variable_name in variables_2d:
            _plot_variable(
                ds_target[variable_name],
                ds_prediction[variable_name],
                str(variable_name),
                level_token="surface",
                qualifier="latbands",
                ens_token=token,
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
                    _plot_variable(
                        da_t_lvl_mean,
                        da_p_lvl_mean,
                        str(variable_name),
                        level_token=str(lvl_clean),
                        qualifier="latbands",
                        ens_token=ensemble_mode_to_token("mean"),
                        level_val=lvl,
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
                            level_val=lvl,
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
                        level_val=lvl,
                    )
