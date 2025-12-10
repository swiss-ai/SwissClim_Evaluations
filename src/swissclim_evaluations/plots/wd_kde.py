from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde, wasserstein_distance

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
)


def _subsample_values(da: xr.DataArray, k: int, seed: int) -> np.ndarray:
    """Dimension-aware uniform subsample across all dims.

    Uses per-dimension index sampling so very large arrays don't need to be fully
    materialized. Always pairs subsamples when given the same seed.
    """
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


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    ds_target_std: xr.Dataset,
    ds_prediction_std: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    ensemble_mode: str | None = None,
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    # Limit number of samples drawn from each band to avoid loading all data into memory
    max_samples = int(plotting_cfg.get("kde_max_samples", 200_000))
    # Global random seed from config for reproducible subsampling
    base_seed = int(plotting_cfg.get("random_seed", 42))
    # Target/prediction always use identical subsamples so that if underlying
    # arrays are equal the KDEs match exactly (paired subsampling is enforced).
    section_output = out_root / "wd_kde"

    # Ensure output directory exists early
    section_output.mkdir(parents=True, exist_ok=True)

    wasserstein_rows: list[dict[str, Any]] = []
    n_bands = 8
    lat_bins = np.linspace(-90, 90, n_bands + 1)
    n_rows = n_bands // 2

    process_3d = bool(plotting_cfg.get("wd_kde_include_3d", True))
    max_levels = plotting_cfg.get("wd_kde_max_levels")
    try:
        max_levels = int(max_levels) if max_levels is not None else None
        if max_levels is not None and max_levels <= 0:
            max_levels = None
    except Exception:
        max_levels = None

    per_lat_band = bool(plotting_cfg.get("wd_kde_per_lat_band", False))

    # Select only genuine 2D variables (no 'level' dimension) and 3D ones
    variables_2d = [v for v in ds_target_std.data_vars if "level" not in ds_target_std[v].dims]
    variables_3d = [v for v in ds_target_std.data_vars if "level" in ds_target_std[v].dims]
    if not variables_2d and (not process_3d or not variables_3d):
        print("[wd_kde] No eligible variables found – skipping.")
        return
    if variables_2d:
        print(f"[wd_kde] Processing {len(variables_2d)} 2D variables (standardized).")
    if process_3d and variables_3d:
        print(f"[wd_kde] Processing {len(variables_3d)} 3D variables (per-level, standardized).")
    # Global KDE curves instead of latitude-binned panels

    # Resolve ensemble handling (pooled/mean/members). Prob not allowed here.
    resolved_mode = resolve_ensemble_mode("wd_kde", ensemble_mode, ds_target_std, ds_prediction_std)
    has_ens = "ensemble" in ds_prediction_std.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for wd_kde")

    def _iter_members():
        if not has_ens:
            yield None, ds_target_std, ds_prediction_std
        else:
            for i in range(int(ds_prediction_std.sizes["ensemble"])):
                tgt_m = (
                    ds_target_std.isel(ensemble=i)
                    if "ensemble" in ds_target_std.dims
                    else ds_target_std
                )
                pred_m = ds_prediction_std.isel(ensemble=i)
                yield i, tgt_m, pred_m

    # Establish dataset views depending on mode
    if resolved_mode == "mean" and has_ens:
        ds_target_std_eff = (
            ds_target_std.mean(dim="ensemble")
            if "ensemble" in ds_target_std.dims
            else ds_target_std
        )
        ds_prediction_std_eff = ds_prediction_std.mean(dim="ensemble")
        ens_token_base = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled":
        ds_target_std_eff = ds_target_std
        ds_prediction_std_eff = ds_prediction_std
        ens_token_base = ensemble_mode_to_token("pooled") if has_ens else None
    else:  # members
        ds_target_std_eff = ds_target_std  # used only for 2D variable discovery
        ds_prediction_std_eff = ds_prediction_std
        ens_token_base = None  # per-member inside loop

    def _process_variable(
        var_name: str,
        da_t_std: xr.DataArray,
        da_p_std: xr.DataArray,
        level_token: str,
        ens_token: str | None,
        level_val: Any = None,
    ):
        # local copy of loop body (with minor modifications to accept arrays directly)
        def _subsample_values(da: xr.DataArray, k: int, seed: int) -> np.ndarray:
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
                # Cast to plain numpy array for mypy/xarray typing compatibility
                indexers[str(d)] = np.asarray(idx)
            sub = da.isel(indexers)
            arr = np.asarray(sub.compute().values).ravel()
            return arr[np.isfinite(arr)]

        print(f"[wd_kde] variable: {var_name} level={level_token}")

        # --- Global KDE ---
        seed_g = base_seed + (hash(var_name + level_token) % 1000) * 1000
        ds_flat_g = _subsample_values(da_t_std, max_samples, seed=seed_g)
        ml_flat_g = _subsample_values(da_p_std, max_samples, seed=seed_g)
        units = get_variable_units(ds_target, var_name)

        if ds_flat_g.size > 0 and ml_flat_g.size > 0:
            w_g = wasserstein_distance(ds_flat_g, ml_flat_g)
            kde_ds_g = gaussian_kde(ds_flat_g)
            kde_ml_g = gaussian_kde(ml_flat_g)

            # Plot Global
            fig_g, ax_g = plt.subplots(figsize=(10, 6), dpi=dpi)
            x_eval_g = np.linspace(
                min(ds_flat_g.min(), ml_flat_g.min()),
                max(ds_flat_g.max(), ml_flat_g.max()),
                100,
            )
            ax_g.plot(x_eval_g, kde_ds_g(x_eval_g), color=COLOR_GROUND_TRUTH, label="Target")
            ax_g.plot(
                x_eval_g, kde_ml_g(x_eval_g), color=COLOR_MODEL_PREDICTION, label="Prediction"
            )

            # Check for single date
            date_str = extract_date_from_dataset(da_t_std)
            lev_part = format_level_label(level_val if level_val is not None else level_token)

            ax_g.set_title(
                f"Global Normalized KDE — {format_variable_name(var_name)}"
                f"{lev_part}{date_str}\nW-dist: {w_g:.3f}"
            )
            ax_g.legend()

            if save_fig:
                section_output.mkdir(parents=True, exist_ok=True)
                out_png_g = section_output / build_output_filename(
                    metric="wd_kde",
                    variable=var_name,
                    level=level_token,
                    qualifier="global",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="png",
                )
                fig_g.savefig(out_png_g, bbox_inches="tight", dpi=200)
                print(f"[wd_kde] saved {out_png_g}")

            if save_npz:
                section_output.mkdir(parents=True, exist_ok=True)
                out_npz_g = section_output / build_output_filename(
                    metric="wd_kde",
                    variable=var_name,
                    level=level_token,
                    qualifier="global",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="npz",
                )
                np.savez(
                    out_npz_g,
                    w_dist=w_g,
                    x=x_eval_g,
                    kde_ds=kde_ds_g(x_eval_g),
                    kde_ml=kde_ml_g(x_eval_g),
                    units=units,
                    allow_pickle=True,
                )
                print(f"[wd_kde] saved {out_npz_g}")
            plt.close(fig_g)

            # Add to CSV rows
            wasserstein_rows.append(
                {
                    "variable": var_name,
                    "hemisphere": "global",
                    "lat_min": -90.0,
                    "lat_max": 90.0,
                    "wasserstein": float(w_g),
                }
            )

        if not per_lat_band:
            return

        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=dpi)
        w_distances: list[float] = []
        combined: dict[str, list[np.ndarray | float]] = {
            "neg_x": [],
            "neg_kde_ds": [],
            "neg_kde_ml": [],
            "neg_lat_min": [],
            "neg_lat_max": [],
            "pos_x": [],
            "pos_kde_ds": [],
            "pos_kde_ml": [],
            "pos_lat_min": [],
            "pos_lat_max": [],
        }
        # Negative latitudes (right column)
        for j in range(n_bands // 2):
            lat_max = lat_bins[j]
            lat_min = lat_bins[j + 1]
            da_target_slice = da_t_std.sel(latitude=slice(lat_min, lat_max))
            da_prediction_slice = da_p_std.sel(latitude=slice(lat_min, lat_max))
            if da_target_slice.size == 0 or da_prediction_slice.size == 0:
                axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue
            seed = base_seed + (hash(var_name + level_token) % 1000) * 1000 + (j + 1) * 10 + 1
            ds_flat = _subsample_values(da_target_slice, max_samples, seed=seed)
            ml_flat = _subsample_values(da_prediction_slice, max_samples, seed=seed)
            if ds_flat.size == 0 or ml_flat.size == 0:
                axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue
            w = wasserstein_distance(ds_flat, ml_flat)
            w_distances.append(w)
            wasserstein_rows.append(
                {
                    "variable": var_name,
                    "hemisphere": "south",
                    "lat_min": float(lat_min),
                    "lat_max": float(lat_max),
                    "wasserstein": float(w),
                }
            )
            kde_ds = gaussian_kde(ds_flat)
            kde_ml = gaussian_kde(ml_flat)
            x_eval = np.linspace(
                min(ds_flat.min(), ml_flat.min()),
                max(ds_flat.max(), ml_flat.max()),
                100,
            )
            axs[j, 1].plot(x_eval, kde_ds(x_eval), color=COLOR_GROUND_TRUTH, label="Target")
            axs[j, 1].plot(x_eval, kde_ml(x_eval), color=COLOR_MODEL_PREDICTION, label="Prediction")
            axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}° (W-dist: {w:.3f})")
            axs[j, 1].legend()
            if save_npz:
                combined["neg_x"].append(x_eval)
                combined["neg_kde_ds"].append(kde_ds(x_eval))
                combined["neg_kde_ml"].append(kde_ml(x_eval))
                combined["neg_lat_min"].append(float(lat_min))
                combined["neg_lat_max"].append(float(lat_max))
        # Positive latitudes (left column)
        for j in range(n_bands // 2):
            idx = -(j + 1)
            lat_max = lat_bins[idx - 1]
            lat_min = lat_bins[idx]
            da_target_slice = da_t_std.sel(latitude=slice(lat_min, lat_max))
            da_prediction_slice = da_p_std.sel(latitude=slice(lat_min, lat_max))
            if da_target_slice.size == 0 or da_prediction_slice.size == 0:
                axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue
            seed = base_seed + (hash(var_name + level_token) % 1000) * 1000 + (j + 1) * 10 + 2
            ds_flat = _subsample_values(da_target_slice, max_samples, seed=seed)
            ml_flat = _subsample_values(da_prediction_slice, max_samples, seed=seed)
            if ds_flat.size == 0 or ml_flat.size == 0:
                axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue
            w = wasserstein_distance(ds_flat, ml_flat)
            w_distances.append(w)
            wasserstein_rows.append(
                {
                    "variable": var_name,
                    "hemisphere": "north",
                    "lat_min": float(lat_min),
                    "lat_max": float(lat_max),
                    "wasserstein": float(w),
                }
            )
            kde_ds = gaussian_kde(ds_flat)
            kde_ml = gaussian_kde(ml_flat)
            x_eval = np.linspace(
                min(ds_flat.min(), ml_flat.min()),
                max(ds_flat.max(), ml_flat.max()),
                100,
            )
            axs[j, 0].plot(x_eval, kde_ds(x_eval), color=COLOR_GROUND_TRUTH, label="Target")
            axs[j, 0].plot(x_eval, kde_ml(x_eval), color=COLOR_MODEL_PREDICTION, label="Prediction")
            axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}° (W-dist: {w:.3f})")
            axs[j, 0].legend()
            if save_npz:
                combined["pos_x"].append(x_eval)
                combined["pos_kde_ds"].append(kde_ds(x_eval))
                combined["pos_kde_ml"].append(kde_ml(x_eval))
                combined["pos_lat_min"].append(float(lat_min))
                combined["pos_lat_max"].append(float(lat_max))
        mean_w = float(np.mean(w_distances)) if w_distances else float("nan")

        # Check for single date
        date_str = extract_date_from_dataset(da_t_std)

        plt.suptitle(
            "Normalized Distribution of "
            f"{var_name} ({level_token}) by latitude bands{date_str}\nMean Wasserstein distance: "
            f"{mean_w:.3f}",
            y=1.02,
        )
        plt.tight_layout()
        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png = section_output / build_output_filename(
                metric="wd_kde",
                variable=var_name,
                level=level_token,
                qualifier="latbands",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token,
                ext="png",
            )
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"[wd_kde] saved {out_png}")
        if save_npz:
            section_output.mkdir(parents=True, exist_ok=True)
            out_npz = section_output / build_output_filename(
                metric="wd_kde",
                variable=var_name,
                level=level_token,
                qualifier="latbands",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token,
                ext="npz",
            )
            np.savez(
                out_npz,
                mean_w=mean_w,
                neg_x=np.array(combined["neg_x"], dtype=object),
                neg_kde_ds=np.array(combined["neg_kde_ds"], dtype=object),
                neg_kde_ml=np.array(combined["neg_kde_ml"], dtype=object),
                neg_lat_min=np.array(combined["neg_lat_min"]),
                neg_lat_max=np.array(combined["neg_lat_max"]),
                pos_x=np.array(combined["pos_x"], dtype=object),
                pos_kde_ds=np.array(combined["pos_kde_ds"], dtype=object),
                pos_kde_ml=np.array(combined["pos_kde_ml"], dtype=object),
                pos_lat_min=np.array(combined["pos_lat_min"]),
                pos_lat_max=np.array(combined["pos_lat_max"]),
                units=units,
                allow_pickle=True,
            )
            print(f"[wd_kde] saved {out_npz}")
        plt.close(fig)

    # 2D standardized variables (run once per variable; avoid prior recursion issue)
    if resolved_mode == "members" and has_ens:
        for member_index, tgt_m, pred_m in _iter_members():
            token_m = ensemble_mode_to_token("members", member_index)
            for variable_name in variables_2d:
                _process_variable(
                    str(variable_name),
                    tgt_m[variable_name],
                    pred_m[variable_name],
                    level_token="surface",
                    ens_token=token_m,
                )
    else:
        for variable_name in variables_2d:
            _process_variable(
                str(variable_name),
                ds_target_std_eff[variable_name],
                ds_prediction_std_eff[variable_name],
                level_token="surface",
                ens_token=ens_token_base,
            )

    # 3D standardized variables per level
    if process_3d:
        for variable_name in variables_3d:
            da_t_std = ds_target_std_eff[variable_name]
            da_p_std = ds_prediction_std_eff[variable_name]
            levels = list(da_t_std["level"].values)
            if max_levels is not None:
                levels = levels[:max_levels]
            for lvl in levels:
                lvl_clean = str(lvl).replace(".", "_")
                if resolved_mode == "members" and has_ens:
                    for member_index, tgt_m, pred_m in _iter_members():
                        token_m = ensemble_mode_to_token("members", member_index)
                        _process_variable(
                            str(variable_name),
                            (
                                tgt_m[variable_name].sel(level=lvl)
                                if "ensemble" in tgt_m[variable_name].dims
                                else tgt_m.sel(level=lvl)
                            ),
                            pred_m[variable_name].sel(level=lvl),
                            level_token=str(lvl_clean),
                            ens_token=token_m,
                            level_val=lvl,
                        )
                else:
                    da_t_lvl = da_t_std.sel(level=lvl)
                    da_p_lvl = da_p_std.sel(level=lvl)
                    _process_variable(
                        str(variable_name),
                        da_t_lvl,
                        da_p_lvl,
                        level_token=str(lvl_clean),
                        ens_token=ens_token_base,
                        level_val=lvl,
                    )

    # Optional: Global KDE evolution over lead_time (3D perspective)
    evolve_flag = bool((plotting_cfg or {}).get("wd_kde_global_evolution", False))
    if evolve_flag and ("lead_time" in ds_prediction_std_eff.dims):
        # Choose a representative 2D standardized variable (no level dim)
        cand_vars = [
            v
            for v in ds_prediction_std_eff.data_vars
            if "level" not in ds_prediction_std_eff[v].dims
        ]
        if cand_vars:
            base_var = str(cand_vars[0])
            # Common evaluation axis from combined sample across all leads (paired with targets)
            # Draw a coarse subsample to set robust evaluation range
            da_t_all = ds_target_std_eff[base_var]
            da_p_all = ds_prediction_std_eff[base_var]
            # Collapse spatial + time to estimate global min/max quickly
            q_t = da_t_all.quantile([0.001, 0.999], skipna=True).compute().values
            q_p = da_p_all.quantile([0.001, 0.999], skipna=True).compute().values
            vmin = float(min(q_t[0], q_p[0]))
            vmax = float(max(q_t[1], q_p[1]))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                vmin, vmax = -3.0, 3.0
            y_eval = np.linspace(vmin, vmax, 200)

            # Build densities per lead for target and model
            leads = list(ds_prediction_std_eff["lead_time"].values)
            lead_hours = []
            Z_t = []
            Z_p = []

            def _eval_kde_1d(da: xr.DataArray) -> np.ndarray:
                arr = np.asarray(da.compute().values).ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size < 10:
                    return np.zeros_like(y_eval)
                kde = gaussian_kde(arr)
                return kde(y_eval)

            for i, lt in enumerate(leads):
                # Convert timedelta leads to hours; fall back to index
                lt_arr = np.asarray(lt)
                if np.issubdtype(lt_arr.dtype, np.timedelta64):
                    hours = int(lt_arr / np.timedelta64(1, "h"))
                else:
                    hours = int(i)
                lead_hours.append(hours)
                da_t = ds_target_std_eff[base_var]
                da_p = ds_prediction_std_eff[base_var]
                # Select single lead slice (drop=False to keep dim for metadata if present)
                if "lead_time" in da_t.dims:
                    da_t = da_t.isel(lead_time=i, drop=True)
                if "lead_time" in da_p.dims:
                    da_p = da_p.isel(lead_time=i, drop=True)
                # Average over remaining time/init dims for stability
                reduce_dims = [d for d in ["time", "init_time", "ensemble"] if d in da_t.dims]
                if reduce_dims:
                    da_t = da_t.mean(dim=reduce_dims, skipna=True)
                reduce_dims_p = [d for d in ["time", "init_time", "ensemble"] if d in da_p.dims]
                if reduce_dims_p:
                    da_p = da_p.mean(dim=reduce_dims_p, skipna=True)
                Z_t.append(_eval_kde_1d(da_t))
                Z_p.append(_eval_kde_1d(da_p))

            X = np.asarray(lead_hours, dtype=float)
            Y = y_eval
            Z_t_arr = np.asarray(Z_t)
            Z_p_arr = np.asarray(Z_p)

            # Keep only: Ridgeline plot

            # 3: Ridgeline plot (joy plot) in 2D
            # Style reversal requested: fill = model densities, outline line = target densities.
            # Color meaning: Viridis gradient keyed by lead index (early → dark, late → bright).
            fig_r, ax_r = plt.subplots(figsize=(10, 6), dpi=dpi * 2, constrained_layout=True)
            offset = 1.05 * max(float(np.max(Z_t_arr)), float(np.max(Z_p_arr)))
            cmap = plt.get_cmap("viridis")
            for i, h in enumerate(X.tolist()):
                color = cmap(i / max(1, len(X) - 1))
                y_target = i * offset + Z_t_arr[i]
                y_model = i * offset + Z_p_arr[i]
                # Filled model ridge (fallback to line if fill_between not available in test stubs)
                if hasattr(ax_r, "fill_between"):
                    ax_r.fill_between(
                        Y, i * offset, y_model, color=color, alpha=0.55, linewidth=0.0
                    )
                else:
                    ax_r.plot(Y, y_model, color=color, lw=1.0)
                # Target outline as thin black line for contrast
                ax_r.plot(Y, y_target, color="black", lw=0.7)
                # Lead hour label
                ax_r.text(Y[-1] + (Y[1] - Y[0]) * 0.5, i * offset + 0.02, f"{int(h)}h", fontsize=8)
                if hasattr(ax_r, "set_yticks"):
                    ax_r.set_yticks([])
            ax_r.set_xlabel(f"{base_var} (standardized)")
            ax_r.set_title("Global KDE evolution — ridgeline (filled=model, line=target)")
            out_png_r = section_output / build_output_filename(
                metric="wd_kde_evolve",
                variable=base_var,
                level=None,
                qualifier="ridgeline",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token_base,
                ext="png",
            )
            if save_fig:
                fig_r.savefig(out_png_r, bbox_inches="tight", dpi=200)
                print(f"[wd_kde] saved {out_png_r}")
            plt.close(fig_r)
            if save_npz:
                out_npz = section_output / build_output_filename(
                    metric="wd_kde_evolve",
                    variable=base_var,
                    level=None,
                    qualifier="ridgeline_data",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token_base,
                    ext="npz",
                )
                np.savez(
                    out_npz,
                    lead_hours=X,
                    y_eval=Y,
                    density_target=Z_t_arr,
                    density_model=Z_p_arr,
                    variable=base_var,
                )
                print(f"[wd_kde] saved {out_npz}")

            # Removed: heatmaps, 3D curves, and NPZ bundle.

    if wasserstein_rows:
        import pandas as pd

        df_w = pd.DataFrame(wasserstein_rows)
        out_csv = section_output / build_output_filename(
            metric="wd_kde",
            variable=None,
            level=None,
            qualifier="wasserstein",
            init_time_range=None,
            lead_time_range=None,
            ensemble=ens_token_base,
            ext="csv",
        )
        df_w.to_csv(out_csv, index=False)
        print(f"[wd_kde] saved {out_csv}")

        # Removed: heatmaps, 3D curves, and NPZ bundle.
