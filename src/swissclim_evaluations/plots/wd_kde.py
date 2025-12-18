from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde, wasserstein_distance

from ..dask_utils import compute_jobs, to_finite_array
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
    lat_bins = np.arange(-90, 91, 10)
    n_bands = len(lat_bins) - 1
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
        print(f"[wd_kde] variable: {var_name} level={level_token}")

        # Collect all jobs
        jobs = []

        # 1. Global KDE job
        seed_g = base_seed + (hash(var_name + level_token) % 1000) * 1000
        global_job = {
            "type": "global",
            "sub_t_lazy": subsample_values(da_t_std, max_samples, seed_g, lazy=True),
            "sub_p_lazy": subsample_values(da_p_std, max_samples, seed_g, lazy=True),
        }
        jobs.append(global_job)

        # 2. Lat band jobs (if needed)
        if per_lat_band:
            # Negative latitudes (right column)
            for j in range(n_bands // 2):
                lat_max = lat_bins[j]
                lat_min = lat_bins[j + 1]
                da_target_slice = da_t_std.sel(latitude=slice(lat_min, lat_max))
                da_prediction_slice = da_p_std.sel(latitude=slice(lat_min, lat_max))

                seed = base_seed + (hash(var_name + level_token) % 1000) * 1000 + (j + 1) * 10 + 1

                job = {
                    "type": "lat_neg",
                    "j": j,
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "sub_t_lazy": subsample_values(da_target_slice, max_samples, seed, lazy=True),
                    "sub_p_lazy": subsample_values(
                        da_prediction_slice, max_samples, seed, lazy=True
                    ),
                }
                jobs.append(job)

            # Positive latitudes (left column)
            for j in range(n_bands // 2):
                idx = -(j + 1)
                lat_max = lat_bins[idx - 1]
                lat_min = lat_bins[idx]
                da_target_slice = da_t_std.sel(latitude=slice(lat_min, lat_max))
                da_prediction_slice = da_p_std.sel(latitude=slice(lat_min, lat_max))

                seed = base_seed + (hash(var_name + level_token) % 1000) * 1000 + (j + 1) * 10 + 2

                job = {
                    "type": "lat_pos",
                    "j": j,
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "sub_t_lazy": subsample_values(da_target_slice, max_samples, seed, lazy=True),
                    "sub_p_lazy": subsample_values(
                        da_prediction_slice, max_samples, seed, lazy=True
                    ),
                }
                jobs.append(job)

        # Compute all
        compute_jobs(
            jobs,
            key_map={"sub_t_lazy": "val_t", "sub_p_lazy": "val_p"},
            post_process={"val_t": to_finite_array, "val_p": to_finite_array},
        )

        # --- Process Global KDE ---
        global_job = jobs[0]
        ds_flat_g = cast(np.ndarray, global_job["val_t"])
        prediction_flat_g = cast(np.ndarray, global_job["val_p"])
        units = get_variable_units(ds_target, var_name)

        if ds_flat_g.size > 0 and prediction_flat_g.size > 0:
            w_g = wasserstein_distance(ds_flat_g, prediction_flat_g)
            kde_ds_g = gaussian_kde(ds_flat_g)
            kde_prediction_g = gaussian_kde(prediction_flat_g)

            # Plot Global
            fig_g, ax_g = plt.subplots(figsize=(10, 6), dpi=dpi)
            x_eval_g = np.linspace(
                min(ds_flat_g.min(), prediction_flat_g.min()),
                max(ds_flat_g.max(), prediction_flat_g.max()),
                100,
            )
            ax_g.plot(x_eval_g, kde_ds_g(x_eval_g), color=COLOR_GROUND_TRUTH, label="Target")
            ax_g.plot(
                x_eval_g,
                kde_prediction_g(x_eval_g),
                color=COLOR_MODEL_PREDICTION,
                label="Prediction",
            )

            # Check for single date
            date_str = extract_date_from_dataset(da_t_std)
            lev_part = format_level_label(level_val if level_val is not None else level_token)

            if units:
                ax_g.set_xlabel(f"{format_variable_name(var_name)} [{units}]")
            else:
                ax_g.set_xlabel(f"{format_variable_name(var_name)}")

            ax_g.set_title(
                f"Global Normalized KDE — {format_variable_name(variable_name)}"
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
                save_figure(fig_g, out_png_g)

            if save_npz:
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
                save_data(
                    out_npz_g,
                    w_dist=w_g,
                    x=x_eval_g,
                    kde_ds=kde_ds_g(x_eval_g),
                    kde_prediction=kde_prediction_g(x_eval_g),
                    units=units,
                    allow_pickle=True,
                )
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
            "neg_kde_prediction": [],
            "neg_lat_min": [],
            "neg_lat_max": [],
            "pos_x": [],
            "pos_kde_ds": [],
            "pos_kde_prediction": [],
            "pos_lat_min": [],
            "pos_lat_max": [],
        }

        # Process lat band results
        for job in jobs[1:]:
            ds_flat = job["val_t"]
            prediction_flat = job["val_p"]
            lat_min = job["lat_min"]
            lat_max = job["lat_max"]
            j = job["j"]

            if job["type"] == "lat_neg":
                ax = axs[j, 1]
                key_prefix = "neg"
            else:
                ax = axs[j, 0]
                key_prefix = "pos"

            if ds_flat.size == 0 or prediction_flat.size == 0:
                ax.set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue

            w = wasserstein_distance(ds_flat, prediction_flat)
            w_distances.append(w)
            wasserstein_rows.append(
                {
                    "variable": var_name,
                    "hemisphere": "south" if job["type"] == "lat_neg" else "north",
                    "lat_min": float(lat_min),
                    "lat_max": float(lat_max),
                    "wasserstein": float(w),
                }
            )
            kde_ds = gaussian_kde(ds_flat)
            kde_prediction = gaussian_kde(prediction_flat)
            x_eval = np.linspace(
                min(ds_flat.min(), prediction_flat.min()),
                max(ds_flat.max(), prediction_flat.max()),
                100,
            )
            ax.plot(x_eval, kde_ds(x_eval), color=COLOR_GROUND_TRUTH, label="Target")
            ax.plot(
                x_eval, kde_prediction(x_eval), color=COLOR_MODEL_PREDICTION, label="Prediction"
            )
            if units:
                ax.set_xlabel(f"{format_variable_name(var_name)} [{units}]")
            else:
                ax.set_xlabel(f"{format_variable_name(var_name)}")
            ax.set_title(f"Lat {lat_min}° to {lat_max}° (W-dist: {w:.3f})")
            ax.legend()
            if save_npz:
                combined[f"{key_prefix}_x"].append(x_eval)
                combined[f"{key_prefix}_kde_ds"].append(kde_ds(x_eval))
                combined[f"{key_prefix}_kde_prediction"].append(kde_prediction(x_eval))
                combined[f"{key_prefix}_lat_min"].append(float(lat_min))
                combined[f"{key_prefix}_lat_max"].append(float(lat_max))

        mean_w = float(np.mean(w_distances)) if w_distances else float("nan")

        # Check for single date
        date_str = extract_date_from_dataset(da_t_std)

        plt.suptitle(
            "Normalized Distribution of "
            f"{format_variable_name(var_name)} ({level_token}) by Latitude Bands{date_str}\n"
            f"Mean Wasserstein distance: {mean_w:.3f}",
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
            save_figure(fig, out_png)
        if save_npz:
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
            save_data(
                out_npz,
                mean_w=mean_w,
                neg_x=np.array(combined["neg_x"], dtype=object),
                neg_kde_ds=np.array(combined["neg_kde_ds"], dtype=object),
                neg_kde_prediction=np.array(combined["neg_kde_prediction"], dtype=object),
                neg_lat_min=np.array(combined["neg_lat_min"]),
                neg_lat_max=np.array(combined["neg_lat_max"]),
                pos_x=np.array(combined["pos_x"], dtype=object),
                pos_kde_ds=np.array(combined["pos_kde_ds"], dtype=object),
                pos_kde_prediction=np.array(combined["pos_kde_prediction"], dtype=object),
                pos_lat_min=np.array(combined["pos_lat_min"]),
                pos_lat_max=np.array(combined["pos_lat_max"]),
                allow_pickle=True,
            )
        plt.close(fig)

    # 2D standardized variables (run once per variable; avoid prior recursion issue)
    has_multi_lead = (
        "lead_time" in ds_prediction_std_eff.dims and ds_prediction_std_eff.sizes["lead_time"] > 1
    )

    if not has_multi_lead:
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
    if process_3d and not has_multi_lead:
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
    if has_multi_lead:
        # Choose all eligible variables (2D and 3D)
        cand_vars = [
            v
            for v in ds_prediction_std_eff.data_vars
            if "level" not in ds_prediction_std_eff[v].dims
        ]
        if process_3d:
            cand_vars.extend(
                [
                    v
                    for v in ds_prediction_std_eff.data_vars
                    if "level" in ds_prediction_std_eff[v].dims
                ]
            )

        for base_var in cand_vars:
            # Handle 3D variables by iterating over levels
            levels = [None]
            if "level" in ds_prediction_std_eff[base_var].dims:
                levels = ds_prediction_std_eff[base_var].level.values

            for lvl in levels:
                # Common evaluation axis from combined sample across all leads (paired with targets)
                # Draw a coarse subsample to set robust evaluation range
                da_t_all = ds_target_std_eff[base_var]
                da_p_all = ds_prediction_std_eff[base_var]

                if lvl is not None:
                    da_t_all = da_t_all.sel(level=lvl)
                    da_p_all = da_p_all.sel(level=lvl)

                # Collapse spatial + time to estimate global min/max quickly
                q_t_lazy = da_t_all.quantile([0.001, 0.999], skipna=True)
                q_p_lazy = da_p_all.quantile([0.001, 0.999], skipna=True)

                # Build densities per lead for target and model
                leads = list(ds_prediction_std_eff["lead_time"].values)
                lead_hours = []

                jobs = []

                # Quantile job
                jobs.append({"type": "quantile", "q_t_lazy": q_t_lazy, "q_p_lazy": q_p_lazy})

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

                    if lvl is not None:
                        da_t = da_t.sel(level=lvl)
                        da_p = da_p.sel(level=lvl)

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

                    jobs.append({"type": "lead", "i": i, "da_t_lazy": da_t, "da_p_lazy": da_p})

            # Compute all
            compute_jobs(
                jobs,
                key_map={
                    "q_t_lazy": "q_t",
                    "q_p_lazy": "q_p",
                    "da_t_lazy": "da_t",
                    "da_p_lazy": "da_p",
                },
            )

            # Process quantiles
            q_t = jobs[0]["q_t"]
            q_p = jobs[0]["q_p"]

            vmin = float(min(q_t[0], q_p[0]))
            vmax = float(max(q_t[1], q_p[1]))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                vmin, vmax = -3.0, 3.0
            y_eval = np.linspace(vmin, vmax, 200)

            Z_t = []
            Z_p = []

            def _eval_kde_from_array(arr: np.ndarray, y_eval_local: np.ndarray) -> np.ndarray:
                arr = arr.ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size < 10:
                    return np.zeros_like(y_eval_local)
                kde = gaussian_kde(arr)
                return kde(y_eval_local)

            # Process leads
            for job in jobs[1:]:
                arr_t = np.asarray(job["da_t"])
                arr_p = np.asarray(job["da_p"])

                Z_t.append(_eval_kde_from_array(arr_t, y_eval))
                Z_p.append(_eval_kde_from_array(arr_p, y_eval))

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

            # Add level info to title if applicable
            level_str = f" (level {lvl})" if lvl is not None else ""
            ax_r.set_xlabel(f"{format_variable_name(base_var)} (standardized)")
            ax_r.set_title(
                f"KDE Evolution (Ridgeline): {format_variable_name(base_var)}{level_str}"
            )

            if save_fig:
                qual = f"ridgeline{'_level' + str(lvl) if lvl is not None else ''}"
                out_png = section_output / build_output_filename(
                    metric="wd_kde_evolve",
                    variable=base_var,
                    level="surface",
                    qualifier=qual,
                    ensemble=ensemble_mode_to_token(ensemble_mode),
                    ext="png",
                )
                save_figure(fig_r, out_png)
            else:
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
                save_data(
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
            qualifier="wasserstein_averaged",
            init_time_range=None,
            lead_time_range=None,
            ensemble=ens_token_base,
            ext="csv",
        )
        df_w.to_csv(out_csv, index=False)
        print(f"[wd_kde] saved {out_csv}")
