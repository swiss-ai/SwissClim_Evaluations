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


def _lat_bands():
    lat_bins = np.arange(-90, 91, 10)
    n_bands = len(lat_bins) - 1
    n_rows = n_bands // 2
    return lat_bins, n_bands, n_rows


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
    # Collect all Wasserstein distances across variables and latitude bands
    # Each row: variable, hemisphere, lat_min, lat_max, wasserstein
    wasserstein_rows: list[dict[str, float | str]] = []

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
    lat_bins, n_bands, n_rows = _lat_bands()

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

    # After processing all variables, write Wasserstein distances CSV summary
    if resolved_mode == "members" and has_ens:
        # No aggregated Wasserstein summary when per-member artifacts produced (could add later)
        wasserstein_rows = []

    ens_token = ens_token_base
    if resolved_mode == "members" and has_ens:
        ens_token = None  # per-member tokens used inside loops already

    if wasserstein_rows:
        import pandas as _pd

        df_w = _pd.DataFrame(wasserstein_rows)
        out_csv = section_output / build_output_filename(
            metric="wd_kde_wasserstein",
            variable=None,  # aggregate across variables -> omit variable token
            level=None,
            qualifier="averaged",
            init_time_range=None,
            lead_time_range=None,
            ensemble=ens_token,
            ext="csv",
        )
        df_w.to_csv(out_csv, index=False)
        print(f"[wd_kde] saved {out_csv}")
    else:
        # Still emit an empty CSV to satisfy expectations
        import pandas as _pd

        out_csv = section_output / build_output_filename(
            metric="wd_kde_wasserstein",
            variable=None,
            level=None,
            qualifier="averaged",
            init_time_range=None,
            lead_time_range=None,
            ensemble=ens_token,
            ext="csv",
        )
        _pd.DataFrame(
            columns=[
                "variable",
                "hemisphere",
                "lat_min",
                "lat_max",
                "wasserstein",
            ]
        ).to_csv(out_csv, index=False)
        print("[wd_kde] WARNING: No Wasserstein rows collected; emitted empty CSV")
