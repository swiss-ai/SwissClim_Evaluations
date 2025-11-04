from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
from scipy.stats import gaussian_kde, wasserstein_distance

from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
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

    # Resolve ensemble handling (pooled/mean/members/none). Prob not allowed here.
    resolved_mode = resolve_ensemble_mode("wd_kde", ensemble_mode, ds_target_std, ds_prediction_std)
    has_ens = "ensemble" in ds_prediction_std.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for wd_kde")
    if resolved_mode == "none" and has_ens:
        # Force user to choose pooling semantics instead of implicitly ignoring ensemble
        raise ValueError(
            "ensemble_mode=none requested but 'ensemble' dimension present; "
            "choose mean|pooled|members"
        )

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
    elif resolved_mode in ("pooled", "none"):
        ds_target_std_eff = ds_target_std
        ds_prediction_std_eff = ds_prediction_std
        ens_token_base = (
            ensemble_mode_to_token("pooled") if (resolved_mode == "pooled" and has_ens) else None
        )
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
    ) -> None:
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
            axs[j, 1].plot(x_eval, kde_ds(x_eval), color="skyblue", label="Ground Truth")
            axs[j, 1].plot(x_eval, kde_ml(x_eval), color="salmon", label="Model Prediction")
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
            # Surface variable, no level dim expected
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
            axs[j, 0].plot(x_eval, kde_ds(x_eval), color="skyblue", label="Ground Truth")
            axs[j, 0].plot(x_eval, kde_ml(x_eval), color="salmon", label="Model Prediction")
            axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}° (W-dist: {w:.3f})")
            axs[j, 0].legend()
            if save_npz:
                combined["pos_x"].append(x_eval)
                combined["pos_kde_ds"].append(kde_ds(x_eval))
                combined["pos_kde_ml"].append(kde_ml(x_eval))
                combined["pos_lat_min"].append(float(lat_min))
                combined["pos_lat_max"].append(float(lat_max))

        mean_w = float(np.mean(w_distances)) if w_distances else float("nan")
        plt.suptitle(
            "Normalized Distribution of "
            f"{var_name} ({level_token}) by latitude bands\nMean Wasserstein distance: "
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
                qualifier="plot",
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
                qualifier="combined",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token,
                ext="npz",
            )
            np.savez(
                out_npz,
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
                    level_token="",
                    ens_token=token_m,
                )
    else:
        for variable_name in variables_2d:
            _process_variable(
                str(variable_name),
                ds_target_std_eff[variable_name],
                ds_prediction_std_eff[variable_name],
                level_token="",
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
        # Try to include init_time range if present to align with naming helpers

        def _extract_init_range(ds: xr.Dataset):
            if "init_time" not in ds:
                return None
            try:
                vals = ds["init_time"].values
                if vals.size == 0:
                    return None
                start = np.datetime64(vals.min()).astype("datetime64[h]")
                end = np.datetime64(vals.max()).astype("datetime64[h]")

                def _fmt(x):
                    return (
                        np.datetime_as_string(x, unit="h")
                        .replace("-", "")
                        .replace(":", "")
                        .replace("T", "")
                    )

                return (_fmt(start), _fmt(end))
            except Exception:
                return None

        init_range_csv = _extract_init_range(ds_prediction_std_eff)
        out_csv = section_output / build_output_filename(
            metric="wd_kde_wasserstein",
            variable=None,  # aggregate across variables -> omit variable token
            level=None,
            qualifier=None,
            init_time_range=init_range_csv,
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

    # Optional: Global KDE evolution over lead_time (3D perspective)
    try:
        evolve_flag = bool((plotting_cfg or {}).get("wd_kde_global_evolution", False))
    except Exception:
        evolve_flag = False
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
            try:
                # Draw a coarse subsample to set robust evaluation range
                da_t_all = ds_target_std_eff[base_var]
                da_p_all = ds_prediction_std_eff[base_var]
                # Collapse spatial + time to estimate global min/max quickly
                q_t = da_t_all.quantile([0.001, 0.999], skipna=True).compute().values
                q_p = da_p_all.quantile([0.001, 0.999], skipna=True).compute().values
                vmin = float(min(q_t[0], q_p[0]))
                vmax = float(max(q_t[1], q_p[1]))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = -3.0, 3.0
            except Exception:
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
                try:
                    kde = gaussian_kde(arr)
                    return kde(y_eval)
                except Exception:
                    return np.zeros_like(y_eval)

            for i, lt in enumerate(leads):
                try:
                    hours = int(np.timedelta64(lt) / np.timedelta64(1, "h"))
                except Exception:
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

            # Normalize z-scale for comparable panels
            zmax = float(max(Z_t_arr.max(initial=0.0), Z_p_arr.max(initial=0.0))) or 1.0

            def _plot_surface(Z: np.ndarray, qualifier: str) -> None:
                fig = plt.figure(figsize=(12, 7), dpi=dpi * 2)
                ax = fig.add_subplot(111, projection="3d")
                # Create a surface by expanding X/Y grids
                Xg, Yg = np.meshgrid(X, Y, indexing="ij")
                ax.plot_surface(Xg, Yg, Z, cmap="viridis", linewidth=0, antialiased=True)
                ax.set_xlabel("lead_time (h)")
                ax.set_ylabel(f"{base_var} (standardized)")
                ax.set_zlabel("density")
                ax.set_zlim(0.0, zmax * 1.05)
                out_png = section_output / build_output_filename(
                    metric="wd_kde_evolve",
                    variable=base_var,
                    level=None,
                    qualifier=qualifier,
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token_base,
                    ext="png",
                )
                plt.tight_layout()
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[wd_kde] saved {out_png}")
                plt.close(fig)

            # Plot target and model
            _plot_surface(Z_t_arr, "target")
            _plot_surface(Z_p_arr, "model")

            # Save NPZ bundle for programmatic use
            out_npz = section_output / build_output_filename(
                metric="wd_kde_evolve",
                variable=base_var,
                level=None,
                qualifier="bundle",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token_base,
                ext="npz",
            )
            np.savez(
                out_npz,
                lead_hours=X,
                value_axis=Y,
                density_target=Z_t_arr,
                density_model=Z_p_arr,
                variable=base_var,
            )
            print(f"[wd_kde] saved {out_npz}")
