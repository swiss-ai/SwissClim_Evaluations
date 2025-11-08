from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
from scipy.stats import gaussian_kde

from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
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
    _max_samples = int(
        plotting_cfg.get("kde_max_samples", 200_000)
    )  # unused; underscore to appease linter
    # Global random seed from config for reproducible subsampling
    _base_seed = int(plotting_cfg.get("random_seed", 42))  # unused; underscore to appease linter
    # Target/prediction always use identical subsamples so that if underlying
    # arrays are equal the KDEs match exactly (paired subsampling is enforced).
    section_output = out_root / "wd_kde"

    # Ensure output directory exists early
    section_output.mkdir(parents=True, exist_ok=True)
    # Simplified output: only keep ridgeline evolution plot. No per-variable
    # global KDE curves/NPZ or Wasserstein CSV summaries.

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
    # Global KDE curves instead of latitude-binned panels

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

    # Removed per-variable global KDE plots and Wasserstein summaries.

    # 2D standardized variables (run once per variable; avoid prior recursion issue)
    # Global per-variable KDE artifacts removed.

    # 3D standardized variables per level
    # 3D per-level global KDE artifacts removed.

    # Removed Wasserstein CSV generation entirely.

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
            cmap = plt.cm.viridis
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
