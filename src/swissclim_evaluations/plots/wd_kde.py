from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde, wasserstein_distance


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

    # Collect all Wasserstein distances across variables and latitude bands
    # Each row: variable, hemisphere, lat_min, lat_max, wasserstein
    wasserstein_rows: list[dict[str, float | str]] = []

    # Select only genuine 2D variables (no 'level' dimension)
    variables_2d = [
        v
        for v in ds_target_std.data_vars
        if "level" not in ds_target_std[v].dims
    ]
    if not variables_2d:
        print("[wd_kde] No 2D variables found – skipping.")
        return
    print(f"[wd_kde] Processing {len(variables_2d)} 2D variables.")
    lat_bins, n_bands, n_rows = _lat_bands()

    for i, variable_name in enumerate(variables_2d):
        print(f"[wd_kde] variable: {variable_name}")
        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=dpi)
        w_distances: list[float] = []
        combined = {
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

            da_target_slice = ds_target_std[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            da_prediction_slice = ds_prediction_std[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            # Surface variable, no level dim expected
            if da_target_slice.size == 0 or da_prediction_slice.size == 0:
                axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue

            # Subsample up to k points by selecting a small slice along each dimension,
            # then materialize only that subset.
            def _subsample_values(
                da: xr.DataArray, k: int, seed: int
            ) -> np.ndarray:
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
                indexers: dict[str, np.ndarray] = {}
                for d in dims:
                    n = int(da.sizes.get(d, 1))
                    take = max(1, int(np.ceil(frac * n)))
                    take = min(take, n)
                    idx = rng.choice(n, size=take, replace=False)
                    idx.sort()
                    indexers[d] = idx
                sub = da.isel(**indexers)
                # Materialize the small subset and keep only finite values (avoid boolean dask indexing)
                arr = np.asarray(sub.compute().values).ravel()
                return arr[np.isfinite(arr)]

            # Derive deterministic seeds per variable/band/hemisphere
            seed = base_seed + (i + 1) * 1000 + (j + 1) * 10 + 1
            ds_flat = _subsample_values(da_target_slice, max_samples, seed=seed)
            ml_flat = _subsample_values(
                da_prediction_slice, max_samples, seed=seed
            )
            w = wasserstein_distance(ds_flat, ml_flat)
            w_distances.append(w)
            # record row for CSV
            wasserstein_rows.append({
                "variable": variable_name,
                "hemisphere": "south",
                "lat_min": float(lat_min),
                "lat_max": float(lat_max),
                "wasserstein": float(w),
            })
            kde_ds = gaussian_kde(ds_flat)
            kde_ml = gaussian_kde(ml_flat)
            x_eval = np.linspace(
                min(ds_flat.min(), ml_flat.min()),
                max(ds_flat.max(), ml_flat.max()),
                100,
            )
            axs[j, 1].plot(
                x_eval,
                kde_ds(x_eval),
                color="skyblue",
                label="Ground Truth",
            )
            axs[j, 1].plot(
                x_eval,
                kde_ml(x_eval),
                color="salmon",
                label="Model Prediction",
            )
            axs[j, 1].set_title(
                f"Lat {lat_min}° to {lat_max}° (W-dist: {w:.3f})"
            )
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
            da_target_slice = ds_target_std[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            da_prediction_slice = ds_prediction_std[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            # Surface variable, no level dim expected
            if da_target_slice.size == 0 or da_prediction_slice.size == 0:
                axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue
            seed = base_seed + (i + 1) * 1000 + (j + 1) * 10 + 2
            ds_flat = _subsample_values(da_target_slice, max_samples, seed=seed)
            ml_flat = _subsample_values(
                da_prediction_slice, max_samples, seed=seed
            )
            w = wasserstein_distance(ds_flat, ml_flat)
            w_distances.append(w)
            # record row for CSV
            wasserstein_rows.append({
                "variable": variable_name,
                "hemisphere": "north",
                "lat_min": float(lat_min),
                "lat_max": float(lat_max),
                "wasserstein": float(w),
            })
            kde_ds = gaussian_kde(ds_flat)
            kde_ml = gaussian_kde(ml_flat)
            x_eval = np.linspace(
                min(ds_flat.min(), ml_flat.min()),
                max(ds_flat.max(), ml_flat.max()),
                100,
            )
            axs[j, 0].plot(
                x_eval,
                kde_ds(x_eval),
                color="skyblue",
                label="Ground Truth",
            )
            axs[j, 0].plot(
                x_eval,
                kde_ml(x_eval),
                color="salmon",
                label="Model Prediction",
            )
            axs[j, 0].set_title(
                f"Lat {lat_min}° to {lat_max}° (W-dist: {w:.3f})"
            )
            axs[j, 0].legend()
            if save_npz:
                combined["pos_x"].append(x_eval)
                combined["pos_kde_ds"].append(kde_ds(x_eval))
                combined["pos_kde_ml"].append(kde_ml(x_eval))
                combined["pos_lat_min"].append(float(lat_min))
                combined["pos_lat_max"].append(float(lat_max))

        mean_w = float(np.mean(w_distances)) if w_distances else float("nan")
        plt.suptitle(
            f"Normalized Distribution of {variable_name} by latitude bands\nMean Wasserstein distance: {mean_w:.3f}",
            y=1.02,
        )
        plt.tight_layout()

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png = section_output / f"{variable_name}_sfc_latbands_norm.png"
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"[wd_kde] saved {out_png}")
        if save_npz:
            section_output.mkdir(parents=True, exist_ok=True)
            out_npz = (
                section_output
                / f"{variable_name}_sfc_latbands_kde_combined.npz"
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
                allow_pickle=True,
            )
            print(f"[wd_kde] saved {out_npz}")
        plt.close(fig)

    # After processing all variables, write all collected Wasserstein distances to CSV
    try:
        if wasserstein_rows:
            section_output.mkdir(parents=True, exist_ok=True)
            df_w = pd.DataFrame(wasserstein_rows)
            # Sort for readability
            df_w = df_w.sort_values(
                by=["variable", "hemisphere", "lat_min"]
            ).reset_index(drop=True)
            out_csv = section_output / "wasserstein_latbands.csv"
            df_w.to_csv(out_csv, index=False)
            print(f"[wd_kde] saved {out_csv}")
        else:
            print(
                "[wd_kde] No Wasserstein distances computed – CSV not written."
            )
    except Exception as e:
        print(f"[wd_kde] Failed to write Wasserstein CSV: {e}")
