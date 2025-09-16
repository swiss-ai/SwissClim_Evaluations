from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde, wasserstein_distance


def _lat_bands():
    lat_bins = np.arange(-90, 91, 10)
    n_bands = len(lat_bins) - 1
    n_rows = n_bands // 2
    return lat_bins, n_bands, n_rows


def run(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    ds_std: xr.Dataset,
    ds_ml_std: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    section_output = out_root / "wd_kde"

    # Select only genuine 2D variables (no 'level' dimension)
    variables_2d = [
        v for v in ds_std.data_vars if "level" not in ds_std[v].dims
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

            ds_slice = ds_std[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            ds_ml_slice = ds_ml_std[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            # Surface variable, no level dim expected
            if ds_slice.size == 0 or ds_ml_slice.size == 0:
                axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue
            ds_flat = ds_slice.values.flatten()
            ml_flat = ds_ml_slice.values.flatten()
            w = wasserstein_distance(ds_flat, ml_flat)
            w_distances.append(w)
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
            ds_slice = ds_std[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            ds_ml_slice = ds_ml_std[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            # Surface variable, no level dim expected
            if ds_slice.size == 0 or ds_ml_slice.size == 0:
                axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}° (No data)")
                continue
            ds_flat = ds_slice.values.flatten()
            ml_flat = ds_ml_slice.values.flatten()
            w = wasserstein_distance(ds_flat, ml_flat)
            w_distances.append(w)
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
