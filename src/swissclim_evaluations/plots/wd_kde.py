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
    outputs = plotting_cfg.get(
        "outputs", ["file"]
    )  # ["file", "cell", "cell-first"]
    dpi = int(plotting_cfg.get("dpi", 48))
    save_plot_data = bool(plotting_cfg.get("save_plot_data", False))
    section_output = out_root / "wd_kde"

    variables_2d = [
        v for v in ds_std.data_vars if "level" not in ds_std[v].dims
    ]
    lat_bins, n_bands, n_rows = _lat_bands()

    for i, variable_name in enumerate(variables_2d):
        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=dpi)
        w_distances: list[float] = []

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
            if save_plot_data:
                section_output.mkdir(parents=True, exist_ok=True)
                np.savez(
                    section_output
                    / f"{variable_name}_neg_{lat_min}_{lat_max}_kde.npz",
                    x_eval=x_eval,
                    kde_ds=kde_ds(x_eval),
                    kde_ml=kde_ml(x_eval),
                    wasserstein=w,
                )

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
            if save_plot_data:
                section_output.mkdir(parents=True, exist_ok=True)
                np.savez(
                    section_output
                    / f"{variable_name}_pos_{lat_min}_{lat_max}_kde.npz",
                    x_eval=x_eval,
                    kde_ds=kde_ds(x_eval),
                    kde_ml=kde_ml(x_eval),
                    wasserstein=w,
                )

        mean_w = float(np.mean(w_distances)) if w_distances else float("nan")
        plt.suptitle(
            f"Normalized Distribution of {variable_name} by latitude bands\nMean Wasserstein distance: {mean_w:.3f}",
            y=1.02,
        )
        plt.tight_layout()

        if "file" in outputs:
            section_output.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                section_output / f"{variable_name}_sfc_latbands_norm.png",
                bbox_inches="tight",
                dpi=200,
            )
        if (
            i == 0
            and "cell-first" in outputs
            and plt.get_backend().lower().find("agg") == -1
        ):
            plt.show()
        if i > 0 and "cell" not in outputs:
            plt.close(fig)
