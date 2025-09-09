from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _lat_bands() -> tuple[np.ndarray, int, int]:
    lat_bins = np.arange(-90, 91, 10)
    n_bands = len(lat_bins) - 1
    n_rows = n_bands // 2
    return lat_bins, n_bands, n_rows


def run(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    outputs = plotting_cfg.get(
        "outputs", ["file"]
    )  # ["file", "cell", "cell-first"]
    dpi = int(plotting_cfg.get("dpi", 48))
    save_plot_data = bool(plotting_cfg.get("save_plot_data", False))
    section_output = out_root / "histograms"

    variables_2d = [v for v in ds.data_vars if "level" not in ds[v].dims]
    lat_bins, n_bands, n_rows = _lat_bands()

    for i, variable_name in enumerate(variables_2d):
        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=dpi)

        # Negative latitudes (right column)
        for j in range(n_bands // 2):
            lat_max = lat_bins[j]
            lat_min = lat_bins[j + 1]
            data_ds = (
                ds[variable_name]
                .sel(latitude=slice(lat_min, lat_max))
                .values.flatten()
            )
            data_ds_ml = (
                ds_ml[variable_name]
                .sel(latitude=slice(lat_min, lat_max))
                .values.flatten()
            )
            counts_ds, bins_ds, _ = axs[j, 1].hist(
                data_ds,
                bins=1000,
                density=True,
                alpha=0.5,
                color="skyblue",
                label="Ground Truth - ERA5",
            )
            counts_ml, bins_ml, _ = axs[j, 1].hist(
                data_ds_ml,
                bins=1000,
                density=True,
                alpha=0.5,
                color="salmon",
                label="Model Prediction - SwissAI",
            )
            axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}°")
            axs[j, 1].legend()
            if save_plot_data:
                section_output.mkdir(parents=True, exist_ok=True)
                np.savez(
                    section_output
                    / f"{variable_name}_neg_{lat_min}_{lat_max}_hist.npz",
                    counts_ds=counts_ds,
                    bins_ds=bins_ds,
                    counts_ml=counts_ml,
                    bins_ml=bins_ml,
                )

        # Positive latitudes (left column)
        for j in range(n_bands // 2):
            idx = -(j + 1)
            lat_max = lat_bins[idx - 1]
            lat_min = lat_bins[idx]
            data_ds = (
                ds[variable_name]
                .sel(latitude=slice(lat_min, lat_max))
                .values.flatten()
            )
            data_ds_ml = (
                ds_ml[variable_name]
                .sel(latitude=slice(lat_min, lat_max))
                .values.flatten()
            )
            counts_ds, bins_ds, _ = axs[j, 0].hist(
                data_ds,
                bins=1000,
                density=True,
                alpha=0.5,
                color="skyblue",
                label="Ground Truth - ERA5",
            )
            counts_ml, bins_ml, _ = axs[j, 0].hist(
                data_ds_ml,
                bins=1000,
                density=True,
                alpha=0.5,
                color="salmon",
                label="Model Prediction - SwissAI",
            )
            axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}°")
            axs[j, 0].legend()
            if save_plot_data:
                section_output.mkdir(parents=True, exist_ok=True)
                np.savez(
                    section_output
                    / f"{variable_name}_pos_{lat_min}_{lat_max}_hist.npz",
                    counts_ds=counts_ds,
                    bins_ds=bins_ds,
                    counts_ml=counts_ml,
                    bins_ml=bins_ml,
                )

        units = ds[variable_name].attrs.get("units", "")
        plt.suptitle(
            f"Distribution of {variable_name} ({units}) by latitude bands",
            y=1.02,
        )
        plt.tight_layout()

        if "file" in outputs:
            section_output.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                section_output / f"{variable_name}_sfc_latbands.png",
                bbox_inches="tight",
                dpi=200,
            )
        if i == 0 and "cell-first" in outputs:
            plt.show()
        if i > 0 and "cell" not in outputs:
            plt.close(fig)
