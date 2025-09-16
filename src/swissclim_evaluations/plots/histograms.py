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
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    section_output = out_root / "histograms"

    # Select only genuine 2D variables (no 'level' dimension)
    variables_2d = [
        v for v in ds_target.data_vars if "level" not in ds_target[v].dims
    ]
    if not variables_2d:
        print("[histograms] No 2D variables found – skipping.")
        return
    print(f"[histograms] Processing {len(variables_2d)} 2D variables.")
    lat_bins, n_bands, n_rows = _lat_bands()

    for i, variable_name in enumerate(variables_2d):
        print(f"[histograms] variable: {variable_name}")
        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=dpi)
        # Collect combined NPZ data across all bands
        combined = {
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
            da_true = ds_target[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            da_pred = ds_prediction[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            # Surface variable, no level dim expected
            data_ds = da_true.values.flatten()
            data_ds_ml = da_pred.values.flatten()
            counts_ds, bins_ds, _ = axs[j, 1].hist(
                data_ds,
                bins=1000,
                density=True,
                alpha=0.5,
                color="skyblue",
                label="Ground Truth",
            )
            counts_ml, bins_ml, _ = axs[j, 1].hist(
                data_ds_ml,
                bins=1000,
                density=True,
                alpha=0.5,
                color="salmon",
                label="Model Prediction",
            )
            axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}°")
            axs[j, 1].legend(loc="upper right")
            if save_npz:
                combined["neg_counts"].append((counts_ds, counts_ml))
                combined["neg_bins"].append((bins_ds, bins_ml))
                combined["neg_lat_min"].append(float(lat_min))
                combined["neg_lat_max"].append(float(lat_max))

        # Positive latitudes (left column)
        for j in range(n_bands // 2):
            idx = -(j + 1)
            lat_max = lat_bins[idx - 1]
            lat_min = lat_bins[idx]
            da_true = ds_target[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            da_pred = ds_prediction[variable_name].sel(
                latitude=slice(lat_min, lat_max)
            )
            # Surface variable, no level dim expected
            data_ds = da_true.values.flatten()
            data_ds_ml = da_pred.values.flatten()
            counts_ds, bins_ds, _ = axs[j, 0].hist(
                data_ds,
                bins=1000,
                density=True,
                alpha=0.5,
                color="skyblue",
                label="Ground Truth",
            )
            counts_ml, bins_ml, _ = axs[j, 0].hist(
                data_ds_ml,
                bins=1000,
                density=True,
                alpha=0.5,
                color="salmon",
                label="Model Prediction",
            )
            axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}°")
            axs[j, 0].legend(loc="upper right")
            if save_npz:
                combined["pos_counts"].append((counts_ds, counts_ml))
                combined["pos_bins"].append((bins_ds, bins_ml))
                combined["pos_lat_min"].append(float(lat_min))
                combined["pos_lat_max"].append(float(lat_max))

        units = ds_target[variable_name].attrs.get("units", "")
        plt.suptitle(
            f"Distribution of {variable_name} ({units}) by latitude bands",
            y=1.02,
        )
        plt.tight_layout()

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png = section_output / f"{variable_name}_sfc_latbands.png"
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"[histograms] saved {out_png}")
        if save_npz:
            # Write one combined NPZ with all band histograms for this variable
            # Convert list of tuples to stacked arrays for easier downstream use
            def _stack_counts_bins(pairs):
                counts = [p[0] for p in pairs]
                bins = [p[1] for p in pairs]
                # counts arrays have equal length (bins-1); bins arrays may have equal length
                return np.stack(counts, axis=0), np.stack(bins, axis=0)

            neg_counts, neg_bins_arr = (
                _stack_counts_bins(combined["neg_counts"])
                if combined["neg_counts"]
                else (np.empty((0,)), np.empty((0,)))
            )
            pos_counts, pos_bins_arr = (
                _stack_counts_bins(combined["pos_counts"])
                if combined["pos_counts"]
                else (np.empty((0,)), np.empty((0,)))
            )

            # Note: counts are tuples (ds, ml). Split into two arrays if present
            def _split_counts(counts_tuple_stack):
                if counts_tuple_stack.size == 0:
                    return np.empty((0,)), np.empty((0,))
                # counts_tuple_stack is shape (bands, 2, N) if stacked correctly; ensure dims
                return counts_tuple_stack[:, 0, :], counts_tuple_stack[:, 1, :]

            def _split_bins(bins_tuple_stack):
                if bins_tuple_stack.size == 0:
                    return np.empty((0,)), np.empty((0,))

                return bins_tuple_stack[:, 0, :], bins_tuple_stack[:, 1, :]

            # The _stack_counts_bins returns stacks of objects; to keep it simple, store ragged lists via allow_pickle
            section_output.mkdir(parents=True, exist_ok=True)
            out_npz = (
                section_output / f"{variable_name}_sfc_latbands_combined.npz"
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
                allow_pickle=True,
            )
            print(f"[histograms] saved {out_npz}")
        plt.close(fig)
