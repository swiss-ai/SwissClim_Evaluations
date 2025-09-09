from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _lat_bands():
    import numpy as np

    lat_bins = np.arange(-90, 91, 10)
    n_bands = len(lat_bins) - 1
    return lat_bins, n_bands


def run(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    outputs = plotting_cfg.get(
        "outputs", ["file"]
    )  # ["file", "cell", "cell-first"]
    dpi = int(plotting_cfg.get("dpi", 48))
    save_plot_data = bool(plotting_cfg.get("save_plot_data", False))
    section_output = out_root / "vertical_profiles"

    variables_3d = [v for v in ds.data_vars if "level" in ds[v].dims]
    levels = select_cfg.get("levels") or list(ds.coords.get("level", []))
    lat_bins, n_bands = _lat_bands()

    fig_count = 0
    for var in variables_3d:
        n_cols = 2
        fig, axes = plt.subplots(
            n_cols, n_bands // 2, figsize=(24, 10), dpi=dpi * 2, sharey=True
        )

        global_min = float("inf")
        global_max = float("-inf")

        # Determine global x-range
        for i in range(n_bands // 2):
            lat_min_neg = lat_bins[i]
            lat_max_neg = lat_bins[i + 1]
            lat_slice_neg = slice(lat_max_neg, lat_min_neg)
            data_ds_neg = ds[var].sel(latitude=lat_slice_neg)
            data_ds_ml_neg = ds_ml[var].sel(latitude=lat_slice_neg)
            rel_err_neg = ((data_ds_ml_neg - data_ds_neg) / data_ds_neg).mean(
                dim=[
                    d
                    for d in ["time", "latitude", "longitude"]
                    if d in data_ds_neg.dims
                ],
                skipna=True,
            ) * 100
            finite_mask = np.isfinite(rel_err_neg)
            if finite_mask.any():
                global_min = min(
                    global_min, float(rel_err_neg.where(finite_mask).min())
                )
                global_max = max(
                    global_max, float(rel_err_neg.where(finite_mask).max())
                )

            idx = -(i + 1)
            lat_min_pos = lat_bins[idx]
            lat_max_pos = lat_bins[idx - 1]
            lat_slice_pos = slice(lat_min_pos, lat_max_pos)
            data_ds_pos = ds[var].sel(latitude=lat_slice_pos)
            data_ds_ml_pos = ds_ml[var].sel(latitude=lat_slice_pos)
            rel_err_pos = ((data_ds_ml_pos - data_ds_pos) / data_ds_pos).mean(
                dim=[
                    d
                    for d in ["time", "latitude", "longitude"]
                    if d in data_ds_pos.dims
                ],
                skipna=True,
            ) * 100
            global_min = min(global_min, float(rel_err_pos.min()))
            global_max = max(global_max, float(rel_err_pos.max()))

        for i in range(n_bands // 2):
            lat_min_neg = lat_bins[i]
            lat_max_neg = lat_bins[i + 1]
            lat_slice_neg = slice(lat_max_neg, lat_min_neg)
            ax_neg = axes[0, i]
            data_ds_neg = ds[var].sel(latitude=lat_slice_neg)
            data_ds_ml_neg = ds_ml[var].sel(latitude=lat_slice_neg)
            rel_err_neg = ((data_ds_ml_neg - data_ds_neg) / data_ds_neg).mean(
                dim=[
                    d
                    for d in ["time", "latitude", "longitude"]
                    if d in data_ds_neg.dims
                ],
                skipna=True,
            ) * 100
            ax_neg.plot(rel_err_neg, levels)
            ax_neg.set_title(f"Lat {lat_min_neg}° to {lat_max_neg}°")
            ax_neg.set_xlabel("Relative Error (%)")
            ax_neg.set_ylabel("Level")
            ax_neg.invert_yaxis()
            ax_neg.set_xlim(global_min, global_max)

            idx = -(i + 1)
            lat_min_pos = lat_bins[idx]
            lat_max_pos = lat_bins[idx - 1]
            lat_slice_pos = slice(lat_min_pos, lat_max_pos)
            ax_pos = axes[1, i]
            data_ds_pos = ds[var].sel(latitude=lat_slice_pos)
            data_ds_ml_pos = ds_ml[var].sel(latitude=lat_slice_pos)
            rel_err_pos = ((data_ds_ml_pos - data_ds_pos) / data_ds_pos).mean(
                dim=[
                    d
                    for d in ["time", "latitude", "longitude"]
                    if d in data_ds_pos.dims
                ],
                skipna=True,
            ) * 100
            ax_pos.plot(rel_err_pos, levels)
            ax_pos.set_title(f"Lat {lat_min_pos}° to {lat_max_pos}°")
            ax_pos.set_xlabel("Relative Error (%)")
            ax_pos.set_ylabel("Level")
            ax_pos.invert_yaxis()
            ax_pos.set_xlim(global_min, global_max)

        plt.gca().invert_yaxis()
        plt.suptitle(
            f"Vertical Profiles of Relative Error for {var} for Each Latitude Band"
        )
        plt.tight_layout()

        if "file" in outputs:
            section_output.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                section_output / f"{var}_pl_rel_error.png",
                bbox_inches="tight",
                dpi=200,
            )
        if save_plot_data:
            # Save relative error curves per band and level
            bands = n_bands // 2
            neg_curves = []
            pos_curves = []
            neg_min = []
            neg_max = []
            pos_min = []
            pos_max = []
            for i in range(bands):
                # Negative latitudes band
                lat_min_neg = lat_bins[i]
                lat_max_neg = lat_bins[i + 1]
                lat_slice_neg = slice(lat_max_neg, lat_min_neg)
                data_ds_neg = ds[var].sel(latitude=lat_slice_neg)
                data_ds_ml_neg = ds_ml[var].sel(latitude=lat_slice_neg)
                rel_err_neg = (
                    (data_ds_ml_neg - data_ds_neg) / data_ds_neg
                ).mean(
                    dim=[
                        d
                        for d in ["time", "latitude", "longitude"]
                        if d in data_ds_neg.dims
                    ],
                    skipna=True,
                ) * 100
                neg_curves.append(rel_err_neg.values)
                neg_min.append(float(lat_min_neg))
                neg_max.append(float(lat_max_neg))

                # Positive latitudes band
                idx = -(i + 1)
                lat_min_pos = lat_bins[idx]
                lat_max_pos = lat_bins[idx - 1]
                lat_slice_pos = slice(lat_min_pos, lat_max_pos)
                data_ds_pos = ds[var].sel(latitude=lat_slice_pos)
                data_ds_ml_pos = ds_ml[var].sel(latitude=lat_slice_pos)
                rel_err_pos = (
                    (data_ds_ml_pos - data_ds_pos) / data_ds_pos
                ).mean(
                    dim=[
                        d
                        for d in ["time", "latitude", "longitude"]
                        if d in data_ds_pos.dims
                    ],
                    skipna=True,
                ) * 100
                pos_curves.append(rel_err_pos.values)
                pos_min.append(float(lat_min_pos))
                pos_max.append(float(lat_max_pos))

            neg_arr = np.stack(neg_curves, axis=0)
            pos_arr = np.stack(pos_curves, axis=0)
            ds_save = xr.Dataset({
                "rel_error_neg": ("band", neg_arr),
                "rel_error_pos": ("band", pos_arr),
            }).assign_coords(
                band=np.arange(bands),
                level=("level", np.array(levels)),
                neg_lat_min=("band", np.array(neg_min)),
                neg_lat_max=("band", np.array(neg_max)),
                pos_lat_min=("band", np.array(pos_min)),
                pos_lat_max=("band", np.array(pos_max)),
            )
            section_output.mkdir(parents=True, exist_ok=True)
            ds_save.to_netcdf(section_output / f"{var}_pl_rel_error.nc")
        if fig_count == 0 and "cell-first" in outputs:
            plt.show()
        if fig_count > 0 and "cell" not in outputs:
            plt.close(fig)
        fig_count += 1
