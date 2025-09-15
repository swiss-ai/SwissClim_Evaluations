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
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    section_output = out_root / "vertical_profiles"

    variables_3d = [v for v in ds.data_vars if "level" in ds[v].dims]
    lat_bins, n_bands = _lat_bands()

    # Note: derive levels per-variable from dataset to avoid mismatches with config

    fig_count = 0
    for var in variables_3d:
        print(f"[vertical_profiles] variable: {var}")
        # Derive actual levels present for this variable; intersect with config if provided
        level_coord = ds[var].coords.get("level", None)
        if level_coord is None or int(level_coord.size) == 0:
            continue
        if select_cfg.get("levels"):
            requested = np.array(select_cfg.get("levels"))
            avail = set(level_coord.values.tolist())
            level_values = np.array([lv for lv in requested if lv in avail])
            if level_values.size == 0:
                # No overlap; skip this variable
                continue
        else:
            level_values = np.array(level_coord.values)
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
            data_ds_neg = (
                ds[var].sel(latitude=lat_slice_neg).sel(level=level_values)
            )
            data_ds_ml_neg = (
                ds_ml[var].sel(latitude=lat_slice_neg).sel(level=level_values)
            )
            reduce_dims = [
                d
                for d in [
                    "time",
                    "init_time",
                    "lead_time",
                    "latitude",
                    "longitude",
                    "ensemble",
                ]
                if d in data_ds_neg.dims
            ]
            rel_err_neg = ((data_ds_ml_neg - data_ds_neg) / data_ds_neg).mean(
                dim=reduce_dims, skipna=True
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
            data_ds_pos = (
                ds[var].sel(latitude=lat_slice_pos).sel(level=level_values)
            )
            data_ds_ml_pos = (
                ds_ml[var].sel(latitude=lat_slice_pos).sel(level=level_values)
            )
            reduce_dims_pos = [
                d
                for d in [
                    "time",
                    "init_time",
                    "lead_time",
                    "latitude",
                    "longitude",
                    "ensemble",
                ]
                if d in data_ds_pos.dims
            ]
            rel_err_pos = ((data_ds_ml_pos - data_ds_pos) / data_ds_pos).mean(
                dim=reduce_dims_pos, skipna=True
            ) * 100
            global_min = min(global_min, float(rel_err_pos.min()))
            global_max = max(global_max, float(rel_err_pos.max()))

        for i in range(n_bands // 2):
            lat_min_neg = lat_bins[i]
            lat_max_neg = lat_bins[i + 1]
            lat_slice_neg = slice(lat_max_neg, lat_min_neg)
            ax_neg = axes[0, i]
            data_ds_neg = (
                ds[var].sel(latitude=lat_slice_neg).sel(level=level_values)
            )
            data_ds_ml_neg = (
                ds_ml[var].sel(latitude=lat_slice_neg).sel(level=level_values)
            )
            reduce_dims = [
                d
                for d in [
                    "time",
                    "init_time",
                    "lead_time",
                    "latitude",
                    "longitude",
                    "ensemble",
                ]
                if d in data_ds_neg.dims
            ]
            rel_err_neg = ((data_ds_ml_neg - data_ds_neg) / data_ds_neg).mean(
                dim=reduce_dims, skipna=True
            ) * 100
            ax_neg.plot(np.ravel(rel_err_neg.squeeze().values), level_values)
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
            data_ds_pos = (
                ds[var].sel(latitude=lat_slice_pos).sel(level=level_values)
            )
            data_ds_ml_pos = (
                ds_ml[var].sel(latitude=lat_slice_pos).sel(level=level_values)
            )
            reduce_dims_pos = [
                d
                for d in [
                    "time",
                    "init_time",
                    "lead_time",
                    "latitude",
                    "longitude",
                    "ensemble",
                ]
                if d in data_ds_pos.dims
            ]
            rel_err_pos = ((data_ds_ml_pos - data_ds_pos) / data_ds_pos).mean(
                dim=reduce_dims_pos, skipna=True
            ) * 100
            ax_pos.plot(np.ravel(rel_err_pos.squeeze().values), level_values)
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

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png = section_output / f"{var}_pl_rel_error.png"
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"[vertical_profiles] saved {out_png}")
        if save_npz:
            # Save a single combined NPZ for this variable containing all bands and metadata
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
                data_ds_neg = (
                    ds[var].sel(latitude=lat_slice_neg).sel(level=level_values)
                )
                data_ds_ml_neg = (
                    ds_ml[var]
                    .sel(latitude=lat_slice_neg)
                    .sel(level=level_values)
                )
                rel_err_neg = (
                    (data_ds_ml_neg - data_ds_neg) / data_ds_neg
                ).mean(
                    dim=[
                        d
                        for d in [
                            "time",
                            "init_time",
                            "lead_time",
                            "latitude",
                            "longitude",
                            "ensemble",
                        ]
                        if d in data_ds_neg.dims
                    ],
                    skipna=True,
                ) * 100
                neg_curves.append(np.ravel(rel_err_neg.squeeze().values))
                neg_min.append(float(lat_min_neg))
                neg_max.append(float(lat_max_neg))

                # Positive latitudes band
                idx = -(i + 1)
                lat_min_pos = lat_bins[idx]
                lat_max_pos = lat_bins[idx - 1]
                lat_slice_pos = slice(lat_min_pos, lat_max_pos)
                data_ds_pos = (
                    ds[var].sel(latitude=lat_slice_pos).sel(level=level_values)
                )
                data_ds_ml_pos = (
                    ds_ml[var]
                    .sel(latitude=lat_slice_pos)
                    .sel(level=level_values)
                )
                rel_err_pos = (
                    (data_ds_ml_pos - data_ds_pos) / data_ds_pos
                ).mean(
                    dim=[
                        d
                        for d in [
                            "time",
                            "init_time",
                            "lead_time",
                            "latitude",
                            "longitude",
                            "ensemble",
                        ]
                        if d in data_ds_pos.dims
                    ],
                    skipna=True,
                ) * 100
                pos_curves.append(np.ravel(rel_err_pos.squeeze().values))
                pos_min.append(float(lat_min_pos))
                pos_max.append(float(lat_max_pos))

            neg_arr = np.stack(neg_curves, axis=0)
            pos_arr = np.stack(pos_curves, axis=0)
            section_output.mkdir(parents=True, exist_ok=True)
            out_npz = section_output / f"{var}_pl_rel_error_combined.npz"
            np.savez(
                out_npz,
                rel_error_neg=neg_arr,
                rel_error_pos=pos_arr,
                band=np.arange(bands),
                level=level_values,
                neg_lat_min=np.array(neg_min),
                neg_lat_max=np.array(neg_max),
                pos_lat_min=np.array(pos_min),
                pos_lat_max=np.array(pos_max),
            )
            print(f"[vertical_profiles] saved {out_npz}")
        plt.close(fig)
        fig_count += 1
