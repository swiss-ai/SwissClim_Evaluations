from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np  # retained only for final serialization (NPZ) and minimal list ops
import xarray as xr

from ..helpers import build_output_filename


def _lat_bands() -> tuple[list[float], int]:
    # Pure Python to avoid forcing concrete numpy arrays early
    lat_bins = [float(x) for x in range(-90, 91, 10)]
    n_bands = len(lat_bins) - 1
    return lat_bins, n_bands


def _compute_nmae(
    true_da: xr.DataArray,
    pred_da: xr.DataArray,
    lat_slice: slice,
    level_values: Sequence[int | float],
) -> xr.DataArray:
    """Compute NMAE per level (percentage) for a latitude slice lazily with xarray.

    NMAE_k = MAE_k / Δ_k * 100, with Δ_k the range of true values over all non-level dims.
    Returns a DataArray with dimension 'level'.
    If there are no selected values, returns a level-aligned DataArray of NaNs.
    """
    sub_true = (
        true_da.sel(latitude=lat_slice)
        .sel(level=level_values)
        .astype("float32")
    )
    sub_pred = (
        pred_da.sel(latitude=lat_slice)
        .sel(level=level_values)
        .astype("float32")
    )
    if sub_true.size == 0:
        return xr.DataArray(
            np.full((len(level_values),), np.nan, dtype="float32"),
            dims=["level"],
            coords={"level": list(level_values)},
            name="nmae",
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
        if d in sub_true.dims
    ]
    diff = (sub_pred - sub_true).astype("float32")
    abs_err = xr.ufuncs.abs(diff)

    mae = abs_err.mean(dim=reduce_dims, skipna=True)
    t_max = sub_true.max(dim=reduce_dims, skipna=True)
    t_min = sub_true.min(dim=reduce_dims, skipna=True)
    delta = (t_max - t_min).astype("float32")
    nmae = (mae / delta.where(delta != 0)).where(delta != 0)
    nmae = (nmae * 100.0).fillna(0.0).astype("float32")
    nmae.name = "nmae"
    # Ensure only level dimension remains
    if "level" not in nmae.dims:
        nmae = nmae.expand_dims("level")
    return nmae


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    section_output = out_root / "vertical_profiles"

    variables_3d = [
        v for v in ds_target.data_vars if "level" in ds_target[v].dims
    ]
    lat_bins, n_bands = _lat_bands()

    # Extract time ranges for naming
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

    def _extract_lead_range(ds: xr.Dataset):
        if "lead_time" not in ds:
            return None
        try:
            vals = ds["lead_time"].values
            if vals.size == 0:
                return None
            hours = (vals / np.timedelta64(1, "h")).astype(int)
            sh = int(hours.min())
            eh = int(hours.max())

            def _fmt(h: int) -> str:
                return f"{h:03d}h"

            return (_fmt(sh), _fmt(eh))
        except Exception:
            return None

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    fig_count = 0
    try:
        from ..progress import iter_progress  # type: ignore

        _iter_vprof = iter_progress(
            variables_3d, module="vertical_profiles", total=len(variables_3d)
        )
    except Exception:  # pragma: no cover
        _iter_vprof = variables_3d
    for var in _iter_vprof:
        print(f"[vertical_profiles] variable: {var}")
        level_coord = ds_target[var].coords.get("level", None)
        if level_coord is None or int(level_coord.size) == 0:
            continue
        if select_cfg.get("levels"):
            requested = select_cfg.get("levels")
            avail = set(level_coord.values.tolist())
            level_values = [lv for lv in requested if lv in avail]
            if len(level_values) == 0:
                continue
        else:
            level_values = list(level_coord.values)

        # Build NMAE curves once per band (southern + northern)
        south_curves: list[xr.DataArray] = []
        north_curves: list[xr.DataArray] = []
        south_meta = []  # (lat_min, lat_max)
        north_meta = []
        half = n_bands // 2
        # Detect latitude ordering once (ERA5 typically descending 90 -> -90). We adapt slice direction
        # instead of sorting (cheaper and preserves original memory layout / lazy dask graph).
        try:
            lat_vals = ds_target[var].latitude
            lat_desc = bool(lat_vals[0] > lat_vals[-1])
        except Exception:
            lat_desc = False

        def _lat_slice(lo: float, hi: float) -> slice:
            """Return a slice selecting [lo, hi] irrespective of coordinate order.
            lo < hi in logical (ascending) definition coming from _lat_bands().
            If coordinate is descending we invert endpoints so that .sel() matches data.
            """
            return slice(hi, lo) if lat_desc else slice(lo, hi)

        for i in range(half):
            # South bands (negative latitudes in array order)
            lat_min_neg = lat_bins[i]
            lat_max_neg = lat_bins[i + 1]
            lat_slice_neg = _lat_slice(lat_min_neg, lat_max_neg)
            south_curves.append(
                _compute_nmae(
                    ds_target[var],
                    ds_prediction[var],
                    lat_slice_neg,
                    level_values,
                )
            )
            south_meta.append((lat_min_neg, lat_max_neg))
            # North bands (from end backwards)
            idx = -(i + 1)
            lat_min_pos = lat_bins[idx]
            lat_max_pos = lat_bins[idx - 1]
            # Provide logical lower/upper to helper; it will flip if needed.
            low = min(lat_min_pos, lat_max_pos)
            high = max(lat_min_pos, lat_max_pos)
            lat_slice_pos = _lat_slice(low, high)
            north_curves.append(
                _compute_nmae(
                    ds_target[var],
                    ds_prediction[var],
                    lat_slice_pos,
                    level_values,
                )
            )
            north_meta.append((lat_min_pos, lat_max_pos))

        band_idx = xr.DataArray(np.arange(half), dims=["band"], name="band")
        south_da = xr.concat(south_curves, dim=band_idx).assign_coords(
            band=band_idx
        )
        north_da = xr.concat(north_curves, dim=band_idx).assign_coords(
            band=band_idx
        )
        hemisphere = xr.DataArray(
            ["south", "north"], dims=["hemisphere"], name="hemisphere"
        )
        combined = xr.concat([south_da, north_da], dim=hemisphere)
        # Materialize full combined array once; then derive global x-range.
        combined = combined.compute()
        vals = combined.values
        # Mask of finite values (ignores NaN/inf). If none are finite, skip gracefully.
        finite_mask = np.isfinite(vals)
        if not finite_mask.any():
            print(
                f"[vertical_profiles] skipping {var}: no finite NMAE values (all selections empty)."
            )
            plt.close("all")
            continue
        # Compute global range only on finite subset to avoid RuntimeWarning from all-NaN slices.
        finite_vals = vals[finite_mask]
        gmin_val = float(finite_vals.min())
        gmax_val = float(finite_vals.max())
        # If degenerate (all identical), expand range slightly so matplotlib doesn't complain.
        if gmin_val == gmax_val:
            pad = 1e-6 if gmin_val == 0 else abs(gmin_val) * 1e-6
            gmin_val -= pad
            gmax_val += pad

        # Plot
        n_cols = 2
        fig, axes = plt.subplots(
            n_cols, half, figsize=(24, 10), dpi=dpi * 2, sharey=True
        )
        for i in range(half):
            # South (row 0)
            ax_s = axes[0, i]
            # Some xarray versions keep a length-1 dimension when selecting a
            # single coordinate (hemisphere/band). Squeeze defensively so we
            # always have shape (n_levels,) for plotting.
            curve_s_da = (
                combined.sel(hemisphere="south").isel(band=i).squeeze(drop=True)
            )
            curve_s = np.asarray(curve_s_da.values).squeeze()
            ax_s.plot(curve_s, level_values)
            lat_min_neg, lat_max_neg = south_meta[i]
            ax_s.set_title(f"Lat {lat_min_neg}° to {lat_max_neg}°")
            ax_s.set_xlabel("NMAE (%)")
            ax_s.set_ylabel("Level")
            ax_s.invert_yaxis()
            ax_s.set_xlim(gmin_val, gmax_val)
            # North (row 1)
            ax_n = axes[1, i]
            curve_n_da = (
                combined.sel(hemisphere="north").isel(band=i).squeeze(drop=True)
            )
            curve_n = np.asarray(curve_n_da.values).squeeze()
            lat_min_pos, lat_max_pos = north_meta[i]
            ax_n.plot(curve_n, level_values)
            ax_n.set_title(f"Lat {lat_min_pos}° to {lat_max_pos}°")
            ax_n.set_xlabel("NMAE (%)")
            ax_n.set_ylabel("Level")
            ax_n.invert_yaxis()
            ax_n.set_xlim(gmin_val, gmax_val)
        plt.gca().invert_yaxis()
        plt.suptitle(f"Vertical Profiles of NMAE for {var} (band-wise)")
        plt.tight_layout()

        if save_fig:
            section_output.mkdir(parents=True, exist_ok=True)
            out_png = section_output / build_output_filename(
                metric="vprof_nmae",
                variable=var,
                level="multi",
                qualifier="plot",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=None,
                ext="png",
            )
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"[vertical_profiles] saved {out_png}")

        if save_npz:
            section_output.mkdir(parents=True, exist_ok=True)
            out_npz = section_output / build_output_filename(
                metric="vprof_nmae",
                variable=var,
                level="multi",
                qualifier="combined",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=None,
                ext="npz",
            )
            south_vals = combined.sel(hemisphere="south").values
            north_vals = combined.sel(hemisphere="north").values
            neg_min = np.asarray([m[0] for m in south_meta])
            neg_max = np.asarray([m[1] for m in south_meta])
            pos_min = np.asarray([m[0] for m in north_meta])
            pos_max = np.asarray([m[1] for m in north_meta])
            np.savez(
                out_npz,
                nmae_neg=south_vals,
                nmae_pos=north_vals,
                band=np.arange(half),
                level=np.asarray(level_values),
                neg_lat_min=neg_min,
                neg_lat_max=neg_max,
                pos_lat_min=pos_min,
                pos_lat_max=pos_max,
            )
            print(f"[vertical_profiles] saved {out_npz}")
        plt.close(fig)
        fig_count += 1
