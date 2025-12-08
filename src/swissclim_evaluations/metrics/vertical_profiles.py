from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np  # retained only for final serialization (NPZ) and minimal list ops
import xarray as xr

from ..aggregations import latitude_weights
from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_init_time_range,
    format_variable_name,
    resolve_ensemble_mode,
)


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
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute NMAE per level (percentage) for a latitude slice lazily with xarray.

    NMAE_k = MAE_k / Δ_k * 100, with Δ_k the range of true values over all non-level dims.
    Returns a DataArray with dimension 'level'.
    If there are no selected values, returns a level-aligned DataArray of NaNs.
    """
    sub_true = true_da.sel(latitude=lat_slice).sel(level=level_values).astype("float32")
    sub_pred = pred_da.sel(latitude=lat_slice).sel(level=level_values).astype("float32")
    if sub_true.size == 0:
        return xr.DataArray(
            np.full((len(level_values),), np.nan, dtype="float32"),
            dims=["level"],
            coords={"level": list(level_values)},
            name="nmae",
        )

    # Reduce over all non-level dims present in either target or prediction,
    # including 'ensemble' when computing pooled semantics.
    candidate_dims = [
        "time",
        "init_time",
        "lead_time",
        "latitude",
        "longitude",
        "ensemble",
    ]
    reduce_dims = [d for d in candidate_dims if (d in sub_true.dims) or (d in sub_pred.dims)]
    reduce_dims_true = [d for d in candidate_dims if d in sub_true.dims]
    diff = (sub_pred - sub_true).astype("float32")
    abs_err = xr.ufuncs.abs(diff)

    if weights is not None:
        mae = abs_err.weighted(weights).mean(dim=reduce_dims, skipna=True)
    else:
        mae = abs_err.mean(dim=reduce_dims, skipna=True)
    # Range is computed from the truth across its own dims only
    t_max = sub_true.max(dim=reduce_dims_true, skipna=True)
    t_min = sub_true.min(dim=reduce_dims_true, skipna=True)
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
    ensemble_mode: str | None = None,
    metrics_cfg: dict[str, Any] | None = None,
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    section_output = out_root / "vertical_profiles"

    variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
    lat_bins, n_bands = _lat_bands()

    latitude_weighting = bool((metrics_cfg or {}).get("latitude_weighting", True))
    weights = None
    if latitude_weighting and "latitude" in ds_target.dims:
        weights = latitude_weights(ds_target.latitude)

    # Extract time ranges for naming
    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            return format_init_time_range(vals)
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

    # Resolve ensemble handling for vertical profiles (support mean, pooled, members; prob invalid).
    resolved_mode = resolve_ensemble_mode(
        "vertical_profiles", ensemble_mode, ds_target, ds_prediction
    )
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for vertical_profiles")
    if resolved_mode == "none" and has_ens:
        raise ValueError(
            "ensemble_mode=none requested but 'ensemble' dimension present; "
            "choose mean|pooled|members"
        )

    def _iter_members():
        if not has_ens:
            yield None, ds_target, ds_prediction
        else:
            for i in range(int(ds_prediction.sizes["ensemble"])):
                tgt_m = ds_target.isel(ensemble=i) if "ensemble" in ds_target.dims else ds_target
                pred_m = ds_prediction.isel(ensemble=i)
                yield i, tgt_m, pred_m

    if resolved_mode == "mean" and has_ens:
        ds_target = ds_target.mean(dim="ensemble") if "ensemble" in ds_target.dims else ds_target
        ds_prediction = ds_prediction.mean(dim="ensemble")
        ens_token = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled" and has_ens:
        ens_token = ensemble_mode_to_token("pooled")
    elif resolved_mode == "members" and has_ens:
        ens_token = None  # per-member tokens will be used
    else:
        ens_token = None

    fig_count = 0
    for var in variables_3d:
        print(f"[vertical_profiles] variable: {var}")
        level_coord = ds_target[var].coords.get("level", None)
        if level_coord is None or int(level_coord.size) == 0:
            continue
        if select_cfg.get("levels"):
            requested = select_cfg.get("levels")
            avail = set(level_coord.values.tolist())
            try:
                level_values = [lv for lv in (requested or []) if lv in avail]
            except Exception:
                level_values = []
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

        # Detect latitude ordering once (ERA5 often descending 90 -> -90). We
        # adapt slice direction instead of sorting (cheaper and preserves
        # original layout / lazy dask graph).
        try:
            lat_vals = ds_target[var].latitude
            lat_desc = bool(lat_vals[0] > lat_vals[-1])
        except Exception:
            lat_desc = False

        def _lat_slice(lo: float, hi: float, *, _lat_desc: bool = lat_desc) -> slice:  # bind order
            """Return a slice selecting [lo, hi] irrespective of coordinate order.

            lo < hi follows logical ascending definition from _lat_bands(). If
            coordinate is descending we invert endpoints so that .sel() matches
            data without an explicit sort.
            """
            return slice(hi, lo) if _lat_desc else slice(lo, hi)

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
                    weights=weights,
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
                    weights=weights,
                )
            )
            north_meta.append((lat_min_pos, lat_max_pos))

        band_idx = xr.DataArray(np.arange(half), dims=["band"], name="band")
        south_da = xr.concat(south_curves, dim=band_idx).assign_coords(band=band_idx)
        north_da = xr.concat(north_curves, dim=band_idx).assign_coords(band=band_idx)
        hemisphere = xr.DataArray(["south", "north"], dims=["hemisphere"], name="hemisphere")
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

        def _emit(
            ens_token_local: str | None,
            member_index: int | None = None,
            *,
            _combined=combined,
            _south_meta=south_meta,
            _north_meta=north_meta,
            _lvl_vals=level_values,
            _gmin=gmin_val,
            _gmax=gmax_val,
            _var_name=var,
            _half=half,
        ) -> None:
            """Emit plot/NPZ for current (or overridden) combined NMAE data.

            Defaulted keyword arguments bind loop variables (Ruff B023 safe).
            """
            n_cols = 2
            fig, axes = plt.subplots(n_cols, _half, figsize=(24, 10), dpi=dpi * 2, sharey=True)
            for bi in range(_half):
                ax_s = axes[0, bi]
                curve_s_da = _combined.sel(hemisphere="south").isel(band=bi).squeeze(drop=True)
                curve_s = np.asarray(curve_s_da.values).squeeze()
                lat_min_neg, lat_max_neg = _south_meta[bi]
                ax_s.plot(curve_s, _lvl_vals)
                ax_s.set_title(f"Lat {lat_min_neg}° to {lat_max_neg}°")
                ax_s.set_xlabel("NMAE (%)")
                ax_s.set_ylabel("Level")
                ax_s.invert_yaxis()
                ax_s.set_xlim(_gmin, _gmax)
                ax_n = axes[1, bi]
                curve_n_da = _combined.sel(hemisphere="north").isel(band=bi).squeeze(drop=True)
                curve_n = np.asarray(curve_n_da.values).squeeze()
                lat_min_pos, lat_max_pos = _north_meta[bi]
                ax_n.plot(curve_n, _lvl_vals)
                ax_n.set_title(f"Lat {lat_min_pos}° to {lat_max_pos}°")
                ax_n.set_xlabel("NMAE (%)")
                ax_n.set_ylabel("Level")
                ax_n.invert_yaxis()
                ax_n.set_xlim(_gmin, _gmax)
            plt.gca().invert_yaxis()
            title_extra = f" member={member_index}" if member_index is not None else ""

            # Check for single date
            date_str = extract_date_from_dataset(ds_target)

            plt.suptitle(
                f"Vertical Profiles of NMAE for {format_variable_name(_var_name)} "
                f"(band-wise){title_extra}{date_str}"
            )
            plt.tight_layout()
            if save_fig:
                section_output.mkdir(parents=True, exist_ok=True)
                out_png = section_output / build_output_filename(
                    metric="vprof_nmae",
                    variable=_var_name,
                    level="multi",
                    qualifier="plot",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token_local,
                    ext="png",
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[vertical_profiles] saved {out_png}")
            if save_npz:
                section_output.mkdir(parents=True, exist_ok=True)
                out_npz = section_output / build_output_filename(
                    metric="vprof_nmae",
                    variable=_var_name,
                    level="multi",
                    qualifier="combined",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token_local,
                    ext="npz",
                )
                south_vals = _combined.sel(hemisphere="south").values
                north_vals = _combined.sel(hemisphere="north").values
                neg_min = np.asarray([m[0] for m in _south_meta])
                neg_max = np.asarray([m[1] for m in _south_meta])
                pos_min = np.asarray([m[0] for m in _north_meta])
                pos_max = np.asarray([m[1] for m in _north_meta])
                np.savez(
                    out_npz,
                    nmae_neg=south_vals,
                    nmae_pos=north_vals,
                    band=np.arange(_half),
                    level=np.asarray(_lvl_vals),
                    neg_lat_min=neg_min,
                    neg_lat_max=neg_max,
                    pos_lat_min=pos_min,
                    pos_lat_max=pos_max,
                )
                print(f"[vertical_profiles] saved {out_npz}")
            plt.close(fig)

        if resolved_mode == "members" and has_ens:
            for member_index, tgt_m, pred_m in _iter_members():
                # Recompute combined curves for this member
                # (Reuse existing logic by calling _compute_nmae per band again)
                south_curves_m: list[xr.DataArray] = []
                north_curves_m: list[xr.DataArray] = []
                for i in range(half):
                    lat_min_neg, lat_max_neg = south_meta[i]
                    lat_slice_neg = _lat_slice(lat_min_neg, lat_max_neg)
                    south_curves_m.append(
                        _compute_nmae(tgt_m[var], pred_m[var], lat_slice_neg, level_values)
                    )
                    idx = -(i + 1)
                    lat_min_pos = lat_bins[idx]
                    lat_max_pos = lat_bins[idx - 1]
                    low = min(lat_min_pos, lat_max_pos)
                    high = max(lat_min_pos, lat_max_pos)
                    lat_slice_pos = _lat_slice(low, high)
                    north_curves_m.append(
                        _compute_nmae(tgt_m[var], pred_m[var], lat_slice_pos, level_values)
                    )
                band_idx = xr.DataArray(np.arange(half), dims=["band"], name="band")
                south_da_m = xr.concat(south_curves_m, dim=band_idx).assign_coords(band=band_idx)
                north_da_m = xr.concat(north_curves_m, dim=band_idx).assign_coords(band=band_idx)
                hemisphere = xr.DataArray(
                    ["south", "north"], dims=["hemisphere"], name="hemisphere"
                )
                combined_m = xr.concat([south_da_m, north_da_m], dim=hemisphere)
                combined_m = combined_m.compute()
                _emit(
                    ensemble_mode_to_token("members", member_index),
                    member_index=member_index,
                    _combined=combined_m,
                )
        else:
            _emit(ens_token)
        fig_count += 1
