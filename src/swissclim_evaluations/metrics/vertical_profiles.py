from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np  # retained only for final serialization (NPZ) and minimal list ops
import xarray as xr
from scores.functions import create_latitude_weights

from ..dask_utils import compute_jobs
from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_variable_name,
    resolve_ensemble_mode,
    save_data,
    save_figure,
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
    preserve_dims: Sequence[str] | None = None,
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

    if preserve_dims:
        reduce_dims = [d for d in reduce_dims if d not in preserve_dims]
        reduce_dims_true = [d for d in reduce_dims_true if d not in preserve_dims]

    diff = (sub_pred - sub_true).astype("float32")
    abs_err = np.abs(diff)

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


def _compute_nmae_per_lead(
    true_da: xr.DataArray,
    pred_da: xr.DataArray,
    level_values: Sequence[int | float],
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute NMAE per level and lead_time (percentage)."""
    # Select levels
    sub_true = true_da.sel(level=level_values).astype("float32")
    sub_pred = pred_da.sel(level=level_values).astype("float32")

    if sub_true.size == 0:
        return xr.DataArray()

    candidate_dims = [
        "time",
        "init_time",
        # "lead_time",  <-- Keep lead_time
        "latitude",
        "longitude",
        "ensemble",
    ]
    reduce_dims = [d for d in candidate_dims if (d in sub_true.dims) or (d in sub_pred.dims)]
    reduce_dims_true = [d for d in candidate_dims if d in sub_true.dims]

    diff = (sub_pred - sub_true).astype("float32")
    abs_err = np.abs(diff)

    if weights is not None:
        mae = abs_err.weighted(weights).mean(dim=reduce_dims, skipna=True)
    else:
        mae = abs_err.mean(dim=reduce_dims, skipna=True)

    t_max = sub_true.max(dim=reduce_dims_true, skipna=True)
    t_min = sub_true.min(dim=reduce_dims_true, skipna=True)
    delta = (t_max - t_min).astype("float32")

    nmae = (mae / delta.where(delta != 0)).where(delta != 0)
    nmae = (nmae * 100.0).fillna(0.0).astype("float32")
    nmae.name = "nmae"
    return nmae


def _plot_vertical_profile_evolution(
    nmae_da: xr.DataArray,
    variable_name: str,
    ens_token: str | None,
    out_root: Path,
    dpi: int,
    save_fig: bool,
) -> None:
    if "lead_time" not in nmae_da.dims or nmae_da.sizes["lead_time"] < 2:
        return

    leads = nmae_da["lead_time"].values
    levels = nmae_da["level"].values

    # Convert leads to hours
    if np.issubdtype(leads.dtype, np.timedelta64):
        lead_hours = (leads / np.timedelta64(1, "h")).astype(int)
    else:
        lead_hours = leads

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    # Contour plot
    # X: lead_time, Y: level (inverted usually for pressure)
    X, Y = np.meshgrid(lead_hours, levels)
    # Transpose to (level, lead_time)
    if nmae_da.dims != ("level", "lead_time"):
        nmae_da = nmae_da.transpose("level", "lead_time")
    Z = nmae_da.values

    im = ax.contourf(X, Y, Z, cmap="viridis", levels=20)
    fig.colorbar(im, ax=ax, label="NMAE [%]")

    ax.set_xlabel("Lead Time [h]")
    ax.set_ylabel("Level")
    # Heuristic for pressure levels: usually descending or large values
    # Standard convention for vertical profiles (top is low pressure)
    ax.invert_yaxis()

    ax.set_title(f"Vertical Profile Evolution — {format_variable_name(variable_name)}")

    if save_fig:
        section_output = out_root / "vertical_profiles"
        out_png = section_output / build_output_filename(
            metric="vertical_profile_evolution",
            variable=variable_name,
            level=None,
            qualifier=None,
            init_time_range=None,
            lead_time_range=None,
            ensemble=ens_token,
            ext="png",
        )
        save_figure(fig, out_png)
    plt.close(fig)


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

    if "latitude" not in ds_target.dims:
        raise ValueError("Latitude dimension required for vertical profiles metrics.")

    weights = create_latitude_weights(ds_target.latitude)
    weights = weights / weights.mean()

    # Extract time ranges for naming
    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
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

    def _extract_lead_range(ds: xr.Dataset):
        if "lead_time" not in ds:
            return None
        vals = ds["lead_time"].values
        if getattr(vals, "size", 0) == 0:
            return None
        hours = (vals / np.timedelta64(1, "h")).astype(int)
        sh = int(hours.min())
        eh = int(hours.max())

        def _fmt(h: int) -> str:
            return f"{h:03d}h"

        return (_fmt(sh), _fmt(eh))

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    # Resolve ensemble handling for vertical profiles (support mean, pooled, members; prob invalid).
    resolved_mode = resolve_ensemble_mode(
        "vertical_profiles", ensemble_mode, ds_target, ds_prediction
    )
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for vertical_profiles")

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

    # Collect all lazy computations
    jobs = []

    for var in variables_3d:
        print(f"[vertical_profiles] preparing {var}...")
        level_values = ds_target[var].level.values
        south_curves: list[xr.DataArray] = []
        north_curves: list[xr.DataArray] = []
        south_meta = []  # (lat_min, lat_max)
        north_meta = []
        half = n_bands // 2

        # Detect latitude ordering once (ERA5 often descending 90 -> -90). We
        # adapt slice direction instead of sorting (cheaper and preserves
        # original layout / lazy dask graph).
        lat_desc = False
        if "latitude" in ds_target[var].coords:
            lat_vals = ds_target[var].latitude
            lat_desc = bool(lat_vals[0] > lat_vals[-1])

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
            w_slice_neg = weights.sel(latitude=lat_slice_neg)
            south_curves.append(
                _compute_nmae(
                    ds_target[var],
                    ds_prediction[var],
                    lat_slice_neg,
                    level_values,
                    weights=w_slice_neg,
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
            w_slice_pos = weights.sel(latitude=lat_slice_pos)
            north_curves.append(
                _compute_nmae(
                    ds_target[var],
                    ds_prediction[var],
                    lat_slice_pos,
                    level_values,
                    weights=w_slice_pos,
                )
            )
            north_meta.append((lat_min_pos, lat_max_pos))

        band_idx = xr.DataArray(np.arange(half), dims=["band"], name="band")
        south_da = xr.concat(south_curves, dim=band_idx).assign_coords(band=band_idx)
        north_da = xr.concat(north_curves, dim=band_idx).assign_coords(band=band_idx)
        hemisphere = xr.DataArray(["south", "north"], dims=["hemisphere"], name="hemisphere")
        combined = xr.concat([south_da, north_da], dim=hemisphere)

        job = {
            "var": var,
            "south_meta": south_meta,
            "north_meta": north_meta,
            "combined_lazy": combined,
            "half": half,
            "ens_token_local": ens_token,
        }

        if resolved_mode == "members" and has_ens:
            # Vectorized calculation for all members
            south_curves_m: list[xr.DataArray] = []
            north_curves_m: list[xr.DataArray] = []
            for i in range(half):
                lat_min_neg, lat_max_neg = south_meta[i]
                lat_slice_neg = _lat_slice(lat_min_neg, lat_max_neg)
                w_slice_neg = weights.sel(latitude=lat_slice_neg)
                south_curves_m.append(
                    _compute_nmae(
                        ds_target[var],
                        ds_prediction[var],
                        lat_slice_neg,
                        level_values,
                        weights=w_slice_neg,
                        preserve_dims=["ensemble"],
                    )
                )
                idx = -(i + 1)
                lat_min_pos = lat_bins[idx]
                lat_max_pos = lat_bins[idx - 1]
                low = min(lat_min_pos, lat_max_pos)
                high = max(lat_min_pos, lat_max_pos)
                lat_slice_pos = _lat_slice(low, high)
                w_slice_pos = weights.sel(latitude=lat_slice_pos)
                north_curves_m.append(
                    _compute_nmae(
                        ds_target[var],
                        ds_prediction[var],
                        lat_slice_pos,
                        level_values,
                        weights=w_slice_pos,
                        preserve_dims=["ensemble"],
                    )
                )
            band_idx = xr.DataArray(np.arange(half), dims=["band"], name="band")
            south_da_m = xr.concat(south_curves_m, dim=band_idx).assign_coords(band=band_idx)
            north_da_m = xr.concat(north_curves_m, dim=band_idx).assign_coords(band=band_idx)
            hemisphere = xr.DataArray(["south", "north"], dims=["hemisphere"], name="hemisphere")
            combined_all = xr.concat([south_da_m, north_da_m], dim=hemisphere)
            job["combined_all_lazy"] = combined_all

        # Optional: overlay vertical profiles for selected lead_time values (evolution)
        evolve = bool((plotting_cfg or {}).get("vertical_profiles_evolve_lead", False))
        if (
            evolve
            and ("lead_time" in ds_prediction.dims)
            and int(ds_prediction.sizes.get("lead_time", 0)) > 1
        ):
            # Use all retained lead_time hours (panel concept removed)
            if np.issubdtype(np.asarray(ds_prediction["lead_time"].values).dtype, np.timedelta64):
                all_hours = [
                    int(np.timedelta64(x) / np.timedelta64(1, "h"))
                    for x in ds_prediction["lead_time"].values
                ]
            else:
                all_hours = [int(x) for x in range(int(ds_prediction.sizes.get("lead_time", 0)))]
            panel_hours = all_hours

            job["evolve_info"] = {
                "panel_hours": panel_hours,
                "all_hours": all_hours,
            }

            # Collect lazy profiles for panel_hours (line plot)
            evolve_profiles_lazy = []
            for idx, h in enumerate(panel_hours):
                li = all_hours.index(int(h)) if int(h) in all_hours else idx
                da_t = ds_target[var]
                da_p = ds_prediction[var]
                if "lead_time" in da_t.dims:
                    da_t = da_t.isel(lead_time=li)
                if "lead_time" in da_p.dims:
                    da_p = da_p.isel(lead_time=li)
                prof = _compute_nmae(da_t, da_p, slice(-90.0, 90.0), level_values)
                evolve_profiles_lazy.append(prof)
            job["evolve_profiles_lazy"] = evolve_profiles_lazy

            # Additionally: full evolution heatmap with x=lead_time (h), y=level, color=NMAE
            hour_index_pairs = []
            for h in panel_hours:
                try:
                    hour_index_pairs.append((int(h), all_hours.index(int(h))))
                except Exception:
                    hour_index_pairs.append((int(h), panel_hours.index(h)))

            job["evolve_info"]["hour_index_pairs"] = hour_index_pairs

            if hour_index_pairs:
                heatmap_profiles_lazy = []
                for _j, (_h, li) in enumerate(hour_index_pairs):
                    da_t = ds_target[var]
                    da_p = ds_prediction[var]
                    if "lead_time" in da_t.dims:
                        da_t = da_t.isel(lead_time=li)
                    if "lead_time" in da_p.dims:
                        da_p = da_p.isel(lead_time=li)
                    prof = _compute_nmae(da_t, da_p, slice(-90.0, 90.0), level_values)
                    heatmap_profiles_lazy.append(prof)
                job["heatmap_profiles_lazy"] = heatmap_profiles_lazy

        # Compute per-lead NMAE vertical profiles
        nmae_lead = _compute_nmae_per_lead(
            ds_target[var], ds_prediction[var], level_values, weights
        )
        job["nmae_lead_lazy"] = nmae_lead

        jobs.append(job)

    # Batch compute
    print(f"[vertical_profiles] computing {len(jobs)} jobs...")
    compute_jobs(
        jobs,
        key_map={
            "combined_lazy": "combined",
            "combined_all_lazy": "combined_all",
            "evolve_profiles_lazy": "evolve_profiles",
            "heatmap_profiles_lazy": "heatmap_profiles",
            "nmae_lead_lazy": "nmae_lead",
        },
    )

    # Process results
    for job in jobs:
        var = job["var"]
        print(f"[vertical_profiles] post-processing {var}...")
        combined = job["combined"]
        south_meta = job["south_meta"]
        north_meta = job["north_meta"]
        half = job["half"]
        ens_token_local = job["ens_token_local"]

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
                out_png = section_output / build_output_filename(
                    metric="vertical_profiles_nmae",
                    variable=str(_var_name),
                    level="multi",
                    qualifier="plot",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token_local,
                    ext="png",
                )
                save_figure(fig, out_png)
            if save_npz:
                out_npz = section_output / build_output_filename(
                    metric="vertical_profiles_nmae",
                    variable=str(_var_name),
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
                save_data(
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
            plt.close(fig)

        if "combined_all" in job:
            combined_all = job["combined_all"]
            for member_index in range(int(ds_prediction.sizes["ensemble"])):
                combined_m = combined_all.isel(ensemble=member_index)
                _emit(
                    ensemble_mode_to_token("members", member_index),
                    member_index=member_index,
                    _combined=combined_m,
                )
        else:
            _emit(ens_token_local)
        fig_count += 1

        # Optional: overlay vertical profiles for selected lead_time values (evolution)
        if "evolve_profiles" in job:
            evolve_info = job["evolve_info"]
            panel_hours = evolve_info["panel_hours"]
            all_hours = evolve_info["all_hours"]
            evolve_profiles = job["evolve_profiles"]

            # Compute global (all-latitudes) profiles per selected lead
            fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi * 2)
            colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(panel_hours)))
            for idx, h in enumerate(panel_hours):
                prof = evolve_profiles[idx]
                ax.plot(prof.values, level_values, label=f"{int(h)}h", color=colors[idx])
            ax.set_xlabel("NMAE (%)")
            ax.set_ylabel("Level")
            ax.invert_yaxis()
            ax.set_title(f"Vertical Profiles NMAE — lead evolution — {var}")
            ax.legend(title="lead_time", fontsize=8, ncols=min(3, len(panel_hours)))
            out_png = section_output / build_output_filename(
                metric="vertical_profiles_nmae",
                variable=str(var),
                level="multi",
                qualifier="evolve",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token if resolved_mode != "members" else None,
                ext="png",
            )
            plt.tight_layout()
            save_figure(fig, out_png)
            plt.close(fig)
            if save_npz:
                out_npz = section_output / build_output_filename(
                    metric="vertical_profiles_nmae",
                    variable=str(var),
                    level="multi",
                    qualifier="evolve_data",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token if resolved_mode != "members" else None,
                    ext="npz",
                )
                profiles = [p.values for p in evolve_profiles]
                save_data(
                    out_npz,
                    lead_hours=np.array(panel_hours, dtype=float),
                    level=np.asarray(level_values),
                    nmae_profiles=np.array(profiles, dtype=object),
                    allow_pickle=True,
                )

            # Additionally: full evolution heatmap with x=lead_time (h), y=level, color=NMAE
            if "heatmap_profiles" in job:
                hour_index_pairs = evolve_info["hour_index_pairs"]
                heatmap_profiles = job["heatmap_profiles"]

                n_levels = len(level_values)
                n_leads = len(hour_index_pairs)
                grid = np.full((n_levels, n_leads), np.nan, dtype=float)
                for j, (_h, _li) in enumerate(hour_index_pairs):
                    prof = heatmap_profiles[j]
                    grid[:, j] = np.asarray(prof.values).ravel()
                lead_hours_plot = [h for h, _ in hour_index_pairs]
                fig2, ax2 = plt.subplots(figsize=(9, 6), dpi=dpi * 2)
                im = ax2.pcolormesh(
                    lead_hours_plot,
                    level_values,
                    grid,
                    shading="nearest",
                    cmap="viridis",
                )
                ax2.invert_yaxis()
                ax2.set_xlabel("lead_time (h)")
                ax2.set_ylabel("Level")
                ax2.set_title(f"Vertical Profiles NMAE — lead-time evolution (heatmap) — {var}")
                cbar = plt.colorbar(im, ax=ax2, orientation="vertical")
                cbar.set_label("NMAE (%)")
                out_png2 = section_output / build_output_filename(
                    metric="vertical_profiles_nmae",
                    variable=str(var),
                    level="multi",
                    qualifier="evolve_heatmap",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token if resolved_mode != "members" else None,
                    ext="png",
                )
                plt.tight_layout()
                save_figure(fig2, out_png2)
                plt.close(fig2)
                if save_npz:
                    out_npz2 = section_output / build_output_filename(
                        metric="vertical_profiles_nmae",
                        variable=str(var),
                        level="multi",
                        qualifier="evolve_heatmap_data",
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token if resolved_mode != "members" else None,
                        ext="npz",
                    )
                    save_data(
                        out_npz2,
                        lead_hours=np.array(lead_hours_plot, dtype=float),
                        level=np.asarray(level_values),
                        nmae_grid=grid,
                    )

        # Compute and plot per-lead NMAE vertical profiles
        nmae_lead = job["nmae_lead"]
        if nmae_lead.size > 0:
            # Emit NPZ with all data
            if save_npz:
                out_npz = section_output / build_output_filename(
                    metric="vertical_profiles_nmae",
                    variable=str(var),
                    level="multi",
                    qualifier="all_leads",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="npz",
                )
                save_dict = {
                    "nmae": nmae_lead.values,
                    "level": np.asarray(level_values),
                }
                if "lead_time" in ds_prediction:
                    save_dict["lead_time"] = ds_prediction["lead_time"].values

                save_data(out_npz, **save_dict)

            # Plot evolution of vertical profile (contour plot)
            _plot_vertical_profile_evolution(
                nmae_lead,
                variable_name=var,
                ens_token=ens_token,
                out_root=out_root,
                dpi=dpi,
                save_fig=save_fig,
            )
