from __future__ import annotations

import contextlib
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from weatherbenchX import aggregation, binning
from weatherbenchX.metrics import base
from weatherbenchX.metrics.probabilistic import (
    CRPSEnsemble,
    EnsembleVariance,
    UnbiasedEnsembleMeanSquaredError,
    UnbiasedSpreadSkillRatio,
)

from ..dask_utils import compute_jobs, dask_histogram
from ..helpers import (
    COLOR_DIAGNOSTIC,
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_init_time_range,
    format_variable_name,
    save_data,
    save_dataframe,
    save_figure,
    time_chunks,
    unwrap_longitude_for_plot,
)


class RobustUnbiasedEnsembleMeanSquaredError(UnbiasedEnsembleMeanSquaredError):
    """Unbiased MSE that filters out negative estimates (statistical artifacts)."""

    def _compute_per_variable(
        self,
        predictions: xr.DataArray,
        targets: xr.DataArray,
    ) -> xr.DataArray:
        # Handle case where targets has ensemble dim of size 1 (causes NaN variance with ddof=1)
        if self._ensemble_dim in targets.dims and targets.sizes[self._ensemble_dim] == 1:
            targets = targets.squeeze(self._ensemble_dim, drop=True)

        val = super()._compute_per_variable(predictions, targets)
        # Replace negative values with NaN to avoid issues in aggregation and SSR calculation
        return val.where(val >= 0)


class RobustUnbiasedSpreadSkillRatio(UnbiasedSpreadSkillRatio):
    """SSR that uses RobustUnbiasedEnsembleMeanSquaredError to avoid NaNs."""

    @property
    def statistics(self) -> Mapping[str, base.Statistic]:
        return {
            "EnsembleVariance": EnsembleVariance(
                ensemble_dim=self._ensemble_dim,
                skipna_ensemble=self._skipna_ensemble,
            ),
            "UnbiasedEnsembleMeanSquaredError": RobustUnbiasedEnsembleMeanSquaredError(
                ensemble_dim=self._ensemble_dim,
                skipna_ensemble=self._skipna_ensemble,
            ),
        }

    def _values_from_mean_statistics_per_variable(
        self,
        statistic_values: Mapping[str, xr.DataArray],
    ) -> xr.DataArray:
        """Computes metrics from aggregated statistics."""
        variance = statistic_values["EnsembleVariance"]
        mse = statistic_values["UnbiasedEnsembleMeanSquaredError"]

        # Handle division by zero or NaN
        ratio = variance / mse
        return np.sqrt(ratio)


def compute_wbx_crps(
    preds: dict[str, xr.DataArray] | xr.Dataset | xr.DataArray,
    targs: dict[str, xr.DataArray] | xr.Dataset | xr.DataArray,
    ensemble_dim: str = "ensemble",
) -> xr.Dataset:
    """Compute Fair CRPS using WeatherBenchX implementation.

    Replicates logic of CRPSEnsemble: CRPS = CRPSSkill - 0.5 * CRPSSpread.
    """
    metric = CRPSEnsemble(ensemble_dim=ensemble_dim)

    # Ensure inputs are dicts if they are Datasets or DataArrays
    if isinstance(preds, xr.DataArray):
        preds = {preds.name or "prediction": preds}
    if isinstance(targs, xr.DataArray):
        targs = {targs.name or "target": targs}

    if isinstance(preds, xr.Dataset):
        preds = {v: preds[v] for v in preds.data_vars}
    if isinstance(targs, xr.Dataset):
        targs = {v: targs[v] for v in targs.data_vars}

    stats = {}
    for name, stat in metric.statistics.items():
        # compute returns dict {var: da}
        res = stat.compute(preds, targs)
        stats[name] = res

    # CRPS = CRPSSkill - 0.5 * CRPSSpread
    crps_results = {}
    for var in preds:
        if var in stats["CRPSSkill"] and var in stats["CRPSSpread"]:
            crps = stats["CRPSSkill"][var] - 0.5 * stats["CRPSSpread"][var]
            crps_results[var] = _add_metric_prefix(crps, "CRPS")

    return xr.Dataset(crps_results)


def _save_npz_with_coords(path: Path, da: xr.DataArray, **kwargs):
    """Save DataArray to NPZ with coordinates, handling missing ones gracefully."""
    coords = {}
    # Standard coordinates we care about
    for coord_name in ["latitude", "longitude", "init_time", "lead_time", "level", "region"]:
        if coord_name in da.coords:
            coords[coord_name] = da.coords[coord_name].values
        else:
            coords[coord_name] = np.array([])

    # Add any extra kwargs
    coords.update(kwargs)

    save_data(path, data=da.values, **coords)

    np.savez(path, data=da.values, **coords)


def _pit(da_target, da_prediction):
    return np.mean(da_prediction < da_target[..., None], axis=-1)


def probability_integral_transform(
    da_target, da_prediction, ensemble_dim="ensemble", name_prefix: str | None = "PIT"
):
    """Compute the probability integral transform for ensemble predictions vs targets."""
    res = xr.apply_ufunc(
        _pit,
        da_target,
        da_prediction,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float],
    )
    return _add_metric_prefix(res, name_prefix) if name_prefix else res


def _add_metric_prefix(da_or_ds: xr.Dataset | xr.DataArray, prefix: str):
    # Accept both Dataset and DataArray; for DataArray, rename the variable name if present
    if isinstance(da_or_ds, xr.DataArray):
        name = da_or_ds.name or "value"
        return da_or_ds.rename(f"{prefix}.{name}")
    else:
        return da_or_ds.rename({var: f"{prefix}.{var}" for var in da_or_ds.data_vars})


# --- Runner helpers and orchestrators (combined) ---


def _common_dims_for_reduce(da: xr.DataArray) -> list[str]:
    return [
        d
        for d in [
            "time",
            "init_time",
            "lead_time",
            "latitude",
            "longitude",
            "level",
        ]
        if d in da.dims
    ]


def _reduce_mean_all(da: xr.DataArray) -> xr.DataArray:
    dims = _common_dims_for_reduce(da)
    return da.mean(dim=dims, skipna=True)


def _pit_histogram_dask_lazy(da: xr.DataArray, bins: int = 50) -> tuple[Any, np.ndarray]:
    """Compute PIT histogram lazily using dask_histogram.
    Returns (lazy_counts, edges).
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = dask_histogram(da, bins=edges)
    return counts, edges


def _pit_histogram_dask(
    da: xr.DataArray, bins: int = 50, density: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PIT histogram using dask.
    Returns (counts, edges). If density=True, return density values.
    """
    counts_lazy, edges = _pit_histogram_dask_lazy(da, bins)
    counts = counts_lazy.compute().astype(np.float64)

    if density:
        total = counts.sum()
        if total > 0:
            bin_width = 1.0 / bins
            counts = counts / (total * bin_width)
    return counts, edges


def _iter_time_chunks(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    init_chunk: int | None = None,
    lead_chunk: int | None = None,
):
    if all(dim in ds_prediction.dims for dim in ("init_time", "lead_time")):
        for init_chunk_vals, lead_chunk_vals in time_chunks(
            ds_prediction["init_time"].values,
            ds_prediction["lead_time"].values,
            init_chunk,
            lead_chunk,
        ):
            idx = {"init_time": init_chunk_vals, "lead_time": lead_chunk_vals}
            # Assumes upstream CLI aligned datasets by init_time/lead_time intersection.
            yield (ds_target.sel(**idx).load(), ds_prediction.sel(**idx).load())
    elif "time" in ds_prediction.dims:
        yield ds_target, ds_prediction
    else:
        yield ds_target, ds_prediction


def run_probabilistic(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    cfg_plot: dict[str, Any],
    cfg_all: dict[str, Any],
    ensemble_mode: str | None = None,
) -> None:
    """Compute CRPS and PIT, save summaries and optional fields.

    Outputs:
    - crps_summary.csv (mean across common dims)
    - {var}_pit_hist.npz (counts, edges)
    - {var}_pit_field.npz (full PIT field with coordinates)
    - {var}_crps_field.npz (full CRPS field with coordinates)

    Note: All field outputs use NPZ format for memory efficiency (avoids OOM).
    """
    section_output = out_root / "probabilistic"
    section_output.mkdir(parents=True, exist_ok=True)
    # Always export numeric artifacts for reproducibility (output_mode does not affect data saves)

    if "ensemble" not in ds_prediction.dims:
        print("[probabilistic] Skipping: model dataset has no 'ensemble' dimension.")
        return

    variables = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not variables:
        print(
            "[probabilistic] No overlapping variables between targets and predictions; "
            "nothing to do."
        )
        return

    crps_rows: list[dict[str, Any]] = []

    # Extract time ranges for naming
    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        vals = ds["init_time"].values
        if vals.size == 0:
            return None

        start = np.datetime64(np.min(vals)).astype("datetime64[h]")
        end = np.datetime64(np.max(vals)).astype("datetime64[h]")

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
        sh = int(np.min(hours))
        eh = int(np.max(hours))

        def _fmt(h: int) -> str:
            return f"{h:03d}h"

        return (_fmt(sh), _fmt(eh))

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    ens_token = ensemble_mode_to_token("prob")
    metrics_cfg = (cfg_all or {}).get("metrics", {})
    prob_cfg = metrics_cfg.get("probabilistic") or (cfg_all or {}).get("probabilistic", {})
    report_per_level = bool(prob_cfg.get("report_per_level", True))

    crps_rows_per_level: list[dict[str, Any]] = []

    # --- Optimization: Batch compute CRPS and PIT ---
    ds_target_sub = ds_target[variables]
    ds_prediction_sub = ds_prediction[variables]

    if "ensemble" in ds_target_sub.dims:
        ds_target_sub = ds_target_sub.isel(ensemble=0, drop=True)
    if "ensemble" in ds_target_sub.coords:
        ds_target_sub = ds_target_sub.drop_vars("ensemble")

    with contextlib.suppress(Exception):
        ds_target_sub, ds_prediction_sub = xr.align(
            ds_target_sub, ds_prediction_sub, join="exact", exclude=["ensemble"]
        )

    # Ensure target does not have ensemble dimension (even after align)
    if "ensemble" in ds_target_sub.dims:
        ds_target_sub = ds_target_sub.isel(ensemble=0, drop=True)

    ds_crps = compute_wbx_crps(ds_prediction_sub, ds_target_sub, ensemble_dim="ensemble")
    ds_pit = probability_integral_transform(
        ds_target_sub, ds_prediction_sub, ensemble_dim="ensemble", name_prefix=None
    )

    # --- Optimization: Collect all lazy computations ---
    jobs = []
    is_multi_lead = "lead_time" in ds_prediction.dims and ds_prediction.sizes["lead_time"] > 1

    for var in variables:
        job: dict[str, Any] = {"var": var}
        crps_da = ds_crps[var]
        pit_da = ds_pit[var]

        # 1. CRPS Mean
        if not is_multi_lead:
            job["crps_mean_lazy"] = _reduce_mean_all(crps_da)
        else:
            job["crps_mean_lazy"] = None

        # 2. CRPS Per Lead
        if "lead_time" in crps_da.dims and crps_da.sizes["lead_time"] > 1:
            dims_to_reduce = [d for d in crps_da.dims if d != "lead_time"]
            job["crps_per_lead_lazy"] = crps_da.mean(dim=dims_to_reduce, skipna=True)
        else:
            job["crps_per_lead_lazy"] = None

        # 3. CRPS Per Level
        if report_per_level and "level" in crps_da.dims:
            dims_to_reduce = [d for d in crps_da.dims if d != "level"]
            job["crps_per_level_lazy"] = crps_da.mean(dim=dims_to_reduce, skipna=True)
        else:
            job["crps_per_level_lazy"] = None

        # 4. PIT Global Histogram
        if not is_multi_lead:
            counts_lazy, edges = _pit_histogram_dask_lazy(pit_da, bins=50)
            job["pit_counts_lazy"] = counts_lazy
            job["pit_edges"] = edges
        else:
            job["pit_counts_lazy"] = None
            job["pit_edges"] = None

        # 6. Full Fields (for saving)
        job["crps_field_lazy"] = crps_da
        job["pit_field_lazy"] = pit_da

        jobs.append(job)

    # --- Batch Compute ---
    compute_jobs(
        jobs,
        key_map={
            "crps_mean_lazy": "crps_mean_res",
            "crps_per_lead_lazy": "crps_per_lead_res",
            "crps_per_level_lazy": "crps_per_level_res",
            "pit_counts_lazy": "pit_counts_res",
            "crps_field_lazy": "crps_field_res",
            "pit_field_lazy": "pit_field_res",
        },
    )

    # --- Process Results (Save/Plot) ---
    for job in jobs:
        var = job["var"]

        # 1. CRPS Mean
        if job["crps_mean_lazy"] is not None:
            try:
                crps_mean = float(job["crps_mean_res"].item())
            except Exception:
                crps_mean = float(job["crps_mean_res"])
            crps_rows.append({"variable": var, "CRPS": crps_mean})

        # 2. Per Lead Time CRPS
        if job["crps_per_lead_lazy"] is not None:
            crps_per_lead = job["crps_per_lead_res"]
            # Re-extract leads from the computed result (coords preserved)
            leads = crps_per_lead["lead_time"].values
            if np.issubdtype(leads.dtype, np.timedelta64):
                lead_hours = (leads / np.timedelta64(1, "h")).astype(int)
            else:
                lead_hours = leads

            values = crps_per_lead.values

            df_lead = pd.DataFrame(
                {
                    "lead_time_hours": lead_hours,
                    "CRPS": values,
                    "variable": str(var),
                }
            )

            out_csv_lead = section_output / build_output_filename(
                metric="temporal_probabilistic_metrics",
                variable=str(var),
                level=None,
                qualifier="per_lead_time",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(df_lead, out_csv_lead, index=False)

            # Plot CRPS vs Lead Time
            fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
            ax.plot(lead_hours, values, marker="o", linestyle="-")
            ax.set_xlabel("Lead Time [h]")
            ax.set_ylabel("CRPS")
            ax.set_title(f"CRPS Evolution — {format_variable_name(str(var))}")
            ax.grid(True, linestyle="--", alpha=0.6)
            out_png_lead = section_output / build_output_filename(
                metric="temporal_probabilistic_metrics",
                variable=str(var),
                level=None,
                qualifier="per_lead_time",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="png",
            )
            save_figure(fig, out_png_lead)
            print(f"[probabilistic] saved {out_png_lead}")

        # 3. Per Level CRPS
        if job["crps_per_level_lazy"] is not None:
            crps_per_level = job["crps_per_level_res"]
            for lvl in crps_per_level.level.values:
                crps_rows_per_level.append(
                    {
                        "variable": var,
                        "level": int(lvl),
                        "CRPS": float(crps_per_level.sel(level=lvl).item()),
                    }
                )

        # 4. PIT Global Histogram
        if job["pit_counts_lazy"] is None:
            if "lead_time" in pit_da.dims and pit_da.sizes["lead_time"] > 1:
                print("[probabilistic] Skipping average PIT histogram (lead_time > 1 present).")
        else:
            counts = job["pit_counts_res"].astype(np.float64)
            edges = job["pit_edges"]
            # Density normalization
            width = np.diff(edges)
            bin_area = counts.sum() * width.mean() if counts.sum() > 0 else 1.0
            counts = counts / bin_area

            pit_npz = section_output / build_output_filename(
                metric="pit_hist",
                variable=str(var),
                level=None,
                qualifier=None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="npz",
            )
            np.savez(
                pit_npz,
                counts=counts,
                edges=edges,
            )
            print(f"[probabilistic] saved {pit_npz}")

        # 6. Save Full Fields
        out_npz_crps = section_output / build_output_filename(
            metric="crps_field",
            variable=str(var),
            level=None,
            qualifier=None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="npz",
        )
        _save_npz_with_coords(out_npz_crps, job["crps_field_res"])
        print(f"[probabilistic] saved {out_npz_crps}")

        out_npz_pit = section_output / build_output_filename(
            metric="pit_field",
            variable=str(var),
            level=None,
            qualifier=None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="npz",
        )
        _save_npz_with_coords(out_npz_pit, job["pit_field_res"])
        print(f"[probabilistic] saved {out_npz_pit}")

    if crps_rows:
        if "lead_time" in ds_prediction.dims and ds_prediction.sizes["lead_time"] > 1:
            print("[probabilistic] Skipping CRPS summary table (lead_time > 1 present).")
        else:
            df = pd.DataFrame(crps_rows).groupby("variable").mean()
            out_csv = section_output / build_output_filename(
                metric="crps_summary",
                variable=None,
                level=None,
                qualifier="averaged" if (init_range or lead_range) else None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            df.to_csv(out_csv)
            print("CRPS summary (per variable):")
            print(df.head())
            print(f"[probabilistic] saved {out_csv}")
        # Backward-compatible copy for tests expecting ensnone naming

    if crps_rows_per_level:
        df_lvl = pd.DataFrame(crps_rows_per_level)
        out_csv_lvl = section_output / build_output_filename(
            metric="crps_summary",
            variable=None,
            level=None,
            qualifier="per_level",
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        df_lvl.to_csv(out_csv_lvl, index=False)
        print(f"[probabilistic] saved {out_csv_lvl}")


def _select_base_variable_for_plot(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    plotting_cfg: dict[str, Any],
) -> str:
    cfg_var = (plotting_cfg or {}).get("map_variable") if isinstance(plotting_cfg, dict) else None
    if cfg_var and isinstance(cfg_var, str):
        if cfg_var.startswith("CRPS."):
            return cfg_var.split(".", 1)[1]
        return cfg_var
    common = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not common:
        raise ValueError(
            "No common variables between targets and predictions for probabilistic plots."
        )
    return str(common[0])


def _time_reduce_dims_for_plot(da: xr.DataArray) -> list[str]:
    return [d for d in ["time", "init_time", "lead_time", "ensemble"] if d in da.dims]


def plot_probabilistic(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    """Generate probabilistic plots (CRPS map + PIT histogram).

    Saves under out_root/probabilistic. If output_mode in {'npz','both'} also
    writes NPZ data artifacts.
    """
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int((plotting_cfg or {}).get("dpi", 48))
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    base_var = _select_base_variable_for_plot(ds_target, ds_prediction, plotting_cfg)

    # Align target/prediction on common coordinates with an OUTER join to preserve all
    # prediction lead_time offsets (especially when targets are sparser). This mirrors
    # the alignment used in run_probabilistic and prevents accidental intersection that
    # could drop requested panel hours.
    da_t = ds_target[base_var]
    da_p = ds_prediction[base_var]

    # Ensure target does not have ensemble dimension (even after align)
    if "ensemble" in da_t.dims:
        da_t = da_t.isel(ensemble=0, drop=True)
    if "ensemble" in da_t.coords:
        da_t = da_t.drop_vars("ensemble")

    _aligned = False
    # For probabilistic plots, target is deterministic (no ensemble) while prediction
    # has ensemble dimension. Check alignment only on non-ensemble dimensions.
    non_ens_dims_p = [d for d in da_p.dims if d != "ensemble"]
    if all(dim in da_t.dims for dim in non_ens_dims_p):
        with contextlib.suppress(Exception):
            da_t, da_p = xr.align(da_t, da_p, join="exact", exclude=["ensemble"])
            _aligned = True
    if not _aligned and da_t.shape != da_p.shape:
        raise RuntimeError(
            f"Failed aligning base variable '{base_var}' for probabilistic plot; shapes differ: "
            f"target={da_t.shape} prediction={da_p.shape}"
        )

    # CRPS values (keep lead_time for per-lead panels)
    # Note: We use compute_wbx_crps which leverages WeatherBenchX statistics (Skill/Spread)
    # to derive the full CRPS field.
    crps_ds = compute_wbx_crps(da_p, da_t, ensemble_dim="ensemble")
    # Extract the single variable
    crps = crps_ds[base_var] if base_var in crps_ds else crps_ds[list(crps_ds.data_vars)[0]]
    # For the single-map preview, we reduce all time-like dims including lead_time
    reduce_dims = _time_reduce_dims_for_plot(crps)
    crps_map = crps.mean(dim=reduce_dims, skipna=True) if reduce_dims else crps

    # Detect lat/lon and sort latitude ascending for pcolormesh compatibility
    lat_name = next((n for n in crps_map.dims if n in ("latitude", "lat", "y")), None)
    lon_name = next((n for n in crps_map.dims if n in ("longitude", "lon", "x")), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot find lat/lon dims in CRPS map dims: {crps_map.dims}")
    lat_vals = crps_map[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        crps_map = crps_map.sortby(lat_name)
    # Unwrap longitudes for wrapped selections (e.g., 335..360 U 0..45 -> -25..45)
    crps_map = unwrap_longitude_for_plot(crps_map, lon_name)

    # Attempt time range extraction for plots
    def _extract_init_range_plot(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        vals = ds["init_time"].values
        if vals.size == 0:
            return None

        start = np.datetime64(np.min(vals)).astype("datetime64[h]")
        end = np.datetime64(np.max(vals)).astype("datetime64[h]")

        def _fmt(x):
            return (
                np.datetime_as_string(x, unit="h")
                .replace("-", "")
                .replace(":", "")
                .replace("T", "")
            )

        return (_fmt(start), _fmt(end))

    def _extract_lead_range_plot(ds: xr.Dataset):
        if "lead_time" not in ds:
            return None
        vals = ds["lead_time"].values
        if getattr(vals, "size", 0) == 0:
            return None
        hours = (vals / np.timedelta64(1, "h")).astype(int)
        if hours.size == 0:
            return None
        sh = int(np.min(hours))
        eh = int(np.max(hours))

        def _fmt(h: int) -> str:
            return f"{h:03d}h"

        return (_fmt(sh), _fmt(eh))

    init_range_plot = _extract_init_range_plot(ds_prediction)
    lead_range_plot = _extract_lead_range_plot(ds_prediction)

    # Check for single date
    date_str = extract_date_from_dataset(ds_target)

    # Plot CRPS map (simple original style, no percentile scaling / fallback)
    if "lead_time" in ds_prediction.dims and ds_prediction.sizes["lead_time"] > 1:
        print("[probabilistic] Skipping average CRPS map (lead_time > 1 present).")
    else:
        fig = plt.figure(figsize=(12, 6), dpi=dpi * 2)
        ax = plt.axes(projection=ccrs.PlateCarree())
        if hasattr(ax, "add_feature"):
            ax.add_feature(cfeature.COASTLINE, lw=0.5)
            ax.add_feature(cfeature.BORDERS, lw=0.3)
        Z = crps_map.values
        # Basic color limits
        vmin = 0.0
        vmax = float(np.nanmax(Z)) if np.isfinite(Z).any() else 1.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = 1.0
        mesh = ax.pcolormesh(
            crps_map[lon_name],
            crps_map[lat_name],
            Z,
            cmap="viridis",
            shading="auto",
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )
        cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        cbar.set_label("CRPS")

        ax.set_title(f"CRPS Map (Mean) — {format_variable_name(base_var)}", loc="left", fontsize=10)
        ax.set_title(date_str, loc="right", fontsize=10)

        if save_fig:
            ens_token_plot = ensemble_mode_to_token("prob")
            # Title set above
            out_png = section / build_output_filename(
                metric="crps_map",
                variable=base_var,
                level=None,
                qualifier=None,
                init_time_range=init_range_plot,
                lead_time_range=lead_range_plot,
                ensemble=ens_token_plot,
                ext="png",
            )
            save_figure(fig, out_png)
        else:
            plt.close(fig)

        if save_npz:
            ens_token_plot = ensemble_mode_to_token("prob")
            out_npz = section / build_output_filename(
                metric="crps_map",
                variable=base_var,
                level=None,
                qualifier=None,
                init_time_range=init_range_plot,
                lead_time_range=lead_range_plot,
                ensemble=ens_token_plot,
                ext="npz",
            )
            save_data(
                out_npz,
                crps=crps_map.values,
                latitude=crps_map[lat_name].values,
                longitude=crps_map[lon_name].values,
                variable=base_var,
                metric="CRPS",
            )

    # Panel: CRPS maps by lead_time across all retained hours (panel concept removed)
    if "lead_time" in crps.dims and int(crps.sizes.get("lead_time", 0)) > 1 and save_fig:

        def _to_hour(val, fallback: int) -> int:
            arr = np.asarray(val)
            if np.issubdtype(arr.dtype, np.timedelta64):
                return int(arr / np.timedelta64(1, "h"))
            return int(arr) if np.isfinite(arr).all() else fallback

        full_hours = [_to_hour(x, idx) for idx, x in enumerate(crps["lead_time"].values)]
        if full_hours:
            # Reduce all dims except latitude/longitude and lead_time
            dims_to_reduce = [d for d in ["time", "init_time", "ensemble"] if d in crps.dims]
            crps_by_lead = crps.mean(dim=dims_to_reduce, skipna=True) if dims_to_reduce else crps
            lat_name = next((n for n in crps_by_lead.dims if n in ("latitude", "lat", "y")), None)
            lon_name = next((n for n in crps_by_lead.dims if n in ("longitude", "lon", "x")), None)
            if lat_name is None or lon_name is None:
                raise ValueError(f"Cannot find lat/lon dims in CRPS dims: {crps_by_lead.dims}")
            lat_vals = crps_by_lead[lat_name].values
            if lat_vals[0] > lat_vals[-1]:
                crps_by_lead = crps_by_lead.sortby(lat_name)
            # Unwrap longitudes for wrapped domains
            crps_by_lead = unwrap_longitude_for_plot(crps_by_lead, lon_name)
            # Build mapping from all available lead hours -> index
            raw_leads = crps_by_lead["lead_time"].values
            crps_hours: list[int] = [
                _to_hour(x, idx) for idx, x in enumerate(raw_leads)
            ]  # renamed from all_hours
            # Debug visibility
            from contextlib import suppress

            with suppress(Exception):
                print(f"[probabilistic] CRPS grid using lead_hours={crps_hours}")
            hour_index_pairs = []
            for h in full_hours:
                try:
                    idx = crps_hours.index(int(h))
                    hour_index_pairs.append((int(h), idx))
                except Exception:
                    continue
            crps_line_rows: list[dict[str, float]] = []
            if hour_index_pairs:
                # 2-column CRPS-only layout (configuration removed).
                ncols = 2
                n = len(hour_index_pairs)
                nrows = (n + ncols - 1) // ncols
                # Width mirrors maps grid proportionally: 7.0 per column (maps uses ~7 per column).
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(7.0 * ncols, 4.0 * nrows),
                    dpi=dpi * 2,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    squeeze=False,
                    constrained_layout=True,
                )
                # Flatten axes for indexing; hide unused later.
                axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
                # Global color scale
                Z_stack = []
                for _, li in hour_index_pairs:
                    Z_stack.append(np.asarray(crps_by_lead.isel(lead_time=li).values))
                Z_stack = np.asarray(Z_stack)
                vmin, vmax = 0.0, float(np.nanmax(Z_stack)) if np.isfinite(Z_stack).any() else 1.0
                first_im = None
                for i, (h, li) in enumerate(hour_index_pairs):
                    ax = axes_flat[i]
                    if hasattr(ax, "add_feature"):
                        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                        ax.coastlines(linewidth=0.5)
                    Z = np.asarray(crps_by_lead.isel(lead_time=li).values)
                    im = ax.pcolormesh(
                        crps_by_lead[lon_name],
                        crps_by_lead[lat_name],
                        Z,
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                        shading="auto",
                    )
                    if first_im is None:
                        first_im = im
                    _lon = crps_by_lead[lon_name].values
                    _lat = crps_by_lead[lat_name].values
                    if _lon.size and _lat.size and hasattr(ax, "set_extent"):
                        ax.set_extent(
                            [
                                float(np.min(_lon)),
                                float(np.max(_lon)),
                                float(np.min(_lat)),
                                float(np.max(_lat)),
                            ],
                            crs=ccrs.PlateCarree(),
                        )
                    ax.set_title(f"CRPS (+{int(h)}h)", fontsize=10)
                    # Mean CRPS per lead for later CSV/NPZ line output
                    crps_line_rows.append(
                        {
                            "lead_time_hours": float(h),
                            "CRPS": float(np.nanmean(Z)),
                        }
                    )
                # Hide unused axes if n not multiple of ncols
                for j in range(n, nrows * ncols):
                    axes_flat[j].axis("off")
                if first_im is not None:
                    # Single vertical colorbar akin to maps grid
                    cb = fig.colorbar(
                        first_im,
                        ax=[ax for ax in axes_flat[:n] if ax.axes.get_visible()],
                        orientation="vertical",
                        fraction=0.025,
                        pad=0.02,
                    )
                    cb.set_label("CRPS")
                ens_token_grid = ensemble_mode_to_token("prob")
                plt.suptitle(
                    f"CRPS Grid by Lead Time — {format_variable_name(base_var)}{date_str}",
                    fontsize=16,
                    y=1.05,
                )
                out_png = section / build_output_filename(
                    metric="crps_map",
                    variable=base_var,
                    level=None,
                    qualifier="grid",
                    init_time_range=init_range_plot,
                    lead_time_range=lead_range_plot,
                    ensemble=ens_token_grid,
                    ext="png",
                )
                save_figure(fig, out_png)
            else:
                plt.close(fig)

            # Persist CRPS line data (hours vs mean CRPS) if we have rows
            if crps_line_rows:
                df_crps_line = pd.DataFrame(crps_line_rows).sort_values("lead_time_hours")
                out_csv_line = section / build_output_filename(
                    metric="crps_line",
                    variable=base_var,
                    level=None,
                    qualifier="by_lead",
                    init_time_range=init_range_plot,
                    lead_time_range=lead_range_plot,
                    ensemble=ensemble_mode_to_token("prob"),
                    ext="csv",
                )
                df_crps_line.to_csv(out_csv_line, index=False)
                print(f"[probabilistic] saved {out_csv_line}")
                if save_npz:
                    out_npz_line = section / build_output_filename(
                        metric="crps_line",
                        variable=base_var,
                        level=None,
                        qualifier="by_lead_data",
                        init_time_range=init_range_plot,
                        lead_time_range=lead_range_plot,
                        ensemble=ensemble_mode_to_token("prob"),
                        ext="npz",
                    )
                    save_data(
                        out_npz_line,
                        lead_hours=df_crps_line["lead_time_hours"].values.astype(float),
                        crps=df_crps_line["CRPS"].values.astype(float),
                        variable=base_var,
                    )

    # PIT histogram (global and by-lead panels) for ALL variables
    # Target ensemble removal is handled before alignment.

    common_vars = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]

    def _to_hour_safe(val, fallback: int) -> int:
        arr = np.asarray(val)
        if np.issubdtype(arr.dtype, np.timedelta64):
            return int(arr / np.timedelta64(1, "h"))
        return int(arr) if np.isfinite(arr).all() else fallback

    for var_name in common_vars:
        da_t_var = ds_target[var_name]
        da_p_var = ds_prediction[var_name]

        # Ensure target does not have ensemble dimension (even after align)
        if "ensemble" in da_t_var.dims:
            da_t_var = da_t_var.isel(ensemble=0, drop=True)
        if "ensemble" in da_t_var.coords:
            da_t_var = da_t_var.drop_vars("ensemble")

        # Align
        non_ens_dims_p = [d for d in da_p_var.dims if d != "ensemble"]
        if all(dim in da_t_var.dims for dim in non_ens_dims_p):
            with contextlib.suppress(Exception):
                da_t_var, da_p_var = xr.align(
                    da_t_var, da_p_var, join="exact", exclude=["ensemble"]
                )

        pit = probability_integral_transform(
            da_t_var,
            da_p_var,
            ensemble_dim="ensemble",
            name_prefix="PIT",
        )

        if pit.size > 0:
            if "lead_time" in pit.dims and pit.sizes["lead_time"] > 1:
                # print(f"[probabilistic] Skipping average PIT histogram for {var_name} (lead_time >
                # 1 present).") Define edges for per-lead plots even if global histogram is skipped
                edges = np.linspace(0.0, 1.0, 21)
            else:
                counts, edges = _pit_histogram_dask(pit, bins=20, density=True)
                fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi * 2)
                widths = np.diff(edges)
                ax.bar(
                    edges[:-1],
                    counts,
                    width=widths,
                    align="edge",
                    color=COLOR_DIAGNOSTIC,
                    edgecolor="white",
                )
                # Check for single date
                date_str = extract_date_from_dataset(ds_target)

                ax.set_title(
                    f"PIT Histogram — {format_variable_name(str(var_name))}",
                    loc="left",
                    fontsize=10,
                )
                ax.set_title(date_str, loc="right", fontsize=10)
                ax.set_xlabel("PIT value")
                ax.set_ylabel("Density")
                ax.axhline(1.0, color="brown", linestyle="--", linewidth=1, label="Uniform")
                ax.legend()
                if save_npz:
                    # Use standardized filename builder for NPZ (with ensprob token)

                    ens_token_plot = ensemble_mode_to_token("prob")
                    out_npz = section / build_output_filename(
                        metric="pit_hist",
                        variable=str(var_name),
                        level=None,
                        qualifier=None,
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=ens_token_plot,
                        ext="npz",
                    )
                    save_data(out_npz, counts=counts, edges=edges, variable=str(var_name))

                if save_fig:
                    ens_token_plot = ensemble_mode_to_token("prob")
                    out_png = section / build_output_filename(
                        metric="pit_hist",
                        variable=str(var_name),
                        level=None,
                        qualifier=None,
                        init_time_range=init_range_plot,
                        lead_time_range=lead_range_plot,
                        ensemble=ens_token_plot,
                        ext="png",
                    )
                    save_figure(fig, out_png)
                else:
                    plt.close(fig)

        # PIT per-lead panel plot (multi-row grid) over all retained hours
        if (
            pit.size > 0
            and "lead_time" in pit.dims
            and int(pit.sizes.get("lead_time", 0)) > 1
            and save_fig
        ):
            full_hours = [_to_hour_safe(x, idx) for idx, x in enumerate(pit["lead_time"].values)]
            if full_hours:
                raw_leads = pit["lead_time"].values
                all_hours: list[int] = [_to_hour_safe(x, idx) for idx, x in enumerate(raw_leads)]
                # Debug visibility
                hour_index_pairs = []
                for h in full_hours:
                    try:
                        idx = all_hours.index(int(h))
                        hour_index_pairs.append((int(h), idx))
                    except Exception:
                        continue
                if hour_index_pairs:
                    n = len(hour_index_pairs)
                    ncols = int((plotting_cfg or {}).get("panel_cols", 2))
                    nrows = (n + ncols - 1) // ncols
                    fig, axes = plt.subplots(
                        nrows,
                        ncols,
                        figsize=(5.4 * ncols, 3.0 * nrows),
                        dpi=dpi * 2,
                        squeeze=False,
                        constrained_layout=True,
                    )
                    axes_flat = axes.flatten()
                    for i, (h, li) in enumerate(hour_index_pairs):
                        r, c = divmod(i, ncols)
                        sub = pit.isel(lead_time=li)
                        data = np.asarray(sub.values).ravel()
                        data = data[np.isfinite(data)]
                        counts_local, _ = np.histogram(data, bins=edges)
                        width = np.diff(edges)
                        total = counts_local.sum()
                        dens = counts_local / (total * width.mean()) if total > 0 else counts_local
                        ax = axes_flat[i]
                        ax.bar(
                            edges[:-1],
                            dens,
                            width=width,
                            align="edge",
                            color="#4C78A8",
                            edgecolor="white",
                        )
                        ax.axhline(1.0, color="brown", linestyle="--", linewidth=1)
                        ax.set_title(f"PIT (+{int(h)}h)", fontsize=10)
                        if r == nrows - 1:
                            ax.set_xlabel("PIT value")
                        if c == 0:
                            ax.set_ylabel("Density")

                    # Hide unused axes if n not multiple of ncols
                    for j in range(n, nrows * ncols):
                        axes_flat[j].axis("off")

                    date_str = extract_date_from_dataset(ds_target)
                    # Adjust layout to leave space for suptitle
                    fig.get_layout_engine().set(rect=[0, 0, 1, 0.92])
                    plt.suptitle(
                        f"PIT Histograms by Lead Time — "
                        f"{format_variable_name(str(var_name))}{date_str}",
                        fontsize=16,
                        y=0.98,
                    )

                    out_png = section / build_output_filename(
                        metric="pit_hist",
                        variable=str(var_name),
                        level=None,
                        qualifier="grid",
                        init_time_range=init_range_plot,
                        lead_time_range=lead_range_plot,
                        ensemble=ensemble_mode_to_token("prob"),
                        ext="png",
                    )
                    save_figure(fig, out_png)
                else:
                    plt.close(fig)

    # CRPS line plots across all retained lead_time hours
    if ("lead_time" in ds_prediction.dims) and int(ds_prediction.sizes.get("lead_time", 0)) > 1:
        panel_hours = [_to_hour(x, idx) for idx, x in enumerate(ds_prediction["lead_time"].values)]
        if panel_hours:
            variables = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
            all_hours = [
                _to_hour(x, idx) for idx, x in enumerate(ds_prediction["lead_time"].values)
            ]
            hour_index_pairs = []
            for h in panel_hours:
                try:
                    hour_index_pairs.append((int(h), all_hours.index(int(h))))
                except Exception:
                    continue
            if hour_index_pairs:
                for var in variables:
                    crps_ds = compute_wbx_crps(
                        ds_prediction[var], ds_target[var], ensemble_dim="ensemble"
                    )
                    # Extract DataArray (compute_wbx_crps returns Dataset)
                    crps = crps_ds[var] if var in crps_ds else crps_ds[list(crps_ds.data_vars)[0]]
                    reduce_dims = [
                        d
                        for d in [
                            "time",
                            "init_time",
                            "latitude",
                            "longitude",
                            "level",
                            "ensemble",
                        ]
                        if d in crps.dims
                    ]
                    crps_lt = crps.mean(dim=reduce_dims, skipna=True) if reduce_dims else crps
                    # Subset to selected lead indices
                    sel_indices = [li for _, li in hour_index_pairs]
                    crps_sel = crps_lt.isel(lead_time=sel_indices)
                    vals = np.asarray(crps_sel.values).ravel()
                    hours_plot = [h for h, _ in hour_index_pairs]
                    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi * 2)
                    ax.plot(hours_plot, vals, marker="o")
                    ax.set_xlabel("Lead Time [h]")
                    ax.set_ylabel("CRPS")
                    ax.set_title(
                        f"CRPS Evolution — {format_variable_name(var)}", loc="left", fontsize=10
                    )
                    ax.set_title(date_str, loc="right", fontsize=10)
                    out_png = section / build_output_filename(
                        metric="crps_line",
                        variable=str(var),
                        level=None,
                        qualifier=None,
                        init_time_range=init_range_plot,
                        lead_time_range=lead_range_plot,
                        ensemble=ensemble_mode_to_token("prob"),
                        ext="png",
                    )
                    plt.tight_layout()
                    save_figure(fig, out_png)
                    # Save NPZ for each variable line plot & accumulate rows for combined CSV
                    if save_npz:
                        out_npz = section / build_output_filename(
                            metric="crps_line",
                            variable=str(var),
                            level=None,
                            qualifier="data",
                            init_time_range=init_range_plot,
                            lead_time_range=lead_range_plot,
                            ensemble=ensemble_mode_to_token("prob"),
                            ext="npz",
                        )
                        save_data(
                            out_npz, lead_hours=np.array(hours_plot), crps=vals, variable=str(var)
                        )


def _wbx_metric_to_df(
    metric: Any,
    ds_prediction: xr.Dataset,
    ds_target: xr.Dataset,
    value_col: str,
) -> pd.DataFrame:
    """Compute a WeatherBenchX PerVariableMetric into a tidy DataFrame.

    Steps:
    - Compute each required statistic via statistic.compute(predictions, targets)
      to get mapping var -> DataArray.
    - Reduce each DataArray by taking mean over common dims.
    - Call metric.values_from_mean_statistics(mean_stats) to obtain final values.
    - Return DataFrame with index 'variable' and a single column 'value_col'.
    """
    # Build var->DataArray mappings using only common variables
    variables = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    pred_map: Mapping[Hashable, xr.DataArray] = {v: ds_prediction[v] for v in variables}
    targ_map: Mapping[Hashable, xr.DataArray] = {v: ds_target[v] for v in variables}

    # Compute and average statistics per variable
    mean_stats: dict[str, dict[Hashable, xr.DataArray]] = {}
    dims_all = [
        "time",
        "init_time",
        "lead_time",
        "latitude",
        "longitude",
        "level",
        "ensemble",
    ]
    for stat_name, stat in metric.statistics.items():
        stat_vals = stat.compute(predictions=pred_map, targets=targ_map)
        reduced: dict[Hashable, xr.DataArray] = {}
        for var, da in stat_vals.items():
            dims = [d for d in dims_all if d in da.dims]
            reduced[var] = da.mean(dim=dims, skipna=True)
        mean_stats[stat_name] = reduced

    # Derive metric values from averaged statistics
    values_map = metric.values_from_mean_statistics(mean_stats)
    rows = []
    for var, da in values_map.items():
        rows.append({"variable": str(var), value_col: float(da.values)})
    df = pd.DataFrame(rows).set_index("variable").sort_index()
    return df


def run_probabilistic_wbx(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    all_cfg: dict[str, Any],
) -> None:
    """Compute WBX temporal/spatial metrics and CSV summaries.

    Outputs (under out_root/probabilistic):
    - spread_skill_ratio.csv
    - crps_ensemble.csv
    - prob_metrics_temporal.npz
    - prob_metrics_spatial.npz
    - Optional: crps_map_wbx_<var>.png if output_mode enables plotting

    Note: All aggregated results use NPZ format for memory efficiency (avoids OOM).
    """
    # Write WBX artifacts into the same probabilistic folder to avoid split outputs
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    if "ensemble" not in ds_prediction.dims:
        print("[probabilistic] Skipping: model dataset has no 'ensemble' dimension.")
        return

    common_vars = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not common_vars:
        print(
            "[probabilistic] No overlapping variables between targets and predictions; "
            "nothing to do."
        )
        return
    ds_pred = ds_prediction[common_vars]
    ds_targ = ds_target[common_vars]

    # CSV summaries using WBX metrics (UnbiasedSpreadSkillRatio)
    # Use .sizes (preferred) instead of .dims.get for forward compatibility
    m_ens = int(getattr(ds_pred, "sizes", {}).get("ensemble", 0))
    is_multi_lead = "lead_time" in ds_pred.dims and ds_pred.sizes["lead_time"] > 1

    if m_ens < 2:
        raise RuntimeError(
            "WBX probabilistic metrics require ensemble size >=2 (UnbiasedSpreadSkillRatio). "
            f"Found ensemble size {m_ens}."
        )

    if not is_multi_lead:
        ssr_metric = RobustUnbiasedSpreadSkillRatio(ensemble_dim="ensemble")
        try:
            ssr_df = _wbx_metric_to_df(
                ssr_metric,
                ds_prediction=ds_pred,
                ds_target=ds_targ,
                value_col="SSR",
            )
        except Exception as e:  # pragma: no cover - defensive clarity wrapper
            raise RuntimeError(
                "Failed computing UnbiasedSpreadSkillRatio via WeatherBenchX. "
                "Ensure ensemble size >=2 and variables overlap. Original error: " + str(e)
            ) from e
    else:
        print("[probabilistic] Skipping average SSR summary (lead_time > 1 present).")
        ssr_df = pd.DataFrame()

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

    ens_token_prob = ensemble_mode_to_token("prob")

    if not ssr_df.empty:
        ssr_csv = section / build_output_filename(
            metric="spread_skill_ratio",
            variable=None,
            level=None,
            qualifier=None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token_prob,
            ext="csv",
        )
        ssr_df.to_csv(ssr_csv)
        print(f"[probabilistic] saved {ssr_csv}")

    def _default_regions() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
        return {
            "global": ((-90, 90), (0, 360)),
            "tropics": ((-20, 20), (0, 360)),
            "northern-hemisphere": ((20, 90), (0, 360)),
            "southern-hemisphere": ((-90, -20), (0, 360)),
            "europe": ((35, 75), (-12.5, 42.5)),
            "north-america": ((25, 60), (360 - 120, 360 - 75)),
            "north-atlantic": ((25, 65), (360 - 70, 360 - 10)),
            "north-pacific": ((25, 60), (145, 360 - 130)),
            "east-asia": ((25, 60), (102.5, 150)),
            "ausnz": ((-45, -12.5), (120, 175)),
            "arctic": ((60, 90), (0, 360)),
            "antarctic": ((-90, -60), (0, 360)),
        }

    regions_cfg = (plotting_cfg or {}).get("regions") if isinstance(plotting_cfg, dict) else None
    regions = regions_cfg or _default_regions()

    # Aggregator that reduces spatial dimensions (lat/lon) -> Produces Temporal Results
    spatial_aggregator = aggregation.Aggregator(
        reduce_dims=["latitude", "longitude"],
        bin_by=[binning.Regions(regions=regions)],
        skipna=True,
    )

    seasonal = (
        bool((plotting_cfg or {}).get("group_by_season", False))
        if isinstance(plotting_cfg, dict)
        else False
    )
    temporal_bin_by = [binning.ByTimeUnit("season", "init_time")] if seasonal else None

    # Aggregator that reduces temporal dimensions (time) -> Produces Spatial Results (Maps)
    temporal_aggregator = aggregation.Aggregator(
        reduce_dims=["init_time"],
        bin_by=temporal_bin_by,
        skipna=True,
    )

    metrics = {}
    metrics["SSR"] = RobustUnbiasedSpreadSkillRatio(ensemble_dim="ensemble")
    metrics["CRPS"] = CRPSEnsemble(ensemble_dim="ensemble")

    variables = list(ds_pred.data_vars)
    pred_map = {v: ds_pred[v] for v in variables}
    targ_map = {v: ds_targ[v] for v in variables}

    # Compute metrics aggregated over space (Regions) -> Result has dimensions (Region, Time)
    # This object contains the TEMPORAL evolution of the metrics (for each region)
    results_temporal = aggregation.compute_metric_values_for_single_chunk(
        metrics, spatial_aggregator, pred_map, targ_map
    )

    # Compute metrics aggregated over time -> Result has dimensions (Lat, Lon) i.e. a Map
    # This object contains the SPATIAL distribution of the metrics (averaged over time)
    results_spatial = aggregation.compute_metric_values_for_single_chunk(
        metrics, temporal_aggregator, pred_map, targ_map
    )

    # Save individual metric/variable combinations (like other modules)
    # results_temporal has vars like "CRPS.2m_temperature", "SSR.2m_temperature", etc.
    for var_name, _da in results_temporal.data_vars.items():
        metric_name, variable = var_name.split(".", 1)
        npz_path = section / build_output_filename(
            metric=f"{metric_name.lower()}_temporal_wbx",
            variable=variable,
            level=None,
            qualifier=None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token_prob,
            ext="npz",
        )
        _save_npz_with_coords(npz_path, results_temporal[var_name])
        print("Wrote:", npz_path)

    for var_name, _da in results_spatial.data_vars.items():
        metric_name, variable = var_name.split(".", 1)
        npz_path = section / build_output_filename(
            metric=f"{metric_name.lower()}_spatial_wbx",
            variable=variable,
            level=None,
            qualifier=None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token_prob,
            ext="npz",
        )
        _save_npz_with_coords(npz_path, results_spatial[var_name])
        print("Wrote:", npz_path)

    # --- Plotting SSR (Temporal and Spatial) ---
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")

    if save_fig:
        date_str = extract_date_from_dataset(ds_target)

        # Plot Temporal (Time Series)
        # We use results_temporal because it preserves the time dimension
        for var_name in results_temporal.data_vars:
            if not str(var_name).startswith("SSR"):
                continue

            da = results_temporal[var_name]
            # da is (Region, Time)

            # Skip if lead_time dimension is size <= 1
            if "lead_time" in da.dims and da.sizes["lead_time"] <= 1:
                continue

            # Select global region if present, else average over regions
            if "region" in da.dims:
                if "global" in da["region"].values:
                    da = da.sel(region="global")
                else:
                    da = da.mean(dim="region", skipna=True)

            # Average over init_time if present (to show evolution over lead_time)
            if "init_time" in da.dims:
                da = da.mean(dim="init_time", skipna=True)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Convert to dataframe for line plot
            df = da.to_dataframe(name="SSR").reset_index()

            # Determine x-axis column (should be lead_time)
            x_col = "lead_time" if "lead_time" in df.columns else df.columns[0]

            # Convert timedelta to hours if needed
            if pd.api.types.is_timedelta64_dtype(df[x_col]):
                df[x_col] = (df[x_col] / pd.Timedelta(hours=1)).astype(int)

            # Save CSV for intercomparison
            display_var = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
            out_csv_line = section / build_output_filename(
                metric="ssr_line",
                variable=display_var,
                level=None,
                qualifier="by_lead",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token_prob,
                ext="csv",
            )
            # Ensure column name is standard for intercompare
            df_save = df.rename(columns={x_col: "lead_time_hours"})
            df_save.to_csv(out_csv_line, index=False)
            print(f"[probabilistic] saved {out_csv_line}")

            # Plot line
            ax.plot(df[x_col], df["SSR"], marker="o")

            display_var = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)

            ax.set_title(
                f"SSR over Lead Time — {format_variable_name(display_var)}", loc="left", fontsize=10
            )
            ax.set_title(date_str, loc="right", fontsize=10)
            ax.set_ylabel("SSR")
            ax.set_xlabel("Lead Time [h]")
            ax.grid(True, axis="y")

            out_png_temp = section / build_output_filename(
                metric="wbx_temporal",
                variable=display_var,
                level=None,
                qualifier=None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token_prob,
                ext="png",
            )
            save_figure(fig, out_png_temp)
            print(f"[probabilistic] saved {out_png_temp}")

        # Plot Regional Comparison (Bar Chart of Regions)
        # Only plot if lead_time dimension is not present or has size 1 (i.e., not multi-lead)
        for var_name in results_temporal.data_vars:
            if not str(var_name).startswith("SSR"):
                continue

            da = results_temporal[var_name]
            # da is (Region, Time)

            # Skip if lead_time dimension is present and size > 1 (multi-lead)
            if "lead_time" in da.dims and da.sizes["lead_time"] > 1:
                continue

            if "region" in da.dims:
                # Average over time to get pure regional view
                dims_to_mean = [d for d in da.dims if d != "region"]
                da_spatial = da.mean(dim=dims_to_mean, skipna=True)

                # Convert to series for plotting
                s_spatial = da_spatial.to_series()

                # Ensure numeric (coercing errors to NaN), but keep NaNs to show missing regions
                s_spatial = pd.to_numeric(s_spatial, errors="coerce")

                if not s_spatial.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    s_spatial.plot(kind="bar", ax=ax)
                    display_var = (
                        str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
                    )
                    ax.set_title(
                        f"SSR by Region (Time-Averaged) — {format_variable_name(display_var)}",
                        loc="left",
                        fontsize=10,
                    )
                    ax.set_title(date_str, loc="right", fontsize=10)
                    ax.set_ylabel("SSR")
                    ax.set_xlabel("")  # Remove Region label
                    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Ideal (1.0)")

                    # Clean up legend labels
                    handles, labels = ax.get_legend_handles_labels()
                    new_labels = []
                    for lbl in labels:
                        if lbl == "Ideal (1.0)":
                            new_labels.append(lbl)
                        else:
                            # Remove SSR. prefix and format variable
                            clean_lbl = lbl.split(".", 1)[1] if "." in lbl else lbl
                            new_labels.append(format_variable_name(clean_lbl))
                    ax.legend(handles, new_labels, fontsize=10)

                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()

                    out_png_spatial = section / build_output_filename(
                        metric="wbx_spatial",
                        variable=display_var,
                        level=None,
                        qualifier=None,
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token_prob,
                        ext="png",
                    )
                    save_figure(fig, out_png_spatial)
                    print(f"[probabilistic] saved {out_png_spatial}")
                else:
                    print(f"[probabilistic] Skipping spatial plot for {var_name}: No numeric data.")
