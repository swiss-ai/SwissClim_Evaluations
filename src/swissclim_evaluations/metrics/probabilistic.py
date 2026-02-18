from __future__ import annotations

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

from .. import console as c
from ..dask_utils import (
    apply_split_to_dataarray,
    build_variable_level_lead_splits,
    compute_jobs,
    dask_histogram,
    resolve_dynamic_batch_size,
    resolve_module_batching_options,
)
from ..helpers import (
    COLOR_DIAGNOSTIC,
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_init_time_range,
    format_level_token,
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
    metric: CRPSEnsemble | None = None,
) -> xr.Dataset:
    """Compute Fair CRPS using WeatherBenchX implementation.

    Replicates logic of CRPSEnsemble: CRPS = CRPSSkill - 0.5 * CRPSSpread.
    """
    metric_obj = metric or CRPSEnsemble(ensemble_dim=ensemble_dim)

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
    for name, stat in metric_obj.statistics.items():
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


def _save_npz_with_coords(path: Path, da: xr.DataArray, module: str | None = None, **kwargs):
    """Save DataArray to NPZ with compact coordinate payload.

    To reduce NPZ I/O overhead, only coordinates that align with DataArray
    dimensions are stored by default; optional explicit extras are merged from
    kwargs.
    """
    coords: dict[str, Any] = {}

    # Keep only dimension-bearing coordinates to avoid serializing large,
    # unrelated auxiliary coordinates repeatedly across many files.
    for dim_name in da.dims:
        if dim_name in da.coords:
            coords[dim_name] = np.asarray(da.coords[dim_name].values)

    # Add explicit extras passed by caller (e.g., scalar level token).
    coords.update(kwargs)

    data_obj = da.data
    data_arr = (
        np.asarray(data_obj.compute()) if hasattr(data_obj, "compute") else np.asarray(da.values)
    )
    save_data(path, module=module, data=data_arr, **coords)


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


def _histogram_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return np.zeros(edges.size - 1, dtype=np.float64)
    counts, _ = np.histogram(finite, bins=edges)
    return counts.astype(np.float64)


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
            yield (ds_target.sel(**idx), ds_prediction.sel(**idx))
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
    performance_cfg: dict[str, Any] | None = None,
    include_wbx_outputs: bool = True,
) -> None:
    """Compute PIT artifacts for probabilistic diagnostics.

    Outputs:
    - {var}_pit_hist.npz (counts, edges)

    When include_wbx_outputs=True (default), CRPS/SSR outputs are additionally
    generated via `run_probabilistic_wbx` from a single WBX CRPS/SSR pass.
    """
    section_output = out_root / "probabilistic"
    section_output.mkdir(parents=True, exist_ok=True)
    mode = str((cfg_plot or {}).get("output_mode", "plot")).lower()
    legacy_save_plot_data = bool((cfg_plot or {}).get("save_plot_data", False))
    save_npz = (mode in ("npz", "both")) or legacy_save_plot_data
    if mode == "none":
        c.print("[probabilistic] Skipping PIT artifacts: output_mode=none.")
        return

    if "ensemble" not in ds_prediction.dims:
        c.print("[probabilistic] Skipping: model dataset has no 'ensemble' dimension.")
        return

    variables = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not variables:
        c.print(
            "[probabilistic] No overlapping variables between targets and predictions; "
            "nothing to do."
        )
        return

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
    batch_opts = resolve_module_batching_options(
        performance_cfg=performance_cfg,
        default_split_level=True,
        default_split_lead_time=True,
        default_split_init_time=True,
        default_lead_time_block_size=6,
        default_init_time_block_size=6,
    )
    split_lead_time = bool(batch_opts["split_lead_time"])
    split_init_time = bool(batch_opts["split_init_time"])
    lead_time_block_size = int(batch_opts["lead_time_block_size"])
    init_time_block_size = int(batch_opts["init_time_block_size"])

    # --- Use Batching Logic (Variable by Variable) ---
    is_multi_lead = (
        split_lead_time
        and ("lead_time" in ds_prediction.dims)
        and ds_prediction.sizes["lead_time"] > 1
    )
    is_multi_init = (
        split_init_time
        and ("init_time" in ds_prediction.dims)
        and ds_prediction.sizes["init_time"] > 1
    )

    dynamic_batch = resolve_dynamic_batch_size(
        performance_cfg,
        ds=ds_target,
    )

    pit_hist_edges = np.linspace(0.0, 1.0, 51)
    pit_hist_counts_parts: dict[tuple[str, Any], np.ndarray] = {}

    def _finalize_fields_and_hist(var: str, lvl: Any) -> None:
        key = (str(var), lvl)

        counts = pit_hist_counts_parts.pop(key, None)
        if counts is not None and save_npz:
            width = np.diff(pit_hist_edges)
            total = counts.sum()
            density = counts / (total * width.mean()) if total > 0 else counts
            pit_npz = section_output / build_output_filename(
                metric="pit_hist",
                variable=str(var),
                level=lvl,
                qualifier=f"level{lvl}" if lvl is not None else None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="npz",
            )
            save_data(
                pit_npz,
                counts=density,
                edges=pit_hist_edges,
                module="probabilistic",
            )

    # Define callback to process results incrementally and free memory
    def _process_batch(batch_jobs: list[dict[str, Any]]):
        for job in batch_jobs:
            var = job["var"]
            lvl = job["level"]

            if job.get("pit_hist_counts_res") is not None and job.get("collect_pit_hist"):
                key = (str(var), lvl)
                if key not in pit_hist_counts_parts:
                    pit_hist_counts_parts[key] = np.zeros(
                        pit_hist_edges.size - 1,
                        dtype=np.float64,
                    )
                pit_hist_counts_parts[key] += np.asarray(
                    job["pit_hist_counts_res"], dtype=np.float64
                )

            # Release memory
            keys_to_clear = [
                "pit_hist_counts_res",
            ]
            for k in keys_to_clear:
                if k in job:
                    job[k] = None

    for var in variables:
        var_jobs: list[dict[str, Any]] = []
        # Slice per variable to decouple dask graphs
        ds_p_var = ds_prediction[[var]]
        ds_t_var = ds_target[[var]]

        if "ensemble" in ds_t_var.dims:
            ds_t_var = ds_t_var.isel(ensemble=0, drop=True)
        if "ensemble" in ds_t_var.coords:
            ds_t_var = ds_t_var.drop_vars("ensemble")
        is_3d = "level" in ds_p_var[var].dims

        split_specs = build_variable_level_lead_splits(
            ds_p_var,
            variables=[str(var)],
            split_level=True,
            split_lead_time=is_multi_lead,
            lead_time_block_size=lead_time_block_size,
            split_init_time=is_multi_init,
            init_time_block_size=init_time_block_size,
        )

        for split_spec in split_specs:
            level_val = split_spec["level"]
            lead_slice = split_spec["lead_slice"]
            init_slice = split_spec.get("init_slice", slice(None))

            pred_slice = apply_split_to_dataarray(
                ds_p_var[var],
                level=level_val,
                lead_slice=lead_slice,
                init_slice=init_slice,
            )
            targ_slice = apply_split_to_dataarray(
                ds_t_var[var],
                level=level_val,
                lead_slice=lead_slice,
                init_slice=init_slice,
            )

            collect_pit_hist = True
            needs_pit = collect_pit_hist
            pit_da = (
                probability_integral_transform(
                    targ_slice,
                    pred_slice,
                    ensemble_dim="ensemble",
                    name_prefix=None,
                )
                if needs_pit
                else None
            )

            job: dict[str, Any] = {
                "var": var,
                "level": level_val,
                "is_3d": is_3d,
                "lead_start": int(split_spec.get("lead_start", 0)),
                "lead_len": int(split_spec.get("lead_len", 1)),
                "init_start": int(split_spec.get("init_start", 0)),
                "init_len": int(split_spec.get("init_len", 1)),
            }

            # 4. PIT Global Histogram
            job["collect_pit_hist"] = collect_pit_hist
            if collect_pit_hist and pit_da is not None:
                job["pit_hist_counts_lazy"], _ = _pit_histogram_dask_lazy(
                    pit_da,
                    bins=pit_hist_edges.size - 1,
                )
            else:
                job["pit_hist_counts_lazy"] = None

            var_jobs.append(job)

        if var_jobs:
            jobs_by_level: dict[Any, list[dict[str, Any]]] = {}
            for job in var_jobs:
                jobs_by_level.setdefault(job.get("level"), []).append(job)

            for level_val, level_jobs in jobs_by_level.items():
                level_suffix = "" if level_val is None else f" level={level_val}"
                desc = f"Computing PIT metrics variable={var}{level_suffix}"

                compute_jobs(
                    level_jobs,
                    key_map={
                        "pit_hist_counts_lazy": "pit_hist_counts_res",
                    },
                    batch_size=dynamic_batch,
                    desc=desc,
                    batch_callback=_process_batch,
                )

                # Flush artifacts eagerly at var+level granularity to keep resident memory low.
                _finalize_fields_and_hist(str(var), level_val)

    # Backward-compatible fallback flush in case any parts remain.
    for var, lvl in list(pit_hist_counts_parts.keys()):
        _finalize_fields_and_hist(var, lvl)

    if include_wbx_outputs:
        run_probabilistic_wbx(
            ds_target=ds_target,
            ds_prediction=ds_prediction,
            out_root=out_root,
            plotting_cfg=cfg_plot,
            all_cfg=cfg_all,
            performance_cfg=performance_cfg,
        )


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
    if not save_fig and not save_npz:
        c.print("[probabilistic] Skipping plot_probabilistic: output_mode=none.")
        return
    dpi = int((plotting_cfg or {}).get("dpi", 48))
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    # Identify common variables
    common_vars = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not common_vars:
        c.print("[probabilistic] No common variables found for plotting.")
        return

    # Helper for converting lead hours
    def _to_hour(val, fallback: int) -> int:
        arr = np.asarray(val)
        if np.issubdtype(arr.dtype, np.timedelta64):
            return int(arr / np.timedelta64(1, "h"))
        return int(arr) if np.isfinite(arr).all() else fallback

    # Attempt time range extraction for plots (once for the dataset)
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

    # CRPS plots are generated in run_probabilistic_wbx using WBX aggregated outputs
    # to avoid duplicate CRPS computation.
    crps_plot_vars: list[str] = []
    for var_name in crps_plot_vars:
        # Align target/prediction on common coordinates with an OUTER join.
        da_t = ds_target[var_name]
        da_p = ds_prediction[var_name]

        if "ensemble" in da_t.dims:
            da_t = da_t.isel(ensemble=0, drop=True)
        if "ensemble" in da_t.coords:
            da_t = da_t.drop_vars("ensemble")

        _aligned = False
        non_ens_dims_p = [d for d in da_p.dims if d != "ensemble"]
        if all(dim in da_t.dims for dim in non_ens_dims_p):
            # Assume aligned or rely on implicit alignment
            _aligned = True

        if da_p.size == 0 or da_t.size == 0:
            c.print(f"[probabilistic] Skipping '{var_name}' plots; data empty.")
            continue

        if not _aligned and da_t.shape != da_p.shape:
            c.print(
                f"[probabilistic] Note: '{var_name}' shapes differ "
                f"{da_p.shape} vs {da_t.shape}. Implicit alignment will be used."
            )

        # CRPS values
        crps_ds = compute_wbx_crps(da_p, da_t, ensemble_dim="ensemble")
        crps = crps_ds[var_name] if var_name in crps_ds else crps_ds[list(crps_ds.data_vars)[0]]

        # Determine levels to iterate
        levels = [None]
        if "level" in crps.dims:
            levels = crps["level"].values.tolist()

        for lvl in levels:
            crps_sub = crps
            if lvl is not None:
                crps_sub = crps.sel(level=lvl, drop=True)

            date_str = extract_date_from_dataset(ds_target)

            # --- Map Plot ---
            # Only plot map if lead_time dim is not > 1 (aggregated map)
            # We'll stick to logic: if lead_time > 1, skip single average map.
            if save_fig and (
                "lead_time" not in ds_prediction.dims or ds_prediction.sizes["lead_time"] <= 1
            ):
                reduce_dims = _time_reduce_dims_for_plot(crps_sub)
                crps_map = crps_sub.mean(dim=reduce_dims, skipna=True) if reduce_dims else crps_sub
                lat_name = next((n for n in crps_map.dims if n in ("latitude", "lat", "y")), None)
                lon_name = next((n for n in crps_map.dims if n in ("longitude", "lon", "x")), None)
                if lat_name is None or lon_name is None:
                    continue

                lat_vals = crps_map[lat_name].values
                if lat_vals.size > 1 and lat_vals[0] > lat_vals[-1]:
                    crps_map = crps_map.sortby(lat_name)
                crps_map = unwrap_longitude_for_plot(crps_map, lon_name)

                fig = plt.figure(figsize=(12, 6), dpi=dpi * 2)
                ax = plt.axes(projection=ccrs.PlateCarree())
                if hasattr(ax, "add_feature"):
                    ax.add_feature(cfeature.COASTLINE, lw=0.5)
                Z = crps_map.values
                vmin, vmax = 0.0, float(np.nanmax(Z)) if np.isfinite(Z).any() else 1.0
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
                plt.colorbar(
                    mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, label="CRPS"
                )
                lvl_str = f" @ {format_level_token(lvl)}" if lvl is not None else ""
                ax.set_title(
                    f"CRPS Map (Mean) — {format_variable_name(str(var_name))}{lvl_str}",
                    loc="left",
                    fontsize=10,
                )
                ax.set_title(date_str, loc="right", fontsize=10)

                if save_fig:
                    ens_token_plot = ensemble_mode_to_token("prob")
                    out_png = section / build_output_filename(
                        metric="crps_map",
                        variable=str(var_name),
                        level=lvl,
                        qualifier=None,
                        init_time_range=init_range_plot,
                        lead_time_range=lead_range_plot,
                        ensemble=ens_token_plot,
                        ext="png",
                    )
                    save_figure(fig, out_png)
                else:
                    plt.close(fig)

            # --- Panel Grid & Line Plots (by lead) ---
            if "lead_time" in crps_sub.dims and int(crps_sub.sizes.get("lead_time", 0)) > 1:
                full_hours = [
                    _to_hour(x, idx) for idx, x in enumerate(crps_sub["lead_time"].values)
                ]
                if full_hours:
                    # Logic to slice leads
                    dims_to_reduce = [
                        d for d in ["time", "init_time", "ensemble"] if d in crps_sub.dims
                    ]
                    crps_by_lead = (
                        crps_sub.mean(dim=dims_to_reduce, skipna=True)
                        if dims_to_reduce
                        else crps_sub
                    )

                    lat_name = None
                    lon_name = None
                    if save_fig:
                        lat_name = next(
                            (n for n in crps_by_lead.dims if n in ("latitude", "lat", "y")),
                            None,
                        )
                        lon_name = next(
                            (n for n in crps_by_lead.dims if n in ("longitude", "lon", "x")),
                            None,
                        )
                        if lat_name is not None and lon_name is not None:
                            lat_vals = crps_by_lead[lat_name].values
                            if lat_vals.size > 1 and lat_vals[0] > lat_vals[-1]:
                                crps_by_lead = crps_by_lead.sortby(lat_name)
                            crps_by_lead = unwrap_longitude_for_plot(crps_by_lead, lon_name)

                    raw_leads = crps_by_lead["lead_time"].values
                    crps_hours = [_to_hour(x, idx) for idx, x in enumerate(raw_leads)]
                    hour_index_pairs = []
                    for h in full_hours:
                        try:
                            idx = crps_hours.index(int(h))
                            hour_index_pairs.append((int(h), idx))
                        except Exception:
                            continue

                    if hour_index_pairs:
                        # 1. Calculation Phase
                        crps_line_rows = []
                        Z_stack: list[np.ndarray] = [] if save_fig else []
                        for h, li in hour_index_pairs:
                            if save_fig:
                                Z = np.asarray(crps_by_lead.isel(lead_time=li).values)
                                Z_stack.append(Z)
                                mean_val = float(np.nanmean(Z))
                            else:
                                mean_val = float(np.nanmean(crps_by_lead.isel(lead_time=li).values))
                            crps_line_rows.append(
                                {
                                    "lead_time_hours": float(h),
                                    "CRPS": mean_val,
                                }
                            )

                        df_crps_line = pd.DataFrame(crps_line_rows).sort_values("lead_time_hours")

                        # 2. Output: CSV
                        if save_fig or save_npz:
                            out_csv_line = section / build_output_filename(
                                metric="crps_line",
                                variable=str(var_name),
                                level=lvl,
                                qualifier="by_lead",
                                init_time_range=init_range_plot,
                                lead_time_range=lead_range_plot,
                                ensemble=ensemble_mode_to_token("prob"),
                                ext="csv",
                            )
                            save_dataframe(
                                df_crps_line, out_csv_line, index=False, module="probabilistic"
                            )

                        # 3. Output: Plots
                        if (
                            save_fig
                            and lat_name is not None
                            and lon_name is not None
                            and len(Z_stack) > 0
                        ):
                            # Panel Grid
                            ncols = 2
                            nrows = (len(hour_index_pairs) + ncols - 1) // ncols
                            fig, axes = plt.subplots(
                                nrows,
                                ncols,
                                figsize=(7.0 * ncols, 4.0 * nrows),
                                dpi=dpi * 2,
                                subplot_kw={"projection": ccrs.PlateCarree()},
                                squeeze=False,
                                constrained_layout=True,
                            )
                            axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

                            Z_stack_np = np.asarray(Z_stack)
                            vmin, vmax = (
                                0.0,
                                float(np.nanmax(Z_stack_np))
                                if np.isfinite(Z_stack_np).any()
                                else 1.0,
                            )

                            first_im = None
                            for i, (h, _) in enumerate(hour_index_pairs):
                                ax = axes_flat[i]
                                if hasattr(ax, "add_feature"):
                                    ax.add_feature(cfeature.COASTLINE, lw=0.5)
                                im = ax.pcolormesh(
                                    crps_by_lead[lon_name],
                                    crps_by_lead[lat_name],
                                    Z_stack[i],
                                    cmap="viridis",
                                    vmin=vmin,
                                    vmax=vmax,
                                    transform=ccrs.PlateCarree(),
                                    shading="auto",
                                )
                                if first_im is None:
                                    first_im = im
                                ax.set_title(f"CRPS (+{int(h)}h)", fontsize=10)

                            for j in range(len(hour_index_pairs), nrows * ncols):
                                axes_flat[j].axis("off")
                            if first_im:
                                cb = fig.colorbar(
                                    first_im,
                                    ax=[
                                        ax
                                        for ax in axes_flat[: len(hour_index_pairs)]
                                        if ax.axes.get_visible()
                                    ],
                                    orientation="vertical",
                                    fraction=0.025,
                                    pad=0.02,
                                )
                                cb.set_label("CRPS")

                            lvl_str = f" @ {format_level_token(lvl)}" if lvl is not None else ""
                            title_text = (
                                f"CRPS Grid by Lead Time — "
                                f"{format_variable_name(str(var_name))}{lvl_str}{date_str}"
                            )
                            plt.suptitle(
                                title_text,
                                fontsize=16,
                                y=1.05,
                            )
                            ens_token_grid = ensemble_mode_to_token("prob")
                            out_png = section / build_output_filename(
                                metric="crps_map",
                                variable=str(var_name),
                                level=lvl,
                                qualifier="grid",
                                init_time_range=init_range_plot,
                                lead_time_range=lead_range_plot,
                                ensemble=ens_token_grid,
                                ext="png",
                            )
                            save_figure(fig, out_png)

                            # Simple Line Plot
                            fig_line, ax_line = plt.subplots(figsize=(7, 3), dpi=dpi * 2)
                            ax_line.plot(
                                df_crps_line["lead_time_hours"], df_crps_line["CRPS"], marker="o"
                            )
                            ax_line.set_xlabel("Lead Time [h]")
                            ax_line.set_ylabel("CRPS")
                            ax_line.set_title(
                                f"CRPS Evolution — {format_variable_name(str(var_name))}{lvl_str}",
                                loc="left",
                                fontsize=10,
                            )
                            ax_line.set_title(date_str, loc="right", fontsize=10)
                            plt.tight_layout()
                            out_png_line = section / build_output_filename(
                                metric="crps_line",
                                variable=str(var_name),
                                level=lvl,
                                qualifier=None,
                                init_time_range=init_range_plot,
                                lead_time_range=lead_range_plot,
                                ensemble=ensemble_mode_to_token("prob"),
                                ext="png",
                            )
                            save_figure(fig_line, out_png_line)

                        # 4. Output: NPZ Data
                        if save_npz:
                            # Only save line data here as 'crps_line' npz
                            out_npz_line = section / build_output_filename(
                                metric="crps_line",
                                variable=str(var_name),
                                level=lvl,
                                qualifier="data",
                                init_time_range=init_range_plot,
                                lead_time_range=lead_range_plot,
                                ensemble=ensemble_mode_to_token("prob"),
                                ext="npz",
                            )
                            save_data(
                                out_npz_line,
                                lead_hours=df_crps_line["lead_time_hours"].values,
                                crps=df_crps_line["CRPS"].values,
                                variable=str(var_name),
                                level=lvl,
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
                        r, col = divmod(i, ncols)
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
                        if col == 0:
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


def _iter_variable_batches(variables: list[str], batch_size: int) -> list[list[str]]:
    if not variables:
        return []
    size = max(1, int(batch_size))
    return [variables[i : i + size] for i in range(0, len(variables), size)]


def _to_python_float(x: xr.DataArray | Any) -> float:
    """Convert scalar-like xarray/dask values to Python float safely."""
    if isinstance(x, xr.DataArray):
        xa = x
        if xa.ndim > 0:
            xa = xa.squeeze(drop=True)
        data = xa.data
        if hasattr(data, "compute"):
            return float(data.compute())
        return float(xa.values)
    if hasattr(x, "compute"):
        return float(x.compute())
    return float(x)


def _wbx_metric_to_df_batched(
    metric: Any,
    ds_prediction: xr.Dataset,
    ds_target: xr.Dataset,
    value_col: str,
    var_batch_size: int,
) -> pd.DataFrame:
    variables = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not variables:
        return pd.DataFrame()

    batches = _iter_variable_batches([str(v) for v in variables], var_batch_size)
    rows: list[pd.DataFrame] = []
    if len(batches) > 1:
        c.print(
            "[Dask] Computing WBX metric "
            f"{value_col}: Processing {len(variables)} variables in {len(batches)} batches "
            f"(batch_size={max(1, int(var_batch_size))})..."
        )

    for batch in batches:
        df_batch = _wbx_metric_to_df(
            metric,
            ds_prediction=ds_prediction[batch],
            ds_target=ds_target[batch],
            value_col=value_col,
        )
        if not df_batch.empty:
            rows.append(df_batch)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).sort_index()


def run_probabilistic_wbx(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    all_cfg: dict[str, Any],
    performance_cfg: dict[str, Any] | None = None,
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
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")

    if "ensemble" not in ds_prediction.dims:
        c.print("[probabilistic] Skipping: model dataset has no 'ensemble' dimension.")
        return

    common_vars = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not common_vars:
        c.print(
            "[probabilistic] No overlapping variables between targets and predictions; "
            "nothing to do."
        )
        return
    ds_pred = ds_prediction[common_vars]
    ds_targ = ds_target[common_vars]

    dynamic_batch = resolve_dynamic_batch_size(
        performance_cfg,
        ds=ds_targ,
    )
    var_batch_size = max(1, min(int(dynamic_batch), max(1, len(common_vars))))

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
            ssr_df = _wbx_metric_to_df_batched(
                ssr_metric,
                ds_prediction=ds_pred,
                ds_target=ds_targ,
                value_col="SSR",
                var_batch_size=var_batch_size,
            )
        except Exception as e:  # pragma: no cover - defensive clarity wrapper
            raise RuntimeError(
                "Failed computing UnbiasedSpreadSkillRatio via WeatherBenchX. "
                "Ensure ensemble size >=2 and variables overlap. Original error: " + str(e)
            ) from e
    else:
        c.print("[probabilistic] Skipping average SSR summary (lead_time > 1 present).")
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
        save_dataframe(ssr_df, ssr_csv, module="probabilistic")

    # Single WBX aggregation pass: reduce init_time and retain lead_time + spatial fields.
    seasonal = (
        bool((plotting_cfg or {}).get("group_by_season", False))
        if isinstance(plotting_cfg, dict)
        else False
    )
    temporal_bin_by = [binning.ByTimeUnit("season", "init_time")] if seasonal else None
    metric_aggregator = aggregation.Aggregator(
        reduce_dims=["init_time"],
        bin_by=temporal_bin_by,
        skipna=True,
    )

    metrics = {}
    metrics["SSR"] = RobustUnbiasedSpreadSkillRatio(ensemble_dim="ensemble")
    metrics["CRPS"] = CRPSEnsemble(ensemble_dim="ensemble")

    variables = [str(v) for v in ds_pred.data_vars]
    variable_batches = _iter_variable_batches(variables, var_batch_size)

    batch_opts = resolve_module_batching_options(
        performance_cfg=performance_cfg,
        default_split_level=True,
        default_split_lead_time=True,
        default_split_init_time=True,
        default_lead_time_block_size=6,
        default_init_time_block_size=6,
    )
    split_lead_time = bool(batch_opts["split_lead_time"])
    lead_time_block_size = int(batch_opts["lead_time_block_size"])
    split_init_time = bool(batch_opts["split_init_time"])
    init_time_block_size = int(batch_opts["init_time_block_size"])
    use_lead_time_batches = (
        split_lead_time
        and ("lead_time" in ds_pred.dims)
        and int(ds_pred.sizes.get("lead_time", 0)) > 1
        and lead_time_block_size > 0
    )
    use_init_time_batches = (
        split_init_time
        and ("init_time" in ds_pred.dims)
        and int(ds_pred.sizes.get("init_time", 0)) > 1
        and init_time_block_size > 0
    )
    if seasonal and use_init_time_batches:
        c.print(
            "[probabilistic] Seasonal binning enabled: disabling init_time batching "
            "for WBX aggregation to preserve seasonal weighting."
        )
        use_init_time_batches = False

    c.print(
        "[Dask] Computing WBX probabilistic metrics: "
        f"Processing {len(variables)} variables in {len(variable_batches)} batches "
        f"(batch_size={var_batch_size})..."
    )

    jobs_meta: list[dict[str, Any]] = []
    for variable_batch in variable_batches:
        ds_pred_batch = ds_pred[variable_batch]
        ds_targ_batch = ds_targ[variable_batch]

        for variable in variable_batch:
            split_specs = build_variable_level_lead_splits(
                ds_pred_batch,
                variables=[str(variable)],
                split_level=True,
                split_lead_time=use_lead_time_batches,
                lead_time_block_size=lead_time_block_size,
                split_init_time=use_init_time_batches,
                init_time_block_size=init_time_block_size,
            )

            for spec in split_specs:
                level_val = spec.get("level")
                lead_slice = spec.get("lead_slice", slice(None))
                init_slice = spec.get("init_slice", slice(None))

                pred_da = ds_pred_batch[str(variable)]
                targ_da = ds_targ_batch[str(variable)]

                if level_val is not None and "level" in pred_da.dims:
                    pred_da = pred_da.sel(level=[level_val])
                if level_val is not None and "level" in targ_da.dims:
                    targ_da = targ_da.sel(level=[level_val])

                if lead_slice != slice(None) and "lead_time" in pred_da.dims:
                    pred_da = pred_da.isel(lead_time=lead_slice)
                if lead_slice != slice(None) and "lead_time" in targ_da.dims:
                    targ_da = targ_da.isel(lead_time=lead_slice)

                if init_slice != slice(None) and "init_time" in pred_da.dims:
                    pred_da = pred_da.isel(init_time=init_slice)
                if init_slice != slice(None) and "init_time" in targ_da.dims:
                    targ_da = targ_da.isel(init_time=init_slice)

                pred_map = {str(variable): pred_da}
                targ_map = {str(variable): targ_da}
                batch_result = aggregation.compute_metric_values_for_single_chunk(
                    metrics, metric_aggregator, pred_map, targ_map
                )

                jobs_meta.append(
                    {
                        "variable": str(variable),
                        "level": level_val,
                        "lead_start": int(spec.get("lead_start", 0)),
                        "init_start": int(spec.get("init_start", 0)),
                        "init_len": int(spec.get("init_len", 1)),
                        "result": batch_result,
                    }
                )

    by_var_level: dict[tuple[str, Any], list[tuple[int, int, int, xr.Dataset]]] = {}
    for job in jobs_meta:
        key = (str(job["variable"]), job.get("level"))
        by_var_level.setdefault(key, []).append(
            (
                int(job.get("lead_start", 0)),
                int(job.get("init_start", 0)),
                int(job.get("init_len", 1)),
                job["result"],
            )
        )

    per_variable_parts: dict[str, list[xr.Dataset]] = {}
    for (variable, level_val), parts in by_var_level.items():
        by_lead_start: dict[int, list[tuple[int, int, xr.Dataset]]] = {}
        for lead_start, init_start, init_len, result in parts:
            by_lead_start.setdefault(int(lead_start), []).append(
                (int(init_start), int(init_len), result)
            )

        lead_parts: list[xr.Dataset] = []
        for lead_start in sorted(by_lead_start.keys()):
            init_parts = sorted(by_lead_start[lead_start], key=lambda item: item[0])
            if len(init_parts) == 1:
                lead_parts.append(init_parts[0][2])
                continue

            weighted_sum: xr.Dataset | None = None
            total_weight = 0
            for _init_start, init_len, result in init_parts:
                weight = max(1, int(init_len))
                weighted_result = result * weight
                weighted_sum = (
                    weighted_result if weighted_sum is None else (weighted_sum + weighted_result)
                )
                total_weight += weight

            if weighted_sum is None:
                continue
            lead_parts.append(weighted_sum / max(1, total_weight))

        if not lead_parts:
            continue
        merged = lead_parts[0] if len(lead_parts) == 1 else xr.concat(lead_parts, dim="lead_time")
        if "lead_time" in merged.dims:
            merged = merged.sortby("lead_time")

        if level_val is not None and "level" not in merged.dims:
            merged = merged.expand_dims(level=[level_val])

        per_variable_parts.setdefault(str(variable), []).append(merged)

    spatial_parts: list[xr.Dataset] = []
    for variable in variables:
        var_parts = per_variable_parts.get(str(variable), [])
        if not var_parts:
            continue
        if len(var_parts) == 1:
            spatial_parts.append(var_parts[0])
            continue

        all_have_level = all("level" in ds_part.dims for ds_part in var_parts)
        if all_have_level:
            var_merged = xr.concat(var_parts, dim="level")
            if "level" in var_merged.dims:
                var_merged = var_merged.sortby("level")
            spatial_parts.append(var_merged)
        else:
            spatial_parts.append(xr.merge(var_parts, compat="override"))

    results_spatial = xr.merge(spatial_parts, compat="override") if spatial_parts else xr.Dataset()
    results_temporal = results_spatial

    if ssr_df.empty:
        ssr_rows_fallback: list[dict[str, Any]] = []
        for var_name, da_metric in results_temporal.data_vars.items():
            if not str(var_name).startswith("SSR"):
                continue
            display_var = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
            red_dims = list(da_metric.dims)
            da_global = da_metric.mean(dim=red_dims, skipna=True) if red_dims else da_metric
            ssr_rows_fallback.append({"variable": display_var, "SSR": _to_python_float(da_global)})

        if ssr_rows_fallback:
            ssr_df_fallback = (
                pd.DataFrame(ssr_rows_fallback).groupby("variable", as_index=False).mean()
            )
            ssr_csv = section / build_output_filename(
                metric="spread_skill_ratio",
                variable=None,
                level=None,
                qualifier="averaged" if (init_range or lead_range) else None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token_prob,
                ext="csv",
            )
            save_dataframe(ssr_df_fallback, ssr_csv, index=False, module="probabilistic")

    # Save individual metric/variable combinations from the single WBX pass.
    if save_npz:
        for var_name, _da in results_spatial.data_vars.items():
            metric_name, variable = var_name.split(".", 1)
            da_metric = results_spatial[var_name]
            if "level" in da_metric.dims:
                for lvl in da_metric["level"].values:
                    npz_path = section / build_output_filename(
                        metric=f"{metric_name.lower()}_spatial_wbx",
                        variable=variable,
                        level=lvl,
                        qualifier=None,
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token_prob,
                        ext="npz",
                    )
                    _save_npz_with_coords(
                        npz_path,
                        da_metric.sel(level=lvl, drop=True),
                        module="probabilistic",
                        level=lvl,
                    )
            else:
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
                _save_npz_with_coords(npz_path, da_metric, module="probabilistic")

    dpi = int((plotting_cfg or {}).get("dpi", 48))
    date_str = extract_date_from_dataset(ds_target) if save_fig else ""

    # --- CRPS outputs from WBX (single CRPS computation path) ---
    crps_summary_rows: list[dict[str, Any]] = []
    crps_summary_rows_per_level: list[dict[str, Any]] = []

    crps_summary_source = results_spatial
    if not any(str(name).startswith("CRPS") for name in crps_summary_source.data_vars):
        crps_summary_source = results_temporal

    for var_name, da_metric in crps_summary_source.data_vars.items():
        if not str(var_name).startswith("CRPS"):
            continue

        display_var = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
        da_global = da_metric
        red_dims_global = list(da_global.dims)
        global_val = (
            da_global.mean(dim=red_dims_global, skipna=True) if red_dims_global else da_global
        )
        crps_summary_rows.append(
            {
                "variable": display_var,
                "CRPS": _to_python_float(global_val),
            }
        )

        if "level" in da_global.dims:
            red_dims_level = [d for d in da_global.dims if d != "level"]
            da_per_level = (
                da_global.mean(dim=red_dims_level, skipna=True) if red_dims_level else da_global
            )
            for lvl in da_per_level["level"].values:
                val = _to_python_float(da_per_level.sel(level=lvl))
                crps_summary_rows_per_level.append(
                    {
                        "variable": display_var,
                        "level": int(lvl) if hasattr(lvl, "item") else lvl,
                        "CRPS": val,
                    }
                )

    if not crps_summary_rows:
        try:
            crps_df_fallback = _wbx_metric_to_df_batched(
                CRPSEnsemble(ensemble_dim="ensemble"),
                ds_prediction=ds_pred,
                ds_target=ds_targ,
                value_col="CRPS",
                var_batch_size=var_batch_size,
            )
            if not crps_df_fallback.empty:
                crps_summary_rows = [
                    {
                        "variable": str(var_name),
                        "CRPS": float(row["CRPS"]),
                    }
                    for var_name, row in crps_df_fallback.iterrows()
                ]
        except Exception:
            pass

    if crps_summary_rows:
        df_crps = pd.DataFrame(crps_summary_rows).groupby("variable", as_index=False).mean()
        out_csv = section / build_output_filename(
            metric="crps_summary",
            variable=None,
            level=None,
            qualifier="averaged" if (init_range or lead_range) else None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token_prob,
            ext="csv",
        )
        save_dataframe(df_crps, out_csv, index=False, module="probabilistic")

    if crps_summary_rows_per_level:
        df_crps_lvl = pd.DataFrame(crps_summary_rows_per_level)
        out_csv_lvl = section / build_output_filename(
            metric="crps_summary",
            variable=None,
            level=None,
            qualifier="per_level",
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token_prob,
            ext="csv",
        )
        save_dataframe(df_crps_lvl, out_csv_lvl, index=False, module="probabilistic")

    crps_plot_source = results_spatial
    if not any(str(name).startswith("CRPS") for name in crps_plot_source.data_vars):
        crps_plot_source = results_temporal

    for var_name, da_metric in crps_plot_source.data_vars.items():
        if not str(var_name).startswith("CRPS"):
            continue

        display_var = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
        levels = [None]
        if "level" in da_metric.dims:
            levels = list(da_metric["level"].values)

        for lvl in levels:
            da = da_metric.sel(level=lvl, drop=True) if lvl is not None else da_metric

            if "region" in da.dims:
                if "global" in da["region"].values:
                    da = da.sel(region="global")
                else:
                    da = da.mean(dim="region", skipna=True)

            lat_name = next((n for n in da.dims if n in ("latitude", "lat", "y")), None)
            lon_name = next((n for n in da.dims if n in ("longitude", "lon", "x")), None)

            has_map_coords = (lat_name is not None) and (lon_name is not None)

            if save_fig and lat_name is not None and lon_name is not None:
                lat_vals = da[lat_name].values
                if lat_vals.size > 1 and lat_vals[0] > lat_vals[-1]:
                    da = da.sortby(lat_name)
                da = unwrap_longitude_for_plot(da, lon_name)

            # Per-lead line and optional map grid
            if "lead_time" in da.dims and int(da.sizes.get("lead_time", 0)) > 1:
                reduce_for_line = [d for d in da.dims if d not in ("lead_time",)]
                crps_line = da.mean(dim=reduce_for_line, skipna=True) if reduce_for_line else da

                leads = crps_line["lead_time"].values
                if np.issubdtype(np.asarray(leads).dtype, np.timedelta64):
                    lead_hours = (leads / np.timedelta64(1, "h")).astype(int)
                else:
                    lead_hours = np.asarray(leads).astype(int)
                values = np.asarray(crps_line.values, dtype=float)

                df_line = pd.DataFrame(
                    {
                        "lead_time_hours": lead_hours,
                        "CRPS": values,
                    }
                )

                out_csv_line = section / build_output_filename(
                    metric="crps_line",
                    variable=display_var,
                    level=lvl,
                    qualifier="by_lead",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token_prob,
                    ext="csv",
                )
                save_dataframe(df_line, out_csv_line, index=False, module="probabilistic")

                if save_npz:
                    out_npz_line = section / build_output_filename(
                        metric="crps_line",
                        variable=display_var,
                        level=lvl,
                        qualifier="data",
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token_prob,
                        ext="npz",
                    )
                    save_data(
                        out_npz_line,
                        lead_hours=df_line["lead_time_hours"].values,
                        crps=df_line["CRPS"].values,
                        variable=display_var,
                        level=lvl,
                        module="probabilistic",
                    )

                if save_fig:
                    fig_line, ax_line = plt.subplots(figsize=(7, 3), dpi=dpi * 2)
                    ax_line.plot(df_line["lead_time_hours"], df_line["CRPS"], marker="o")
                    lvl_str = f" @ {format_level_token(lvl)}" if lvl is not None else ""
                    ax_line.set_title(
                        f"CRPS Evolution — {format_variable_name(display_var)}{lvl_str}",
                        loc="left",
                        fontsize=10,
                    )
                    ax_line.set_title(date_str, loc="right", fontsize=10)
                    ax_line.set_xlabel("Lead Time [h]")
                    ax_line.set_ylabel("CRPS")
                    ax_line.grid(True, linestyle="--", alpha=0.6)
                    out_png_line = section / build_output_filename(
                        metric="crps_line",
                        variable=display_var,
                        level=lvl,
                        qualifier=None,
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token_prob,
                        ext="png",
                    )
                    save_figure(fig_line, out_png_line, module="probabilistic")

                    if not has_map_coords:
                        continue

                    reduce_for_map = [
                        d for d in da.dims if d not in ("lead_time", lat_name, lon_name)
                    ]
                    crps_by_lead_map = (
                        da.mean(dim=reduce_for_map, skipna=True) if reduce_for_map else da
                    )

                    n_leads = int(crps_by_lead_map.sizes.get("lead_time", 0))
                    if n_leads > 0:
                        ncols = 2
                        nrows = (n_leads + ncols - 1) // ncols
                        fig_grid, axes = plt.subplots(
                            nrows,
                            ncols,
                            figsize=(7.0 * ncols, 4.0 * nrows),
                            dpi=dpi * 2,
                            subplot_kw={"projection": ccrs.PlateCarree()},
                            squeeze=False,
                            constrained_layout=True,
                        )
                        axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
                        z_stack = np.asarray(crps_by_lead_map.values)
                        vmax = float(np.nanmax(z_stack)) if np.isfinite(z_stack).any() else 1.0
                        first_im = None
                        for idx in range(n_leads):
                            ax = axes_flat[idx]
                            if hasattr(ax, "add_feature"):
                                ax.add_feature(cfeature.COASTLINE, lw=0.5)
                            z = np.asarray(crps_by_lead_map.isel(lead_time=idx).values)
                            im = ax.pcolormesh(
                                crps_by_lead_map[lon_name],
                                crps_by_lead_map[lat_name],
                                z,
                                cmap="viridis",
                                vmin=0.0,
                                vmax=vmax,
                                transform=ccrs.PlateCarree(),
                                shading="auto",
                            )
                            if first_im is None:
                                first_im = im
                            ax.set_title(f"CRPS (+{int(lead_hours[idx])}h)", fontsize=10)

                        for idx in range(n_leads, nrows * ncols):
                            if hasattr(axes_flat[idx], "axis"):
                                axes_flat[idx].axis("off")

                        if first_im is not None:
                            cb = fig_grid.colorbar(
                                first_im,
                                ax=axes_flat[:n_leads],
                                orientation="vertical",
                                fraction=0.025,
                                pad=0.02,
                            )
                            cb.set_label("CRPS")

                        lvl_str = f" @ {format_level_token(lvl)}" if lvl is not None else ""
                        fig_grid.suptitle(
                            (
                                "CRPS Grid by Lead Time — "
                                f"{format_variable_name(display_var)}{lvl_str}{date_str}"
                            ),
                            fontsize=16,
                            y=1.05,
                        )
                        out_png_grid = section / build_output_filename(
                            metric="crps_map",
                            variable=display_var,
                            level=lvl,
                            qualifier="grid",
                            init_time_range=init_range,
                            lead_time_range=lead_range,
                            ensemble=ens_token_prob,
                            ext="png",
                        )
                        save_figure(fig_grid, out_png_grid, module="probabilistic")

            elif save_fig and has_map_coords:
                reduce_for_map = [d for d in da.dims if d not in (lat_name, lon_name)]
                crps_map = da.mean(dim=reduce_for_map, skipna=True) if reduce_for_map else da

                fig = plt.figure(figsize=(12, 6), dpi=dpi * 2)
                ax = plt.axes(projection=ccrs.PlateCarree())
                if hasattr(ax, "add_feature"):
                    ax.add_feature(cfeature.COASTLINE, lw=0.5)
                z = np.asarray(crps_map.values)
                vmax = float(np.nanmax(z)) if np.isfinite(z).any() else 1.0
                mesh = ax.pcolormesh(
                    crps_map[lon_name],
                    crps_map[lat_name],
                    z,
                    cmap="viridis",
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                    vmin=0.0,
                    vmax=vmax,
                )
                plt.colorbar(
                    mesh,
                    ax=ax,
                    orientation="horizontal",
                    pad=0.05,
                    shrink=0.8,
                    label="CRPS",
                )
                lvl_str = f" @ {format_level_token(lvl)}" if lvl is not None else ""
                ax.set_title(
                    f"CRPS Map (Mean) — {format_variable_name(display_var)}{lvl_str}",
                    loc="left",
                    fontsize=10,
                )
                ax.set_title(date_str, loc="right", fontsize=10)
                out_png = section / build_output_filename(
                    metric="crps_map",
                    variable=display_var,
                    level=lvl,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token_prob,
                    ext="png",
                )
                save_figure(fig, out_png, module="probabilistic")

    ssr_rows_per_level: list[dict[str, Any]] = []
    for var_name, da_metric in results_temporal.data_vars.items():
        if not str(var_name).startswith("SSR") or "level" not in da_metric.dims:
            continue
        variable = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
        red_dims = [d for d in da_metric.dims if d != "level"]
        da_level = da_metric.mean(dim=red_dims, skipna=True) if red_dims else da_metric
        for lvl in da_level["level"].values:
            val = _to_python_float(da_level.sel(level=lvl))
            if val is None:
                continue
            ssr_rows_per_level.append(
                {
                    "variable": variable,
                    "level": int(lvl) if hasattr(lvl, "item") else lvl,
                    "SSR": float(val),
                }
            )

    if ssr_rows_per_level:
        ssr_df_per_level = pd.DataFrame(ssr_rows_per_level)
        ssr_csv_per_level = section / build_output_filename(
            metric="spread_skill_ratio",
            variable=None,
            level=None,
            qualifier="per_level",
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token_prob,
            ext="csv",
        )
        save_dataframe(ssr_df_per_level, ssr_csv_per_level, index=False, module="probabilistic")

    for var_name in results_temporal.data_vars:
        if not str(var_name).startswith("SSR"):
            continue

        da_base = results_temporal[var_name]
        display_var = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
        levels = [None]
        if "level" in da_base.dims:
            levels = list(da_base["level"].values)

        for lvl in levels:
            da = da_base.sel(level=lvl, drop=True) if lvl is not None else da_base

            if "lead_time" not in da.dims or da.sizes["lead_time"] <= 1:
                continue

            reduce_dims = [d for d in da.dims if d != "lead_time"]
            da_line = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da

            leads = da_line["lead_time"].values
            if np.issubdtype(np.asarray(leads).dtype, np.timedelta64):
                lead_hours = (leads / np.timedelta64(1, "h")).astype(int)
            else:
                lead_hours = np.asarray(leads).astype(int)
            values = np.asarray(da_line.values, dtype=float)

            df_save = pd.DataFrame({"lead_time_hours": lead_hours, "SSR": values})
            out_csv_line = section / build_output_filename(
                metric="ssr_line",
                variable=display_var,
                level=lvl,
                qualifier="by_lead",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token_prob,
                ext="csv",
            )
            save_dataframe(df_save, out_csv_line, index=False, module="probabilistic")

    # --- Plotting SSR (Temporal and Spatial) ---
    if save_fig:
        # Plot Temporal (Time Series)
        # We use results_temporal because it preserves the time dimension
        for var_name in results_temporal.data_vars:
            if not str(var_name).startswith("SSR"):
                continue

            da_base = results_temporal[var_name]
            display_var = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
            levels = [None]
            if "level" in da_base.dims:
                levels = list(da_base["level"].values)

            for lvl in levels:
                da = da_base.sel(level=lvl, drop=True) if lvl is not None else da_base

                # Skip if lead_time dimension is size <= 1
                if "lead_time" in da.dims and da.sizes["lead_time"] <= 1:
                    continue

                reduce_dims = [d for d in da.dims if d != "lead_time"]
                da = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da

                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Convert to dataframe for line plot
                df = da.to_dataframe(name="SSR").reset_index()

                # Determine x-axis column (should be lead_time)
                x_col = "lead_time" if "lead_time" in df.columns else df.columns[0]

                # Convert timedelta to hours if needed
                if pd.api.types.is_timedelta64_dtype(df[x_col]):
                    df[x_col] = (df[x_col] / pd.Timedelta(hours=1)).astype(int)

                # Plot line
                ax.plot(df[x_col], df["SSR"], marker="o")

                lvl_str = f" @ {format_level_token(lvl)}" if lvl is not None else ""
                ax.set_title(
                    f"SSR over Lead Time — {format_variable_name(display_var)}{lvl_str}",
                    loc="left",
                    fontsize=10,
                )
                ax.set_title(date_str, loc="right", fontsize=10)
                ax.set_ylabel("SSR")
                ax.set_xlabel("Lead Time [h]")
                ax.grid(True, axis="y")

                out_png_temp = section / build_output_filename(
                    metric="wbx_temporal",
                    variable=display_var,
                    level=lvl,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token_prob,
                    ext="png",
                )
                save_figure(fig, out_png_temp, module="probabilistic")

        # Plot Regional Comparison (Bar Chart of Regions)
        # Only plot if lead_time dimension is not present or has size 1 (i.e., not multi-lead)
        for var_name in results_temporal.data_vars:
            if not str(var_name).startswith("SSR"):
                continue

            da_base = results_temporal[var_name]
            display_var = str(var_name).split(".", 1)[1] if "." in str(var_name) else str(var_name)
            levels = [None]
            if "level" in da_base.dims:
                levels = list(da_base["level"].values)

            for lvl in levels:
                da = da_base.sel(level=lvl, drop=True) if lvl is not None else da_base

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
                        x_pos = np.arange(len(s_spatial), dtype=float)
                        ax.bar(x_pos, s_spatial.values)
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels([str(i) for i in s_spatial.index])
                        lvl_str = f" @ {format_level_token(lvl)}" if lvl is not None else ""
                        ax.set_title(
                            f"SSR by Region (Time-Averaged) — "
                            f"{format_variable_name(display_var)}{lvl_str}",
                            loc="left",
                            fontsize=10,
                        )
                        ax.set_title(date_str, loc="right", fontsize=10)
                        ax.set_ylabel("SSR")
                        ax.set_xlabel("")  # Remove Region label
                        ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Ideal (1.0)")

                        # Clean up legend labels
                        if hasattr(ax, "get_legend_handles_labels"):
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
                        elif hasattr(ax, "legend"):
                            ax.legend(fontsize=10)

                        if hasattr(ax, "tick_params"):
                            ax.tick_params(axis="x", labelrotation=30)
                        plt.tight_layout()

                        out_png_spatial = section / build_output_filename(
                            metric="wbx_spatial",
                            variable=display_var,
                            level=lvl,
                            qualifier=None,
                            init_time_range=init_range,
                            lead_time_range=lead_range,
                            ensemble=ens_token_prob,
                            ext="png",
                        )
                        save_figure(fig, out_png_spatial, module="probabilistic")
                    else:
                        c.print(
                            f"[probabilistic] Skipping spatial plot for {var_name}: "
                            f"No numeric data."
                        )
