from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from . import console as c, data as data_mod
from .dask_utils import (
    describe_batch_size_mode,
    resolve_dynamic_batch_details,
    resolve_dynamic_batch_size,
)
from .helpers import (
    format_ensemble_log,
    resolve_ensemble_mode,
    validate_and_normalize_ensemble_config,
)
from .lead_time_policy import (
    LeadTimePolicy,
    apply_lead_time_selection,
    parse_lead_time_policy,
)

# Public: expected module subdirectories produced by the evaluation pipeline
# These names correspond to folders created under each model output directory.
EXPECTED_SUBDIRS: set[str] = {
    "deterministic",
    "probabilistic",
    "energy_spectra",
    "ets",
    "histograms",
    "wd_kde",
    "maps",
    "vertical_profiles",
}


def _ensemble_handling_message(
    ds_prediction: xr.Dataset, cfg: dict[str, Any], resolved_modes: dict[str, str] | None = None
) -> str:
    if "ensemble" not in ds_prediction.dims:
        return "Ensemble: No 'ensemble' dimension present."

    # Ensemble present
    ens_size = ds_prediction.sizes.get("ensemble", -1)
    base_msg = f"Ensemble Size: {ens_size}."

    if ens_size == 1:
        return f"{base_msg} Ensemble settings disregarded; probabilistic module disabled."

    if resolved_modes:
        modules_cfg = cfg.get("modules", {})
        # Filter modes by enabled modules
        active_modes = set()
        for module, mode in resolved_modes.items():
            if modules_cfg.get(module):
                active_modes.add(mode)

        details = []
        if "prob" in active_modes:
            details.append("Probabilistic")
        if "members" in active_modes:
            details.append("Members")
        if "pooled" in active_modes:
            details.append("Pooled")

        if details:
            return f"{base_msg} Active evaluation modes: {', '.join(details)}."

        if "mean" in active_modes:
            return f"{base_msg} All modules use mean reduction."

        return f"{base_msg} No ensemble operations active."

    # Fallback if resolved_modes not provided
    modules_cfg = cfg.get("modules", {})
    if bool(modules_cfg.get("probabilistic")):
        return f"{base_msg} Probabilistic evaluation enabled."

    return f"{base_msg} Deterministic/Mean evaluation."


def _parse_time_ranges(values) -> list[tuple[str | None, str | None]]:
    if not isinstance(values, list | tuple):  # ruff UP038
        return []
    # Case: exactly two entries without ':' → treat as single [start, end]
    if (
        len(values) == 2
        and all(isinstance(v, str) for v in values)
        and all(":" not in v for v in values)  # plain bounds
    ):
        return [(values[0], values[1])]
    ranges: list[tuple[str | None, str | None]] = []
    for it in values:
        if isinstance(it, list | tuple) and len(it) == 2:  # ruff UP038
            s, e = it[0], it[1]
            ranges.append(
                (
                    str(s) if s else None,
                    str(e) if e else None,
                )
            )
        elif isinstance(it, str) and ":" in it:
            s, e = it.split(":", 1)
            ranges.append((s or None, e or None))
        elif isinstance(it, str):
            # A single timestamp → treat as [t, t]
            ranges.append((it, it))
    return ranges


def _slice_common(ds: xr.Dataset, cfg: dict[str, Any], extend_end_hours: int = 0) -> xr.Dataset:
    # Ensure selection dict exists in config so we can update it if needed
    sel = cfg.setdefault("selection", {})
    levels: list[int] | None = sel.get("levels")

    # If levels are not specified but present in the dataset, select ALL available levels
    # and update the configuration so downstream modules are aware of the full list.
    if levels is None and "level" in ds.dims:
        try:
            # Use sorted list of available levels
            all_levels = sorted(ds.coords["level"].values.tolist())
            sel["levels"] = all_levels
            levels = all_levels
        except Exception:
            # Fallback: if we can't read levels, proceed without selection
            pass

    latitudes: list[float] | None = sel.get("latitudes")
    longitudes: list[float] | None = sel.get("longitudes")
    datetimes: list[str] | None = sel.get("datetimes")
    datetimes_list: list[str] | None = sel.get("datetimes_list")

    if levels is not None and "level" in ds.dims:
        try:
            available = set(ds.coords["level"].values.tolist())
        except Exception:
            available = set()
        requested = list(levels)
        present = [lv for lv in requested if lv in available]
        missing = [lv for lv in requested if lv not in available]
        if missing:
            raise KeyError(
                f"Requested pressure levels not found: {missing}. Available: {sorted(available)}"
            )
        if present:
            ds = ds.sel(level=present)

        else:
            if requested:
                c.warn(
                    "None of the requested pressure levels are present; "
                    f"requested={requested}, available={sorted(available)}. "
                    "Skipping level selection."
                )

    if latitudes is not None and "latitude" in ds.dims:
        if len(latitudes) == 0:
            raise ValueError("Empty list provided for latitudes; at least one value is required.")
        # Latitude is NOT cyclic. Support a single contiguous band and adapt slice order to
        # the coordinate orientation (ascending or descending) to avoid empty selections.
        if isinstance(latitudes, (list | tuple)) and len(latitudes) == 2:  # ruff UP038
            lat0, lat1 = float(latitudes[0]), float(latitudes[1])
            # Detect axis orientation (monotonic increasing vs decreasing)
            vals = ds["latitude"].values
            asc = bool(vals[1] >= vals[0]) if vals.size >= 2 else True
            lo = min(lat0, lat1)
            hi = max(lat0, lat1)
            # For descending coordinates, slice should be (hi -> lo)
            slc = slice(lo, hi) if asc else slice(hi, lo)
            ds = ds.sel(latitude=slc)
        elif len(latitudes) == 1:
            ds = ds.sel(latitude=slice(latitudes[0], latitudes[0]))
        else:
            ds = ds.sel(latitude=slice(*latitudes))
        # Guard against empty spatial selection early
        if "latitude" in ds.dims and int(ds.sizes.get("latitude", 0)) == 0:
            raise ValueError(
                "Latitude selection resulted in an empty dataset. "
                f"Requested latitudes={latitudes}. Check bounds and coordinate convention."
            )
    if longitudes is not None and "longitude" in ds.dims:
        if len(longitudes) == 0:
            raise ValueError("Empty list provided for longitudes; at least one value is required.")
        # For longitude: if first <= second, apply a normal slice.
        # If first > second, select union of [first..max_lon] ∪ [min_lon..second].
        # Works regardless of coordinate convention (0..360 or -180..180). We sort ascending
        # for deterministic behavior before masking.
        if isinstance(longitudes, (list | tuple)) and len(longitudes) == 2:  # ruff UP038
            lon0, lon1 = float(longitudes[0]), float(longitudes[1])
            if lon0 > lon1:
                ds = ds.sortby("longitude", ascending=True)
                lon_vals = ds["longitude"]
                mask = (lon_vals >= lon0) | (lon_vals <= lon1)
                ds = ds.sel(longitude=lon_vals[mask])
            else:
                ds = ds.sel(longitude=slice(lon0, lon1))
        elif len(longitudes) == 1:
            ds = ds.sel(longitude=slice(longitudes[0], longitudes[0]))
        else:
            ds = ds.sel(longitude=slice(*longitudes))
        # Guard against empty spatial selection early
        if "longitude" in ds.dims and int(ds.sizes.get("longitude", 0)) == 0:
            raise ValueError(
                "Longitude selection resulted in an empty dataset. "
                f"Requested longitudes={longitudes}. Check bounds and coordinate "
                "convention (0–360 vs -180..180)."
            )

    # Non-contiguous explicit timestamps take precedence if provided
    if datetimes_list is not None and len(datetimes_list) > 0:
        try:
            req = [np.datetime64(x).astype("datetime64[ns]") for x in datetimes_list]
        except Exception:
            req = list(datetimes_list)
        dim_name = (
            "init_time" if "init_time" in ds.dims else ("time" if "time" in ds.dims else None)
        )
        if dim_name is not None:
            try:
                available = set(ds[dim_name].values.tolist())
            except Exception:
                available = set()
            present = [x for x in req if x in available]
            missing = [x for x in req if x not in available]
            if missing:
                raise KeyError(
                    "Requested timestamps not found in "
                    f"{dim_name}: {missing[:6]}"
                    f"{' ...' if len(missing) > 6 else ''}"
                )
            if present:
                ds = ds.sel({dim_name: present})
        return ds

    if datetimes is not None:
        dim_name = (
            "init_time" if "init_time" in ds.dims else ("time" if "time" in ds.dims else None)
        )
        if dim_name is None:
            return ds

        ranges = _parse_time_ranges(datetimes)
        if not ranges:
            if len(datetimes) >= 2:
                # If selecting on 'time' (typical for ERA5 targets) and multi-lead with a
                # max_hour horizon is configured, extend the upper bound so that
                # valid_time = init+lead remains covered by the ground-truth window.
                end = datetimes[1]
                try:
                    if dim_name == "time" and extend_end_hours > 0:
                        end_dt = np.datetime64(end).astype("datetime64[ns]")
                        end_ext = end_dt + np.timedelta64(int(extend_end_hours), "h")
                        end = str(end_ext)
                    else:
                        lt_cfg = (cfg or {}).get("lead_time", {})
                        mode = str(lt_cfg.get("mode", "first")).lower()
                        max_h = lt_cfg.get("max_hour")
                        if dim_name == "time" and mode != "first" and max_h is not None:
                            end_dt = np.datetime64(end).astype("datetime64[ns]")
                            end_ext = end_dt + np.timedelta64(int(max_h), "h")
                            end = str(end_ext)
                except Exception:
                    # If any error occurs while extending the end bound, fall back to the original
                    # end value.
                    pass
                ds = ds.sel({dim_name: slice(datetimes[0], end)})
            return ds

        vals = ds[dim_name].values.astype("datetime64[ns]")
        mask_arr = np.zeros(vals.shape, dtype=bool)
        for start_s, end_s in ranges:
            try:
                start = np.datetime64(start_s).astype("datetime64[ns]") if start_s else vals.min()
            except Exception:
                start = vals.min()
            try:
                end = np.datetime64(end_s).astype("datetime64[ns]") if end_s else vals.max()
            except Exception:
                end = vals.max()
            # Extend end bound for ERA5 targets when multi-lead horizon is requested
            try:
                if dim_name == "time" and extend_end_hours > 0:
                    end = end + np.timedelta64(int(extend_end_hours), "h")
                else:
                    lt_cfg = (cfg or {}).get("lead_time", {})
                    mode = str(lt_cfg.get("mode", "first")).lower()
                    max_h = lt_cfg.get("max_hour")
                    if dim_name == "time" and mode != "first" and max_h is not None:
                        end = end + np.timedelta64(int(max_h), "h")
            except Exception:
                # If any error occurs while extending the end bound,
                # fall back to the original end value.
                pass
            mask_arr |= (vals >= start) & (vals <= end)

        count = int(mask_arr.sum())
        if count == 0:
            msg = f"No timestamps within requested ranges on {dim_name}."
            raise KeyError(msg)
        # isel by positions retains labels and avoids building a long explicit label list
        idx = np.nonzero(mask_arr)[0]
        ds = ds.isel({dim_name: idx})
        return ds

    return ds


def _apply_temporal_resolution(ds: xr.Dataset, hours: int | None) -> xr.Dataset:
    """Downsample temporal axes to the requested hourly resolution.

    Preference order:
    - If both init_time and lead_time exist, stride along lead_time if it is evenly spaced.
    - Else, stride along init_time if present.
    - Else, no-op.
    """
    if hours is None:
        return ds
    # (debug logging removed)
    if "lead_time" in ds.dims and ds.lead_time.size >= 2:
        step_ns = int(
            (ds.lead_time[1] - ds.lead_time[0]).astype("timedelta64[h]") / np.timedelta64(1, "h")
        )
        step_ns = max(1, step_ns)
        factor = max(1, hours // step_ns)
        out = ds.isel(lead_time=slice(None, None, factor))
        # (debug logging removed)
        return out
    if "init_time" in ds.dims and ds.init_time.size >= 2:
        # Only stride if cadence appears uniform; else skip to avoid dropping labels.
        try:
            vals = ds.init_time.values.astype("datetime64[h]").astype("int64")
            diffs = np.diff(vals)
            is_uniform = diffs.size > 0 and np.all(diffs == diffs[0])
        except Exception:
            is_uniform = False
        if is_uniform:
            dt_hours = int(diffs[0]) if diffs.size > 0 else 1
            dt_hours = max(1, dt_hours)
            factor = max(1, hours // dt_hours)
            out = ds.isel(init_time=slice(None, None, factor))
            # (debug logging removed)
            return out
        else:
            c.warn(
                "Requested temporal downsampling on irregular init_time cadence → skipping stride."
            )
            return ds
    return ds


def _select_variables(
    ds: xr.Dataset,
    variables_2d: Iterable[str] | None,
    variables_3d: Iterable[str] | None,
) -> xr.Dataset:
    vars_sel: list[str] = []
    if variables_2d:
        vars_sel.extend([v for v in variables_2d if v in ds.data_vars])
    if variables_3d:
        vars_sel.extend([v for v in variables_3d if v in ds.data_vars])
    if vars_sel:
        return ds[vars_sel]
    return ds


def _parse_iso_datetime(value: str) -> np.datetime64:
    return np.datetime64(value).astype("datetime64[ns]")


def _select_plot_datetime(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    cfg: dict[str, Any],
) -> tuple[xr.Dataset, xr.Dataset]:
    """If plotting.plot_datetime is set, select that init_time for plotting datasets.

    Validates that the requested datetime lies within selection.datetimes range (if provided)
    and matches an available init_time label. Returns filtered copies with a length-1
    init_time dimension. If not set, defaults to selecting the first available init_time.
    """
    plot_dt_str = (cfg.get("plotting", {}) or {}).get("plot_datetime")
    if not plot_dt_str:
        # Default behavior: pick the first available init_time from predictions
        if "init_time" in ds_prediction.dims and int(ds_prediction.init_time.size) > 0:
            plot_dt = ds_prediction["init_time"].values[0]
            ds_target_plot = (
                ds_target.sel(init_time=[plot_dt]) if "init_time" in ds_target.dims else ds_target
            )
            ds_prediction_plot = ds_prediction.sel(init_time=[plot_dt])
            return ds_target_plot, ds_prediction_plot
        return ds_target, ds_prediction

    # Validate against selection.datetimes boundaries, if present
    sel = cfg.get("selection", {}) or {}
    bounds = sel.get("datetimes")
    plot_dt = _parse_iso_datetime(plot_dt_str)
    if bounds and len(bounds) == 2 and all(bounds):
        start = _parse_iso_datetime(bounds[0])
        end = _parse_iso_datetime(bounds[1])
        if not (start <= plot_dt <= end):
            raise ValueError(
                f"plot_datetime={plot_dt_str} must lie within selection.datetimes={bounds}"
            )

    if "init_time" not in ds_prediction.dims:
        raise ValueError("plot_datetime requires datasets with 'init_time' dimension.")

    # Ensure exact label present in predictions (targets are aligned to prediction labels)
    available = ds_prediction["init_time"].values
    if plot_dt not in available:
        raise ValueError(
            "Requested plot_datetime not found in predictions init_time. "
            f"Requested={plot_dt_str}. Available examples: {available[:8]} "
            f"(total {available.size})."
        )

    ds_target_plot = (
        ds_target.sel(init_time=[plot_dt]) if "init_time" in ds_target.dims else ds_target
    )
    ds_prediction_plot = ds_prediction.sel(init_time=[plot_dt])
    return ds_target_plot, ds_prediction_plot


def _select_plot_ensemble(
    ds_prediction: xr.Dataset,
    ds_prediction_std: xr.Dataset,
    cfg: dict[str, Any],
) -> tuple[xr.Dataset, xr.Dataset]:
    """Optionally select specific ensemble members for plotting datasets.

    plotting.plot_ensemble_members: list of integer indices. If provided, subset
    the 'ensemble' dimension in predictions and predictions_std to those indices.
    Targets are unaffected (typically non-ensemble). If 'ensemble' is absent and
    a list is provided, raise a ValueError.
    """
    members = (cfg.get("plotting", {}) or {}).get("plot_ensemble_members")
    if not members:
        return ds_prediction, ds_prediction_std
    if "ensemble" not in ds_prediction.dims:
        raise ValueError(
            "plot_ensemble_members specified but 'ensemble' dim not present in predictions."
        )
    # Validate indices
    ens_size = int(ds_prediction.sizes.get("ensemble", 0))
    idx = [int(m) for m in members]
    if any((m < 0 or m >= ens_size) for m in idx):
        raise ValueError(
            "plot_ensemble_members indices out of range. Requested="
            f"{idx}, available range=0..{ens_size - 1}."
        )
    ds_prediction = ds_prediction.isel(ensemble=idx)
    if "ensemble" in ds_prediction_std.dims:
        ds_prediction_std = ds_prediction_std.isel(ensemble=idx)
    return ds_prediction, ds_prediction_std


def _standardize_pair(
    targets: xr.Dataset, predictions: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    # Explicit join ensures compatibility with upcoming xarray default change
    combined = xr.concat([targets, predictions], dim="__concat__", join="outer")
    mean = combined.mean()
    std = combined.std()
    return (targets - mean) / std, (predictions - mean) / std


def validate_requirements(ds: xr.Dataset, cfg: dict[str, Any], dataset_name: str) -> list[str]:
    errors = []
    sel = cfg.get("selection", {})

    # 1. Variables
    variables_2d = sel.get("variables_2d") or []
    variables_3d = sel.get("variables_3d") or []
    requested_vars = set(variables_2d) | set(variables_3d)

    missing_vars = [v for v in requested_vars if v not in ds.data_vars]
    if missing_vars:
        errors.append(f"{dataset_name}: Missing variables: {sorted(missing_vars)}")

    # 2. Levels
    levels = sel.get("levels")
    if levels:
        if "level" in ds.dims:
            available_levels = set(ds["level"].values.tolist())
            missing_levels = [l_val for l_val in levels if l_val not in available_levels]
            if missing_levels:
                errors.append(f"{dataset_name}: Missing pressure levels: {sorted(missing_levels)}")
        elif variables_3d:
            # If we have 3D variables but no level dimension, and levels are requested
            errors.append(
                f"{dataset_name}: Missing 'level' dimension "
                f"but levels {levels} requested for 3D variables."
            )

    # 3. Time
    datetimes = sel.get("datetimes")
    datetimes_list = sel.get("datetimes_list")

    dim_name = "init_time" if "init_time" in ds.dims else ("time" if "time" in ds.dims else None)

    if dim_name:
        try:
            available_times = ds[dim_name].values.astype("datetime64[ns]")
        except Exception:
            available_times = ds[dim_name].values

        if datetimes_list:
            try:
                req_times = [np.datetime64(x).astype("datetime64[ns]") for x in datetimes_list]
            except Exception:
                req_times = []

            available_set = set(available_times)
            missing_times = [str(t) for t in req_times if t not in available_set]
            if missing_times:
                errors.append(
                    f"{dataset_name}: Missing timestamps (from datetimes_list): "
                    f"{len(missing_times)} missing, e.g. {missing_times[:3]}"
                )

        elif datetimes:
            ranges = _parse_time_ranges(datetimes)
            if ranges and len(available_times) > 0:
                min_time = available_times.min()
                max_time = available_times.max()

                for start_s, end_s in ranges:
                    try:
                        start = (
                            np.datetime64(start_s).astype("datetime64[ns]") if start_s else min_time
                        )
                    except Exception:
                        start = min_time
                    try:
                        end = np.datetime64(end_s).astype("datetime64[ns]") if end_s else max_time
                    except Exception:
                        end = max_time

                    if start < min_time or end > max_time:
                        errors.append(
                            f"{dataset_name}: Requested time range [{start}, {end}] is not fully"
                            f" covered by available range [{min_time}, {max_time}]"
                        )
            elif ranges and len(available_times) == 0:
                errors.append(f"{dataset_name}: No timestamps available.")

    else:
        if datetimes or datetimes_list:
            errors.append(
                f"{dataset_name}: Missing time dimension ('init_time' or 'time') but time "
                f"selection requested."
            )

    # 4. Lat/Lon
    latitudes = sel.get("latitudes")
    if latitudes and "latitude" in ds.dims:
        req_lat_min = min(latitudes)
        req_lat_max = max(latitudes)
        avail_lat_min = ds.latitude.min().item()
        avail_lat_max = ds.latitude.max().item()

        tol = 0.01
        if req_lat_min < avail_lat_min - tol or req_lat_max > avail_lat_max + tol:
            errors.append(
                f"{dataset_name}: Requested latitude range [{req_lat_min}, {req_lat_max}] is not "
                f"fully covered by available range [{avail_lat_min:.2f}, {avail_lat_max:.2f}]"
            )

    longitudes = sel.get("longitudes")
    if longitudes and "longitude" in ds.dims:
        req_lon_min = min(longitudes)
        req_lon_max = max(longitudes)
        avail_lon_min = ds.longitude.min().item()
        avail_lon_max = ds.longitude.max().item()

        tol = 0.01
        if req_lon_min < avail_lon_min - tol or req_lon_max > avail_lon_max + tol:
            errors.append(
                f"{dataset_name}: Requested longitude range [{req_lon_min}, {req_lon_max}] is not "
                f"fully covered by available range [{avail_lon_min:.2f}, {avail_lon_max:.2f}]"
            )

    return errors


def prepare_datasets(
    cfg: dict[str, Any],
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    sel = cfg.get("selection", {})
    paths = cfg.get("paths", {})
    variables_2d = sel.get("variables_2d")
    variables_3d = sel.get("variables_3d")
    hours = sel.get("temporal_resolution_hours")
    # Handle ensemble selection (new plural key) with backward compatibility for legacy singular
    if "ensemble_members" in sel:
        ensemble_members = sel.get("ensemble_members")
    else:
        ensemble_members = sel.get("ensemble_member")
        if ensemble_members is not None:
            c.warn(
                "Config key 'selection.ensemble_member' is deprecated; "
                "use 'selection.ensemble_members'."
            )
    # Normalize to int | list[int] | None
    if isinstance(ensemble_members, list):
        try:
            ensemble_members = [int(i) for i in ensemble_members]
        except Exception:
            c.warn("Invalid values in ensemble_members list; ignoring selection.")
            ensemble_members = None
        if isinstance(ensemble_members, list) and len(ensemble_members) == 1:
            ensemble_members = ensemble_members[0]

    # Open datasets from paths with optional variable selection
    var_list = None
    if variables_2d or variables_3d:
        var_list = list(variables_2d or []) + list(variables_3d or [])

    target_path = paths.get("target") or paths.get("nwp")
    prediction_path = paths.get("prediction") or paths.get("ml")

    ds_target = data_mod.open_target(target_path, variables=var_list)
    ds_prediction = data_mod.open_prediction(prediction_path, variables=var_list)

    # Validate requirements
    errors = []
    errors.extend(validate_requirements(ds_target, cfg, "Target"))
    errors.extend(validate_requirements(ds_prediction, cfg, "Prediction"))

    if errors:
        raise ValueError("Missing data requirements:\n" + "\n".join(errors))

    # Debug visibility: show raw lead_time counts in prediction/targets before any selection
    def _lead_hours(ds: xr.Dataset) -> list[int]:
        if "lead_time" not in ds.dims:
            return []
        try:
            raw = ds["lead_time"].values
            hrs = (raw // np.timedelta64(1, "h")).astype(int).tolist()
            return [int(h) for h in hrs]
        except Exception:
            return []

    # Lead-time audit collector to persist exact points of any reduction
    lead_audit: list[dict[str, Any]] = []

    def _audit(
        step: str, ds_prediction: xr.Dataset | None = None, ds_target: xr.Dataset | None = None
    ) -> None:
        """Record lead_time hours and sizes for prediction and target at a named step."""

        def _hours(ds: xr.Dataset | None) -> list[int]:
            if ds is None or "lead_time" not in ds.dims:
                return []
            try:
                vals = ds["lead_time"].values
                return (vals // np.timedelta64(1, "h")).astype(int).tolist()
            except Exception:
                return []

        entry: dict[str, Any] = {"step": step}
        if ds_prediction is not None:
            hrs_prediction = _hours(ds_prediction)
            entry["prediction"] = {"count": len(hrs_prediction), "sample": hrs_prediction[:12]}
        if ds_target is not None:
            hrs_target = _hours(ds_target)
            entry["target"] = {"count": len(hrs_target), "sample": hrs_target[:12]}
        lead_audit.append(entry)

    # Defer detailed lead-time prints; record via audit only here
    _ = _lead_hours(ds_prediction)
    _ = _lead_hours(ds_target)
    _audit("after open", ds_prediction, ds_target)

    # Align dims to match config expectations
    # Lead time policy parsing (opt-in multi-lead) ---------------------------------
    # Parse policy early to standardize predictions and correctly calc extension hours
    lt_cfg = cfg.get("lead_time") if isinstance(cfg, dict) else None
    if lt_cfg is None and isinstance(cfg, dict):
        lt_cfg = cfg.get("selection", {}).get("lead_time")
    lead_policy: LeadTimePolicy = parse_lead_time_policy(lt_cfg)
    # preserve_all_leads when mode != first
    cfg["__lead_time_policy"] = lead_policy  # attach for downstream modules

    # Standardize predictions immediately to allow robust lead_time detection
    ds_prediction = data_mod.standardize_dims(
        ds_prediction,
        dataset_name="prediction",
        first_lead_only=not lead_policy.preserve_all_leads,
        preserve_all_leads=lead_policy.preserve_all_leads,
    )

    # Align dims to match config expectations
    ds_prediction = _slice_common(ds_prediction, cfg)

    # Determine required target extension based on prediction lead times
    extend_hours = 0
    if "lead_time" in ds_prediction.coords:
        try:
            # We can use the parsed policy directly now
            mode = str(lead_policy.mode).lower()

            leads = ds_prediction["lead_time"].values
            leads_h = (leads // np.timedelta64(1, "h")).astype(int)
            if len(leads_h) > 0:
                # If mode is first, we need init + first_lead; otherwise use max_lead
                is_first = mode == "first"
                extend_hours = int(leads_h[0]) if is_first else int(leads_h.max())
        except Exception:
            # If anything goes wrong while inferring extend_hours from lead_time,
            # fall back to the default extend_hours=0; downstream selection still works
            # with shorter targets, and we prefer not to fail the evaluation here.
            pass

    ds_target = _slice_common(ds_target, cfg, extend_end_hours=extend_hours)
    # Guard: empty init_time at this stage leads to cryptic errors later.
    # Check predictions primarily, as they drive alignment.
    if "init_time" in ds_prediction.sizes and int(ds_prediction.sizes["init_time"]) == 0:
        raise ValueError(
            "No init_time labels remain after selection. Check selection.datetimes window "
            "and ensure it overlaps the prediction store's init_time coordinates."
        )
    # Defer selection prints; capture only in audit
    _audit("after selection", ds_prediction, ds_target)

    # Sanity-check: avoid empty temporal dims before heavy alignment/stacking
    def _ensure_nonempty_temporal(ds: xr.Dataset, label: str) -> None:
        it_sz = int(ds.sizes.get("init_time", 0)) if "init_time" in ds.dims else -1
        lt_sz = int(ds.sizes.get("lead_time", 0)) if "lead_time" in ds.dims else -1
        if it_sz == 0:
            raise ValueError(
                f"{label} has zero init_time after selection; check selection.datetimes "
                "and that your source covers the requested window."
            )
        if lt_sz == 0:
            raise ValueError(
                f"{label} has zero lead_time after selection; the dataset appears to have "
                "no forecast steps."
            )

    _ensure_nonempty_temporal(ds_prediction, "Predictions")
    _ensure_nonempty_temporal(ds_target, "Ground-truth")

    # Standardize target (keep original location for target to support 'time' slicing above)
    ds_target = data_mod.standardize_dims(
        ds_target,
        dataset_name="target",
        first_lead_only=not lead_policy.preserve_all_leads,
        preserve_all_leads=lead_policy.preserve_all_leads,
    )
    # ds_prediction already standardized above

    # Handle optional ensemble dimension according to config and selected modules
    ds_prediction = data_mod.apply_ensemble_policy(
        ds_prediction,
        ensemble_members=ensemble_members,
    )
    ds_target = data_mod.apply_ensemble_policy(
        ds_target,
        ensemble_members=None,
    )

    ds_target = _apply_temporal_resolution(ds_target, hours)
    ds_prediction = _apply_temporal_resolution(ds_prediction, hours)
    _audit("after temporal_resolution", ds_prediction, ds_target)

    # Validate ensemble consistency between aligned datasets
    if "ensemble" in ds_prediction.dims and "ensemble" in ds_target.dims:
        e_pred = int(ds_prediction.sizes["ensemble"])
        e_tgt = int(ds_target.sizes["ensemble"])

        if e_pred > 1 and e_tgt == 1:
            c.info(
                f"Ensemble Info: Prediction has {e_pred} members, Target has 1. "
                "Target will be broadcast/reused for all prediction members."
            )
        elif e_pred == 1 and e_tgt > 1:
            c.warn(
                f"Ensemble Warning: Prediction has 1 member, Target has {e_tgt}. "
                "In 'members' mode, only the first target member (idx=0) will be used "
                "for comparison. Other target members are ignored."
            )
        elif e_pred != e_tgt:
            c.warn(
                f"Ensemble Mismatch: Prediction has {e_pred} members, Target has {e_tgt}. "
                "Modules operating in 'members' mode typically iterate over prediction members. "
                "Indices matching available target members will be compared; "
                "excessive prediction indices may fail."
            )

    sel_block = cfg.get("selection", {}) or {}
    bounds = sel_block.get("datetimes")
    if (
        lead_policy.preserve_all_leads
        and lead_policy.max_hour is not None
        and bounds
        and len(bounds) == 2
    ):
        end = np.datetime64(bounds[1]).astype("datetime64[ns]")
        extend_h = int(lead_policy.max_hour)
        end_ext = end + np.timedelta64(extend_h, "h")
        # Only widen targets; predictions keep the user-selected init_time window
        if "init_time" in ds_target.dims:
            vals = ds_target["init_time"].values.astype("datetime64[ns]")
            if vals.size > 0 and end_ext > vals.max():
                ds_target = ds_target.sel(init_time=slice(None, end_ext))
                _audit("after extend_target_window", ds_prediction, ds_target)
        elif "time" in ds_target.dims:
            vals = ds_target["time"].values.astype("datetime64[ns]")
            if vals.size > 0 and end_ext > vals.max():
                ds_target = ds_target.sel(time=slice(None, end_ext))
                _audit("after extend_target_window", ds_prediction, ds_target)

    if lead_policy.preserve_all_leads:
        # Always filter predictions per policy
        ds_prediction_before_policy = ds_prediction
        ds_prediction = apply_lead_time_selection(ds_prediction, lead_policy)
        # If policy selection resulted in empty lead_time, fall back to pre-policy dataset
        try:
            if "lead_time" in ds_prediction.sizes and int(ds_prediction.sizes["lead_time"]) == 0:
                c.warn(
                    "lead_time policy produced an empty selection for predictions. "
                    "Skipping lead_time filtering and keeping original leads."
                )
                ds_prediction = ds_prediction_before_policy
        except (KeyError, AttributeError, TypeError) as e:
            c.warn(f"Exception occurred during lead_time policy check: {e}")
        # Targets: apply only in 'full' mode (respects max_hour) where safe/meaningful
        if lead_policy.mode == "full":
            ds_target = apply_lead_time_selection(ds_target, lead_policy)
        # (summary log removed during cleanup)
        _audit(f"after lead_time policy ({lead_policy.mode})", ds_prediction, ds_target)

    ds_target = _select_variables(ds_target, variables_2d, variables_3d)
    ds_prediction = _select_variables(ds_prediction, variables_2d, variables_3d)

    # Align by valid_time using stack/unstack to ensure identical (init_time, lead_time) dims
    # Cases:
    # 1) targets have (init_time, lead_time) → compute valid_time from both
    # 2) targets only have time (continuous) → use that as valid_time
    # 3) predictions have (init_time, lead_time) required
    if "init_time" in ds_prediction.dims:
        # Ensure predictions have lead_time
        if "lead_time" not in ds_prediction.dims:
            ds_prediction = ds_prediction.expand_dims(
                {
                    "lead_time": np.array([np.timedelta64(0, "h")], dtype="timedelta64[h]").astype(
                        "timedelta64[ns]"
                    )
                }
            )

        # Capture pre-alignment lead_time hours (after policy, before valid_time intersection)
        try:
            pre_align_hours = (
                (ds_prediction["lead_time"].values // np.timedelta64(1, "h")).astype(int).tolist()
            )
        except Exception:
            pre_align_hours = []

        # Build stacked predictions with valid_time
        pred_init = ds_prediction["init_time"].astype("datetime64[ns]")
        pred_lead = ds_prediction["lead_time"].astype("timedelta64[ns]")
        pred_valid_2d = pred_init + pred_lead
        ds_pred_stacked = ds_prediction.stack(pair=("init_time", "lead_time"))
        pred_valid_1d = pred_valid_2d.stack(pair=("init_time", "lead_time"))
        ds_pred_stacked = ds_pred_stacked.assign_coords(valid_time=pred_valid_1d)

        # Targets: support either (init_time, lead_time) or standalone time
        if "init_time" in ds_target.dims:
            if "lead_time" not in ds_target.dims:
                ds_target = ds_target.expand_dims(
                    {
                        "lead_time": np.array(
                            [np.timedelta64(0, "h")], dtype="timedelta64[h]"
                        ).astype("timedelta64[ns]")
                    }
                )
            target_init = ds_target["init_time"].astype("datetime64[ns]")
            target_lead = ds_target["lead_time"].astype("timedelta64[ns]")
            ds_tgt_stacked = ds_target.stack(pair=("init_time", "lead_time"))
            target_valid_1d = (target_init + target_lead).stack(pair=("init_time", "lead_time"))
            ds_tgt_stacked = ds_tgt_stacked.assign_coords(valid_time=target_valid_1d)
        elif "time" in ds_target.dims:
            # Target has only 'time' dim. Reindex target onto the prediction valid_time grid to
            # preserve all available lead_time offsets instead of collapsing to a single dummy lead.
            tvals = ds_target["time"].values.astype("datetime64[ns]")
            # Give target a valid_time coordinate for reindexing
            ds_tgt_time = ds_target.assign_coords(valid_time=("time", tvals)).swap_dims(
                {"time": "valid_time"}
            )
            pred_valid_times = ds_pred_stacked["valid_time"].values
            ds_tgt_reindexed = ds_tgt_time.reindex(valid_time=pred_valid_times)
            # Rename to 'pair' and attach same MultiIndex as predictions so that unstack works
            ds_tgt_stacked = ds_tgt_reindexed.rename({"valid_time": "pair"}).assign_coords(
                pair=ds_pred_stacked["pair"]
            )
            # Keep valid_time coord for auditing symmetry
            ds_tgt_stacked = ds_tgt_stacked.assign_coords(valid_time=("pair", pred_valid_times))
        else:
            raise ValueError(
                "Targets must have either ('init_time','lead_time') or 'time' "
                "dimension for alignment."
            )

        # Intersect valid_time and align order to predictions
        common_valid = np.intersect1d(
            ds_pred_stacked["valid_time"].values,
            ds_tgt_stacked["valid_time"].values,
        )
        if common_valid.size == 0:
            raise ValueError(
                "No overlapping valid times between ground_truth (time/init+lead) and "
                "predictions (init+lead) after selection."
            )
        prediction_mask = np.isin(ds_pred_stacked["valid_time"].values, common_valid)
        target_mask = np.isin(ds_tgt_stacked["valid_time"].values, common_valid)
        # Cast masks to boolean arrays to satisfy mypy (avoiding Hashable/Any ambiguity)
        prediction_mask_bool = np.asarray(prediction_mask, dtype=bool)
        target_mask_bool = np.asarray(target_mask, dtype=bool)
        ds_pred_stacked = ds_pred_stacked.isel(pair=prediction_mask_bool)
        ds_tgt_stacked = ds_tgt_stacked.isel(pair=target_mask_bool)

        # Order targets to match predictions
        target_vt = ds_tgt_stacked["valid_time"].values
        prediction_vt = ds_pred_stacked["valid_time"].values
        index_map: dict[np.datetime64, int] = {}
        for i, vt in enumerate(target_vt):
            index_map.setdefault(vt, i)
        try:
            take_idx = np.array([index_map[vt] for vt in prediction_vt], dtype=int)
        except KeyError as err:
            raise ValueError(
                "Internal alignment error: Prediction valid_time not found in targets "
                "after intersection."
            ) from err
        ds_tgt_stacked = ds_tgt_stacked.isel(pair=take_idx)
        # Replace pair labels on targets to match predictions exactly for clean unstack
        to_drop = [n for n in ("pair", "init_time", "lead_time") if n in ds_tgt_stacked.coords]
        if to_drop:
            ds_tgt_stacked = ds_tgt_stacked.drop_vars(to_drop)
            ds_tgt_stacked = ds_tgt_stacked.assign_coords(
                pair=ds_pred_stacked["pair"]
            )  # pairs aligned

        # Unstack back to (init_time, lead_time)
        ds_prediction = ds_pred_stacked.unstack("pair")
        ds_target = ds_tgt_stacked.unstack("pair")
        # (debug alignment summary removed)
        _audit("after alignment", ds_prediction, ds_target)

        # Warn if alignment dropped lead_time hours selected by the policy due to missing
        # ground-truth valid_time overlap.
        try:
            if lead_policy.preserve_all_leads:
                try:
                    after_align_hours = (
                        (ds_prediction["lead_time"].values // np.timedelta64(1, "h"))
                        .astype(int)
                        .tolist()
                    )
                except Exception:
                    after_align_hours = []
                if (
                    pre_align_hours
                    and after_align_hours
                    and len(after_align_hours) < len(pre_align_hours)
                ):
                    lost = [h for h in pre_align_hours if h not in after_align_hours]
                    if lost:
                        msg = (
                            "Alignment clipped forecast leads due to limited target time range. "
                            f"Kept {len(after_align_hours)} of {len(pre_align_hours)} hours; "
                            "dropped: "
                            f"{lost[:14]}{' ...' if len(lost) > 14 else ''}. "
                            "To retain these, widen selection.datetimes to include their "
                            "valid times "
                            "(init_time + lead_time) or provide a longer ground-truth window."
                        )
                        c.warn(msg)
        except Exception as e:
            c.warn(f"Failed to check for dropped forecast leads: {e}")

    # Enforce repository-wide chunking policy to ensure predictable performance
    ds_target = data_mod.enforce_chunking(ds_target, dataset_name="target")
    ds_prediction = data_mod.enforce_chunking(ds_prediction, dataset_name="prediction")

    # Optional: strict check for missing values in inputs
    try:
        check_missing_flag = bool(cfg.get("selection", {}).get("check_missing", False))
    except Exception:
        check_missing_flag = False
    if check_missing_flag:
        problems: list[str] = []
        for name, ds in (
            ("ground_truth", ds_target),
            ("predictions", ds_prediction),
        ):
            missing_counts: dict[str, int] = {}
            totals: dict[str, int] = {}
            for var in ds.data_vars:
                da = ds[var]
                try:
                    nan_sum = da.isnull().sum()
                    # nan_sum is a 0-D DataArray (possibly dask-backed) → compute
                    count = int(nan_sum.compute())
                    if count > 0:
                        missing_counts[str(var)] = count
                        totals[str(var)] = int(da.size)
                except Exception:
                    # Fallback: attempt an 'any' check
                    try:
                        has_nan = bool(da.isnull().any().compute())
                    except Exception:
                        has_nan = False
                    if has_nan:
                        missing_counts[str(var)] = -1  # unknown exact count
                        totals[str(var)] = int(getattr(da, "size", 0) or 0)
            if missing_counts:
                lines = []
                for v, cnt in missing_counts.items():
                    tot = totals.get(v, 0)
                    if cnt >= 0 and tot > 0:
                        lines.append(f"  - {str(v)}: {cnt}/{tot} missing values")
                    else:
                        lines.append(f"  - {str(v)}: missing values present (count unavailable)")
                problems.append(f"{name} dataset contains missing data:\n" + "\n".join(lines))
        # Always print the result to the terminal; do not abort here
        if problems:
            c.panel(
                "Missing-value check (enabled): issues found after selection/alignment.\n\n"
                + "\n\n".join(problems),
                title="Missing Values — Summary",
                style="yellow",
            )
        else:
            c.success(
                "Missing-value check (enabled): no NaNs detected in ground_truth or predictions."
            )

    # Persist audit for run_selected to write out alongside outputs
    cfg["__lead_time_audit"] = lead_audit
    ds_target_std, ds_prediction_std = _standardize_pair(ds_target, ds_prediction)
    return ds_target, ds_prediction, ds_target_std, ds_prediction_std


def _ensure_output(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _maybe_copy_config_to_output(cfg: dict[str, Any], out_root: Path) -> None:
    """If the CLI provided a config file path, copy it into the output folder.

    - Uses the original basename (e.g., config.yaml) to aid reproducibility.
    - Overwrites any existing file with the same name to reflect the last run.
    - Silently skips if the path is missing or invalid.
    """
    try:
        src_path = cfg.get("_config_path")
        if not src_path:
            return
        src = Path(str(src_path))
        if not src.exists() or not src.is_file():
            return
        dst = out_root / src.name
        try:
            # Avoid SameFileError if output_root is same folder as config
            if dst.resolve() == src.resolve():
                return
        except Exception:
            # If resolve fails (e.g., permissions), proceed with copy best-effort
            pass
        shutil.copy2(src, dst)
    except Exception:
        # Best-effort only; do not fail the run because of a copy issue
        pass


def _setup_dask_logging(log_file: str = "logs/dask_distributed.log") -> None:
    """Redirects dask distributed logs to a file and suppresses them from stderr."""
    import logging
    import os
    from pathlib import Path

    try:
        # Append SLURM_JOB_ID and SLURM_PROCID to filename if available to avoid collisions
        job_id = os.environ.get("SLURM_JOB_ID")
        proc_id = os.environ.get("SLURM_PROCID")

        if job_id:
            p = Path(log_file)
            suffix = f"_{job_id}"
            if proc_id:
                suffix += f"_{proc_id}"
            log_file = str(p.parent / f"{p.stem}{suffix}{p.suffix}")

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        dask_logger = logging.getLogger("distributed")
        dask_logger.propagate = False

        # Clear existing handlers
        if dask_logger.hasHandlers():
            dask_logger.handlers.clear()

        fh = logging.FileHandler(log_path, mode="w")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        dask_logger.addHandler(fh)

        # Capture standard warnings (like those from contextlib or distributed.client)
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.propagate = False
        if warnings_logger.hasHandlers():
            warnings_logger.handlers.clear()
        warnings_logger.addHandler(fh)
    except Exception as e:
        c.print(f"Failed to setup dask logging: {e}")


def _resolve_dask_profile(performance_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve Dask worker/client settings from performance config.

    Safety-first defaults are used unless explicitly overridden in YAML.
    """

    def _as_int(value: Any, default: int) -> int:
        try:
            out = int(value)
            return out if out > 0 else default
        except Exception:
            return default

    def _as_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "no", "n", "off"}:
                return False
        try:
            return bool(value)
        except Exception:
            return default

    cpu_count = max(1, int(os.cpu_count() or 1))
    is_gh200_class = cpu_count >= 192
    profile = str(performance_cfg.get("dask_profile", "safe")).strip().lower()

    if is_gh200_class:
        total_memory_gib = 440.0

        def _per_worker_memory_limit(workers: int) -> str:
            mem = total_memory_gib / max(1, int(workers))
            return f"{mem:.2f}GiB"

        safe_workers = max(1, min(cpu_count, 6))
        balanced_workers = max(1, min(cpu_count, 12))
        fast_workers = max(1, min(cpu_count, 24))
        gh200_defaults = {
            "safe": {
                "profile": "safe",
                "n_workers": safe_workers,
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": _per_worker_memory_limit(safe_workers),
            },
            "balanced": {
                "profile": "balanced",
                "n_workers": balanced_workers,
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": _per_worker_memory_limit(balanced_workers),
            },
            "fast": {
                "profile": "fast",
                "n_workers": fast_workers,
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": _per_worker_memory_limit(fast_workers),
            },
        }
        defaults = gh200_defaults.get(profile, gh200_defaults["safe"])
    else:
        if profile == "fast":
            defaults = {
                "profile": "fast",
                "n_workers": max(1, min(cpu_count, 8)),
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": "auto",
            }
        elif profile == "balanced":
            defaults = {
                "profile": "balanced",
                "n_workers": max(1, min(cpu_count, 2)),
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": "auto",
            }
        else:
            defaults = {
                "profile": "safe",
                "n_workers": 1,
                "threads_per_worker": 1,
                "processes": False,
                "memory_limit": "auto",
            }

    default_n_workers = _as_int(defaults.get("n_workers"), 1)
    default_threads_per_worker = _as_int(defaults.get("threads_per_worker"), 1)
    default_processes = _as_bool(defaults.get("processes"), False)

    n_workers = _as_int(performance_cfg.get("dask_n_workers"), default_n_workers)
    threads_per_worker = _as_int(
        performance_cfg.get("dask_threads_per_worker"),
        default_threads_per_worker,
    )
    processes = _as_bool(performance_cfg.get("dask_processes"), default_processes)
    memory_limit = performance_cfg.get("dask_memory_limit", defaults["memory_limit"])

    return {
        "profile": defaults["profile"],
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "processes": processes,
        "memory_limit": memory_limit,
    }


def run_selected(cfg: dict[str, Any]) -> None:
    c.header("SwissClim Evaluations")

    out_root = _ensure_output(cfg.get("paths", {}).get("output_root", "output/verification_esfm"))
    # Persist the exact configuration used for this run into the output directory
    _maybe_copy_config_to_output(cfg, out_root)

    t0 = time.time()
    module_timings: list[tuple[str, float]] = []
    # Track per-module outcomes: name, status(success|failed|skipped), seconds, optional error
    module_results: list[dict[str, Any]] = []
    # Prepare datasets (this also parses and attaches __lead_time_policy onto cfg)
    ds_target, ds_prediction, ds_target_std, ds_prediction_std = prepare_datasets(cfg)
    # Retrieve the parsed lead time policy AFTER preparation
    lead_policy = cfg.get("__lead_time_policy")

    try:
        plotting_block = cfg.setdefault("plotting", {})
        if "lead_panel_hours" in plotting_block:
            del plotting_block["lead_panel_hours"]
    except Exception:
        # Ignore errors when removing legacy config keys; not critical for pipeline execution.
        pass

    # Derive per-plot datasets if a specific plot datetime is requested
    ds_target_plot, ds_prediction_plot = _select_plot_datetime(ds_target, ds_prediction, cfg)
    # For maps only: optionally subset ensemble members and/or a single datetime
    # Other modules use full datasets (no plot-time/ensemble filtering)
    ds_prediction_plot, _ = _select_plot_ensemble(ds_prediction_plot, ds_prediction_std, cfg)

    chapter_flags = cfg.get("modules", {})
    plotting = cfg.get("plotting", {})
    mode = str(plotting.get("output_mode", "plot")).lower()
    # Validate / normalize ensemble block early to surface typos (e.g. 'member').
    # Support ensemble block under either top-level or selection (example_config uses selection).
    raw_ensemble_top = cfg.get("ensemble", {}) or {}
    raw_ensemble_sel = (cfg.get("selection", {}) or {}).get("ensemble", {}) or {}
    if raw_ensemble_top and raw_ensemble_sel:
        # Merge giving precedence to top-level definitions.
        merged = {**raw_ensemble_sel, **raw_ensemble_top}
        c.warn(
            "Both top-level 'ensemble' and 'selection.ensemble' blocks present; "
            "top-level keys take precedence where duplicated."
        )
        raw_ensemble_cfg = merged
    elif raw_ensemble_top:
        raw_ensemble_cfg = raw_ensemble_top
    else:
        raw_ensemble_cfg = raw_ensemble_sel
    has_ens = "ensemble" in ds_prediction.dims
    ensemble_cfg, ens_warnings = validate_and_normalize_ensemble_config(raw_ensemble_cfg, has_ens)
    # Defer printing of warnings until after dataset summary so all ensemble info appears together.
    fallback_notes = [w for w in ens_warnings if w.startswith("[ensemble-fallback]")]
    other_notes = [w for w in ens_warnings if w not in fallback_notes]
    # Compute resolved modes (post-normalization) using resolver for transparency.
    module_names = [
        "maps",
        "histograms",
        "wd_kde",
        "energy_spectra",
        "vertical_profiles",
        "deterministic",
        "ets",
        "probabilistic",
    ]
    resolved_modes: dict[str, str] = {}
    for _m in module_names:
        req = ensemble_cfg.get(_m)
        try:
            resolved_modes[_m] = resolve_ensemble_mode(_m, req, ds_target, ds_prediction)
        except Exception:
            # Fallback; shouldn't happen
            resolved_modes[_m] = req or "mean"
    # We'll show resolved modes later with fallbacks & summary

    # Basic overview
    all_vars = list(ds_target.data_vars)
    # Classify variables: check if 'level' is in dims, regardless of size
    if "level" in ds_target.dims:
        vars_3d = [v for v in all_vars if "level" in ds_target[v].dims]
        vars_2d = [v for v in all_vars if v not in vars_3d]
    else:
        vars_3d = []
        vars_2d = all_vars
    c.panel(
        (
            f"Output: [bold]{out_root}[/]"
            f"\nPlotting Mode: [bold]{mode}[/]"
            f"\nVariables → 2D: [bold]{len(vars_2d)}[/], 3D: [bold]{len(vars_3d)}[/]"
        ),
        title="Run Overview",
        style="cyan",
    )

    # Show the prepared model dataset and describe ensemble handling
    c.section("Model dataset (prepared)")
    # printing the Dataset object provides a concise summary (dims/coords/vars)
    try:
        from .console import (
            USE_RICH,
            console as _rc,
        )

        if USE_RICH:
            from rich.pretty import Pretty

            _rc.print(Pretty(ds_prediction))
        else:
            c.print(ds_prediction)
    except Exception:
        c.print(ds_prediction)
    # Consolidated ensemble information (fallbacks + resolved modes + high-level message)
    try:
        ens_msg = _ensemble_handling_message(ds_prediction, cfg, resolved_modes)
        blocks: list[str] = []
        if fallback_notes:
            blocks.append("Fallbacks:\n" + "\n".join(fallback_notes))
        if other_notes:
            blocks.append("Notes:\n" + "\n".join(other_notes))
        blocks.append(
            "Resolved Modes:\n" + "\n".join(f"{m}: {resolved_modes[m]}" for m in module_names)
        )
        blocks.append("ℹ️  Summary:\n" + ens_msg)
        c.panel(
            "\n\n".join(blocks),
            title="Ensemble Configuration",
            style="blue",
        )
    except Exception:
        pass

    # Lead time policy summary (multi-lead visibility)
    if lead_policy is not None:
        try:
            from .lead_time_policy import LeadTimePolicy

            if isinstance(lead_policy, LeadTimePolicy):  # runtime guard
                hours = []
                if "lead_time" in ds_prediction.dims:
                    hrs = (ds_prediction["lead_time"].values // np.timedelta64(1, "h")).astype(int)
                    hours = hrs.tolist()
                mode = lead_policy.mode
                details: list[str] = [f"mode={mode}"]
                if lead_policy.stride_hours:
                    details.append(f"stride={lead_policy.stride_hours}h")
                if lead_policy.subset_hours:
                    details.append(f"subset={lead_policy.subset_hours}")
                if lead_policy.max_hour is not None:
                    details.append(f"max_hour={lead_policy.max_hour}")
                # 'bins' mode removed
                policy_text = "Lead time policy → " + ", ".join(details)
                if USE_RICH:
                    c.print(f"[magenta][bold]{policy_text}[/]")
                else:
                    c.print(policy_text)

                # Compact lead-time snapshot (moved down near policy)
                try:
                    audit = cfg.get("__lead_time_audit") or []

                    def _fmt(step_key: str, label: str) -> str | None:
                        for e in audit:
                            if e.get("step") == step_key and "prediction" in e:
                                pred = e["prediction"]
                                sample = pred.get("sample", [])
                                return (
                                    f"Available lead hours {label}: count={pred.get('count', 0)} "
                                    f"sample={sample[:8]}"
                                )
                        return None

                    lines = []
                    s1 = _fmt("after open", "before selection")
                    if s1:
                        lines.append(s1)
                    s2 = _fmt("after selection", "after selection")
                    if s2:
                        lines.append(s2)

                    for ln in lines:
                        if USE_RICH:
                            c.print(f"[cyan][bold]{ln}[/]")
                        else:
                            c.print(ln)
                except Exception as exc:
                    c.warn(f"Failed to log lead-time audit information: {exc}")
                # Extra visibility: warn if multi-lead requested but only a single lead remains
                try:
                    if isinstance(lead_policy, LeadTimePolicy):
                        multi_requested = lead_policy.mode != "first"
                    else:
                        multi_requested = False
                except Exception:
                    multi_requested = False
                if (
                    multi_requested
                    and ("lead_time" in ds_prediction.dims)
                    and int(ds_prediction.lead_time.size) == 1
                ):
                    c.warn(
                        "Multi-lead policy requested but only a single lead is present after "
                        "selection/alignment. This typically means either your prediction store "
                        "has a single lead, your date window clips most valid times, or "
                        "target/prediction time alignment leaves only one overlapping offset. "
                        "Consider widening 'selection.datetimes', setting lead_time.mode=full "
                        "temporarily, or inspecting the prediction Zarr."
                    )
        except Exception as ex:
            c.warn(f"Exception occurred during lead time policy handling: {ex}")

    # Get performance configuration
    performance_cfg = cfg.get("performance", {}) or {}

    # Calculate and print dynamic batch size
    batch_size = resolve_dynamic_batch_size(performance_cfg, ds=ds_prediction)
    chunk_mode = describe_batch_size_mode(performance_cfg)
    msg = f"Dask Execution: Base Batch Size: {batch_size} (mode={chunk_mode})"
    if USE_RICH:
        c.print(f"[cyan][bold]{msg}[/]")
    else:
        c.print(msg)

    if chunk_mode.startswith("auto"):
        try:
            chunk_details = resolve_dynamic_batch_details(performance_cfg, ds=ds_prediction)
            avg_points = chunk_details.get("avg_points_per_var", "n/a")
            avg_points_text = f"{avg_points:,}" if isinstance(avg_points, int) else str(avg_points)
            details_msg = (
                "Dask Auto Batch: "
                f"{chunk_details.get('batch_size', 'n/a')} jobs/batch "
                f"(cap {chunk_details.get('effective_cap', 'n/a')}, "
                f"vars {chunk_details.get('num_vars', 'n/a')}, "
                f"~{avg_points_text} points/var)"
            )
            if USE_RICH:
                c.print(f"[cyan]{details_msg}[/]")
            else:
                c.print(details_msg)
        except Exception:
            pass

    # Import lazily to avoid import time if not needed
    if chapter_flags.get("maps"):
        from .plots import maps as maps_mod

        c.module_status("maps", "run", f"vars_2d={len(vars_2d)}, vars_3d={len(vars_3d)}")
        if "ensemble" in ds_prediction.dims:
            ens_full = int(ds_prediction.sizes.get("ensemble", 0))
            ens_plot = int(ds_prediction_plot.sizes.get("ensemble", ens_full))
            use_mode = resolved_modes.get("maps", "members")
            msg = format_ensemble_log(
                "maps",
                use_mode,
                ens_full,
                None if ens_plot == ens_full else f"selected {ens_plot} of {ens_full}",
            )
            c.info(msg)
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            maps_mod.run(
                ds_target_plot,
                ds_prediction_plot,
                out_root,
                plotting,
                ensemble_mode=ensemble_cfg.get("maps"),
            )
            dt = time.time() - _t
            module_timings.append(("maps", dt))
            module_results.append(
                {
                    "name": "maps",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                }
            )
        except Exception as ex:  # pragma: no cover - robustness
            dt = time.time() - _t
            c.error(f"maps failed: {ex}")
            module_results.append(
                {
                    "name": "maps",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                }
            )

    if chapter_flags.get("histograms"):
        from .plots import histograms as hist_mod

        c.module_status("histograms", "run", f"vars_2d={len(vars_2d)}")
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            c.info(
                format_ensemble_log(
                    "histograms", resolved_modes.get("histograms", "pooled"), ens_size
                )
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            hist_mod.run(
                ds_target,
                ds_prediction,
                out_root,
                plotting,
                ensemble_mode=ensemble_cfg.get("histograms"),
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("histograms", dt))
            module_results.append(
                {
                    "name": "histograms",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                }
            )
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"histograms failed: {ex}")
            module_results.append(
                {
                    "name": "histograms",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                }
            )

    if chapter_flags.get("wd_kde"):
        from .plots import wd_kde as wd_mod

        c.module_status("wd_kde", "run", f"vars_2d={len(vars_2d)} (standardized)")
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            c.info(format_ensemble_log("wd_kde", resolved_modes.get("wd_kde", "pooled"), ens_size))
        else:
            c.info("No ensemble dimension → deterministic standardized inputs.")
        _t = time.time()
        try:
            wd_mod.run(
                ds_target,
                ds_prediction,
                ds_target_std,
                ds_prediction_std,
                out_root,
                plotting,
                ensemble_mode=ensemble_cfg.get("wd_kde"),
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("wd_kde", dt))
            module_results.append(
                {
                    "name": "wd_kde",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                }
            )
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"wd_kde failed: {ex}")
            module_results.append(
                {
                    "name": "wd_kde",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                }
            )

    if chapter_flags.get("energy_spectra"):
        from .plots import energy_spectra as es_mod

        c.module_status("energy_spectra", "run", f"vars_2d={len(vars_2d)}")
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            c.info(
                format_ensemble_log(
                    "energy_spectra", resolved_modes.get("energy_spectra", "mean"), ens_size
                )
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            es_mod.run(
                ds_target,
                ds_prediction,
                out_root,
                plotting,
                cfg.get("selection", {}),
                ensemble_mode=ensemble_cfg.get("energy_spectra"),
                cfg=cfg,
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("energy_spectra", dt))
            module_results.append(
                {
                    "name": "energy_spectra",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                }
            )
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"energy_spectra failed: {ex}")
            module_results.append(
                {
                    "name": "energy_spectra",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                }
            )

    if chapter_flags.get("vertical_profiles"):
        from .metrics import vertical_profiles as vp_mod

        c.module_status("vertical_profiles", "run", f"vars_3d={len(vars_3d)}")
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            c.info(
                format_ensemble_log(
                    "vertical_profiles", resolved_modes.get("vertical_profiles", "mean"), ens_size
                )
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            vp_mod.run(
                ds_target,
                ds_prediction,
                out_root,
                plotting,
                cfg.get("selection", {}),
                ensemble_mode=ensemble_cfg.get("vertical_profiles"),
                metrics_cfg=cfg.get("metrics", {}),
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("vertical_profiles", dt))
            module_results.append(
                {
                    "name": "vertical_profiles",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                }
            )
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"vertical_profiles failed: {ex}")
            module_results.append(
                {
                    "name": "vertical_profiles",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                }
            )

    # Deterministic (previously called objective metrics)
    if chapter_flags.get("deterministic"):
        from .metrics import deterministic as det_mod

        c.module_status("deterministic", "run", f"variables={len(all_vars)}")
        if "ensemble" in ds_prediction.dims:
            ens_size_det: int = int(ds_prediction.sizes.get("ensemble", 0))
            use_mode = resolved_modes.get("deterministic", "mean")
            c.info(format_ensemble_log("deterministic", use_mode, ens_size_det))
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            # Pass lead_policy through so per-lead deterministic artifacts are generated
            det_mod.run(
                ds_target,
                ds_prediction,
                ds_target_std,
                ds_prediction_std,
                out_root,
                plotting,
                cfg.get("metrics", {}),
                ensemble_mode=ensemble_cfg.get("deterministic"),
                lead_policy=lead_policy,
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("deterministic", dt))
            module_results.append(
                {
                    "name": "deterministic",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                }
            )
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"deterministic failed: {ex}")
            module_results.append(
                {
                    "name": "deterministic",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                }
            )

    if chapter_flags.get("ets"):
        from .metrics import ets as ets_mod

        c.module_status("ets", "run", f"variables={len(all_vars)}")
        if "ensemble" in ds_prediction.dims:
            ens_size_ets = int(ds_prediction.sizes.get("ensemble", 0))
            use_mode = resolved_modes.get("ets", "mean")
            c.info(format_ensemble_log("ets", use_mode, ens_size_ets))
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            # Provide lead_policy to unlock by-lead ETS artifacts
            ets_mod.run(
                ds_target,
                ds_prediction,
                out_root,
                cfg.get("metrics", {}),
                ensemble_mode=ensemble_cfg.get("ets"),
                lead_policy=lead_policy,
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("ets", dt))
            module_results.append(
                {
                    "name": "ets",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                }
            )
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"ets failed: {ex}")
            module_results.append(
                {
                    "name": "ets",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                }
            )

    # Combined probabilistic: run both xarray (CRPS/PIT) and WBX (SSR/CRPS) when enabled
    if chapter_flags.get("probabilistic"):
        from .metrics.probabilistic import (
            plot_probabilistic,
            run_probabilistic,
            run_probabilistic_wbx,
        )

        c.module_status(
            "probabilistic",
            "run",
            "PIT (xarray) + WBX SSR/CRPS",
        )
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            if ens_size < 2:
                c.warn(
                    "Ensemble size="
                    f"{ens_size} <2 → skipping probabilistic metrics (CRPS/PIT + WBX require >=2)."
                )
                # Register skipped modules
                module_results.append(
                    {
                        "name": "probabilistic",
                        "status": "skipped",
                        "seconds": 0.0,
                        "error": "ensemble size <2",
                    }
                )
                # Continue to completion without executing probabilistic submodules
                pass
            else:
                c.info(format_ensemble_log("probabilistic", "prob", ens_size))
                # Combined probabilistic module
                _t = time.time()
                try:
                    # 1. Xarray-based CRPS/PIT
                    run_probabilistic(
                        ds_target,
                        ds_prediction,
                        out_root,
                        plotting,
                        cfg,
                        ensemble_mode=ensemble_cfg.get("probabilistic"),
                        performance_cfg=performance_cfg,
                    )
                    # 2. Plots
                    plot_probabilistic(ds_target, ds_prediction, out_root, plotting)
                    # 3. WBX Metrics
                    run_probabilistic_wbx(
                        ds_target,
                        ds_prediction,
                        out_root,
                        plotting,
                        cfg,
                        performance_cfg=performance_cfg,
                    )

                    dt = time.time() - _t
                    module_timings.append(("probabilistic", dt))
                    module_results.append(
                        {
                            "name": "probabilistic",
                            "status": "success",
                            "seconds": dt,
                            "error": None,
                        }
                    )
                except Exception as ex:  # pragma: no cover
                    dt = time.time() - _t
                    c.error(f"probabilistic failed: {ex}")
                    module_results.append(
                        {
                            "name": "probabilistic",
                            "status": "failed",
                            "seconds": dt,
                            "error": str(ex),
                        }
                    )
        else:
            c.warn("No ensemble dimension → skipping probabilistic metrics (requires 'ensemble').")
            module_results.append(
                {
                    "name": "probabilistic",
                    "status": "skipped",
                    "seconds": 0.0,
                    "error": "no ensemble dimension",
                }
            )
    # Final completion message + timings summary + module results summary (pass/fail)
    elapsed = time.time() - t0
    try:
        if "module_results" in locals() and module_results:
            # Build a text table (avoid adding dependency). Use rich if available.
            from .console import (
                USE_RICH,
                console as _rc,
            )

            successes = sum(1 for r in module_results if r["status"] == "success")
            failures = [r for r in module_results if r["status"] == "failed"]
            skipped = [r for r in module_results if r["status"] == "skipped"]
            if USE_RICH:
                try:  # pragma: no cover
                    from rich import box
                    from rich.table import Table

                    tbl = Table(title="Module Results", box=box.SIMPLE_HEAVY)
                    tbl.add_column("Module", style="bold")
                    tbl.add_column("Status")
                    tbl.add_column("Time (s)", justify="right")
                    tbl.add_column("Error")
                    for r in module_results:
                        status = r["status"]
                        style = {
                            "success": "green",
                            "failed": "red",
                            "skipped": "yellow",
                        }.get(status, "white")
                        err = r.get("error") or ""
                        if len(err) > 80:
                            err = err[:77] + "..."
                        tbl.add_row(
                            r["name"],
                            f"[{style}]{status}[/]",
                            f"{r['seconds']:.2f}",
                            err,
                        )
                    _rc.print(tbl)
                except Exception:
                    c.print("Module Results:")
                    for r in module_results:
                        c.print(
                            f" - {r['name']}: {r['status']} ({r['seconds']:.2f}s)"
                            + (f" error={r['error']}" if r["error"] else "")
                        )
            else:
                c.print("Module Results:")
                for r in module_results:
                    c.print(
                        f" - {r['name']}: {r['status']} ({r['seconds']:.2f}s)"
                        + (f" error={r['error']}" if r["error"] else "")
                    )
            if failures:
                from . import console as c2

                c2.warn(
                    f"{len(failures)} module(s) failed: {', '.join(r['name'] for r in failures)}"
                )
            else:
                from . import console as c2

                c2.success(f"All {successes} module(s) succeeded.")

    except Exception:
        pass
    # Always print a plain-text completion line so logs are readable without Rich
    c.print(
        f"FINISHED: duration={elapsed:,.1f}s • outputs={out_root}",
    )
    # If Rich is enabled, also show a styled panel
    try:
        from .console import USE_RICH

        if USE_RICH:
            c.panel(
                (
                    f"Completed in [bold]{elapsed:,.1f}[/] seconds"
                    f"\nOutputs written to: [bold]{out_root}[/]"
                ),
                title="✅ Finished",
                style="green",
            )
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SwissClim Evaluations runner")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file into a dict, returning an empty dict on empty files.

    Kept local to this module to avoid cross-imports so the CLI can run in isolation
    during tests where other modules are monkeypatched.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Ensure we return a plain dict for downstream mutation
    if not isinstance(data, dict):
        return {}
    return data


def main(argv: list[str] | None = None) -> None:
    # Try to enforce line-buffered stdout/stderr so Slurm logs update promptly
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    # In non-interactive (no TTY) environments like Slurm, force plain output
    try:
        from .console import set_color_mode

        is_tty = False
        try:
            is_tty = bool(sys.stdout.isatty())
        except Exception:
            is_tty = False
        if not is_tty:
            set_color_mode("never")
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    # Record the original config path for reproducibility actions (not part of user schema)
    with contextlib.suppress(Exception):
        cfg["_config_path"] = args.config

    # Check if user wants to use distributed scheduler (default: True for backwards compat)
    performance_cfg = cfg.get("performance", {}) or {}
    use_distributed = performance_cfg.get("dask_scheduler", "distributed").lower() != "threaded"
    dask_profile = _resolve_dask_profile(performance_cfg)

    if use_distributed:
        # Initialize Dask Client if available to enable spillover and distributed scheduling
        try:
            import socket

            import dask.config

            # Disable worker profiling to avoid AttributeError: '_AllCompletedWaiter' object has no
            # attribute 'f_back' on Python 3.11+ with recent distributed versions.
            # Also set timeouts to avoid hanging clients when workers die.
            c.print("Configuring Dask Client with timeouts and retry limits...")
            dask.config.set(
                {
                    "distributed.worker.profile.enabled": False,
                    "distributed.comm.timeouts.connect": "30s",
                    "distributed.comm.timeouts.tcp": "30s",
                    "distributed.comm.retry.count": 3,
                    "distributed.scheduler.allowed-failures": 3,
                    "distributed.worker.memory.target": 0.60,
                    "distributed.worker.memory.spill": 0.70,
                    "distributed.worker.memory.pause": 0.80,
                    "distributed.worker.memory.terminate": 0.95,
                }
            )

            from dask.distributed import Client

            _setup_dask_logging()

            try:
                _ = Client.current()
                run_selected(cfg)
            except (ValueError, OSError):
                c.print("Initializing Dask Client for distributed scheduling and spillover...")
                c.print(
                    "Dask profile: "
                    f"{dask_profile['profile']} "
                    f"(workers={dask_profile['n_workers']}, "
                    f"threads/worker={dask_profile['threads_per_worker']}, "
                    f"processes={dask_profile['processes']}, "
                    f"memory_limit={dask_profile['memory_limit']})"
                )
                with Client(
                    n_workers=int(dask_profile["n_workers"]),
                    threads_per_worker=int(dask_profile["threads_per_worker"]),
                    processes=bool(dask_profile["processes"]),
                    memory_limit=dask_profile["memory_limit"],
                ) as client:
                    # Retrieve dashboard port and hostname for tunneling instructions
                    try:
                        dask_info = client.scheduler_info()
                        dashboard_port = dask_info["services"]["dashboard"]
                        host = socket.gethostname()
                        c.print(f"Dask Dashboard running on node: {host}")
                        c.print(f"Dashboard URL: {client.dashboard_link}")
                        c.print(
                            "To view the dashboard in VS Code, "
                            "add these entries to your SSH config:"
                        )
                        c.print("'AddKeysToAgent yes'")
                        c.print("'ForwardAgent yes'")
                        c.print("1. Open a new terminal in VS Code")
                        c.print(
                            f"2. Run: ssh -N -L {dashboard_port}:127.0.0.1:"
                            f"{dashboard_port} {host} &"
                        )
                        c.print(
                            f"3. Open 'Simple Browser' in VS Code and go to: http://localhost:{dashboard_port}/status"
                        )
                    except Exception:
                        c.print(f"Dask dashboard available at: {client.dashboard_link}")

                    run_selected(cfg)
                    c.print("Evaluation finished, closing Dask Client...")
        except ImportError:
            run_selected(cfg)
    else:
        # Use threaded scheduler with conservative worker count by default
        threaded_workers = max(1, int(dask_profile["n_workers"]))
        try:
            import dask.config

            dask.config.set(
                scheduler="threads",
                num_workers=threaded_workers,
            )
        except Exception:
            pass
        c.print(
            "Using threaded Dask scheduler (no distributed client), "
            f"num_workers={threaded_workers}."
        )
        run_selected(cfg)


if __name__ == "__main__":
    main()
