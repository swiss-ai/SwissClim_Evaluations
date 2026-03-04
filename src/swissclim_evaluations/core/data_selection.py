from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr

from .. import console as c, data as data_mod
from ..lead_time_policy import (
    LeadTimePolicy,
    apply_lead_time_selection,
    parse_lead_time_policy,
)


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
            ranges.append((
                str(s) if s else None,
                str(e) if e else None,
            ))
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
    if "lead_time" in ds.dims and ds.lead_time.size >= 2:
        step_ns = int(
            (ds.lead_time[1] - ds.lead_time[0]).astype("timedelta64[h]") / np.timedelta64(1, "h")
        )
        step_ns = max(1, step_ns)
        factor = max(1, hours // step_ns)
        out = ds.isel(lead_time=slice(None, None, factor))
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


def select_plot_datetime(
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


def select_plot_ensemble(
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


def standardize_pair(targets: xr.Dataset, predictions: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Z-score normalise *targets* and *predictions* jointly (shared mean/std).

    Public so that callers outside this module (e.g. the runner) can
    re-standardise after adding derived variables.
    """
    # Explicit join ensures compatibility with upcoming xarray default change
    combined = xr.concat([targets, predictions], dim="__concat__", join="outer")
    mean = combined.mean()
    std = combined.std()
    return (targets - mean) / std, (predictions - mean) / std


# Keep a private alias for any callers that may still reference the old name.
_standardize_pair = standardize_pair


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
            ds_prediction = ds_prediction.expand_dims({
                "lead_time": np.array([np.timedelta64(0, "h")], dtype="timedelta64[h]").astype(
                    "timedelta64[ns]"
                )
            })

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
                ds_target = ds_target.expand_dims({
                    "lead_time": np.array([np.timedelta64(0, "h")], dtype="timedelta64[h]").astype(
                        "timedelta64[ns]"
                    )
                })
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
            ds_tgt_time = ds_target.assign_coords(valid_time=("time", tvals)).swap_dims({
                "time": "valid_time"
            })
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
    ds_target_std, ds_prediction_std = standardize_pair(ds_target, ds_prediction)
    return ds_target, ds_prediction, ds_target_std, ds_prediction_std
