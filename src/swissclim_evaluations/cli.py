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
import yaml  # type: ignore[import-untyped]

from . import console as c, data as data_mod
from .helpers import (
    format_ensemble_log,
    resolve_ensemble_mode,
    validate_and_normalize_ensemble_config,
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


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _ensemble_handling_message(ds_prediction: xr.Dataset, cfg: dict[str, Any]) -> str:
    sel = cfg.get("selection", {})
    modules_cfg = cfg.get("modules", {})
    probabilistic_enabled = bool(modules_cfg.get("probabilistic"))
    # Prefer new plural key; fall back to legacy singular.
    if "ensemble_members" in sel:
        ensemble_members = sel.get("ensemble_members")
    else:
        ensemble_members = sel.get("ensemble_member")
        if ensemble_members is not None:
            c.warn(
                "Config key 'selection.ensemble_member' is deprecated; "
                "use 'selection.ensemble_members'."
            )
    # Normalize ensemble_members to int | list[int] | None for apply_ensemble_policy
    if isinstance(ensemble_members, list):
        try:
            ensemble_members = [int(i) for i in ensemble_members]
        except Exception:
            c.warn("Invalid values in ensemble_members list; ignoring selection.")
            ensemble_members = None
        if isinstance(ensemble_members, list) and len(ensemble_members) == 1:
            ensemble_members = ensemble_members[0]
    # Normalize possible list/int forms for messaging only
    if isinstance(ensemble_members, list):
        if len(ensemble_members) == 1:
            ensemble_member_norm: int | list[int] | None = int(ensemble_members[0])
        else:
            ensemble_member_norm = [int(i) for i in ensemble_members]
    else:
        ensemble_member_norm = ensemble_members

    if "ensemble" not in ds_prediction.dims:
        if ensemble_member_norm is not None and not probabilistic_enabled:
            return (
                "Ensemble: deterministic mode with "
                f"ensemble_members={ensemble_member_norm} → selected single member; "
                "'ensemble' removed."
            )
        return (
            "Ensemble: no 'ensemble' dimension present (either source is single-member "
            "or reduced deterministically)."
        )
    # ensemble present
    ens_size = ds_prediction.sizes.get("ensemble", -1)
    if probabilistic_enabled:
        return (
            "Ensemble: probabilistic mode active "
            f"(size={ens_size}) → token=ensprob for probabilistic outputs."
        )
    # Deterministic paths
    if ensemble_member_norm is None:
        return (
            "Ensemble: deterministic mode without explicit member → reduced to mean (ensmean)."
            if "ensemble" not in ds_prediction.dims or ens_size == 0
            else "Ensemble: deterministic mode without explicit member; original size="
            f"{ens_size} (reduced internally where applicable)."
        )
    if isinstance(ensemble_member_norm, list):
        return (
            "Ensemble: deterministic mode with subset members="
            f"{ensemble_member_norm} (size={len(ensemble_member_norm)} retained)."
        )
    return (
        "Ensemble: deterministic mode with ensemble_members="
        f"{ensemble_member_norm} → single member path."
    )


def _slice_common(ds: xr.Dataset, cfg: dict[str, Any]) -> xr.Dataset:
    sel = cfg.get("selection", {})
    levels: list[int] | None = sel.get("levels")
    latitudes: list[float] | None = sel.get("latitudes")
    longitudes: list[float] | None = sel.get("longitudes")
    datetimes: list[str] | None = sel.get("datetimes")
    datetimes_list: list[str] | None = sel.get("datetimes_list")
    check_missing: bool = bool(sel.get("check_missing", False))

    if levels is not None and "level" in ds.dims:
        # Only select levels that exist to avoid KeyError when upstream datasets
        # have been pre-trimmed to a subset of pressure levels.
        try:
            available = set(ds.coords["level"].values.tolist())
        except Exception:
            available = set()
        requested = list(levels)
        present = [lv for lv in requested if lv in available]
        missing = [lv for lv in requested if lv not in available]
        if missing and check_missing:
            raise KeyError(
                f"Requested pressure levels not found: {missing}. Available: {sorted(available)}"
            )
        if present:
            ds = ds.sel(level=present)
        else:
            # No overlap; keep dataset unchanged but warn to stdout for visibility.
            if requested:
                c.warn(
                    "None of the requested pressure levels are present; "
                    f"requested={requested}, available={sorted(available)}. "
                    "Skipping level selection."
                )
    if latitudes is not None:
        ds = ds.sel(latitude=slice(*latitudes))
    if longitudes is not None:
        ds = ds.sel(longitude=slice(*longitudes))
    # Non-contiguous explicit timestamps take precedence if provided
    if datetimes_list is not None and len(datetimes_list) > 0:
        try:
            # normalize to datetime64[ns]
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
            if missing and check_missing:
                raise KeyError(
                    "Requested timestamps not found in "
                    f"{dim_name}: {missing[:6]}"
                    f"{' ...' if len(missing) > 6 else ''}"
                )
            if missing and not check_missing and len(missing) > 0:
                c.warn(
                    "Some requested timestamps are missing in "
                    f"{dim_name}: {len(missing)} missing; proceeding with "
                    f"{len(present)} present."
                )
            if present:
                ds = ds.sel({dim_name: present})
        # If neither init_time nor time exist, ignore silently
    elif datetimes is not None:
        # Support either a single [start,end] or multiple "start:end" ranges or [[start,end], ...]
        dim_name = (
            "init_time" if "init_time" in ds.dims else ("time" if "time" in ds.dims else None)
        )
        if dim_name is None:
            return ds

        def _parse_ranges(values) -> list[tuple[str | None, str | None]]:
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

        ranges = _parse_ranges(datetimes)
        if not ranges:
            # Fallback to original behavior if we couldn't parse anything meaningful
            if len(datetimes) >= 2:
                ds = ds.sel({dim_name: slice(datetimes[0], datetimes[1])})
            return ds

        vals = ds[dim_name].values.astype("datetime64[ns]")
        mask = np.zeros(vals.shape, dtype=bool)
        for start_s, end_s in ranges:
            try:
                start = np.datetime64(start_s).astype("datetime64[ns]") if start_s else vals.min()
            except Exception:
                start = vals.min()
            try:
                end = np.datetime64(end_s).astype("datetime64[ns]") if end_s else vals.max()
            except Exception:
                end = vals.max()
            mask |= (vals >= start) & (vals <= end)

        count = int(mask.sum())
        if count == 0:
            msg = f"No timestamps within requested ranges on {dim_name}."
            if check_missing:
                raise KeyError(msg)
            c.warn(msg + " Keeping dataset unchanged.")
            return ds
        # isel by positions retains labels and avoids building a long explicit label list
        idx = np.nonzero(mask)[0]
        ds = ds.isel({dim_name: idx})
        return ds
    # No datetime filtering requested → return as-is
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
        return ds.isel(lead_time=slice(None, None, factor))
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
            return ds.isel(init_time=slice(None, None, factor))
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

    # Ensure exact label present in predictions (targets are aligned to ML labels)
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
    ds_target = data_mod.era5(paths.get("nwp"), variables=var_list)
    ds_prediction = data_mod.open_ml(paths.get("ml"), variables=var_list)

    # Align dims to match config expectations
    ds_target = _slice_common(ds_target, cfg)
    ds_prediction = _slice_common(ds_prediction, cfg)

    # Standardize temporal dims and enforce required schema
    ds_target = data_mod.standardize_dims(
        ds_target, dataset_name="ground_truth", first_lead_only=True
    )
    ds_prediction = data_mod.standardize_dims(
        ds_prediction, dataset_name="ml", first_lead_only=True
    )

    # Handle optional ensemble dimension according to config and selected modules
    modules_cfg = cfg.get("modules", {})
    probabilistic_enabled = bool(modules_cfg.get("probabilistic"))
    # Decide if we must retain full ensemble (avoid mean collapse) because user intends
    # pooled/members handling in downstream modules. We inspect raw ensemble blocks before
    # normalization (top-level and selection) since prepare_datasets runs early.
    raw_ensemble_top = cfg.get("ensemble", {}) or {}
    raw_ensemble_sel = (cfg.get("selection", {}) or {}).get("ensemble", {}) or {}
    if raw_ensemble_top and raw_ensemble_sel:
        raw_ensemble_cfg = {**raw_ensemble_sel, **raw_ensemble_top}
    elif raw_ensemble_top:
        raw_ensemble_cfg = raw_ensemble_top
    else:
        raw_ensemble_cfg = raw_ensemble_sel

    def _wants_preserve(block: dict | None) -> bool:
        if not block:
            return False
        try:
            vals = [str(v).strip().lower() for v in block.values() if v is not None]
        except Exception:
            return False
        return any(v in {"pooled", "members"} for v in vals)

    preserve_full = bool(_wants_preserve(raw_ensemble_cfg))

    ds_prediction = data_mod.apply_ensemble_policy(
        ds_prediction,
        ensemble_members=ensemble_members,
        probabilistic_enabled=probabilistic_enabled,
        preserve_full_when_unselected=preserve_full,
    )
    ds_target = data_mod.apply_ensemble_policy(
        ds_target,
        ensemble_members=None,
        probabilistic_enabled=probabilistic_enabled,
        preserve_full_when_unselected=preserve_full,
    )

    ds_target = _apply_temporal_resolution(ds_target, hours)
    ds_prediction = _apply_temporal_resolution(ds_prediction, hours)

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

        # Build stacked predictions with valid_time
        ml_init = ds_prediction["init_time"].astype("datetime64[ns]")
        ml_lead = ds_prediction["lead_time"].astype("timedelta64[ns]")
        ml_valid_2d = ml_init + ml_lead
        ds_pred_stacked = ds_prediction.stack(pair=("init_time", "lead_time"))
        ml_valid_1d = ml_valid_2d.stack(pair=("init_time", "lead_time"))
        ds_pred_stacked = ds_pred_stacked.assign_coords(valid_time=ml_valid_1d)

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
            nwp_init = ds_target["init_time"].astype("datetime64[ns]")
            nwp_lead = ds_target["lead_time"].astype("timedelta64[ns]")
            ds_tgt_stacked = ds_target.stack(pair=("init_time", "lead_time"))
            nwp_valid_1d = (nwp_init + nwp_lead).stack(pair=("init_time", "lead_time"))
            ds_tgt_stacked = ds_tgt_stacked.assign_coords(valid_time=nwp_valid_1d)
        elif "time" in ds_target.dims:
            # Convert time to a stacked structure with a dummy lead_time=0 to keep unstack symmetry
            tvals = ds_target["time"].values.astype("datetime64[ns]")
            # Build a pair index aligned to each time with synthetic init_time=time and lead_time=0
            ds_tgt_stacked = ds_target.expand_dims({"lead_time": [np.timedelta64(0, "ns")]})
            ds_tgt_stacked = ds_tgt_stacked.rename({"time": "init_time"})
            ds_tgt_stacked = ds_tgt_stacked.stack(pair=("init_time", "lead_time"))
            ds_tgt_stacked = ds_tgt_stacked.assign_coords(valid_time=("pair", tvals))
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
        ml_mask = np.isin(ds_pred_stacked["valid_time"].values, common_valid)
        nwp_mask = np.isin(ds_tgt_stacked["valid_time"].values, common_valid)
        ds_pred_stacked = ds_pred_stacked.isel(pair=ml_mask)
        ds_tgt_stacked = ds_tgt_stacked.isel(pair=nwp_mask)

        # Order targets to match predictions
        nwp_vt = ds_tgt_stacked["valid_time"].values
        ml_vt = ds_pred_stacked["valid_time"].values
        index_map: dict[np.datetime64, int] = {}
        for i, vt in enumerate(nwp_vt):
            index_map.setdefault(vt, i)
        try:
            take_idx = np.array([index_map[vt] for vt in ml_vt], dtype=int)
        except KeyError as err:
            raise ValueError(
                "Internal alignment error: ML valid_time not found in targets after intersection."
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

    # Enforce repository-wide chunking policy to ensure predictable performance
    ds_target = data_mod.enforce_chunking(ds_target, dataset_name="ground_truth")
    ds_prediction = data_mod.enforce_chunking(ds_prediction, dataset_name="ml")

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


def run_selected(cfg: dict[str, Any]) -> None:
    c.header("SwissClim Evaluations")
    t0 = time.time()
    module_timings: list[tuple[str, float]] = []
    # Track per-module outcomes: name, status(success|failed|skipped), seconds, optional error
    module_results: list[dict[str, Any]] = []
    ds_target, ds_prediction, ds_target_std, ds_prediction_std = prepare_datasets(cfg)

    # Derive per-plot datasets if a specific plot datetime is requested
    ds_target_plot, ds_prediction_plot = _select_plot_datetime(ds_target, ds_prediction, cfg)
    # For maps only: optionally subset ensemble members and/or a single datetime
    # Other modules use full datasets (no plot-time/ensemble filtering)
    ds_prediction_plot, _ = _select_plot_ensemble(ds_prediction_plot, ds_prediction_std, cfg)

    out_root = _ensure_output(cfg.get("paths", {}).get("output_root", "output/verification_esfm"))
    # Persist the exact configuration used for this run into the output directory
    _maybe_copy_config_to_output(cfg, out_root)
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
            resolved_modes[_m] = req or "none"
    # We'll show resolved modes later with fallbacks & summary

    # Basic overview
    all_vars = list(ds_target.data_vars)
    # Classify variables: treat singleton level dimension (size==1) as non-3D
    if "level" in ds_target.dims and int(ds_target.level.size) > 1:
        vars_3d = [v for v in all_vars if "level" in ds_target[v].dims]
        vars_2d = [v for v in all_vars if v not in vars_3d]
    else:
        vars_3d = []
        vars_2d = all_vars
    c.panel(
        (
            f"Output: [bold]{out_root}[/]"
            f"\nMode: [bold]{mode}[/]"
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
            print(ds_prediction)
    except Exception:
        print(ds_prediction)
    # Consolidated ensemble information (fallbacks + resolved modes + high-level message)
    try:
        ens_msg = _ensemble_handling_message(ds_prediction, cfg)
        blocks: list[str] = []
        if fallback_notes:
            blocks.append("Fallbacks:\n" + "\n".join(fallback_notes))
        if other_notes:
            blocks.append("Notes:\n" + "\n".join(other_notes))
        blocks.append(
            "Resolved Modes:\n" + "\n".join(f"{m}: {resolved_modes[m]}" for m in module_names)
        )
        blocks.append("Summary:\n" + ens_msg)
        c.panel(
            "\n\n".join(blocks),
            title="Ensemble Configuration",
            style="blue",
        )
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

        c.module_status(
            "energy_spectra",
            "run",
            f"vars_2d={len(vars_2d)}, vars_3d={len(vars_3d)}",
        )
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
            c.info(
                format_ensemble_log(
                    "deterministic",
                    use_mode,
                    ens_size_det,
                )
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            det_mod.run(
                ds_target,
                ds_prediction,
                ds_target_std,
                ds_prediction_std,
                out_root,
                plotting,
                cfg.get("metrics", {}),
                ensemble_mode=ensemble_cfg.get("deterministic"),
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
            ets_mod.run(
                ds_target,
                ds_prediction,
                out_root,
                cfg.get("metrics", {}),
                ensemble_mode=ensemble_cfg.get("ets"),
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
            "CRPS/PIT (xarray) + WBX SSR/CRPS",
        )
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            if ens_size < 2:
                c.warn(
                    "Ensemble size="
                    f"{ens_size} <2 → skipping probabilistic metrics (CRPS/PIT + WBX require >=2)."
                )
                # Register skipped modules
                for suffix in ("xarray", "plots", "wbx"):
                    module_results.append(
                        {
                            "name": f"probabilistic:{suffix}",
                            "status": "skipped",
                            "seconds": 0.0,
                            "error": "ensemble size <2",
                        }
                    )
                # Continue to completion without executing probabilistic submodules
                pass
            else:
                c.success(format_ensemble_log("probabilistic", "prob", ens_size))
                # Xarray-based CRPS/PIT + plots
                _t = time.time()
                try:
                    run_probabilistic(
                        ds_target,
                        ds_prediction,
                        out_root,
                        plotting,
                        cfg,
                        ensemble_mode=ensemble_cfg.get("probabilistic"),
                    )
                    dt = time.time() - _t
                    module_timings.append(("probabilistic:xarray", dt))
                    module_results.append(
                        {
                            "name": "probabilistic:xarray",
                            "status": "success",
                            "seconds": dt,
                            "error": None,
                        }
                    )
                except Exception as ex:  # pragma: no cover
                    dt = time.time() - _t
                    c.error(f"probabilistic:xarray failed: {ex}")
                    module_results.append(
                        {
                            "name": "probabilistic:xarray",
                            "status": "failed",
                            "seconds": dt,
                            "error": str(ex),
                        }
                    )
                _t = time.time()
                try:
                    plot_probabilistic(ds_target, ds_prediction, out_root, plotting)
                    dt = time.time() - _t
                    module_timings.append(("probabilistic:plots", dt))
                    module_results.append(
                        {
                            "name": "probabilistic:plots",
                            "status": "success",
                            "seconds": dt,
                            "error": None,
                        }
                    )
                except Exception as ex:  # pragma: no cover
                    dt = time.time() - _t
                    c.error(f"probabilistic:plots failed: {ex}")
                    module_results.append(
                        {
                            "name": "probabilistic:plots",
                            "status": "failed",
                            "seconds": dt,
                            "error": str(ex),
                        }
                    )
                _t = time.time()
                try:
                    run_probabilistic_wbx(ds_target, ds_prediction, out_root, plotting, cfg)
                    dt = time.time() - _t
                    module_timings.append(("probabilistic:wbx", dt))
                    module_results.append(
                        {
                            "name": "probabilistic:wbx",
                            "status": "success",
                            "seconds": dt,
                            "error": None,
                        }
                    )
                except Exception as ex:  # pragma: no cover
                    dt = time.time() - _t
                    c.error(f"probabilistic:wbx failed: {ex}")
                    module_results.append(
                        {
                            "name": "probabilistic:wbx",
                            "status": "failed",
                            "seconds": dt,
                            "error": str(ex),
                        }
                    )
        else:
            c.warn("No ensemble dimension → skipping probabilistic metrics (requires 'ensemble').")
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
                    print("Module Results:")
                    for r in module_results:
                        print(
                            f" - {r['name']}: {r['status']} ({r['seconds']:.2f}s)"
                            + (f" error={r['error']}" if r["error"] else "")
                        )
            else:
                print("Module Results:")
                for r in module_results:
                    print(
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
    print(
        f"FINISHED: duration={elapsed:,.1f}s • outputs={out_root}",
        flush=True,
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
    run_selected(cfg)


if __name__ == "__main__":
    main()
