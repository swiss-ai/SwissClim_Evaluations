from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr
import yaml

from . import console as c
from . import data as data_mod


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _ensemble_handling_message(
    ds_prediction: xr.Dataset, cfg: dict[str, Any]
) -> str:
    sel = cfg.get("selection", {})
    modules_cfg = cfg.get("modules", {})
    probabilistic_enabled = bool(modules_cfg.get("probabilistic"))
    ensemble_member = sel.get("ensemble_member")

    if "ensemble" not in ds_prediction.dims:
        if ensemble_member is not None and not probabilistic_enabled:
            return f"Ensemble: deterministic mode with ensemble_member={ensemble_member} → selected single member; 'ensemble' removed."
        return "Ensemble: no 'ensemble' dimension present (either source is single-member or reduced deterministically)."
    # ensemble present
    ens_size = ds_prediction.sizes.get("ensemble", -1)
    if probabilistic_enabled:
        return f"Ensemble: probabilistic modules enabled → ensemble (size={ens_size})."
    return f"Ensemble: deterministic mode without explicit member → expected reduction to mean, but 'ensemble' still present (size={ens_size})."


def _slice_common(ds: xr.Dataset, cfg: dict[str, Any]) -> xr.Dataset:
    sel = cfg.get("selection", {})
    levels: list[int] | None = sel.get("levels")
    latitudes: list[float] | None = sel.get("latitudes")
    longitudes: list[float] | None = sel.get("longitudes")
    datetimes: list[str] | None = sel.get("datetimes")
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
                    f"requested={requested}, available={sorted(available)}. Skipping level selection."
                )
    if latitudes is not None:
        ds = ds.sel(latitude=slice(*latitudes))
    if longitudes is not None:
        ds = ds.sel(longitude=slice(*longitudes))
    if datetimes is not None:
        # Interpret provided datetimes as temporal range (pre-standardization)
        if "init_time" in ds.dims:
            ds = ds.sel(init_time=slice(*datetimes))
            # except for legacy datasets with 'time' dim (e.g. ERA5)
        elif "time" in ds.dims:
            ds = ds.sel(time=slice(*datetimes))
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
            (ds.lead_time[1] - ds.lead_time[0]).astype("timedelta64[h]")
            / np.timedelta64(1, "h")
        )
        step_ns = max(1, step_ns)
        factor = max(1, hours // step_ns)
        return ds.isel(lead_time=slice(None, None, factor))
    if "init_time" in ds.dims and ds.init_time.size >= 2:
        # Treat init_time as hourly series for downsampling
        dt_hours = int(
            (ds.init_time[1] - ds.init_time[0]).astype("timedelta64[h]")
            / np.timedelta64(1, "h")
        )
        dt_hours = max(1, dt_hours)
        factor = max(1, hours // dt_hours)
        return ds.isel(init_time=slice(None, None, factor))
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
        if (
            "init_time" in ds_prediction.dims
            and int(ds_prediction.init_time.size) > 0
        ):
            plot_dt = ds_prediction["init_time"].values[0]
            ds_target_plot = (
                ds_target.sel(init_time=[plot_dt])
                if "init_time" in ds_target.dims
                else ds_target
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
        raise ValueError(
            "plot_datetime requires datasets with 'init_time' dimension."
        )

    # Ensure exact label present in predictions (targets are aligned to ML labels)
    available = ds_prediction["init_time"].values
    if plot_dt not in available:
        raise ValueError(
            "Requested plot_datetime not found in predictions init_time. "
            f"Requested={plot_dt_str}. Available examples: {available[:8]} (total {available.size})."
        )

    ds_target_plot = (
        ds_target.sel(init_time=[plot_dt])
        if "init_time" in ds_target.dims
        else ds_target
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
            f"plot_ensemble_members indices out of range. Requested={idx}, available range=0..{ens_size - 1}."
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
    ensemble_member = sel.get("ensemble_member")

    # Open datasets from paths with optional variable selection
    var_list = None
    if variables_2d or variables_3d:
        var_list = list((variables_2d or [])) + list((variables_3d or []))
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
    ds_prediction = data_mod.apply_ensemble_policy(
        ds_prediction,
        ensemble_member=ensemble_member,
        probabilistic_enabled=probabilistic_enabled,
    )
    ds_target = data_mod.apply_ensemble_policy(
        ds_target,
        ensemble_member=None,
        probabilistic_enabled=probabilistic_enabled,
    )

    ds_target = _apply_temporal_resolution(ds_target, hours)
    ds_prediction = _apply_temporal_resolution(ds_prediction, hours)

    ds_target = _select_variables(ds_target, variables_2d, variables_3d)
    ds_prediction = _select_variables(ds_prediction, variables_2d, variables_3d)

    # Align by valid_time using stack/unstack to ensure identical (init_time, lead_time) dims
    if "init_time" in ds_target.dims and "init_time" in ds_prediction.dims:
        # Ensure both have lead_time coordinate
        if "lead_time" not in ds_prediction.dims:
            ds_prediction = ds_prediction.expand_dims({
                "lead_time": np.array(
                    [np.timedelta64(0, "h")], dtype="timedelta64[h]"
                ).astype("timedelta64[ns]")
            })
        if "lead_time" not in ds_target.dims:
            ds_target = ds_target.expand_dims({
                "lead_time": np.array(
                    [np.timedelta64(0, "h")], dtype="timedelta64[h]"
                ).astype("timedelta64[ns]")
            })

        # Stack to a single dimension 'pair' (MultiIndex of init_time, lead_time)
        ml_init = ds_prediction["init_time"].astype("datetime64[ns]")
        ml_lead = ds_prediction["lead_time"].astype("timedelta64[ns]")
        nwp_init = ds_target["init_time"].astype("datetime64[ns]")
        nwp_lead = ds_target["lead_time"].astype("timedelta64[ns]")

        ds_pred_stacked = ds_prediction.stack(pair=("init_time", "lead_time"))
        ds_tgt_stacked = ds_target.stack(pair=("init_time", "lead_time"))

        # Compute valid_time coordinate for each pair
        ml_valid = (ml_init.values[:, None] + ml_lead.values[None, :]).ravel()
        nwp_valid = (
            nwp_init.values[:, None] + nwp_lead.values[None, :]
        ).ravel()
        ds_pred_stacked = ds_pred_stacked.assign_coords(
            valid_time=("pair", ml_valid.astype("datetime64[ns]"))
        )
        ds_tgt_stacked = ds_tgt_stacked.assign_coords(
            valid_time=("pair", nwp_valid.astype("datetime64[ns]"))
        )

        # Intersection of valid times
        common_valid = np.intersect1d(
            ds_pred_stacked["valid_time"].values,
            ds_tgt_stacked["valid_time"].values,
        )
        if common_valid.size == 0:
            raise ValueError(
                "No overlapping valid times between ground_truth (ERA5 time) and predictions (init_time+lead_time) after selection."
            )

        # Filter both by common valid_time
        ml_mask = np.isin(ds_pred_stacked["valid_time"].values, common_valid)
        nwp_mask = np.isin(ds_tgt_stacked["valid_time"].values, common_valid)
        ds_pred_stacked = ds_pred_stacked.isel(pair=ml_mask)
        ds_tgt_stacked = ds_tgt_stacked.isel(pair=nwp_mask)

        # Build mapping from valid_time -> first index in NWP; then align NWP order to ML order
        nwp_vt = ds_tgt_stacked["valid_time"].values
        ml_vt = ds_pred_stacked["valid_time"].values
        # Use a dict of int indices for quick lookup
        index_map: dict[np.datetime64, int] = {}
        for i, vt in enumerate(nwp_vt):
            # only first occurrence kept; duplicates in NWP unlikely with zero lead_time
            index_map.setdefault(vt, i)
        try:
            take_idx = np.array([index_map[vt] for vt in ml_vt], dtype=int)
        except KeyError:
            # This should not happen due to intersection, but guard anyway
            raise ValueError(
                "Internal alignment error: ML valid_time not found in NWP after intersection."
            )
        ds_tgt_stacked = ds_tgt_stacked.isel(pair=take_idx)
        # Make NWP stacked coordinate exactly the ML 'pair' MultiIndex for identical labels
        # Drop existing pair and its level coords first to avoid future MultiIndex inconsistency
        to_drop = [
            name
            for name in ("pair", "init_time", "lead_time")
            if name in ds_tgt_stacked.coords
        ]
        if to_drop:
            ds_tgt_stacked = ds_tgt_stacked.drop_vars(to_drop)
        ds_tgt_stacked = ds_tgt_stacked.assign_coords(
            pair=ds_pred_stacked["pair"]
        )  # type: ignore[index]

    # Unstack back to (init_time, lead_time) using the ML's pair labels for both
    ds_prediction = ds_pred_stacked.unstack("pair")
    ds_target = ds_tgt_stacked.unstack("pair")

    # Enforce repository-wide chunking policy to ensure predictable performance
    ds_target = data_mod.enforce_chunking(
        ds_target, dataset_name="ground_truth"
    )
    ds_prediction = data_mod.enforce_chunking(ds_prediction, dataset_name="ml")

    # Optional: strict check for missing values in inputs
    try:
        check_missing_flag = bool(
            cfg.get("selection", {}).get("check_missing", False)
        )
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
                    count = int(nan_sum.compute())  # type: ignore[arg-type]
                    if count > 0:
                        missing_counts[var] = count
                        totals[var] = int(da.size)
                except Exception:
                    # Fallback: attempt an 'any' check
                    try:
                        has_nan = bool(da.isnull().any().compute())
                    except Exception:
                        has_nan = False
                    if has_nan:
                        missing_counts[var] = -1  # unknown exact count
                        totals[var] = int(getattr(da, "size", 0) or 0)
            if missing_counts:
                lines = []
                for v, cnt in missing_counts.items():
                    tot = totals.get(v, 0)
                    if cnt >= 0 and tot > 0:
                        lines.append(f"  - {v}: {cnt}/{tot} missing values")
                    else:
                        lines.append(
                            f"  - {v}: missing values present (count unavailable)"
                        )
                problems.append(
                    f"{name} dataset contains missing data:\n"
                    + "\n".join(lines)
                )
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

    ds_target_std, ds_prediction_std = _standardize_pair(
        ds_target, ds_prediction
    )
    return ds_target, ds_prediction, ds_target_std, ds_prediction_std


def _ensure_output(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_selected(cfg: dict[str, Any]) -> None:
    c.header("SwissClim Evaluations")
    t0 = time.time()
    ds_target, ds_prediction, ds_target_std, ds_prediction_std = (
        prepare_datasets(cfg)
    )

    # Derive per-plot datasets if a specific plot datetime is requested
    ds_target_plot, ds_prediction_plot = _select_plot_datetime(
        ds_target, ds_prediction, cfg
    )
    # For maps only: optionally subset ensemble members and/or a single datetime
    # Other modules use full datasets (no plot-time/ensemble filtering)
    ds_prediction_plot, _ = _select_plot_ensemble(
        ds_prediction_plot, ds_prediction_std, cfg
    )

    out_root = _ensure_output(
        cfg.get("paths", {}).get("output_root", "output/verification_esfm")
    )
    chapter_flags = cfg.get("modules", {})
    plotting = cfg.get("plotting", {})
    mode = str(plotting.get("output_mode", "plot")).lower()

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
        f"Output: [bold]{out_root}[/]\nMode: [bold]{mode}[/]\nVariables → 2D: [bold]{len(vars_2d)}[/], 3D: [bold]{len(vars_3d)}[/]",
        title="Run Overview",
        style="cyan",
    )

    # Show the prepared model dataset and describe ensemble handling
    c.section("Model dataset (prepared)")
    # printing the Dataset object provides a concise summary (dims/coords/vars)
    try:
        from rich.pretty import Pretty  # type: ignore

        from .console import console as _rc

        _rc.print(Pretty(ds_prediction))
    except Exception:
        print(ds_prediction)
    # Ensemble handling summary
    ens_msg = _ensemble_handling_message(ds_prediction, cfg)
    level = "ok" if "probabilistic modules enabled" in ens_msg else "info"
    c.ensemble_panel(ens_msg, level=level)

    # Import lazily to avoid import time if not needed
    if chapter_flags.get("maps"):
        from .plots import maps as maps_mod

        c.module_status(
            "maps", "run", f"vars_2d={len(vars_2d)}, vars_3d={len(vars_3d)}"
        )
        if "ensemble" in ds_prediction.dims:
            ens_full = int(ds_prediction.sizes.get("ensemble"))
            ens_plot = int(ds_prediction_plot.sizes.get("ensemble", ens_full))
            if ens_plot < ens_full:
                c.info(
                    f"Ensemble present (selected size={ens_plot} of total {ens_full}) → generating maps for selected members."
                )
            else:
                c.info(
                    f"Ensemble present (size={ens_full}) → generating maps for all members."
                )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        maps_mod.run(ds_target_plot, ds_prediction_plot, out_root, plotting)

    if chapter_flags.get("histograms"):
        from .plots import histograms as hist_mod

        c.module_status("histograms", "run", f"vars_2d={len(vars_2d)}")
        if "ensemble" in ds_prediction.dims:
            ens_size = ds_prediction.sizes.get("ensemble")
            c.info(
                f"Ensemble present (size={ens_size}) → module displays deterministic reduction."
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        hist_mod.run(ds_target, ds_prediction, out_root, plotting)

    if chapter_flags.get("wd_kde"):
        from .plots import wd_kde as wd_mod

        c.module_status(
            "wd_kde", "run", f"vars_2d={len(vars_2d)} (standardized)"
        )
        if "ensemble" in ds_prediction.dims:
            ens_size = ds_prediction.sizes.get("ensemble")
            c.info(
                f"Ensemble present (size={ens_size}) → module uses reduced standardized mean."
            )
        else:
            c.info("No ensemble dimension → deterministic standardized inputs.")
        wd_mod.run(
            ds_target,
            ds_prediction,
            ds_target_std,
            ds_prediction_std,
            out_root,
            plotting,
        )

    if chapter_flags.get("energy_spectra"):
        from .plots import energy_spectra as es_mod

        c.module_status(
            "energy_spectra",
            "run",
            f"vars_2d={len(vars_2d)}, vars_3d={len(vars_3d)}",
        )
        if "ensemble" in ds_prediction.dims:
            ens_size = ds_prediction.sizes.get("ensemble")
            c.info(
                f"Ensemble present (size={ens_size}) → spectra computed for reduced mean field."
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        es_mod.run(
            ds_target,
            ds_prediction,
            out_root,
            plotting,
            cfg.get("selection", {}),
        )

    if chapter_flags.get("vertical_profiles"):
        from .metrics import vertical_profiles as vp_mod

        c.module_status("vertical_profiles", "run", f"vars_3d={len(vars_3d)}")
        if "ensemble" in ds_prediction.dims:
            ens_size = ds_prediction.sizes.get("ensemble")
            c.info(
                f"Ensemble present (size={ens_size}) → profiles shown for deterministic reduction only."
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        vp_mod.run(
            ds_target,
            ds_prediction,
            out_root,
            plotting,
            cfg.get("selection", {}),
        )

    # Deterministic (previously called objective metrics)
    if chapter_flags.get("deterministic"):
        from .metrics import deterministic as det_mod

        c.module_status("deterministic", "run", f"variables={len(all_vars)}")
        if "ensemble" in ds_prediction.dims:
            ens_size = ds_prediction.sizes.get("ensemble")
            c.info(
                f"Ensemble present (size={ens_size}) → metrics computed on deterministic reduction."
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        det_mod.run(
            ds_target,
            ds_prediction,
            ds_target_std,
            ds_prediction_std,
            out_root,
            plotting,
            cfg.get("metrics", {}),
        )

    if chapter_flags.get("ets"):
        from .metrics import ets as ets_mod

        c.module_status("ets", "run", f"variables={len(all_vars)}")
        if "ensemble" in ds_prediction.dims:
            ens_size = ds_prediction.sizes.get("ensemble")
            c.info(
                f"Ensemble present (size={ens_size}) → ETS computed on deterministic reduction."
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        ets_mod.run(ds_target, ds_prediction, out_root, cfg.get("metrics", {}))

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
            ens_size = ds_prediction.sizes.get("ensemble")
            c.success(
                f"Ensemble present (size={ens_size}) → running both xarray and WBX probabilistic metrics."
            )
        else:
            c.warn(
                "No ensemble dimension → skipping probabilistic metrics (requires 'ensemble')."
            )
        if "ensemble" in ds_prediction.dims:
            # Xarray-based CRPS/PIT + plots
            run_probabilistic(ds_target, ds_prediction, out_root, plotting, cfg)
            plot_probabilistic(ds_target, ds_prediction, out_root, plotting)
            # WeatherBenchX-based summaries and aggregates
            run_probabilistic_wbx(
                ds_target, ds_prediction, out_root, plotting, cfg
            )

    # Final completion message
    elapsed = time.time() - t0
    c.panel(
        f"Completed in [bold]{elapsed:,.1f}[/] seconds\nOutputs written to: [bold]{out_root}[/]",
        title="✅ Finished",
        style="green",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SwissClim Evaluations runner")
    p.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    run_selected(cfg)


if __name__ == "__main__":
    main()
