from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr
import yaml

from . import data as data_mod


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


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
                print(
                    "[swissclim_evaluations] Warning: none of the requested levels are present; "
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


def _maybe_subsample_time(
    ds: xr.Dataset, n: int | None, seed: int | None
) -> xr.Dataset:
    """Optionally subsample along init_time for quick previews."""
    if n is None or "init_time" not in ds.dims or ds.init_time.size == 0:
        return ds
    n = int(min(n, ds.init_time.size))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ds.init_time.size, size=n, replace=False))
    return ds.isel(init_time=idx)


def _standardize_pair(
    ds: xr.Dataset, ds_ml: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    combined = xr.concat([ds, ds_ml], dim="__concat__")
    mean = combined.mean()
    std = combined.std()
    return (ds - mean) / std, (ds_ml - mean) / std


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
    ds_nwp = data_mod.era5(paths.get("nwp"), variables=var_list)
    ds_ml = data_mod.open_ml(paths.get("ml"), variables=var_list)

    # Align dims to match config expectations
    ds_nwp = _slice_common(ds_nwp, cfg)
    ds_ml = _slice_common(ds_ml, cfg)

    # Standardize temporal dims and enforce required schema
    ds_nwp = data_mod.standardize_dims(ds_nwp, dataset_name="ground_truth")
    ds_ml = data_mod.standardize_dims(ds_ml, dataset_name="ml")

    if "ensemble" in ds_ml.dims and ensemble_member is not None:
        # Keep the 'ensemble' dimension by slicing to a length-1 selection
        ds_ml = ds_ml.isel(ensemble=[int(ensemble_member)])

    ds_nwp = _apply_temporal_resolution(ds_nwp, hours)
    ds_ml = _apply_temporal_resolution(ds_ml, hours)

    ds_nwp = _select_variables(ds_nwp, variables_2d, variables_3d)
    ds_ml = _select_variables(ds_ml, variables_2d, variables_3d)

    # Ensure same time subsampling later
    plot_cfg = cfg.get("plotting", {})
    ds_nwp = _maybe_subsample_time(
        ds_nwp, plot_cfg.get("time_subsamples"), plot_cfg.get("random_seed")
    )
    # Align ML dataset to NWP along init_time intersection if available
    if "init_time" in ds_nwp.dims and "init_time" in ds_ml.dims:
        common_inits = np.intersect1d(
            ds_nwp.init_time.values, ds_ml.init_time.values
        )
        if common_inits.size == 0:
            raise ValueError(
                "No overlapping init_time values between ground_truth and ml datasets after selection."
            )
        ds_nwp = ds_nwp.sel(init_time=common_inits)
        ds_ml = ds_ml.sel(init_time=common_inits)
    # If both have lead_time, enforce identical coords by intersection
    if "lead_time" in ds_nwp.dims and "lead_time" in ds_ml.dims:
        common_leads = np.intersect1d(
            ds_nwp.lead_time.values, ds_ml.lead_time.values
        )
        if common_leads.size == 0:
            raise ValueError(
                "No overlapping lead_time values between ground_truth and ml datasets after selection."
            )
        ds_nwp = ds_nwp.sel(lead_time=common_leads)
        ds_ml = ds_ml.sel(lead_time=common_leads)

    ds_std, ds_ml_std = _standardize_pair(ds_nwp, ds_ml)
    return ds_nwp, ds_ml, ds_std, ds_ml_std


def _ensure_output(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_selected(
    cfg: dict[str, Any], selected_modules: list[str] | None
) -> None:
    ds, ds_ml, ds_std, ds_ml_std = prepare_datasets(cfg)

    out_root = _ensure_output(
        cfg.get("paths", {}).get("output_root", "output/verification_esfm")
    )
    chapter_flags = cfg.get("modules", {})
    plotting = cfg.get("plotting", {})

    # Import lazily to avoid import time if not needed
    if (not selected_modules and chapter_flags.get("maps")) or (
        selected_modules and "maps" in selected_modules
    ):
        from .plots import maps as maps_mod

        maps_mod.run(ds, ds_ml, out_root, plotting)

    if (not selected_modules and chapter_flags.get("histograms")) or (
        selected_modules and "histograms" in selected_modules
    ):
        from .plots import histograms as hist_mod

        hist_mod.run(ds, ds_ml, out_root, plotting)

    if (not selected_modules and chapter_flags.get("wd_kde")) or (
        selected_modules and "wd_kde" in selected_modules
    ):
        from .plots import wd_kde as wd_mod

        wd_mod.run(ds, ds_ml, ds_std, ds_ml_std, out_root, plotting)

    if (not selected_modules and chapter_flags.get("energy_spectra")) or (
        selected_modules and "energy_spectra" in selected_modules
    ):
        from .metrics import energy_spectra as es_mod

        es_mod.run(ds, ds_ml, out_root, plotting, cfg.get("selection", {}))

    if (not selected_modules and chapter_flags.get("vertical_profiles")) or (
        selected_modules and "vertical_profiles" in selected_modules
    ):
        from .plots import vertical_profiles as vp_mod

        vp_mod.run(ds, ds_ml, out_root, plotting, cfg.get("selection", {}))

    # Deterministic (previously called objective metrics)
    if (not selected_modules and chapter_flags.get("deterministic")) or (
        selected_modules and ("deterministic" in selected_modules)
    ):
        from .metrics import deterministic as det_mod

        det_mod.run(
            ds,
            ds_ml,
            ds_std,
            ds_ml_std,
            out_root,
            plotting,
            cfg.get("metrics", {}),
        )

    if (not selected_modules and chapter_flags.get("ets")) or (
        selected_modules and "ets" in selected_modules
    ):
        from .metrics import ets as ets_mod

        ets_mod.run(ds, ds_ml, out_root, cfg.get("metrics", {}))

    if (not selected_modules and chapter_flags.get("probabilistic")) or (
        selected_modules and "probabilistic" in selected_modules
    ):
        from .metrics.probabilistic import run_probabilistic

        run_probabilistic(ds, ds_ml, out_root, plotting, cfg)

    if (not selected_modules and chapter_flags.get("probabilistic_wbx")) or (
        selected_modules and "probabilistic_wbx" in selected_modules
    ):
        from .metrics.probabilistic import run_probabilistic_wbx

        run_probabilistic_wbx(ds, ds_ml, out_root, plotting, cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SwissClim Evaluations runner")
    p.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    p.add_argument(
        "--modules",
        type=str,
        nargs="*",
        choices=[
            "maps",
            "histograms",
            "wd_kde",
            "energy_spectra",
            "vertical_profiles",
            "deterministic",
            "ets",
            "probabilistic",
            "probabilistic_wbx",
        ],
        help=(
            "Optional subset of modules to run. If omitted, uses module toggles from the config file."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    # Pass through optional module subset directly
    run_selected(cfg, args.modules)


if __name__ == "__main__":
    main()
