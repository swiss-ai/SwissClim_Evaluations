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

    if levels is not None and "level" in ds.dims:
        ds = ds.sel(level=levels)
    if latitudes is not None:
        ds = ds.sel(latitude=slice(*latitudes))
    if longitudes is not None:
        ds = ds.sel(longitude=slice(*longitudes))
    if datetimes is not None:
        ds = ds.sel(time=slice(*datetimes)) if "time" in ds.dims else ds
    return ds


def _apply_temporal_resolution(ds: xr.Dataset, hours: int | None) -> xr.Dataset:
    if hours is None:
        return ds
    if "time" not in ds.dims or ds.time.size < 2:
        return ds
    timestep = int(
        (ds.time.isel(time=1) - ds.time.isel(time=0)).dt.total_seconds() / 3600
    )
    factor = max(1, hours // timestep) if timestep > 0 else 1
    return ds.isel(time=slice(None, None, factor))


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
    if n is None or "time" not in ds.dims or ds.time.size == 0:
        return ds
    n = int(min(n, ds.time.size))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ds.time.size, size=n, replace=False))
    return ds.isel(time=idx)


def _standardize_pair(
    ds: xr.Dataset, ds_ml: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    combined = (
        xr.concat([ds, ds_ml], dim="time")
        if "time" in ds.dims
        else xr.concat([ds, ds_ml], dim="combine")
    )
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

    ds_nwp = data_mod.era5(paths.get("nwp"))
    ds_ml = xr.open_zarr(paths.get("ml"))

    # Align dims to match notebook expectations
    ds_nwp = _slice_common(ds_nwp, cfg)
    ds_ml = _slice_common(ds_ml, cfg)

    if "ensemble" in ds_ml.dims and ensemble_member is not None:
        ds_ml = ds_ml.isel(ensemble=int(ensemble_member))

    ds_nwp = _apply_temporal_resolution(ds_nwp, hours)
    ds_ml = _apply_temporal_resolution(ds_ml, hours)

    ds_nwp = _select_variables(ds_nwp, variables_2d, variables_3d)
    ds_ml = _select_variables(ds_ml, variables_2d, variables_3d)

    # Ensure same time subsampling later
    plot_cfg = cfg.get("plotting", {})
    ds_nwp = _maybe_subsample_time(
        ds_nwp, plot_cfg.get("time_subsamples"), plot_cfg.get("random_seed")
    )
    ds_ml = (
        ds_ml.sel(time=ds_nwp.time)
        if "time" in ds_nwp.dims and "time" in ds_ml.dims
        else ds_ml
    )

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
    chapter_flags = cfg.get("chapters", {})
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
        "--chapters",
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
        help="[Deprecated] Use --modules. Subset of modules to run. If omitted, uses config toggles.",
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
        help="Subset of modules to run. If omitted, uses config toggles.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    # Combine deprecated --chapters and new --modules
    selected: list[str] | None = None
    if args.modules or args.chapters:
        selected = []
        if args.modules:
            selected.extend(args.modules)
        if args.chapters:
            selected.extend(args.chapters)
    run_selected(cfg, selected)


if __name__ == "__main__":
    main()
