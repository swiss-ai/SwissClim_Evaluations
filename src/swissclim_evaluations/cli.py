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
    # Explicit join ensures compatibility with upcoming xarray default change
    combined = xr.concat([ds, ds_ml], dim="__concat__", join="outer")
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

    # Handle optional ensemble dimension according to config and selected modules
    modules_cfg = cfg.get("modules", {})
    probabilistic_enabled = bool(modules_cfg.get("probabilistic")) or bool(
        modules_cfg.get("probabilistic_wbx")
    )
    if "ensemble" in ds_ml.dims:
        if ensemble_member is not None:
            # Select the given member and drop the ensemble dim to behave as deterministic
            ds_ml = ds_ml.isel(ensemble=int(ensemble_member), drop=True)
        elif not probabilistic_enabled:
            # No specific member requested and no probabilistic metrics: use ensemble mean
            ds_ml = ds_ml.mean(dim="ensemble", keep_attrs=True)

    ds_nwp = _apply_temporal_resolution(ds_nwp, hours)
    ds_ml = _apply_temporal_resolution(ds_ml, hours)

    ds_nwp = _select_variables(ds_nwp, variables_2d, variables_3d)
    ds_ml = _select_variables(ds_ml, variables_2d, variables_3d)

    # Ensure same time subsampling later
    plot_cfg = cfg.get("plotting", {})
    ds_nwp = _maybe_subsample_time(
        ds_nwp, plot_cfg.get("time_subsamples"), plot_cfg.get("random_seed")
    )

    # Align by valid_time using stack/unstack to ensure identical (init_time, lead_time) dims
    if "init_time" in ds_nwp.dims and "init_time" in ds_ml.dims:
        # Ensure both have lead_time coordinate
        if "lead_time" not in ds_ml.dims:
            ds_ml = ds_ml.expand_dims({
                "lead_time": np.array(
                    [np.timedelta64(0, "h")], dtype="timedelta64[h]"
                ).astype("timedelta64[ns]")
            })
        if "lead_time" not in ds_nwp.dims:
            ds_nwp = ds_nwp.expand_dims({
                "lead_time": np.array(
                    [np.timedelta64(0, "h")], dtype="timedelta64[h]"
                ).astype("timedelta64[ns]")
            })

        # Stack to a single dimension 'pair' (MultiIndex of init_time, lead_time)
        ml_init = ds_ml["init_time"].astype("datetime64[ns]")
        ml_lead = ds_ml["lead_time"].astype("timedelta64[ns]")
        nwp_init = ds_nwp["init_time"].astype("datetime64[ns]")
        nwp_lead = ds_nwp["lead_time"].astype("timedelta64[ns]")

        ds_ml_stacked = ds_ml.stack(pair=("init_time", "lead_time"))
        ds_nwp_stacked = ds_nwp.stack(pair=("init_time", "lead_time"))

        # Compute valid_time coordinate for each pair
        ml_valid = (ml_init.values[:, None] + ml_lead.values[None, :]).ravel()
        nwp_valid = (
            nwp_init.values[:, None] + nwp_lead.values[None, :]
        ).ravel()
        ds_ml_stacked = ds_ml_stacked.assign_coords(
            valid_time=("pair", ml_valid.astype("datetime64[ns]"))
        )
        ds_nwp_stacked = ds_nwp_stacked.assign_coords(
            valid_time=("pair", nwp_valid.astype("datetime64[ns]"))
        )

        # Intersection of valid times
        common_valid = np.intersect1d(
            ds_ml_stacked["valid_time"].values,
            ds_nwp_stacked["valid_time"].values,
        )
        if common_valid.size == 0:
            raise ValueError(
                "No overlapping valid times between ground_truth (ERA5 time) and ml (init_time+lead_time) after selection."
            )

        # Filter both by common valid_time
        ml_mask = np.isin(ds_ml_stacked["valid_time"].values, common_valid)
        nwp_mask = np.isin(ds_nwp_stacked["valid_time"].values, common_valid)
        ds_ml_stacked = ds_ml_stacked.isel(pair=ml_mask)
        ds_nwp_stacked = ds_nwp_stacked.isel(pair=nwp_mask)

        # Build mapping from valid_time -> first index in NWP; then align NWP order to ML order
        nwp_vt = ds_nwp_stacked["valid_time"].values
        ml_vt = ds_ml_stacked["valid_time"].values
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
        ds_nwp_stacked = ds_nwp_stacked.isel(pair=take_idx)
        # Make NWP stacked coordinate exactly the ML 'pair' MultiIndex for identical labels
        # Drop existing pair and its level coords first to avoid future MultiIndex inconsistency
        to_drop = [
            name
            for name in ("pair", "init_time", "lead_time")
            if name in ds_nwp_stacked.coords
        ]
        if to_drop:
            ds_nwp_stacked = ds_nwp_stacked.drop_vars(to_drop)
        ds_nwp_stacked = ds_nwp_stacked.assign_coords(
            pair=ds_ml_stacked["pair"]
        )  # type: ignore[index]

    # Unstack back to (init_time, lead_time) using the ML's pair labels for both
    ds_ml = ds_ml_stacked.unstack("pair")
    ds_nwp = ds_nwp_stacked.unstack("pair")

    ds_std, ds_ml_std = _standardize_pair(ds_nwp, ds_ml)
    return ds_nwp, ds_ml, ds_std, ds_ml_std


def _ensure_output(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_selected(cfg: dict[str, Any]) -> None:
    ds, ds_ml, ds_std, ds_ml_std = prepare_datasets(cfg)

    out_root = _ensure_output(
        cfg.get("paths", {}).get("output_root", "output/verification_esfm")
    )
    chapter_flags = cfg.get("modules", {})
    plotting = cfg.get("plotting", {})
    mode = str(plotting.get("output_mode", "plot")).lower()

    # Basic overview
    all_vars = list(ds.data_vars)
    vars_2d = [v for v in all_vars if "level" not in ds[v].dims]
    vars_3d = [v for v in all_vars if "level" in ds[v].dims]
    print(
        f"[swissclim] Starting run → output_root='{out_root}'. Output mode={mode}. Variables: 2D={len(vars_2d)}, 3D={len(vars_3d)}. Modules from config toggles."
    )

    # Import lazily to avoid import time if not needed
    if chapter_flags.get("maps"):
        from .plots import maps as maps_mod

        print(
            f"[swissclim] Module: maps — vars_2d={len(vars_2d)}, vars_3d={len(vars_3d)}"
        )
        maps_mod.run(ds, ds_ml, out_root, plotting)

    if chapter_flags.get("histograms"):
        from .plots import histograms as hist_mod

        print(f"[swissclim] Module: histograms — vars_2d={len(vars_2d)}")
        hist_mod.run(ds, ds_ml, out_root, plotting)

    if chapter_flags.get("wd_kde"):
        from .plots import wd_kde as wd_mod

        print(
            f"[swissclim] Module: wd_kde — vars_2d={len(vars_2d)} (standardized)"
        )
        wd_mod.run(ds, ds_ml, ds_std, ds_ml_std, out_root, plotting)

    if chapter_flags.get("energy_spectra"):
        from .metrics import energy_spectra as es_mod

        print(
            f"[swissclim] Module: energy_spectra — vars_2d={len(vars_2d)}, vars_3d={len(vars_3d)}"
        )
        es_mod.run(ds, ds_ml, out_root, plotting, cfg.get("selection", {}))

    if chapter_flags.get("vertical_profiles"):
        from .plots import vertical_profiles as vp_mod

        print(f"[swissclim] Module: vertical_profiles — vars_3d={len(vars_3d)}")
        vp_mod.run(ds, ds_ml, out_root, plotting, cfg.get("selection", {}))

    # Deterministic (previously called objective metrics)
    if chapter_flags.get("deterministic"):
        from .metrics import deterministic as det_mod

        print(f"[swissclim] Module: deterministic — variables={len(all_vars)}")

        det_mod.run(
            ds,
            ds_ml,
            ds_std,
            ds_ml_std,
            out_root,
            plotting,
            cfg.get("metrics", {}),
        )

    if chapter_flags.get("ets"):
        from .metrics import ets as ets_mod

        print(f"[swissclim] Module: ets — variables={len(all_vars)}")
        ets_mod.run(ds, ds_ml, out_root, cfg.get("metrics", {}))

    if chapter_flags.get("probabilistic"):
        from .metrics.probabilistic import run_probabilistic

        print(
            "[swissclim] Module: probabilistic — using ensemble metrics if available"
        )
        run_probabilistic(ds, ds_ml, out_root, plotting, cfg)

    if chapter_flags.get("probabilistic_wbx"):
        from .metrics.probabilistic import run_probabilistic_wbx

        print(
            "[swissclim] Module: probabilistic_wbx — WBX spread–skill and CRPS"
        )
        run_probabilistic_wbx(ds, ds_ml, out_root, plotting, cfg)


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
