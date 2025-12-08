from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_init_time_range,
    format_level_label,
    format_level_token,
    format_variable_name,
    get_variable_units,
    resolve_ensemble_mode,
)


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    ensemble_mode: str | None = None,
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    seed = int(plotting_cfg.get("random_seed", 42))

    section_output = out_root / "maps"

    rng = np.random.default_rng(seed)
    time_index = 0
    lead_index = 0
    if "init_time" in ds_target.dims and ds_target.init_time.size > 0:
        time_index = int(rng.integers(0, ds_target.init_time.size))
    if "lead_time" in ds_target.dims and ds_target.lead_time.size > 0:
        lead_index = 0
    if "time" in ds_target.dims and ds_target.time.size > 0:
        time_index = 0

    # Extract time/lead ranges for naming helper
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

    # Determine variables
    if "level" in ds_target.dims and int(ds_target.level.size) > 1:
        variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
        variables_2d = [v for v in ds_target.data_vars if v not in variables_3d]
    else:
        variables_3d = []
        variables_2d = list(ds_target.data_vars)

    # Assume no missing data per project requirement; use direct min/max.

    # Resolve ensemble handling (maps: mean/pooled/members). Prob invalid.
    resolved_mode = resolve_ensemble_mode("maps", ensemble_mode, ds_target, ds_prediction)
    has_ens = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for maps")

    if resolved_mode == "mean" and has_ens:
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble")
        if "ensemble" in ds_prediction.dims:
            ds_prediction = ds_prediction.mean(dim="ensemble")
        ensemble_members: list[int | None] = [None]
        ens_token_global = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled" and has_ens:
        ensemble_members = [None]
        ens_token_global = ensemble_mode_to_token("pooled")
    elif resolved_mode == "members" and has_ens:
        # enumerate members; token set per iteration
        if "ensemble" in ds_prediction.dims:
            ensemble_members = [int(i) for i in range(int(ds_prediction.sizes["ensemble"]))]
        else:
            ensemble_members = [int(i) for i in range(int(ds_target.sizes["ensemble"]))]
        ens_token_global = None
    else:
        ensemble_members = [None]
        ens_token_global = None

    # 2D maps (one figure per ensemble member if present)
    for _i, var in enumerate(variables_2d):  # _i unused (ruff B007)
        print(f"[maps] 2D variable: {var}")
        for ens in ensemble_members:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(14, 4),
                dpi=dpi * 2,
                subplot_kw={"projection": ccrs.PlateCarree()},
                constrained_layout=True,
            )

            ds_var = ds_target[var]
            ds_ml_var = ds_prediction[var]

            # Check if original variable has multiple init times (before slicing)
            is_single_init = True
            if "init_time" in ds_var.dims and ds_var.sizes["init_time"] > 1:
                is_single_init = False

            if ens is not None:
                if "ensemble" in ds_var.dims:
                    ds_var = ds_var.isel(ensemble=ens)
                if "ensemble" in ds_ml_var.dims:
                    ds_ml_var = ds_ml_var.isel(ensemble=ens)
            if "init_time" in ds_var.dims:
                ds_var = ds_var.isel(init_time=time_index)
            if "lead_time" in ds_var.dims:
                ds_var = ds_var.isel(lead_time=lead_index)
            if "time" in ds_var.dims:
                ds_var = ds_var.isel(time=time_index)
            if "init_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(init_time=time_index)
            if "lead_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(lead_time=lead_index)
            if "time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(time=time_index)
            # Drop any remaining singleton temporal dims
            for dim_drop in ("time", "init_time", "lead_time"):
                if dim_drop in ds_var.dims and ds_var.sizes[dim_drop] == 1:
                    ds_var = ds_var.isel({dim_drop: 0})
                if dim_drop in ds_ml_var.dims and ds_ml_var.sizes[dim_drop] == 1:
                    ds_ml_var = ds_ml_var.isel({dim_drop: 0})
            ds_var = ds_var.squeeze()
            ds_ml_var = ds_ml_var.squeeze()
            vmin = min(float(ds_var.min()), float(ds_ml_var.min()))
            vmax = max(float(ds_var.max()), float(ds_ml_var.max()))

            lon = ds_var.coords.get("longitude", None)
            lat = ds_var.coords.get("latitude", None)
            im0 = axes[0].pcolormesh(
                lon if lon is not None else ds_var.longitude,
                lat if lat is not None else ds_var.latitude,
                ds_var.values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )
            axes[0].coastlines(linewidth=0.5)

            date_str = extract_date_from_dataset(ds_var) if is_single_init else ""
            axes[0].set_title(f"{format_variable_name(var)} — Target{date_str}")

            lon_ml = ds_ml_var.coords.get("longitude", None)
            lat_ml = ds_ml_var.coords.get("latitude", None)
            axes[1].pcolormesh(
                lon_ml if lon_ml is not None else ds_ml_var.longitude,
                lat_ml if lat_ml is not None else ds_ml_var.latitude,
                ds_ml_var.values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )
            axes[1].coastlines(linewidth=0.5)
            axes[1].set_title("Prediction")

            # Use a colorbar compatible with constrained_layout, spanning both axes
            cb = fig.colorbar(
                im0,
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.08,
            )
            # In test mode, colorbar may be a dummy (None); guard the label call
            try:
                if cb is not None:
                    cb.set_label(get_variable_units(ds_target, str(var)))
            except Exception:
                # Non-fatal: continue without setting label
                pass

            # title_extra = "" if ens is None else f" (Ensemble {ens})"
            # date_suffix = extract_date_from_dataset(ds_var)
            # plt.suptitle(f"{format_variable_name(str(var))}{title_extra}{date_suffix}")

            # Determine filename ensemble token
            ens_token = (
                ensemble_mode_to_token("members", ens)
                if (resolved_mode == "members" and ens is not None)
                else ens_token_global
            )
            if save_fig:
                section_output.mkdir(parents=True, exist_ok=True)
                out_png = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level="surface",
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[maps] saved {out_png}")
            if save_npz:
                section_output.mkdir(parents=True, exist_ok=True)
                npz_path = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level="surface",
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="npz",
                )
                np.savez(
                    npz_path,
                    target=ds_var.values,
                    prediction=ds_ml_var.values,
                    latitude=(
                        ds_var.latitude.values if "latitude" in ds_var.coords else np.array([])
                    ),
                    longitude=(
                        ds_var.longitude.values if "longitude" in ds_var.coords else np.array([])
                    ),
                    ensemble=int(ens) if ens is not None else -1,
                    variable=str(var),
                    units=get_variable_units(ds_target, str(var)),
                )
                print(f"[maps] saved {npz_path}")

            plt.close(fig)

    # 3D maps per level
    for _i, var in enumerate(variables_3d):  # _i unused (ruff B007)
        print(f"[maps] 3D variable: {var}")
        levels = list(ds_target[var].coords.get("level", []))
        if not levels:
            continue

        level_tokens = [format_level_token(lvl) for lvl in levels]
        if not level_tokens:
            level_label = "levels"
        elif len(level_tokens) == 1:
            level_label = level_tokens[0]
        else:
            level_label = f"{level_tokens[0]}_to_{level_tokens[-1]}"
        num_rows = len(levels)
        for ens in ensemble_members:
            fig, axes = plt.subplots(
                num_rows,
                2,
                figsize=(14, 4 * num_rows),
                dpi=dpi * 2,
                subplot_kw={"projection": ccrs.PlateCarree()},
                squeeze=False,
                constrained_layout=True,
            )

            ds_var = ds_target[var]
            ds_ml_var = ds_prediction[var]

            # Check if original variable has multiple init times (before slicing)
            is_single_init = True
            if "init_time" in ds_var.dims and ds_var.sizes["init_time"] > 1:
                is_single_init = False

            if ens is not None:
                if "ensemble" in ds_var.dims:
                    ds_var = ds_var.isel(ensemble=ens)
                if "ensemble" in ds_ml_var.dims:
                    ds_ml_var = ds_ml_var.isel(ensemble=ens)
            if "init_time" in ds_var.dims:
                ds_var = ds_var.isel(init_time=time_index)
            if "lead_time" in ds_var.dims:
                ds_var = ds_var.isel(lead_time=lead_index)
            if "init_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(init_time=time_index)
            if "lead_time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(lead_time=lead_index)
            # Ensure we also select a single time slice if a free 'time' dimension exists
            # (mirrors handling in 2D map section). Without this, pcolormesh receives a 3D
            # array (time, lat, lon) and raises a dimension mismatch error.
            if "time" in ds_var.dims:
                ds_var = ds_var.isel(time=time_index)
            if "time" in ds_ml_var.dims:
                ds_ml_var = ds_ml_var.isel(time=time_index)

            for idx, level in enumerate(levels):
                level_val = int(level.values) if hasattr(level, "values") else int(level)
                ax_ds, ax_ds_ml = axes[idx]
                ds_var_lev = ds_var.sel(level=level)
                ds_ml_var_lev = ds_ml_var.sel(level=level)
                # After selecting level we expect dims (latitude, longitude). If an extra
                # singleton dimension remains (e.g., time/init_time/lead_time), squeeze it.
                ds_var_lev = ds_var_lev.squeeze()
                ds_ml_var_lev = ds_ml_var_lev.squeeze()

                vmin = min(float(ds_var_lev.min()), float(ds_ml_var_lev.min()))
                vmax = max(float(ds_var_lev.max()), float(ds_ml_var_lev.max()))

                vmin = min(float(ds_var_lev.min()), float(ds_ml_var_lev.min()))
                vmax = max(float(ds_var_lev.max()), float(ds_ml_var_lev.max()))

                im_ds = ax_ds.pcolormesh(
                    ds_var_lev.coords.get("longitude"),
                    ds_var_lev.coords.get("latitude"),
                    ds_var_lev.values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                ax_ds.coastlines(linewidth=0.5)

                date_str = extract_date_from_dataset(ds_var) if is_single_init else ""
                ax_ds.set_title(
                    f"{format_variable_name(str(var))} — Target"
                    f"{format_level_label(level_val)}{date_str}"
                )

                ax_ds_ml.pcolormesh(
                    ds_ml_var_lev.coords.get("longitude"),
                    ds_ml_var_lev.coords.get("latitude"),
                    ds_ml_var_lev.values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                )
                ax_ds_ml.coastlines(linewidth=0.5)
                ax_ds_ml.set_title(f"Model{format_level_label(level_val)}")

                fig.colorbar(
                    im_ds,
                    ax=[ax_ds, ax_ds_ml],
                    orientation="horizontal",
                    fraction=0.05,
                    pad=0.07,
                    label=f"{get_variable_units(ds_target, str(var))} (level {level_val})",
                )

            # date_suffix = extract_date_from_dataset(ds_var.isel(level=0))
            # plt.suptitle(f"{format_variable_name(str(var))}{title_extra}{date_suffix}")
            # With constrained_layout=True above, avoid calling tight_layout (incompatible).
            # Adjust padding via constrained layout pads instead.
            # Leave default constrained_layout behavior; avoid calling private methods for safety

            ens_token = (
                ensemble_mode_to_token("members", ens)
                if (resolved_mode == "members" and ens is not None)
                else ens_token_global
            )
            if save_fig:
                section_output.mkdir(parents=True, exist_ok=True)
                out_png = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level=level_label,
                    qualifier=None,  # drop legacy 'pl' qualifier
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[maps] saved {out_png}")
            if save_npz:
                section_output.mkdir(parents=True, exist_ok=True)
                npz_path = section_output / build_output_filename(
                    metric="map",
                    variable=str(var),
                    level=level_label,
                    qualifier=None,  # drop legacy 'pl' qualifier
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="npz",
                )
                np.savez(
                    npz_path,
                    target=ds_var.values,
                    prediction=ds_ml_var.values,
                    latitude=(
                        ds_var.latitude.values if "latitude" in ds_var.coords else np.array([])
                    ),
                    longitude=(
                        ds_var.longitude.values if "longitude" in ds_var.coords else np.array([])
                    ),
                    level=(ds_var.level.values if "level" in ds_var.coords else np.array([])),
                    ensemble=int(ens) if ens is not None else -1,
                    variable=str(var),
                    units=get_variable_units(ds_target, str(var)),
                )
                print(f"[maps] saved {npz_path}")
            plt.close(fig)
