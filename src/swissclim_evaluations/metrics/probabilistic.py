from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as dsa
import matplotlib.pyplot as plt

# plotting dependencies will be used in plot_probabilistic() and WBX map (optional)
import numpy as np
import pandas as pd
import xarray as xr
from weatherbenchX.metrics.probabilistic import (
    CRPSEnsemble,
    UnbiasedSpreadSkillRatio,
)

# Use official WeatherBenchX metrics instead of local copies
from ..helpers import (
    COLOR_DIAGNOSTIC,
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_init_time_range,
    format_variable_name,
    time_chunks,
)


def compute_wbx_crps(
    da_target: xr.DataArray, da_prediction: xr.DataArray, ensemble_dim: str = "ensemble"
) -> xr.DataArray:
    """Compute Fair CRPS using WeatherBenchX implementation.

    Replicates logic of CRPSEnsemble: CRPS = CRPSSkill - 0.5 * CRPSSpread.
    """
    metric = CRPSEnsemble(ensemble_dim=ensemble_dim)
    # WBX expects dicts of DataArrays
    # We use a dummy variable name 'v'
    preds = {"v": da_prediction}
    targs = {"v": da_target}

    stats = {}
    for name, stat in metric.statistics.items():
        # compute returns dict {var: da}
        res = stat.compute(preds, targs)
        stats[name] = res["v"]

    # CRPS = CRPSSkill - 0.5 * CRPSSpread
    crps = stats["CRPSSkill"] - 0.5 * stats["CRPSSpread"]
    return _add_metric_prefix(crps, "CRPS")


def _pit(da_target, da_prediction):
    return np.mean(da_prediction < da_target[..., None], axis=-1)


def probability_integral_transform(
    da_target, da_prediction, ensemble_dim="ensemble", name_prefix: str = "PIT"
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


def _pit_histogram_dask(
    da: xr.DataArray, bins: int = 50, density: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PIT histogram using dask.array.histogram.
    Returns (counts, edges). If density=True, return density values.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Use dask-backed data when available; otherwise wrap numpy data lazily
    data = getattr(da, "data", da)
    darr = dsa.asarray(data)
    darr = darr.ravel()
    darr = darr[~dsa.isnan(darr)]
    counts = dsa.histogram(darr, bins=np.asarray(edges))[0].compute().astype(np.float64)
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
    - Optional: {var}_pit.nc and {var}_crps.nc when plotting.output_mode is 'npz' or 'both'
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
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            return format_init_time_range(vals)
        except Exception:
            pass
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
    prob_cfg = (cfg_all or {}).get("probabilistic", {})
    report_per_level = bool(prob_cfg.get("report_per_level", True))
    crps_rows_per_level: list[dict[str, Any]] = []

    for var in variables:
        # Extract and align targets and predictions along shared coordinates
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

        # Drop ensemble from target if present (it's a dummy for strict compliance)
        # to allow broadcasting against the full prediction ensemble.
        if "ensemble" in da_target.dims:
            da_target = da_target.isel(ensemble=0, drop=True)

        try:
            da_target, da_prediction = xr.align(
                da_target, da_prediction, join="exact", exclude=["ensemble"]
            )
        except Exception:
            # Fallback to by-position if shapes match exactly
            if da_target.shape == da_prediction.shape:
                da_target = da_target.copy()
                da_prediction = da_prediction.copy()
            else:
                raise
        crps_da = compute_wbx_crps(da_target, da_prediction, ensemble_dim="ensemble")
        crps_mean = float(_reduce_mean_all(crps_da).compute().item())
        crps_rows.append({"variable": var, "CRPS": crps_mean})

        if report_per_level and "level" in crps_da.dims:
            dims_to_reduce = [d for d in crps_da.dims if d != "level"]
            crps_per_level = crps_da.mean(dim=dims_to_reduce, skipna=True).compute()

            for lvl in crps_per_level.level.values:
                crps_rows_per_level.append(
                    {
                        "variable": var,
                        "level": int(lvl),
                        "CRPS": float(crps_per_level.sel(level=lvl).item()),
                    }
                )

        pit_da = probability_integral_transform(
            da_target,
            da_prediction,
            ensemble_dim="ensemble",
            name_prefix="PIT",
        )
        counts, edges = _pit_histogram_dask(pit_da, bins=50, density=True)
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
        # Always save PIT and CRPS fields for reproducibility
        pit_nc = section_output / build_output_filename(
            metric="pit_field",
            variable=str(var),
            level=None,
            qualifier=None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="nc",
        )
        crps_nc = section_output / build_output_filename(
            metric="crps_field",
            variable=str(var),
            level=None,
            qualifier=None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="nc",
        )
        pit_da.to_netcdf(pit_nc)
        crps_da.to_netcdf(crps_nc)
        print(f"[probabilistic] saved {pit_nc}")
        print(f"[probabilistic] saved {crps_nc}")

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
    _aligned = False
    # For probabilistic plots, target is deterministic (no ensemble) while prediction
    # has ensemble dimension. Check alignment only on non-ensemble dimensions.
    non_ens_dims_p = [d for d in da_p.dims if d != "ensemble"]
    if all(dim in da_t.dims for dim in non_ens_dims_p):
        da_t, da_p = xr.align(da_t, da_p, join="outer")
        _aligned = True
    if not _aligned and da_t.shape != da_p.shape:
        raise RuntimeError(
            f"Failed aligning base variable '{base_var}' for probabilistic plot; shapes differ: "
            f"target={da_t.shape} prediction={da_p.shape}"
        )
    # CRPS values (keep lead_time for per-lead panels)
    crps = compute_wbx_crps(da_t, da_p, ensemble_dim="ensemble")
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
    lon_vals = np.asarray(crps_map[lon_name].values)
    if lon_vals.size:
        lmin, lmax = float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))
        if (lmax - lmin) > 180 and np.any(lon_vals < 90) and np.any(lon_vals > 270):
            new = lon_vals.copy()
            new[new > 180] -= 360
            order = np.argsort(new)
            crps_map = crps_map.isel({lon_name: order}).assign_coords(
                {lon_name: (lon_name, new[order])}
            )

    # Plot CRPS map (simple original style, no percentile scaling / fallback)
    fig = plt.figure(figsize=(12, 6), dpi=dpi * 2)
    ax = plt.axes(projection=ccrs.PlateCarree())
    if hasattr(ax, "add_feature"):
        ax.add_feature(cfeature.COASTLINE, lw=0.5)
        ax.add_feature(cfeature.BORDERS, lw=0.3)
    Z = crps_map.values
    # Debug info for empty map diagnosis
    finite_count = int(np.isfinite(Z).sum())
    print(f"[probabilistic-plots] CRPS finite values: {finite_count} / {Z.size}")
    if finite_count == 0:
        print("[probabilistic-plots] WARNING: CRPS map has no finite values (all NaN or inf).")
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

    # Check for single date
    date_str = extract_date_from_dataset(ds_target)

    ax.set_title(f"CRPS Map — {format_variable_name(base_var)}{date_str}")

    # Attempt time range extraction for plots
    def _extract_init_range_plot(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            return format_init_time_range(vals)
        except Exception:
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
    if save_fig:
        from ..helpers import build_output_filename, ensemble_mode_to_token

        ens_token_plot = ensemble_mode_to_token("prob")
        ax.set_title(f"CRPS map (mean over time): {base_var}", fontsize=11)
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
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"[probabilistic-plots] saved {out_png}")
    if save_npz:
        from ..helpers import build_output_filename, ensemble_mode_to_token

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
        np.savez(
            out_npz,
            crps=crps_map.values,
            latitude=crps_map[lat_name].values,
            longitude=crps_map[lon_name].values,
            variable=base_var,
            metric="CRPS",
        )
        print(f"[probabilistic-plots] saved {out_npz}")
    plt.close(fig)

    # Panel: CRPS maps by lead_time across all retained hours (panel concept removed)
    if "lead_time" in crps.dims and int(crps.sizes.get("lead_time", 0)) >= 1 and save_fig:

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
            lon_vals2 = np.asarray(crps_by_lead[lon_name].values)
            if lon_vals2.size:
                lmin2, lmax2 = float(np.nanmin(lon_vals2)), float(np.nanmax(lon_vals2))
                if (lmax2 - lmin2) > 180 and np.any(lon_vals2 < 90) and np.any(lon_vals2 > 270):
                    new2 = lon_vals2.copy()
                    new2[new2 > 180] -= 360
                    order2 = np.argsort(new2)
                    crps_by_lead = crps_by_lead.isel({lon_name: order2}).assign_coords(
                        {lon_name: (lon_name, new2[order2])}
                    )
            # Build mapping from all available lead hours -> index
            raw_leads = crps_by_lead["lead_time"].values
            crps_hours: list[int] = [
                _to_hour(x, idx) for idx, x in enumerate(raw_leads)
            ]  # renamed from all_hours
            # Debug visibility
            from contextlib import suppress

            with suppress(Exception):
                print(f"[probabilistic-plots] CRPS grid using lead_hours={crps_hours}")
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
                    ax.set_title(f"CRPS — lead {int(h)}h", fontsize=11)
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
                from ..helpers import ensemble_mode_to_token

                ens_token_grid = ensemble_mode_to_token("prob")
                plt.suptitle(
                    f"CRPS grid — {base_var} | ensemble={ens_token_grid}",
                    fontsize=14,
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
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[probabilistic-plots] saved {out_png}")
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
                print(f"[probabilistic-plots] saved {out_csv_line}")
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
                    np.savez(
                        out_npz_line,
                        lead_hours=df_crps_line["lead_time_hours"].values.astype(float),
                        crps=df_crps_line["CRPS"].values.astype(float),
                        variable=base_var,
                    )
                    print(f"[probabilistic-plots] saved {out_npz_line}")

    # PIT histogram (global and by-lead panels)
    pit = probability_integral_transform(
        da_t,
        da_p,
        ensemble_dim="ensemble",
        name_prefix="PIT",
    )
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

    ax.set_title(f"PIT Histogram — {format_variable_name(base_var)}{date_str}")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.axhline(1.0, color="brown", linestyle="--", linewidth=1, label="Uniform")
    ax.legend()
    if save_npz:
        # Use standardized filename builder for NPZ (with ensprob token)
        from ..helpers import build_output_filename, ensemble_mode_to_token

        ens_token_plot = ensemble_mode_to_token("prob")
        out_npz = section / build_output_filename(
            metric="pit_hist",
            variable=base_var,
            level=None,
            qualifier=None,
            init_time_range=None,
            lead_time_range=None,
            ensemble=ens_token_plot,
            ext="npz",
        )
        np.savez(out_npz, counts=counts, edges=edges, variable=base_var)
        print(f"[probabilistic-plots] saved {out_npz}")

    if save_fig:
        out_png = (
            section / f"pit_hist_{base_var}.png"
        )  # legacy non-tokenized image filename retained
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"[probabilistic-plots] saved {out_png}")
    plt.close(fig)

    # PIT per-lead panel plot (multi-row grid) over all retained hours
    if "lead_time" in pit.dims and int(pit.sizes.get("lead_time", 0)) >= 1 and save_fig:
        full_hours = [_to_hour(x, idx) for idx, x in enumerate(pit["lead_time"].values)]
        if full_hours:
            raw_leads = pit["lead_time"].values
            all_hours: list[int] = [_to_hour(x, idx) for idx, x in enumerate(raw_leads)]
            # Debug visibility
            hour_index_pairs = []
            for h in full_hours:
                try:
                    idx = all_hours.index(int(h))
                    hour_index_pairs.append((int(h), idx))
                except Exception:
                    continue
            pit_line_rows: list[dict[str, float]] = []
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
                )
                for i, (h, li) in enumerate(hour_index_pairs):
                    r, c = divmod(i, ncols)
                    sub = pit.isel(lead_time=li)
                    data = np.asarray(sub.values).ravel()
                    data = data[np.isfinite(data)]
                    counts_local, _ = np.histogram(data, bins=edges)
                    width = np.diff(edges)
                    total = counts_local.sum()
                    dens = counts_local / (total * width.mean()) if total > 0 else counts_local
                    ax = axes[r][c]
                    ax.bar(
                        edges[:-1],
                        dens,
                        width=width,
                        align="edge",
                        color="#4C78A8",
                        edgecolor="white",
                    )
                    ax.axhline(1.0, color="brown", linestyle="--", linewidth=1)
                    ax.set_title(f"Lead {int(h)}h", fontsize=11)
                    pit_line_rows.append(
                        {
                            "lead_time_hours": float(h),
                            "uniform_diff": float(np.nanmean(np.abs(dens - 1.0))),
                        }
                    )
                    if r == nrows - 1:
                        ax.set_xlabel("PIT value")
                    if c == 0:
                        ax.set_ylabel("Density")
                plt.suptitle(f"PIT histograms per lead — {base_var}", fontsize=16)
                from ..helpers import ensemble_mode_to_token

                out_png = section / build_output_filename(
                    metric="pit_hist",
                    variable=base_var,
                    level=None,
                    qualifier="grid",
                    init_time_range=init_range_plot,
                    lead_time_range=lead_range_plot,
                    ensemble=ensemble_mode_to_token("prob"),
                    ext="png",
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[probabilistic-plots] saved {out_png}")
                plt.close(fig)
            if pit_line_rows:
                df_pit = pd.DataFrame(pit_line_rows).sort_values("lead_time_hours")
                out_csv_pit = section / build_output_filename(
                    metric="pit_hist",
                    variable=base_var,
                    level=None,
                    qualifier="uniform_diff_by_lead",
                    init_time_range=init_range_plot,
                    lead_time_range=lead_range_plot,
                    ensemble=ensemble_mode_to_token("prob"),
                    ext="csv",
                )
                df_pit.to_csv(out_csv_pit, index=False)
                print(f"[probabilistic-plots] saved {out_csv_pit}")
                if save_npz:
                    out_npz_pit = section / build_output_filename(
                        metric="pit_hist",
                        variable=base_var,
                        level=None,
                        qualifier="uniform_diff_by_lead_data",
                        init_time_range=init_range_plot,
                        lead_time_range=lead_range_plot,
                        ensemble=ensemble_mode_to_token("prob"),
                        ext="npz",
                    )
                    np.savez(
                        out_npz_pit,
                        lead_hours=df_pit["lead_time_hours"].values.astype(float),
                        uniform_diff=df_pit["uniform_diff"].values.astype(float),
                        variable=base_var,
                    )
                    print(f"[probabilistic-plots] saved {out_npz_pit}")

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
                    crps = compute_wbx_crps(
                        ds_target[var], ds_prediction[var], ensemble_dim="ensemble"
                    )
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
                    ax.set_xlabel("lead_time (h)")
                    ax.set_ylabel("CRPS")
                    ax.set_title(f"CRPS vs lead_time — {var}")
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
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    print(f"[probabilistic-plots] saved {out_png}")
                    plt.close(fig)
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
                        np.savez(
                            out_npz, lead_hours=np.array(hours_plot), crps=vals, variable=str(var)
                        )
                        print(f"[probabilistic-plots] saved {out_npz}")


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
    - probabilistic_metrics_temporal.nc
    - probabilistic_metrics_spatial.nc
    """
    # Write WBX artifacts into the same probabilistic folder to avoid split outputs
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    # Imports only when needed to avoid hard dependency during other runs
    from weatherbenchX import aggregation, binning

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
    if m_ens < 2:
        raise RuntimeError(
            "WBX probabilistic metrics require ensemble size >=2 (UnbiasedSpreadSkillRatio). "
            f"Found ensemble size {m_ens}."
        )
    ssr_metric = UnbiasedSpreadSkillRatio(ensemble_dim="ensemble")
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

    spatial_aggregator = aggregation.Aggregator(
        reduce_dims=["latitude", "longitude"],
        bin_by=[binning.Regions(regions=regions)],
    )

    seasonal = (
        bool((plotting_cfg or {}).get("group_by_season", False))
        if isinstance(plotting_cfg, dict)
        else False
    )
    temporal_bin_by = [binning.ByTimeUnit("season", "init_time")] if seasonal else None
    temporal_aggregator = aggregation.Aggregator(
        reduce_dims=["init_time"],
        bin_by=temporal_bin_by,
    )

    metrics = {}
    metrics["SSR"] = UnbiasedSpreadSkillRatio(ensemble_dim="ensemble")
    metrics["CRPS"] = CRPSEnsemble(ensemble_dim="ensemble")

    variables = list(ds_pred.data_vars)
    pred_map = {v: ds_pred[v] for v in variables}
    targ_map = {v: ds_targ[v] for v in variables}
    # Temporal results: reduce spatial dims, keep time dims (and region)
    # This contains (Region, Time)
    region_time_results = aggregation.compute_metric_values_for_single_chunk(
        metrics, spatial_aggregator, pred_map, targ_map
    )
    # Spatial results: reduce init_time (and optionally bin by season)
    # This contains (Lat, Lon) - Map
    map_results = aggregation.compute_metric_values_for_single_chunk(
        metrics, temporal_aggregator, pred_map, targ_map
    )

    # Derive "Spatial" (Region, averaged over time) from region_time_results
    dims_to_reduce_time = [d for d in region_time_results.dims if "time" in d]
    spatial_results = region_time_results.mean(dim=dims_to_reduce_time, skipna=True)

    # Derive "Temporal" (Time, averaged over region/global) from region_time_results
    if "region" in region_time_results.dims:
        if "global" in region_time_results.region.values:
            temporal_results = region_time_results.sel(region="global")
        else:
            temporal_results = region_time_results.mean(dim="region", skipna=True)
    else:
        temporal_results = region_time_results

    def _build_time_encoding(ds: xr.Dataset) -> dict:
        enc: dict = {}
        names = list(ds.data_vars) + list(ds.coords)
        for name in names:
            try:
                da = ds[name]
            except Exception:
                continue
            if hasattr(da, "dtype"):
                kind = getattr(da.dtype, "kind", "")
                if kind == "M":  # datetime64
                    enc[name] = {
                        "units": "seconds since 1970-01-01",
                        "dtype": "i4",
                    }
                elif kind == "m":  # timedelta64
                    enc[name] = {"units": "seconds", "dtype": "i4"}
        return enc

    enc_t = _build_time_encoding(temporal_results)
    enc_s = _build_time_encoding(spatial_results)
    enc_m = _build_time_encoding(map_results)

    temporal_fn = section / build_output_filename(
        metric="prob_metrics_temporal",
        variable=None,
        level=None,
        qualifier=None,
        init_time_range=init_range,
        lead_time_range=lead_range,
        ensemble=ens_token_prob,
        ext="nc",
    )
    spatial_fn = section / build_output_filename(
        metric="prob_metrics_spatial",
        variable=None,
        level=None,
        qualifier=None,
        init_time_range=init_range,
        lead_time_range=lead_range,
        ensemble=ens_token_prob,
        ext="nc",
    )
    map_fn = section / build_output_filename(
        metric="prob_metrics_map",
        variable=None,
        level=None,
        qualifier=None,
        init_time_range=init_range,
        lead_time_range=lead_range,
        ensemble=ens_token_prob,
        ext="nc",
    )

    temporal_results.to_netcdf(temporal_fn, engine="scipy", encoding=enc_t)
    spatial_results.to_netcdf(spatial_fn, engine="scipy", encoding=enc_s)
    map_results.to_netcdf(map_fn, engine="scipy", encoding=enc_m)

    print("Wrote:", temporal_fn)
    print("Wrote:", spatial_fn)
    print("Wrote:", map_fn)

    # --- Plotting SSR (Temporal and Spatial) ---
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")

    if save_fig:
        # Plot Temporal (Time)
        for var_name in temporal_results.data_vars:
            if not str(var_name).startswith("SSR"):
                continue

            da = temporal_results[var_name]
            # da is (Time)

            # Average over lead_time if present
            if "lead_time" in da.dims:
                da = da.mean(dim="lead_time")

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Convert to dataframe for bar plot
            df = da.to_dataframe(name="SSR").reset_index()

            # Determine x-axis column
            x_col = "init_time" if "init_time" in df.columns else df.columns[0]

            # Plot bar
            ax.bar(df[x_col].astype(str), df["SSR"])

            ax.set_title(f"SSR over Init Time - {var_name}")
            ax.set_ylabel("SSR")
            ax.set_xlabel("")  # Remove x-label as requested
            ax.grid(True, axis="y")

            out_png_temp = section / build_output_filename(
                metric="ssr_temporal",
                variable=str(var_name),
                level=None,
                qualifier=None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token_prob,
                ext="png",
            )
            plt.savefig(out_png_temp, bbox_inches="tight")
            plt.close(fig)
            print(f"[probabilistic] saved {out_png_temp}")

        # Plot Spatial (Region)
        for var_name in spatial_results.data_vars:
            if not str(var_name).startswith("SSR"):
                continue

            da = spatial_results[var_name]
            # da is (Region)

            if "region" in da.dims:
                # Convert to series for plotting
                s_spatial = da.to_series()

                # Filter NaNs (robust plotting)
                s_spatial = pd.to_numeric(s_spatial, errors="coerce").dropna()

                if not s_spatial.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    s_spatial.plot(kind="bar", ax=ax)
                    ax.set_title(f"SSR by Region (Time-Averaged) - {var_name}")
                    ax.set_ylabel("SSR")
                    ax.set_xlabel("")  # Remove Region label
                    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Ideal (1.0)")
                    ax.legend()
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()

                    out_png_spatial = section / build_output_filename(
                        metric="ssr_spatial",
                        variable=str(var_name),
                        level=None,
                        qualifier=None,
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token_prob,
                        ext="png",
                    )
                    plt.savefig(out_png_spatial, bbox_inches="tight")
                    plt.close(fig)
                    print(f"[probabilistic] saved {out_png_spatial}")
                else:
                    print(f"[probabilistic] Skipping spatial plot for {var_name}: No numeric data.")


def _per_variable_mean_df(da_or_ds: xr.Dataset | xr.DataArray) -> pd.DataFrame:
    ds = da_or_ds.to_dataset(name="value") if isinstance(da_or_ds, xr.DataArray) else da_or_ds
    dims = [
        d
        for d in [
            "time",
            "init_time",
            "lead_time",
            "latitude",
            "longitude",
            "level",
            "ensemble",
        ]
        if d in ds.dims
    ]
    return ds.mean(dim=dims, skipna=True).to_dataframe()
