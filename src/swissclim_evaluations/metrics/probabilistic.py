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

# Use official WeatherBenchX metrics instead of local copies
from weatherbenchX.metrics.probabilistic import (
    CRPSEnsemble as WBXCRPSEnsemble,
    SpreadSkillRatio as WBXSpreadSkillRatio,
)

from ..helpers import build_output_filename, time_chunks


def _crps_e1(da_target: np.ndarray, da_prediction: np.ndarray) -> np.ndarray:
    M: int = da_prediction.shape[-1]
    e_1 = np.sum(np.abs(da_target[..., None] - da_prediction), axis=-1) / M
    return e_1


def crps_e1(da_target, da_prediction, name_prefix: str = "CRPS", ensemble_dim="ensemble"):
    """Compute the CRPS (e1 component) for ensemble predictions vs targets."""
    return xr.apply_ufunc(
        _crps_e1,
        da_target,
        da_prediction,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
    )


def _crps_e2(da_prediction: np.ndarray) -> np.ndarray:
    M: int = da_prediction.shape[-1]
    # Require at least 2 members; upstream runner enforces this. Keep explicit check for clarity.
    if M < 2:
        raise ValueError("CRPS e2 component requires ensemble size >=2")
    e_2 = np.sum(
        np.abs(da_prediction[..., None] - da_prediction[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_2


def crps_e2(da_prediction, name_prefix: str = "CRPS", ensemble_dim="ensemble"):
    """Compute the CRPS (e2 component) for ensemble predictions."""
    return xr.apply_ufunc(
        _crps_e2,
        da_prediction,
        input_core_dims=[[ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
    )


def _crps_ensemble_fair(da_target: np.ndarray, da_prediction: np.ndarray) -> np.ndarray:
    M: int = da_prediction.shape[-1]
    e_1 = np.sum(np.abs(da_target[..., None] - da_prediction), axis=-1) / max(M, 1)
    if M < 2:
        raise ValueError("Fair CRPS requires ensemble size >=2 (got 1)")
    e_2 = np.sum(
        np.abs(da_prediction[..., None] - da_prediction[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2


def crps_ensemble(da_target, da_prediction, name_prefix: str = "CRPS", ensemble_dim="ensemble"):
    """Compute the fair CRPS for ensemble predictions vs targets."""
    # Ensure ensemble dimension lives in a single chunk to avoid Dask providing
    # singleton blocks to gufunc core (which can trigger M=1 checks inside the
    # function even when global size >1).
    try:
        if hasattr(da_prediction.data, "chunks"):
            # Rechunk only along ensemble dim; keep others unchanged.
            current = da_prediction.data.chunks
            if ensemble_dim in da_prediction.dims:
                axis = da_prediction.dims.index(ensemble_dim)
                if len(current[axis]) > 1:  # multiple chunks along ensemble dim
                    da_prediction = da_prediction.chunk({ensemble_dim: -1})
            # Mirror target chunking for broadcasting safety
            if (
                hasattr(da_target.data, "chunks")
                and ensemble_dim in da_target.dims
                and len(da_target.data.chunks[da_target.dims.index(ensemble_dim)]) > 1
            ):
                da_target = da_target.chunk({ensemble_dim: -1})
    except Exception:
        pass  # Best effort; fall back silently
    res = xr.apply_ufunc(
        _crps_ensemble_fair,
        da_target,
        da_prediction,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float],
    )
    return _add_metric_prefix(res, name_prefix)


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


def _ens_mean_se(da_target, da_prediction):
    return (da_prediction.mean(axis=-1) - da_target) ** 2


def ensemble_mean_se(da_target, da_prediction, name_prefix: str = "EnsembleMeanSquaredError"):
    """Compute the ensemble mean squared error of predictions vs targets."""
    res = xr.apply_ufunc(
        _ens_mean_se,
        da_target,
        da_prediction,
        input_core_dims=[[], ["ensemble"]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float],
    )
    return _add_metric_prefix(res, name_prefix)


def _ens_std(da_prediction):
    return da_prediction.std(axis=-1)


def ensemble_std(da_prediction, name_prefix: str = "EnsembleSTD"):
    """Compute the ensemble standard deviation of predictions."""
    res = xr.apply_ufunc(
        _ens_std,
        da_prediction,
        input_core_dims=[["ensemble"]],
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
            start = np.datetime64(vals.min()).astype("datetime64[h]")
            end = np.datetime64(vals.max()).astype("datetime64[h]")

            def _fmt(x):
                return (
                    np.datetime_as_string(x, unit="h")
                    .replace("-", "")
                    .replace(":", "")
                    .replace("T", "")
                )

            return (_fmt(start), _fmt(end))
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

    from ..helpers import ensemble_mode_to_token

    ens_token = ensemble_mode_to_token("prob")

    for var in variables:
        # Extract and align targets and predictions along shared coordinates
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]
        # Align coordinates so both arrays share the same indexing. Use an
        # outer join so we preserve all prediction lead_time offsets even when
        # targets are missing some valid_time points. Missing entries become
        # NaN and are handled by downstream skipna averaging.
        try:
            da_target, da_prediction = xr.align(da_target, da_prediction, join="outer")
        except Exception:
            # Fallback to by-position if shapes match exactly
            if da_target.shape == da_prediction.shape:
                da_target = da_target.copy()
                da_prediction = da_prediction.copy()
            else:
                raise
        crps_da = crps_ensemble(da_target, da_prediction, ensemble_dim="ensemble")
        crps_mean = float(_reduce_mean_all(crps_da).compute().item())
        crps_rows.append({"variable": var, "CRPS": crps_mean})

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
    try:
        da_t = ds_target[base_var]
        da_p = ds_prediction[base_var]
        da_t, da_p = xr.align(da_t, da_p, join="outer")
    except Exception:
        # Fallback to by-position if shapes match exactly
        da_t = ds_target[base_var]
        da_p = ds_prediction[base_var]
    # CRPS values (keep lead_time for per-lead panels)
    crps = crps_ensemble(da_t, da_p, ensemble_dim="ensemble")
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
    try:
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
    except Exception:
        pass

    # Plot CRPS map (simple original style, no percentile scaling / fallback)
    fig = plt.figure(figsize=(12, 6), dpi=dpi * 2)
    ax = plt.axes(projection=ccrs.PlateCarree())
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
    try:
        units = ds_target[base_var].attrs.get("units", "")
        label = f"CRPS ({units})" if units else "CRPS"
        cbar.set_label(label)
    except Exception:
        cbar.set_label("CRPS")
    ax.set_title(f"CRPS map (mean over time): {base_var}", fontsize=12)

    # Attempt time range extraction for plots
    def _extract_init_range_plot(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            start = np.datetime64(vals.min()).astype("datetime64[h]")
            end = np.datetime64(vals.max()).astype("datetime64[h]")

            def _fmt(x):
                return (
                    np.datetime_as_string(x, unit="h")
                    .replace("-", "")
                    .replace(":", "")
                    .replace("T", "")
                )

            return (_fmt(start), _fmt(end))
        except Exception:
            return None

    def _extract_lead_range_plot(ds: xr.Dataset):
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
        try:
            full_hours = [
                int(np.timedelta64(x) / np.timedelta64(1, "h")) for x in crps["lead_time"].values
            ]
        except Exception:
            full_hours = list(range(int(crps.sizes.get("lead_time", 0))))
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
            try:
                lon_vals = np.asarray(crps_by_lead[lon_name].values)
                if lon_vals.size:
                    lmin, lmax = float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))
                    if (lmax - lmin) > 180 and np.any(lon_vals < 90) and np.any(lon_vals > 270):
                        new = lon_vals.copy()
                        new[new > 180] -= 360
                        order = np.argsort(new)
                        crps_by_lead = crps_by_lead.isel({lon_name: order}).assign_coords(
                            {lon_name: (lon_name, new[order])}
                        )
            except Exception:
                pass
            # Build mapping from all available lead hours -> index
            raw_leads = crps_by_lead["lead_time"].values
            all_hours: list[int] = []
            for x in raw_leads:
                try:
                    # Timedelta-like
                    h = int(np.timedelta64(x) / np.timedelta64(1, "h"))
                except Exception:
                    try:
                        h = int(x)
                    except Exception:
                        h = len(all_hours)
                all_hours.append(h)
            # Debug visibility
            try:
                print(f"[probabilistic-plots] CRPS grid using lead_hours={all_hours}")
            except Exception:
                pass
            hour_index_pairs = []
            for h in full_hours:
                try:
                    idx = all_hours.index(int(h))
                    hour_index_pairs.append((int(h), idx))
                except Exception:
                    continue
            if hour_index_pairs:
                # 2-column CRPS-only layout: fixed at 2 columns (configuration removed per user request).
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
                    try:
                        _lon = crps_by_lead[lon_name].values
                        _lat = crps_by_lead[lat_name].values
                        ax.set_extent(
                            [
                                float(np.min(_lon)),
                                float(np.max(_lon)),
                                float(np.min(_lat)),
                                float(np.max(_lat)),
                            ],
                            crs=ccrs.PlateCarree(),
                        )
                    except Exception:
                        pass
                    ax.set_title(f"CRPS — lead {int(h)}h", fontsize=11)
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
        color="#4C78A8",
        edgecolor="white",
    )
    ax.set_title(f"PIT histogram — {base_var}")
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
    plt.close(fig)

    # PIT per-lead panel plot (multi-row grid) over all retained hours
    if "lead_time" in pit.dims and int(pit.sizes.get("lead_time", 0)) >= 1 and save_fig:
        try:
            full_hours = [
                int(np.timedelta64(x) / np.timedelta64(1, "h")) for x in pit["lead_time"].values
            ]
        except Exception:
            full_hours = list(range(int(pit.sizes.get("lead_time", 0))))
        if full_hours:
            raw_leads = pit["lead_time"].values
            all_hours: list[int] = []
            for x in raw_leads:
                try:
                    h = int(np.timedelta64(x) / np.timedelta64(1, "h"))
                except Exception:
                    try:
                        h = int(x)
                    except Exception:
                        h = len(all_hours)
                all_hours.append(h)
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

    # CRPS line plots across all retained lead_time hours
    if ("lead_time" in ds_prediction.dims) and int(ds_prediction.sizes.get("lead_time", 0)) > 1:
        try:
            panel_hours = [
                int(np.timedelta64(x) / np.timedelta64(1, "h"))
                for x in ds_prediction["lead_time"].values
            ]
        except Exception:
            panel_hours = list(range(int(ds_prediction.sizes.get("lead_time", 0))))
        if panel_hours:
            variables = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
            try:
                all_hours = [
                    int(np.timedelta64(x) / np.timedelta64(1, "h"))
                    for x in ds_prediction["lead_time"].values
                ]
            except Exception:
                all_hours = list(range(int(ds_prediction.sizes.get("lead_time", 0))))
            hour_index_pairs = []
            for h in panel_hours:
                try:
                    hour_index_pairs.append((int(h), all_hours.index(int(h))))
                except Exception:
                    continue
            if hour_index_pairs:
                for var in variables:
                    crps = crps_ensemble(
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
                    try:
                        crps_sel = crps_lt.isel(lead_time=sel_indices)
                        vals = np.asarray(crps_sel.values).ravel()
                    except Exception:
                        continue
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


"""
Expose WeatherBenchX metric classes under this module for convenient imports.
Public API: CRPSEnsemble, SpreadSkillRatio.
"""
CRPSEnsemble = WBXCRPSEnsemble
SpreadSkillRatio = WBXSpreadSkillRatio


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
    """Compute WBX temporal/spatial metrics, CSV summaries, and optional CRPS map.

    Outputs (under out_root/probabilistic):
    - spread_skill_ratio.csv
    - crps_ensemble.csv
    - probabilistic_metrics_temporal.nc
    - probabilistic_metrics_spatial.nc
    - Optional: crps_map_<var>.png if output_mode enables plotting
    """
    # Write WBX artifacts into the same probabilistic folder to avoid split outputs
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    # Imports only when needed to avoid hard dependency during other runs
    from weatherbenchX import aggregation, binning, weighting

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

    # CSV summaries using WBX metrics (SpreadSkillRatio, CRPSEnsemble)
    # Use .sizes (preferred) instead of .dims.get for forward compatibility
    m_ens = int(getattr(ds_pred, "sizes", {}).get("ensemble", 0))
    if m_ens < 2:
        raise RuntimeError(
            "WBX probabilistic metrics require ensemble size >=2 (SpreadSkillRatio/CRPS ensemble). "
            f"Found ensemble size {m_ens}."
        )
    ssr_metric = SpreadSkillRatio(ensemble_dim="ensemble")
    try:
        ssr_df = _wbx_metric_to_df(
            ssr_metric,
            ds_prediction=ds_pred,
            ds_target=ds_targ,
            value_col="SSR",
        )
    except Exception as e:  # pragma: no cover - defensive clarity wrapper
        raise RuntimeError(
            "Failed computing SpreadSkillRatio via WeatherBenchX. "
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
            start = np.datetime64(vals.min()).astype("datetime64[h]")
            end = np.datetime64(vals.max()).astype("datetime64[h]")

            def _fmt(x):
                return (
                    np.datetime_as_string(x, unit="h")
                    .replace("-", "")
                    .replace(":", "")
                    .replace("T", "")
                )

            return (_fmt(start), _fmt(end))
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

    from ..helpers import ensemble_mode_to_token

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

    crps_metric = CRPSEnsemble(ensemble_dim="ensemble")
    try:
        crps_df = _wbx_metric_to_df(
            crps_metric, ds_prediction=ds_pred, ds_target=ds_targ, value_col="CRPS"
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed computing CRPSEnsemble via WeatherBenchX. "
            "Check ensemble size (>=2) and data alignment. Original error: " + str(e)
        ) from e
    crps_csv = section / build_output_filename(
        metric="crps_ensemble",
        variable=None,
        level=None,
        qualifier=None,
        init_time_range=init_range,
        lead_time_range=lead_range,
        ensemble=ens_token_prob,
        ext="csv",
    )
    crps_df.to_csv(crps_csv)
    print(f"[probabilistic] saved {crps_csv}")

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
        weigh_by=[weighting.GridAreaWeighting()],
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
    metrics["CRPS"] = CRPSEnsemble(ensemble_dim="ensemble")
    metrics["SSR"] = SpreadSkillRatio(ensemble_dim="ensemble")

    variables = list(ds_pred.data_vars)
    pred_map = {v: ds_pred[v] for v in variables}
    targ_map = {v: ds_targ[v] for v in variables}
    # Temporal results: reduce spatial dims, keep time dims
    temporal_results = aggregation.compute_metric_values_for_single_chunk(
        metrics, spatial_aggregator, pred_map, targ_map
    )
    # Spatial results: reduce init_time (and optionally bin by season)
    spatial_results = aggregation.compute_metric_values_for_single_chunk(
        metrics, temporal_aggregator, pred_map, targ_map
    )

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
    temporal_results.to_netcdf(temporal_fn, engine="scipy", encoding=enc_t)
    spatial_results.to_netcdf(spatial_fn, engine="scipy", encoding=enc_s)
    print("Wrote:", temporal_fn)
    print("Wrote:", spatial_fn)

    # Optional CRPS map similar to notebook for a selected variable
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    if mode in ("plot", "both"):
        # Choose base variable
        cfg_var = (
            (plotting_cfg or {}).get("map_variable") if isinstance(plotting_cfg, dict) else None
        )
        base_var = cfg_var or variables[0]
        reduce_dims = [d for d in ["init_time", "lead_time", "time"] if d in ds_pred[base_var].dims]
        # Compute CRPS map using a single-chunk aggregator for simplicity
        pred_map = {base_var: ds_pred[base_var]}
        targ_map = {base_var: ds_targ[base_var]}
        from weatherbenchX import aggregation as agg2

        metrics_map = {"CRPS": CRPSEnsemble(ensemble_dim="ensemble")}
        map_ds = agg2.compute_metric_values_for_single_chunk(
            metrics_map,
            agg2.Aggregator(reduce_dims=reduce_dims),
            pred_map,
            targ_map,
        )
        crps_name = f"CRPS.{base_var}"
        if crps_name in map_ds:
            mean_map = map_ds[crps_name]
            lat_name = next(
                (n for n in mean_map.dims if n in ("latitude", "lat", "y")),
                None,
            )
            lon_name = next(
                (n for n in mean_map.dims if n in ("longitude", "lon", "x")),
                None,
            )
            if lat_name and lon_name:
                lat_vals = mean_map[lat_name].values
                if lat_vals[0] > lat_vals[-1]:
                    mean_map = mean_map.sortby(lat_name)
                fig = plt.figure(figsize=(10, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.coastlines()
                mesh = ax.pcolormesh(
                    mean_map[lon_name],
                    mean_map[lat_name],
                    mean_map.values,
                    cmap="viridis",
                    shading="auto",
                )
                plt.colorbar(mesh, ax=ax, orientation="vertical", label=crps_name)
                ax.set_title(f"CRPS map: {base_var}")
                # Avoid clashing with non-WBX CRPS map by using a distinct filename
                out_png = section / build_output_filename(
                    metric="crps_map_wbx",
                    variable=base_var,
                    level=None,
                    qualifier=None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token_prob,
                    ext="png",
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[probabilistic] saved {out_png}")
                plt.close(fig)


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
