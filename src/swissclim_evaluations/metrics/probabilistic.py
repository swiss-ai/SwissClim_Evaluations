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

from ..helpers import (
    COLOR_DIAGNOSTIC,
    build_output_filename,
    time_chunks,
)


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

    # Extract time ranges for common naming
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
    prob_cfg = (cfg_all or {}).get("probabilistic", {})
    report_per_level = bool(prob_cfg.get("report_per_level", True))
    crps_rows_per_level: list[dict[str, Any]] = []

    for var in variables:
        # Extract and align targets and predictions along shared coordinates
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]
        try:
            da_target, da_prediction = xr.align(da_target, da_prediction, join="exact")
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

    if crps_rows:
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

    # CRPS map (reduce over time-like dims, keep lat/lon)
    crps = crps_ensemble(ds_target[base_var], ds_prediction[base_var], ensemble_dim="ensemble")
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

    # Plot CRPS map
    fig = plt.figure(figsize=(10, 6), dpi=dpi * 2)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    mesh = ax.pcolormesh(
        crps_map[lon_name],
        crps_map[lat_name],
        crps_map.values,
        cmap="viridis",
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label(f"CRPS — {base_var}")
    ax.set_title(f"CRPS map (mean over time): {base_var}")

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

    # PIT histogram (global)
    pit = probability_integral_transform(
        ds_target[base_var],
        ds_prediction[base_var],
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
    ax.set_title(f"PIT histogram — {base_var}")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.axhline(1.0, color="brown", linestyle="--", linewidth=1, label="Uniform")
    ax.legend()

    if save_fig:
        out_png = (
            section / f"pit_hist_{base_var}.png"
        )  # legacy non-tokenized image filename retained
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"[probabilistic-plots] saved {out_png}")
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
