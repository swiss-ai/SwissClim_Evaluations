from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Mapping

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as dsa
import matplotlib.pyplot as plt

# plotting dependencies will be used in plot_probabilistic() and WBX map (optional)
import numpy as np
import pandas as pd
import xarray as xr
from weatherbenchX.metrics.probabilistic import (
    CRPSEnsemble as WBXCRPSEnsemble,
)

# Use official WeatherBenchX metrics instead of local copies
from weatherbenchX.metrics.probabilistic import (
    SpreadSkillRatio as WBXSpreadSkillRatio,
)

from ..helpers import time_chunks
from ..lead_time_policy import LeadTimePolicy


def _crps_e1(da_target, da_prediction):
    M: int = da_prediction.shape[-1]
    e_1 = np.sum(np.abs(da_target[..., None] - da_prediction), axis=-1) / M
    return e_1


def crps_e1(
    da_target, da_prediction, name_prefix: str = "CRPS", ensemble_dim="ensemble"
):
    """Compute the CRPS (e1 component) for ensemble predictions vs targets."""
    return xr.apply_ufunc(
        _crps_e1,
        da_target,
        da_prediction,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
    )


def _crps_e2(da_prediction):
    M: int = da_prediction.shape[-1]
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


def _crps_ensemble_fair(da_target, da_prediction):
    M: int = da_prediction.shape[-1]
    e_1 = np.sum(np.abs(da_target[..., None] - da_prediction), axis=-1) / M
    e_2 = np.sum(
        np.abs(da_prediction[..., None] - da_prediction[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2


def crps_ensemble(
    da_target, da_prediction, name_prefix: str = "CRPS", ensemble_dim="ensemble"
):
    """Compute the fair CRPS for ensemble predictions vs targets."""
    res = xr.apply_ufunc(
        _crps_ensemble_fair,
        da_target,
        da_prediction,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
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
    )
    return _add_metric_prefix(res, name_prefix) if name_prefix else res


def _ens_mean_se(da_target, da_prediction):
    return (da_prediction.mean(axis=-1) - da_target) ** 2


def ensemble_mean_se(
    da_target, da_prediction, name_prefix: str = "EnsembleMeanSquaredError"
):
    """Compute the ensemble mean squared error of predictions vs targets."""
    res = xr.apply_ufunc(
        _ens_mean_se,
        da_target,
        da_prediction,
        input_core_dims=[[], ["ensemble"]],
        output_core_dims=[[]],
        dask="parallelized",
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
    )
    return _add_metric_prefix(res, name_prefix) if name_prefix else res


def _add_metric_prefix(da_or_ds: xr.Dataset | xr.DataArray, prefix: str):
    # Accept both Dataset and DataArray; for DataArray, rename the variable name if present
    if isinstance(da_or_ds, xr.DataArray):
        name = da_or_ds.name or "value"
        return da_or_ds.rename(f"{prefix}.{name}")
    else:
        return da_or_ds.rename({
            var: f"{prefix}.{var}" for var in da_or_ds.data_vars
        })


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
    counts = (
        dsa.histogram(darr, bins=np.asarray(edges))[0]
        .compute()
        .astype(np.float64)
    )
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
            # Assumes CLI aligned targets and predictions by init_time/lead_time intersection already.
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
    lead_policy: LeadTimePolicy | None = None,
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
        print(
            "[probabilistic] Skipping: model dataset has no 'ensemble' dimension."
        )
        return

    variables = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not variables:
        print(
            "[probabilistic] No overlapping variables between targets and predictions; nothing to do."
        )
        return

    crps_rows: list[dict[str, Any]] = []

    per_lead = (
        lead_policy is not None
        and "lead_time" in ds_prediction.dims
        and int(ds_prediction.lead_time.size) > 1
        and lead_policy.mode != "first"
    )

    if not per_lead:
        for var in variables:
            da_target = ds_target[var]
            da_prediction = ds_prediction[var]
            try:
                da_target, da_prediction = xr.align(
                    da_target, da_prediction, join="exact"
                )
            except Exception:
                if da_target.shape == da_prediction.shape:
                    da_target = da_target.copy()
                    da_prediction = da_prediction.copy()
                else:
                    raise
            crps_da = crps_ensemble(
                da_target, da_prediction, ensemble_dim="ensemble"
            )
            crps_mean = float(_reduce_mean_all(crps_da).compute().item())
            crps_rows.append({"variable": var, "CRPS": crps_mean})

            pit_da = probability_integral_transform(
                da_target,
                da_prediction,
                ensemble_dim="ensemble",
                name_prefix=None,
            )
            counts, edges = _pit_histogram_dask(pit_da, bins=50, density=True)
            pit_npz = section_output / f"{var}_pit_hist.npz"
            np.savez(
                pit_npz,
                counts=counts,
                edges=edges,
            )
            print(f"[probabilistic] saved {pit_npz}")
            pit_nc = section_output / f"{var}_pit.nc"
            crps_nc = section_output / f"{var}_crps.nc"
            pit_da.to_netcdf(pit_nc)
            crps_da.to_netcdf(crps_nc)
            print(f"[probabilistic] saved {pit_nc}")
            print(f"[probabilistic] saved {crps_nc}")
    else:
        lead_hours = (
            ds_prediction["lead_time"].values // np.timedelta64(1, "h")
        ).astype(int)
        crps_lead_rows: list[dict[str, Any]] = []
        for li, h in enumerate(lead_hours):
            for var in variables:
                da_target = ds_target[var].isel(lead_time=li)
                da_prediction = ds_prediction[var].isel(lead_time=li)
                try:
                    da_target, da_prediction = xr.align(
                        da_target, da_prediction, join="exact"
                    )
                except Exception:
                    if da_target.shape == da_prediction.shape:
                        da_target = da_target.copy()
                        da_prediction = da_prediction.copy()
                    else:
                        raise
                crps_da = crps_ensemble(
                    da_target, da_prediction, ensemble_dim="ensemble"
                )
                crps_mean = float(_reduce_mean_all(crps_da).compute().item())
                crps_lead_rows.append({
                    "variable": var,
                    "lead_time_hours": int(h),
                    "CRPS": crps_mean,
                })
                if lead_policy.store_full_fields:
                    crps_nc = section_output / f"{var}_crps_lead{int(h)}.nc"
                    crps_da.to_netcdf(crps_nc)
                # Only compute PIT histogram for panel-selected leads to limit cost
                # Select panel hours once
        if crps_lead_rows:
            crps_df = pd.DataFrame(crps_lead_rows)
            crps_df.to_csv(section_output / "crps_by_lead.csv", index=False)
            # Aggregate mean across leads for single-lead summary (backward compatibility)
            summary = crps_df.groupby("variable").mean(numeric_only=True)[
                "CRPS"
            ]
            for var, val in summary.items():
                crps_rows.append({"variable": var, "CRPS": float(val)})

    if crps_rows:
        df = pd.DataFrame(crps_rows).groupby("variable").mean()
        out_csv = section_output / "crps_summary.csv"
        df.to_csv(out_csv)
        print("CRPS summary (per variable):")
        print(df.head())
        print(f"[probabilistic] saved {out_csv}")


def _select_base_variable_for_plot(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    plotting_cfg: dict[str, Any],
) -> str:
    cfg_var = (
        (plotting_cfg or {}).get("map_variable")
        if isinstance(plotting_cfg, dict)
        else None
    )
    if cfg_var and isinstance(cfg_var, str):
        if cfg_var.startswith("CRPS."):
            return cfg_var.split(".", 1)[1]
        return cfg_var
    common = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    if not common:
        raise ValueError(
            "No common variables between targets and predictions for probabilistic plots."
        )
    return common[0]


def _time_reduce_dims_for_plot(da: xr.DataArray) -> list[str]:
    return [
        d
        for d in ["time", "init_time", "lead_time", "ensemble"]
        if d in da.dims
    ]


def plot_probabilistic(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
) -> None:
    """Generate probabilistic plots (CRPS map and PIT histogram) and save to disk.

    Saves under out_root/probabilistic as PNGs and optionally NPZ with data if output_mode is 'npz' or 'both'.
    """
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int((plotting_cfg or {}).get("dpi", 48))
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)

    base_var = _select_base_variable_for_plot(
        ds_target, ds_prediction, plotting_cfg
    )

    # CRPS map (reduce over time-like dims, keep lat/lon)
    crps = crps_ensemble(
        ds_target[base_var], ds_prediction[base_var], ensemble_dim="ensemble"
    )
    reduce_dims = _time_reduce_dims_for_plot(crps)
    crps_map = crps.mean(dim=reduce_dims, skipna=True) if reduce_dims else crps

    # Detect lat/lon and sort latitude ascending for pcolormesh compatibility
    lat_name = next(
        (n for n in crps_map.dims if n in ("latitude", "lat", "y")), None
    )
    lon_name = next(
        (n for n in crps_map.dims if n in ("longitude", "lon", "x")), None
    )
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot find lat/lon dims in CRPS map dims: {crps_map.dims}"
        )
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
    cbar = plt.colorbar(
        mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8
    )
    cbar.set_label(f"CRPS — {base_var}")
    ax.set_title(f"CRPS map (mean over time): {base_var}")

    if save_fig:
        out_png = section / f"crps_map_{base_var}.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"[probabilistic-plots] saved {out_png}")
    if save_npz:
        out_npz = section / f"crps_map_{base_var}.npz"
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
        name_prefix=None,
    )
    counts, edges = _pit_histogram_dask(pit, bins=20, density=True)

    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi * 2)
    # Draw histogram from counts/edges to avoid materializing all values
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

    if save_fig:
        out_png = section / f"pit_hist_{base_var}.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"[probabilistic-plots] saved {out_png}")
    if save_npz:
        out_npz = section / f"pit_hist_{base_var}.npz"
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
    pred_map: Mapping[Hashable, xr.DataArray] = {
        v: ds_prediction[v] for v in variables
    }
    targ_map: Mapping[Hashable, xr.DataArray] = {
        v: ds_target[v] for v in variables
    }

    # Compute and average statistics per variable
    mean_stats: dict[str, dict[Hashable, xr.DataArray]] = {}
    dims_all = [
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
        print(
            "[probabilistic] Skipping: model dataset has no 'ensemble' dimension."
        )
        return

    common_vars = [
        v for v in ds_prediction.data_vars if v in ds_target.data_vars
    ]
    if not common_vars:
        print(
            "[probabilistic] No overlapping variables between targets and predictions; nothing to do."
        )
        return
    ds_pred = ds_prediction[common_vars]
    ds_targ = ds_target[common_vars]

    # CSV summaries using WBX metrics (SpreadSkillRatio, CRPSEnsemble)
    ssr_metric = SpreadSkillRatio(ensemble_dim="ensemble")
    ssr_df = _wbx_metric_to_df(
        ssr_metric, ds_prediction=ds_pred, ds_target=ds_targ, value_col="SSR"
    )
    ssr_csv = section / "spread_skill_ratio.csv"
    ssr_df.to_csv(ssr_csv)
    print(f"[probabilistic] saved {ssr_csv}")

    crps_metric = CRPSEnsemble(ensemble_dim="ensemble")
    crps_df = _wbx_metric_to_df(
        crps_metric, ds_prediction=ds_pred, ds_target=ds_targ, value_col="CRPS"
    )
    crps_csv = section / "crps_ensemble.csv"
    crps_df.to_csv(crps_csv)
    print(f"[probabilistic] saved {crps_csv}")

    def _default_regions() -> dict[
        str, tuple[tuple[float, float], tuple[float, float]]
    ]:
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

    regions_cfg = (
        (plotting_cfg or {}).get("regions")
        if isinstance(plotting_cfg, dict)
        else None
    )
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
    temporal_bin_by = (
        [binning.ByTimeUnit("season", "init_time")] if seasonal else None
    )
    temporal_aggregator = aggregation.Aggregator(
        reduce_dims=["init_time"],
        bin_by=temporal_bin_by,
    )

    metrics = {
        "CRPS": CRPSEnsemble(ensemble_dim="ensemble"),
        "SSR": SpreadSkillRatio(ensemble_dim="ensemble"),
    }

    variables = [v for v in ds_pred.data_vars]
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
    temporal_fn = section / "probabilistic_metrics_temporal.nc"
    spatial_fn = section / "probabilistic_metrics_spatial.nc"
    temporal_results.to_netcdf(temporal_fn, engine="scipy", encoding=enc_t)
    spatial_results.to_netcdf(spatial_fn, engine="scipy", encoding=enc_s)
    print("Wrote:", temporal_fn)
    print("Wrote:", spatial_fn)

    # Optional CRPS map similar to notebook for a selected variable
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    if mode in ("plot", "both"):
        # Choose base variable
        cfg_var = (
            (plotting_cfg or {}).get("map_variable")
            if isinstance(plotting_cfg, dict)
            else None
        )
        base_var = cfg_var or variables[0]
        reduce_dims = [
            d
            for d in ["init_time", "lead_time", "time"]
            if d in ds_pred[base_var].dims
        ]
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
                plt.colorbar(
                    mesh, ax=ax, orientation="vertical", label=crps_name
                )
                ax.set_title(f"CRPS map: {base_var}")
                # Avoid clashing with non-WBX CRPS map by using a distinct filename
                out_png = section / f"crps_map_wbx_{base_var}.png"
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[probabilistic] saved {out_png}")
                plt.close(fig)


def _per_variable_mean_df(da_or_ds: xr.Dataset | xr.DataArray) -> pd.DataFrame:
    if isinstance(da_or_ds, xr.DataArray):
        ds = da_or_ds.to_dataset(name="value")
    else:
        ds = da_or_ds
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
