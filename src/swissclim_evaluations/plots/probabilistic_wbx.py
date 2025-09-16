from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Mapping

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..metrics.probabilistic import CRPSEnsemble, SpreadSkillRatio


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
                enc[name] = {"units": "seconds since 1970-01-01", "dtype": "i4"}
            elif kind == "m":  # timedelta64
                enc[name] = {"units": "seconds", "dtype": "i4"}
    return enc


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    all_cfg: dict[str, Any],
) -> None:
    """Compute WBX temporal/spatial metrics and optional CRPS map using WeatherBenchX.

    Outputs (under out_root/probabilistic_wbx):
    - spread_skill_ratio.csv
    - crps_ensemble.csv
    - probabilistic_metrics_temporal.nc
    - probabilistic_metrics_spatial.nc
    - Optional: crps_map_<var>.png if output_mode enables plotting
    """
    section = out_root / "probabilistic_wbx"
    section.mkdir(parents=True, exist_ok=True)

    # Imports only when needed to avoid hard dependency during other runs
    from weatherbenchX import aggregation, binning, time_chunks, weighting

    if "ensemble" not in ds_prediction.dims:
        print(
            "[probabilistic_wbx-plots] Skipping: predictions lack 'ensemble'."
        )
        return

    common_vars = [
        v for v in ds_prediction.data_vars if v in ds_target.data_vars
    ]
    if not common_vars:
        print(
            "[probabilistic_wbx-plots] No overlapping variables; nothing to do."
        )
        return
    ds_pred = ds_prediction[common_vars]
    ds_targ = ds_target[common_vars]

    # CSV summaries using WBX metrics (SpreadSkillRatio, CRPSEnsemble)
    def _wbx_metric_to_df(
        metric, predictions: xr.Dataset, targets: xr.Dataset, value_col: str
    ):
        import pandas as pd

        variables = [v for v in predictions.data_vars if v in targets.data_vars]
        pred_map: Mapping[Hashable, xr.DataArray] = {
            v: predictions[v] for v in variables
        }
        targ_map: Mapping[Hashable, xr.DataArray] = {
            v: targets[v] for v in variables
        }
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
        values_map = metric.values_from_mean_statistics(mean_stats)
        rows = []
        for var, da in values_map.items():
            rows.append({"variable": str(var), value_col: float(da.values)})
        df = pd.DataFrame(rows).set_index("variable").sort_index()
        return df

    ssr_metric = SpreadSkillRatio(ensemble_dim="ensemble")
    ssr_df = _wbx_metric_to_df(ssr_metric, ds_pred, ds_targ, value_col="SSR")
    ssr_csv = section / "spread_skill_ratio.csv"
    ssr_df.to_csv(ssr_csv)
    print(f"[probabilistic_wbx-plots] saved {ssr_csv}")

    crps_metric = CRPSEnsemble(ensemble_dim="ensemble")
    crps_df = _wbx_metric_to_df(crps_metric, ds_pred, ds_targ, value_col="CRPS")
    crps_csv = section / "crps_ensemble.csv"
    crps_df.to_csv(crps_csv)
    print(f"[probabilistic_wbx-plots] saved {crps_csv}")

    # Chunk iteration and aggregations
    prob_cfg = (
        (all_cfg or {}).get("probabilistic", {})
        if isinstance(all_cfg, dict)
        else {}
    )
    init_chunk_size = int(prob_cfg.get("init_time_chunk_size", 20))
    lead_chunk_size = int(prob_cfg.get("lead_time_chunk_size", 1))
    lead_times_override = prob_cfg.get("lead_times_ns")

    init_times = ds_pred["init_time"].values
    lead_times = ds_pred["lead_time"].values
    if (
        getattr(lead_times, "dtype", None) is not None
        and str(lead_times.dtype) != "timedelta64[ns]"
    ):
        try:
            lead_times = lead_times.astype("timedelta64[ns]")
        except Exception:
            pass
    if lead_times_override is not None:
        lead_times = np.array(lead_times_override, dtype="timedelta64[ns]")

    times = time_chunks.TimeChunks(
        init_times,
        lead_times,
        init_time_chunk_size=init_chunk_size,
        lead_time_chunk_size=lead_chunk_size,
    )

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
    spatial_results = None
    temporal_results_list = []
    chunks_count = 0

    for init_chunk, lead_chunk in times:
        pred_chunk = ds_pred.sel(init_time=init_chunk, lead_time=lead_chunk)
        targ_chunk = ds_targ.sel(init_time=init_chunk, lead_time=lead_chunk)
        pred_map = {v: pred_chunk[v] for v in variables}
        targ_map = {v: targ_chunk[v] for v in variables}

        # Temporal results: spatial aggregation, keep time axes
        temporal_results_list.append(
            aggregation.compute_metric_values_for_single_chunk(
                metrics, spatial_aggregator, pred_map, targ_map
            )
        )

        # Spatial results: temporal aggregation (reduce init_time), average across chunks
        chunk_spatial = aggregation.compute_metric_values_for_single_chunk(
            metrics, temporal_aggregator, pred_map, targ_map
        )
        spatial_results = (
            chunk_spatial
            if spatial_results is None
            else (spatial_results + chunk_spatial)
        )
        chunks_count += 1

    temporal_results = (
        xr.merge(temporal_results_list)
        if temporal_results_list
        else xr.Dataset()
    )
    if spatial_results is None or chunks_count == 0:
        spatial_results = xr.Dataset()
    else:
        spatial_results = spatial_results / float(chunks_count)

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
                out_png = section / f"crps_map_{base_var}.png"
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[probabilistic_wbx-plots] saved {out_png}")
                plt.close(fig)
