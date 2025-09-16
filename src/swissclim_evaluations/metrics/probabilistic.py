from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Mapping

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


def _crps_e1(obs, fct):
    M: int = fct.shape[-1]
    e_1 = np.sum(np.abs(obs[..., None] - fct), axis=-1) / M
    return e_1


def crps_e1(obs, fct, name_prefix: str = "CRPS", ensemble_dim="ensemble"):
    """Compute the CRPS for ensemble forecasts."""
    return xr.apply_ufunc(
        _crps_e1,
        obs,
        fct,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
    )


def _crps_e2(fct):
    M: int = fct.shape[-1]
    e_2 = np.sum(
        np.abs(fct[..., None] - fct[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_2


def crps_e2(fct, name_prefix: str = "CRPS", ensemble_dim="ensemble"):
    """Compute the CRPS for ensemble forecasts."""
    return xr.apply_ufunc(
        _crps_e2,
        fct,
        input_core_dims=[[ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
    )


def _crps_ensemble_fair(obs, fct):
    M: int = fct.shape[-1]
    e_1 = np.sum(np.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = np.sum(
        np.abs(fct[..., None] - fct[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2


def crps_ensemble(obs, fct, name_prefix: str = "CRPS", ensemble_dim="ensemble"):
    """Compute the CRPS for ensemble forecasts."""
    res = xr.apply_ufunc(
        _crps_ensemble_fair,
        obs,
        fct,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
    )
    return _add_metric_prefix(res, name_prefix)


def _pit(obs, fct):
    return np.mean(fct < obs[..., None], axis=-1)


def probability_integral_transform(
    obs, fct, ensemble_dim="ensemble", name_prefix: str = "PIT"
):
    """Compute the probability integral transform."""
    res = xr.apply_ufunc(
        _pit,
        obs,
        fct,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
    )
    return _add_metric_prefix(res, name_prefix) if name_prefix else res


def _ens_mean_se(obs, fct):
    return (fct.mean(axis=-1) - obs) ** 2


def ensemble_mean_se(obs, fct, name_prefix: str = "EnsembleMeanSquaredError"):
    """Compute the ensemble mean squared error."""
    res = xr.apply_ufunc(
        _ens_mean_se,
        obs,
        fct,
        input_core_dims=[[], ["ensemble"]],
        output_core_dims=[[]],
        dask="parallelized",
    )
    return _add_metric_prefix(res, name_prefix)


def _ens_std(fct):
    return fct.std(axis=-1)


def ensemble_std(fct, name_prefix: str = "EnsembleSTD"):
    """Compute the ensemble standard deviation."""
    res = xr.apply_ufunc(
        _ens_std,
        fct,
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


def _pit_histogram_np(
    da: xr.DataArray, bins: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(da.values).ravel()
    counts, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0), density=True)
    return counts, edges


def _iter_time_chunks(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    init_chunk: int | None = None,
    lead_chunk: int | None = None,
):
    if all(dim in ds_ml.dims for dim in ("init_time", "lead_time")):
        for init_chunk_vals, lead_chunk_vals in time_chunks(
            ds_ml["init_time"].values,
            ds_ml["lead_time"].values,
            init_chunk,
            lead_chunk,
        ):
            idx = {"init_time": init_chunk_vals, "lead_time": lead_chunk_vals}
            # Assumes CLI aligned ds and ds_ml by init_time/lead_time intersection already.
            yield (ds.sel(**idx).load(), ds_ml.sel(**idx).load())
    elif "time" in ds_ml.dims:
        yield ds, ds_ml
    else:
        yield ds, ds_ml


def run_probabilistic(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    out_root: Path,
    cfg_plot: dict[str, Any],
    cfg_all: dict[str, Any],
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

    if "ensemble" not in ds_ml.dims:
        print(
            "[probabilistic] Skipping: model dataset has no 'ensemble' dimension."
        )
        return

    variables = [v for v in ds_ml.data_vars if v in ds.data_vars]
    if not variables:
        print(
            "[probabilistic] No overlapping variables between obs and forecast; nothing to do."
        )
        return

    prob_cfg = (cfg_all or {}).get("probabilistic", {})
    init_chunk = prob_cfg.get("init_time_chunk_size")
    lead_chunk = prob_cfg.get("lead_time_chunk_size")

    crps_rows: list[dict[str, Any]] = []

    for ds_obs_chunk, ds_fct_chunk in _iter_time_chunks(
        ds, ds_ml, init_chunk=init_chunk, lead_chunk=lead_chunk
    ):
        for var in variables:
            obs = ds_obs_chunk[var]
            fct = ds_fct_chunk[var]
            crps_da = crps_ensemble(obs, fct, ensemble_dim="ensemble")
            crps_mean = float(_reduce_mean_all(crps_da))
            crps_rows.append({"variable": var, "CRPS": crps_mean})

            pit_da = probability_integral_transform(
                obs, fct, ensemble_dim="ensemble", name_prefix=None
            )
            counts, edges = _pit_histogram_np(pit_da, bins=50)
            pit_npz = section_output / f"{var}_pit_hist.npz"
            np.savez(
                pit_npz,
                counts=counts,
                edges=edges,
            )
            print(f"[probabilistic] saved {pit_npz}")
            # Always save PIT and CRPS fields for reproducibility
            pit_nc = section_output / f"{var}_pit.nc"
            crps_nc = section_output / f"{var}_crps.nc"
            pit_da.to_netcdf(pit_nc)
            crps_da.to_netcdf(crps_nc)
            print(f"[probabilistic] saved {pit_nc}")
            print(f"[probabilistic] saved {crps_nc}")

    if crps_rows:
        df = pd.DataFrame(crps_rows).groupby("variable").mean()
        out_csv = section_output / "crps_summary.csv"
        df.to_csv(out_csv)
        print("CRPS summary (per variable):")
        print(df.head())
        print(f"[probabilistic] saved {out_csv}")


# --- Thin re-exports for compatibility ---
# Keep names imported by notebooks while delegating to WeatherBenchX
CRPSEnsemble = WBXCRPSEnsemble
SpreadSkillRatio = WBXSpreadSkillRatio


def _wbx_metric_to_df(
    metric: Any,
    predictions: xr.Dataset,
    targets: xr.Dataset,
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
    variables = [v for v in predictions.data_vars if v in targets.data_vars]
    pred_map: Mapping[Hashable, xr.DataArray] = {
        v: predictions[v] for v in variables
    }
    targ_map: Mapping[Hashable, xr.DataArray] = {
        v: targets[v] for v in variables
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
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    all_cfg: dict[str, Any],
) -> None:
    """Compute WBX Spread–Skill Ratio and CRPS (ensemble) and save CSV summaries.

    Uses the official WeatherBenchX metric implementations and their API.
    """

    section_output = out_root / "probabilistic_wbx"
    section_output.mkdir(parents=True, exist_ok=True)

    if "ensemble" not in ds_ml.dims:
        print(
            "[probabilistic_wbx] Skipping: model dataset has no 'ensemble' dimension."
        )
        return

    common_vars = [v for v in ds_ml.data_vars if v in ds.data_vars]
    if not common_vars:
        print(
            "[probabilistic_wbx] No overlapping variables between obs and forecast; nothing to do."
        )
        return
    fct = ds_ml[common_vars]
    obs = ds[common_vars]

    ssr_metric = SpreadSkillRatio(ensemble_dim="ensemble")
    ssr_df = _wbx_metric_to_df(
        ssr_metric, predictions=fct, targets=obs, value_col="SSR"
    )
    ssr_csv = section_output / "spread_skill_ratio.csv"
    ssr_df.to_csv(ssr_csv)
    print(f"[probabilistic_wbx] saved {ssr_csv}")

    crps_metric = CRPSEnsemble(ensemble_dim="ensemble")
    crps_df = _wbx_metric_to_df(
        crps_metric, predictions=fct, targets=obs, value_col="CRPS"
    )
    crps_csv = section_output / "crps_ensemble.csv"
    crps_df.to_csv(crps_csv)
    print(f"[probabilistic_wbx] saved {crps_csv}")


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
