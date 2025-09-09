from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# WeatherBench-X base classes used by WBX metric wrappers
from weatherbenchX.metrics.base import PerVariableMetric, PerVariableStatistic
from weatherbenchX.metrics.deterministic import SquaredError
from weatherbenchX.metrics.wrappers import EnsembleMean, WrappedStatistic

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


def _add_metric_prefix(ds: xr.Dataset, prefix: str):
    return ds.rename({var: f"{prefix}.{var}" for var in ds.data_vars})


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
            yield (
                ds.sel(valid_time=ds_ml.sel(**idx).valid_time).load(),
                ds_ml.sel(**idx).load(),
            )
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
    - Optional: {var}_pit.nc and {var}_crps.nc when plotting.save_plot_data is True
    """
    section_output = out_root / "probabilistic"
    section_output.mkdir(parents=True, exist_ok=True)
    save_plot_data = bool(cfg_plot.get("save_plot_data", False))

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
            np.savez(
                section_output / f"{var}_pit_hist.npz",
                counts=counts,
                edges=edges,
            )
            if save_plot_data:
                pit_da.to_netcdf(section_output / f"{var}_pit.nc")
                crps_da.to_netcdf(section_output / f"{var}_crps.nc")

    if crps_rows:
        pd.DataFrame(crps_rows).groupby("variable").mean().to_csv(
            section_output / "crps_summary.csv"
        )


# --- WBX-compatible metric classes (moved from wbx.py) ---


class ProbabilityIntegralTransform(PerVariableMetric):
    """Compute the PIT for ensemble forecasts."""

    def __init__(self, ensemble_dim: str = "ensemble"):
        self.ensemble_dim = ensemble_dim

    def _compute_per_variable(self, predictions, targets):
        return probability_integral_transform(
            targets,
            predictions,
            ensemble_dim=self.ensemble_dim,
            name_prefix=None,
        )


class EnsembleVariance(PerVariableStatistic):
    """Compute the ensemble variance."""

    def __init__(self, ensemble_dim: str = "ensemble"):
        self.ensemble_dim = ensemble_dim

    def _compute_per_variable(self, predictions, targets):
        return xr.apply_ufunc(
            lambda x: np.var(x, axis=-1),
            predictions,
            input_core_dims=[[self.ensemble_dim]],
            output_core_dims=[[]],
            dask="parallelized",
        )


class SpreadSkillRatio(PerVariableMetric):
    """Computes the (biased) spread-skill ratio."""

    def __init__(self, ensemble_dim: str = "ensemble"):
        self.ensemble_dim = ensemble_dim

    @property
    def statistics(self):
        return {
            "EnsembleVariance": EnsembleVariance(
                ensemble_dim=self.ensemble_dim
            ),
            "EnsembleMeanSquaredError": WrappedStatistic(
                SquaredError(),
                EnsembleMean(
                    which="predictions",
                    ensemble_dim=self.ensemble_dim,
                ),
            ),
        }

    def _values_from_mean_statistics_per_variable(
        self, statistic_values
    ) -> xr.DataArray:
        """Computes metrics from aggregated statistics."""
        return np.sqrt(
            statistic_values["EnsembleVariance"]
            / statistic_values["EnsembleMeanSquaredError"]
        ).compute(scheduler="threads", num_workers=8)


class CRPSAccuracyTerm(PerVariableStatistic):
    """Compute the CRPS accuracy term E|y - f|."""

    def __init__(self, ensemble_dim: str = "ensemble"):
        self.ensemble_dim = ensemble_dim

    def _compute_per_variable(self, predictions, targets):
        return crps_e1(targets, predictions, ensemble_dim=self.ensemble_dim)


class CRPSSpreadTerm(PerVariableStatistic):
    """Compute the CRPS spread term E|f - f"|."""

    def __init__(self, ensemble_dim: str = "ensemble"):
        self.ensemble_dim = ensemble_dim

    def _compute_per_variable(self, predictions, targets):
        return crps_e2(predictions, ensemble_dim=self.ensemble_dim)


class CRPSEnsemble(PerVariableMetric):
    """Compute the CRPS for ensemble forecasts."""

    def __init__(self, ensemble_dim: str = "ensemble"):
        self.ensemble_dim = ensemble_dim

    @property
    def statistics(self):
        return {
            "CRPSAccuracyTerm": CRPSAccuracyTerm(self.ensemble_dim),
            "CRPSSpreadTerm": CRPSSpreadTerm(self.ensemble_dim),
        }

    def _values_from_mean_statistics_per_variable(self, statistic_values):
        return (
            statistic_values["CRPSAccuracyTerm"]
            - 0.5 * statistic_values["CRPSSpreadTerm"]
        ).compute(scheduler="threads", num_workers=8)


def run_probabilistic_wbx(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    all_cfg: dict[str, Any],
) -> None:
    """Compute WBX spread–skill ratio and CRPS ensemble and save CSV summaries."""

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
    try:
        ssr_values = ssr_metric(obs=obs, predictions=fct)
    except TypeError:
        ssr_values = ssr_metric.compute(predictions=fct, targets=obs)  # type: ignore[attr-defined]
    ssr_df = _per_variable_mean_df(ssr_values)
    ssr_df.to_csv(section_output / "spread_skill_ratio.csv")

    crps_metric = CRPSEnsemble(ensemble_dim="ensemble")
    try:
        crps_values = crps_metric(obs=obs, predictions=fct)
    except TypeError:
        crps_values = crps_metric.compute(predictions=fct, targets=obs)  # type: ignore[attr-defined]
    crps_df = _per_variable_mean_df(crps_values)
    crps_df.to_csv(section_output / "crps_ensemble.csv")


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
