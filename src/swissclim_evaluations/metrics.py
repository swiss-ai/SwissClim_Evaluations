import numpy as np
import xarray as xr



def _crps_ensemble_fair(obs, fct):
    M: int = fct.shape[-1]
    e_1 = np.sum(np.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = np.sum(
        np.abs(fct[..., None] - fct[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2

def crps_ensemble(obs, fct, name_prefix: str = "CRPS"):
    """Compute the CRPS for ensemble forecasts."""
    # Compute the CRPS for each ensemble member
    res = xr.apply_ufunc(
        _crps_ensemble_fair,
        obs,
        fct,
        input_core_dims=[[], ["ensemble"]],
        output_core_dims=[[]],
        dask="parallelized",
    )
    return _add_metric_prefix(res, name_prefix)


def _pit(obs, fct):
    return np.mean(fct < obs[...,None], axis=-1)

def probability_integral_transform(obs, fct, name_prefix: str = "PIT"):
    """Compute the probability integral transform."""
    res = xr.apply_ufunc(
        _pit,
        obs,
        fct,
        input_core_dims=[[], ["ensemble"]],
        output_core_dims=[[]],
        dask="parallelized",
    )
    return _add_metric_prefix(res, name_prefix)


def _ssr(obs, fct):
    ensemble_mean = fct.mean(axis=-1)
    spread = np.var(fct, axis=-1)
    rmse = (ensemble_mean - obs)**2
    ssr = np.sqrt(spread / rmse)
    return ssr

def spread_skill_ratio(obs, fct, name_prefix: str = "SSR"):
    """Compute the spread skill ratio."""
    res = xr.apply_ufunc(
        _ssr,
        obs,
        fct,
        input_core_dims=[[], ["ensemble"]],
        output_core_dims=[[]],
        dask="parallelized",
    )
    return _add_metric_prefix(res, name_prefix)


def _add_metric_prefix(ds: xr.Dataset, prefix: str):
    return ds.rename({var: f"{prefix}.{var}" for var in ds.data_vars})

