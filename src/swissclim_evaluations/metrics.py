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


def _ens_mean_se(obs, fct):
    return (fct.mean(axis=-1) - obs)**2

def ensemble_mean_se(obs, fct, name_prefix: str = "EnsembleMeanSquaredError"):
    """Compute the ensemble mean root mean squared error."""
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
    return _add_metric_prefix(res, name_prefix)


def _add_metric_prefix(ds: xr.Dataset, prefix: str):
    return ds.rename({var: f"{prefix}.{var}" for var in ds.data_vars})

