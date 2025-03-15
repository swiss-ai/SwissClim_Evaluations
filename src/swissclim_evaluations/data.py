import numpy as np
import xarray as xr


LEVELS = [500, 700, 850]
LATITUDES = [90.0, -89.75]
LONGITUDES = [0.0, 359.75]
VARIABLES_2D = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]
VARIABLES_3D = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]


def _esmf_patch_time(ds: xr.Dataset) -> xr.Dataset:
    return (ds
    .rename(time="valid_time")
    .expand_dims(lead_time=np.array([6], dtype="timedelta64[h]").astype("timedelta64[ns]"), axis=1)
    .assign_coords(init_time=lambda x: x["valid_time"] - np.timedelta64(6, "h"))
    .swap_dims({"valid_time": "init_time"})
    .assign_coords(valid_time=lambda x: x["init_time"] + x["lead_time"])
    )

def esmf_deterministic(path=None, variables=None) -> xr.Dataset:
    PATH = "/iopsstor/scratch/cscs/sadamov/pyprojects_data/swissai/preds_20250219/aurora.zarr"
    ds = xr.open_zarr(path or PATH)[variables or VARIABLES_2D + VARIABLES_3D]
    # ds = ds.sel(latitude=slice(*LATITUDES), longitude=slice(*LONGITUDES), level=LEVELS)
    return _esmf_patch_time(ds)

def esmf_ensemble(path=None, variables=None) -> xr.Dataset:
    PATH = "/capstor/store/cscs/swissai/a01/ESFM_Results/preds_20250219/aurora_tail.zarr"
    # PATH = "/iopsstor/scratch/cscs/sadamov/pyprojects_data/swissai/preds_20250219/aurora_tail.zarr"
    ds = xr.open_zarr(path or PATH)[variables or VARIABLES_2D + VARIABLES_3D]
    indexers = {}
    if "level" in ds.dims:
        indexers["level"] = LEVELS
    ds = ds.sel(indexers)
    return _esmf_patch_time(ds)

def ifs(path=None, variables=None) -> xr.Dataset:
    PATH = "/capstor/store/cscs/swissai/a01/IFSensemble-2020-1440x721.zarr"
    ds = xr.open_zarr(path or PATH, decode_timedelta=True)[variables or VARIABLES_2D + VARIABLES_3D]
    indexers = {}
    if "level" in ds.dims:
        indexers["level"] = LEVELS
    ds = ds.sel(indexers)
    ds = ds.isel(time=list(set(range(734)) - set([106, 300, 456, 511]))) # remove 4 duplicates from the time indices
    ds = ds.rename({"time": "init_time", "prediction_timedelta": "lead_time", "number": "ensemble"})
    ds = ds.assign_coords(valid_time=ds.init_time + ds.lead_time)
    return ds

def era5(path=None, variables=None) -> xr.Dataset:
    PATH = "/capstor/store/cscs/ERA5/weatherbench2_original"
    ds = xr.open_zarr(path or PATH, decode_timedelta=True)[variables or VARIABLES_2D + VARIABLES_3D]
    indexers = {}
    if "level" in ds.dims:
        indexers["level"] = LEVELS
    ds = ds.sel(indexers)
    return ds

def land_sea_mask(path=None, variables=None) -> xr.Dataset:
    PATH = "/capstor/store/cscs/ERA5/weatherbench2_original"
    da = xr.open_zarr(path or PATH, decode_timedelta=True)["land_sea_mask"]
    da = da.sel(latitude=slice(*LATITUDES), longitude=slice(*LONGITUDES))
    return da