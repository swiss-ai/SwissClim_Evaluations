import xarray as xr

def modify_ds(ds: xr.Dataset, path: str):
    ''' User-defined customization to the dataset based on the full path to the zarr file '''

    print("inside custom", path)
    print(ds)
    if path == "/capstor/store/cscs/swissai/a122/hydrological_data/IFS_ensmean_2020_totalprecipitation6hr.zarr":
        ds = ds.sel(prediction_timedelta = ds.prediction_timedelta.values[1:]) # the first step is 0 hour and full of NaN

    if path == "/capstor/store/cscs/swissai/a122/hydrological_data/FuXi_2020_totalprecipitation6hr.zarr":
        ds['total_precipitation_6hr'] = ds['total_precipitation_6hr']*1e-3

    return ds
    