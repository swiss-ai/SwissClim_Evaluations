import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(os.path.abspath("src"))
from swissclim_evaluations.helpers import extract_date_from_dataset


def create_dummy_ds(var_name, init_times):
    lats = np.linspace(45, 48, 10)
    lons = np.linspace(6, 10, 10)

    dims = ["init_time", "latitude", "longitude"]
    coords = {"init_time": pd.to_datetime(init_times), "latitude": lats, "longitude": lons}
    shape = (len(init_times), len(lats), len(lons))

    data = np.random.rand(*shape)

    ds = xr.Dataset(
        data_vars={var_name: (dims, data, {"units": "K"})},
        coords=coords,
    )
    return ds


ds = create_dummy_ds("temp", ["2023-01-01T12:00:00"])
print(f"Original DS coords: {ds.coords}")

# Simulate what maps.py does
ds_var = ds["temp"]
if "init_time" in ds_var.dims:
    ds_var = ds_var.isel(init_time=0)

print(f"Sliced DS coords: {ds_var.coords}")
print(f"Init time size: {ds_var.coords['init_time'].size}")
print(f"Init time values: {ds_var.coords['init_time'].values}")
print(f"Type of values: {type(ds_var.coords['init_time'].values)}")

try:
    its = ds_var.coords["init_time"]
    if its.size == 1:
        val = its.values.item() if hasattr(its.values, "item") else its.values[0]
        print(f"Val: {val}, Type: {type(val)}")
        ts = np.datetime64(val).astype("datetime64[h]")
        print(f"TS: {ts}")
        res = f" ({np.datetime_as_string(ts, unit='h').replace(':', '')})"
        print(f"Formatted: {res}")
except Exception as e:
    print(f"Exception: {e}")

result = extract_date_from_dataset(ds_var)
print(f"Result: '{result}'")
