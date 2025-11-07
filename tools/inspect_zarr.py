import sys

import numpy as np
import xarray as xr

p = sys.argv[1]
try:
    ds = xr.open_zarr(p)
except Exception:
    ds = xr.open_zarr(p, consolidated=False)
print("dims:", dict(ds.dims))
print("coords:", list(ds.coords))
print("data_vars:", list(ds.data_vars))
# lead_time hours
if "lead_time" in ds.coords:
    raw = ds["lead_time"].values
    try:
        hours = (raw // np.timedelta64(1, "h")).astype(int).tolist()
    except Exception:
        hours = []
    print("lead_time count:", len(hours), "sample:", hours[:10])
# ensemble size
if "ensemble" in ds.dims:
    print("ensemble size:", int(ds.dims["ensemble"]))
