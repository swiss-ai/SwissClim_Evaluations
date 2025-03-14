import itertools

import numpy as np
import xarray as xr

def time_chunks(init_times, lead_times, init_time_chunk_size=None, lead_time_chunk_size=None):
    init_times = init_times.astype('datetime64[ns]')
    init_time_chunks = [
        init_times[i : i + (init_time_chunk_size or len(init_times))]
        for i in range(0, len(init_times), init_time_chunk_size or len(init_times))
    ]

    if isinstance(lead_times, slice):
        lead_time_chunks = [lead_times]
    else:
        lead_times = lead_times.astype('timedelta64[ns]')
        chunk_size = lead_time_chunk_size or len(lead_times)
        lead_time_chunks = [
            lead_times[i : i + chunk_size] for i in range(0, len(lead_times), chunk_size)
        ]

    return itertools.product(init_time_chunks, lead_time_chunks)


def patch_time_dimensions(ds: xr.Dataset) -> xr.Dataset:
    return (ds
    .rename(time="valid_time")
    .expand_dims(lead_time=np.array([6], dtype="timedelta64[h]").astype("timedelta64[ns]"), axis=1)
    .assign_coords(init_time=lambda x: x["valid_time"] - np.timedelta64(6, "h"))
    .swap_dims({"valid_time": "init_time"})
    .assign_coords(valid_time=lambda x: x["valid_time"] - x["lead_time"])
    )