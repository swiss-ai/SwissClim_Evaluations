from __future__ import annotations

import re

import numpy as np
import xarray as xr

from swissclim_evaluations.helpers import time_range_suffix


def test_time_range_suffix_init_and_lead():
    init_time = np.array(
        [
            np.datetime64("2025-01-01T00"),
            np.datetime64("2025-01-02T06"),
        ]
    )
    lead_time = np.array(
        [
            np.timedelta64(0, "h"),
            np.timedelta64(36, "h"),
        ]
    )
    ds = xr.Dataset(
        {"foo": ("lead_time", [1, 2])},
        coords={"init_time": init_time, "lead_time": lead_time},
    )
    suffix = time_range_suffix(ds)
    # Expect both segments joined by '__' with lead_time using numeric offsets
    # truncated to hour precision
    pattern = r"init_time_\d{4}-\d{2}-\d{2}T\d{2}_to_\d{4}-\d{2}-\d{2}T\d{2}__lead_time_0_to_36"
    assert re.fullmatch(pattern, suffix), suffix


def test_time_range_suffix_only_lead_time():
    lead_time = np.array(
        [
            np.timedelta64(0, "h"),
            np.timedelta64(6, "h"),
        ]
    )
    ds = xr.Dataset({"foo": ("lead_time", [1, 2])}, coords={"lead_time": lead_time})
    suffix = time_range_suffix(ds)
    assert suffix.startswith("lead_time_") and "__" not in suffix
