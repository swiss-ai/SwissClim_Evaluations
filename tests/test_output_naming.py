from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.helpers import (
    build_output_filename,
    time_range_suffix,
)


def _make_dataset(init_count=2, lead_hours=(0, 36)):
    init_time = np.array([
        np.datetime64("2025-01-01T00"),
        np.datetime64("2025-01-02T06"),
    ])[:init_count]
    lead_time = np.array([
        np.timedelta64(h, "h")
        for h in range(lead_hours[0], lead_hours[1] + 1, 6)
    ])
    ds = xr.Dataset(
        {"dummy": ("lead_time", np.arange(lead_time.size))},
        coords={"init_time": init_time, "lead_time": lead_time},
    )
    return ds


def test_time_range_suffix_both_dims():
    ds = _make_dataset()
    suffix = time_range_suffix(ds)
    pattern = r"init_time_\d{10}_to_\d{10}__lead_time_0_to_36"
    assert re.fullmatch(pattern, suffix), suffix


def test_time_range_suffix_lead_only():
    # Build dataset containing only lead_time (no init_time) to exercise single segment path
    lead_time = np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
    ds = xr.Dataset(
        {"foo": ("lead_time", [1, 2])}, coords={"lead_time": lead_time}
    )
    suffix = time_range_suffix(ds)
    assert suffix.startswith("lead_time_") and "__" not in suffix


def test_probabilistic_first_lead_integration(tmp_path: Path):
    # Lightweight reproduction of the first-lead selection: we simulate trimming lead_time
    ds = _make_dataset()
    trimmed = ds.isel(lead_time=0)
    # Build a filename using build_output_filename to ensure lead-only and init present logic integrates
    init_range = ("2025010100", "2025010206")
    lead_range = ("000h", "000h")
    fn = build_output_filename(
        metric="crps_summary",
        variable=None,
        init_time_range=init_range,
        lead_time_range=lead_range,
        ensemble=None,
    )
    assert (
        fn == "crps_summary_init2025010100-2025010206_lead000h-000h_ensnone.csv"
    )
    assert trimmed.lead_time.size == 1


def test_build_output_filename_matrix():
    # Parametric quick sweep over combinations
    cases = [
        dict(metric="metrics", ensemble=None, exp="metrics_ensnone.csv"),
        dict(
            metric="map",
            variable="temperature",
            level=500,
            qualifier="averaged",
            init_time_range=("2023010100", "2023010200"),
            lead_time_range=("000h", "024h"),
            ensemble=0,
            ext="png",
            exp="map_temperature_500_averaged_init2023010100-2023010200_lead000h-024h_ens0.png",
        ),
        dict(
            metric="crps_summary",
            variable=["a", "b"],
            ensemble="mean",
            exp="crps_summary_ensmean.csv",
        ),
        dict(
            metric="hist",
            variable="u10",
            ensemble=3,
            ext="npz",
            exp="hist_u10_ens3.npz",
        ),
        dict(
            metric="ets_metrics",
            lead_time_range=("000h", "048h"),
            ensemble=None,
            exp="ets_metrics_lead000h-048h_ensnone.csv",
        ),
        dict(
            metric="wd_kde_wasserstein",
            init_time_range=("2023010100", "2023010300"),
            ensemble=None,
            exp="wd_kde_wasserstein_init2023010100-2023010300_ensnone.csv",
        ),
    ]
    for c in cases:
        exp = c.pop("exp")
        got = build_output_filename(**c)
        assert got == exp
