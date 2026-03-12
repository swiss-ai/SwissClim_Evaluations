from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.lead_time_policy import LeadTimePolicy
from swissclim_evaluations.metrics import ets as ets_mod


def _build_pair(n_leads=3):
    times = np.array(["2023-01-01T00"], dtype="datetime64[h]").astype("datetime64[ns]")
    leads = np.arange(n_leads).astype("timedelta64[h]").astype("timedelta64[ns]")
    arr = xr.DataArray(
        np.random.rand(1, n_leads, 2, 2),
        dims=("init_time", "lead_time", "latitude", "longitude"),
        coords={
            "init_time": times,
            "lead_time": leads,
            "latitude": [0, 1],
            "longitude": [0, 1],
        },
    )
    ds_target = xr.Dataset({"var": arr})
    ds_pred = xr.Dataset({"var": arr * 1.01})
    return ds_target, ds_pred


def test_ets_per_lead(tmp_path: Path):
    ds_t, ds_p = _build_pair(4)
    policy = LeadTimePolicy(mode="full")
    ets_mod.run(ds_t, ds_p, tmp_path, {"ets": {"thresholds": [50]}}, lead_policy=policy)
    wide = tmp_path / "ets" / "ets_metrics_by_lead_wide.csv"
    assert wide.exists(), "ETS per-lead wide CSV missing"
