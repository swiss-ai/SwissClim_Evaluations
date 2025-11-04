from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from swissclim_evaluations.lead_time_policy import LeadTimePolicy
from swissclim_evaluations.metrics import deterministic as det, ets as ets_mod


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


def test_deterministic_per_lead(tmp_path: Path):
    ds_t, ds_p = _build_pair(3)
    policy = LeadTimePolicy(mode="full")
    # Standardized copies trivial
    det.run(
        ds_t,
        ds_p,
        ds_t,
        ds_p,
        tmp_path,
        {},
        {},
        lead_policy=policy,
    )
    long_csv = tmp_path / "deterministic" / "metrics_by_lead_long.csv"
    assert long_csv.exists(), "Per-lead long CSV not written"
    df = pd.read_csv(long_csv)
    assert {0, 1, 2}.issubset(set(df.lead_time_hours.unique()))


def test_ets_per_lead(tmp_path: Path):
    ds_t, ds_p = _build_pair(4)
    policy = LeadTimePolicy(mode="full")
    ets_mod.run(ds_t, ds_p, tmp_path, {"ets": {"thresholds": [50]}}, lead_policy=policy)
    wide = tmp_path / "ets" / "ets_metrics_by_lead_wide.csv"
    assert wide.exists(), "ETS per-lead wide CSV missing"
