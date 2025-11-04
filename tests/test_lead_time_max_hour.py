import numpy as np
import xarray as xr

from swissclim_evaluations.lead_time_policy import (
    LeadTimePolicy,
    apply_lead_time_selection,
)


def _make_ds(hours):
    leads = np.array(hours).astype("timedelta64[h]").astype("timedelta64[ns]")
    data = xr.DataArray(
        np.zeros((1, len(leads))),
        dims=("init_time", "lead_time"),
        coords={
            "init_time": np.array(["2023-01-01T00"], dtype="datetime64[h]").astype(
                "datetime64[ns]"
            ),
            "lead_time": leads,
        },
    )
    return xr.Dataset({"var": data})


def test_max_hour_stride():
    ds = _make_ds([0, 6, 12, 18, 24, 30, 36])
    policy = LeadTimePolicy(mode="stride", stride_hours=12, max_hour=24)
    out = apply_lead_time_selection(ds, policy)
    kept = (out.lead_time.values // np.timedelta64(1, "h")).astype(int).tolist()
    assert kept == [0, 12, 24]


def test_max_hour_subset():
    ds = _make_ds([0, 6, 12, 18, 24, 30])
    policy = LeadTimePolicy(mode="subset", subset_hours=[0, 6, 12, 36], max_hour=18)
    out = apply_lead_time_selection(ds, policy)
    kept = (out.lead_time.values // np.timedelta64(1, "h")).astype(int).tolist()
    # 18 is not in subset_hours; 36 filtered out by max_hour → expect only 0,6,12
    assert kept == [0, 6, 12]


def test_max_hour_full():
    ds = _make_ds([0, 6, 12, 18, 24, 30])
    policy = LeadTimePolicy(mode="full", max_hour=18)
    out = apply_lead_time_selection(ds, policy)
    kept = (out.lead_time.values // np.timedelta64(1, "h")).astype(int).tolist()
    assert kept == [0, 6, 12, 18]
