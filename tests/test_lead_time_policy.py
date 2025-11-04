import numpy as np
import xarray as xr

from swissclim_evaluations.lead_time_policy import (
    apply_lead_time_selection,
    parse_lead_time_policy,
)


def _dummy_ds(n_leads=5):
    leads = np.arange(0, n_leads * 6, 6)  # every 6 hours
    ds = xr.Dataset(
        {"var": ("lead_time", np.arange(n_leads, dtype=float))},
        coords={"lead_time": leads.astype("timedelta64[h]").astype("timedelta64[ns]")},
    )
    return ds


def test_policy_first():
    p = parse_lead_time_policy({"mode": "first"})
    ds = _dummy_ds(4)
    sel = apply_lead_time_selection(ds, p)
    assert sel.lead_time.size == 1


def test_policy_subset():
    p = parse_lead_time_policy({"mode": "subset", "subset_hours": [0, 12]})
    ds = _dummy_ds(5)  # 0,6,12,18,24
    sel = apply_lead_time_selection(ds, p)
    hours = (sel.lead_time.values // np.timedelta64(1, "h")).astype(int).tolist()
    assert hours == [0, 12]


def test_policy_stride():
    p = parse_lead_time_policy({"mode": "stride", "stride_hours": 12})
    ds = _dummy_ds(6)  # 0..30 step6
    sel = apply_lead_time_selection(ds, p)
    hours = (sel.lead_time.values // np.timedelta64(1, "h")).astype(int).tolist()
    assert hours == [0, 12, 24]
