import numpy as np
import pytest
import xarray as xr

from swissclim_evaluations.lead_time_policy import (
    apply_lead_time_selection,
    parse_lead_time_policy,
)


def _ds_with_hours(hours):
    lt = np.asarray(hours, dtype="timedelta64[h]").astype("timedelta64[ns]")
    return xr.Dataset(
        {"var": ("lead_time", np.arange(len(hours), dtype=float))},
        coords={"lead_time": lt},
    )


def test_stride_selection_fails_when_no_multiple():
    # hours: 3, 7, 11 (no multiples of 6)
    ds = _ds_with_hours([3, 7, 11])
    p = parse_lead_time_policy({"mode": "stride", "stride_hours": 6})
    with pytest.raises(ValueError):
        _ = apply_lead_time_selection(ds, p)


def test_panel_specific_fails_when_missing_hours():
    """Legacy: previously panel-specific selection raised when requested hour absent.

    Panel concept is deprecated; select_panel_hours now returns available hours unchanged.
    This test updated to assert graceful fallback instead of raising.
    """
    p = parse_lead_time_policy(
        {
            "mode": "full",
            "panel": {"strategy": "specific", "hours": [0, 9]},
        }
    )
    out = p.select_panel_hours([0, 6, 12])
    # Expect the full available list (no filtering or error)
    assert out == [0, 6, 12]
