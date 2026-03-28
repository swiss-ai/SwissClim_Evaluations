"""Tests that diverging colorbars (wind, etc.) are centred at 0.

No real data files are needed — synthetic xarray datasets are constructed
with deliberately asymmetric values so that a naïve min/max range would
NOT be centred at 0.  After the fix, the pcolormesh vmin/vmax must satisfy
vmin == -vmax.

The conftest autouse fixture _fast_plots replaces plt.subplots with
_DummyAxis objects, so we intercept _DummyAxis.pcolormesh to capture calls.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tests.conftest import _DummyAxis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wind_ds(lo: float = -1.0, hi: float = 8.0) -> xr.Dataset:
    """Tiny (1 time × 2 lat × 2 lon) wind dataset with values in [lo, hi]."""
    data = np.array([[[lo, 0.0], [hi / 2, hi]]])  # shape (1, 2, 2)
    return xr.Dataset(
        {
            "10m_u_component_of_wind": (
                ["time", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "time": [np.datetime64("2021-01-01T00")],
            "latitude": [45.0, 46.0],
            "longitude": [7.0, 8.0],
        },
    )


def _make_temperature_ds() -> xr.Dataset:
    """Non-diverging variable: values entirely positive [270, 300]."""
    data = np.array([[[270.0, 280.0], [290.0, 300.0]]])
    return xr.Dataset(
        {
            "2m_temperature": (
                ["time", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "time": [np.datetime64("2021-01-01T00")],
            "latitude": [45.0, 46.0],
            "longitude": [7.0, 8.0],
        },
    )


def _capture_pcolormesh(monkeypatch):
    """Return a list that will be filled with each pcolormesh kwargs dict."""
    captured: list[dict] = []
    _orig = _DummyAxis.pcolormesh

    def _intercepting(self, *args, **kwargs):
        captured.append(kwargs.copy())
        return _orig(self, *args, **kwargs)

    monkeypatch.setattr(_DummyAxis, "pcolormesh", _intercepting)
    return captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_diverging_colorbar_symmetric_for_wind_field(tmp_path, monkeypatch):
    """pcolormesh vmin/vmax must be symmetric around 0 for wind (RdBu_r) fields.

    Data range is [-1, 8] — without the fix, vmin=-1 and vmax=8 (not symmetric).
    With the fix, both should be ±8.
    """
    from swissclim_evaluations.plots import maps as maps_module

    ds = _make_wind_ds(lo=-1.0, hi=8.0)
    captured = _capture_pcolormesh(monkeypatch)

    maps_module.run(
        ds_target=ds,
        ds_prediction=ds,
        out_root=tmp_path,
        plotting_cfg={"output_mode": "plot", "dpi": 24},
    )

    assert captured, "pcolormesh was never called"
    for kwargs in captured:
        vmin, vmax = kwargs.get("vmin"), kwargs.get("vmax")
        assert vmin is not None and vmax is not None
        assert vmin == pytest.approx(-vmax), (
            f"Colorbar not centred at 0: vmin={vmin}, vmax={vmax}"
        )


def test_non_diverging_colorbar_not_forced_symmetric(tmp_path, monkeypatch):
    """Non-diverging variables (e.g. temperature) must NOT be forced symmetric."""
    from swissclim_evaluations.plots import maps as maps_module

    ds = _make_temperature_ds()
    captured = _capture_pcolormesh(monkeypatch)

    maps_module.run(
        ds_target=ds,
        ds_prediction=ds,
        out_root=tmp_path,
        plotting_cfg={"output_mode": "plot", "dpi": 24},
    )

    assert captured, "pcolormesh was never called"
    for kwargs in captured:
        vmin = kwargs.get("vmin")
        # Temperature range is [270, 300] — vmin must remain positive
        assert vmin is not None and vmin > 0, (
            f"Temperature colorbar was wrongly forced negative: vmin={vmin}"
        )


def test_diverging_colorbar_symmetric_when_mostly_negative(tmp_path, monkeypatch):
    """Symmetry works both ways: if data is mostly negative, extend positive side.

    Data spans [-9, 2]: without fix vmin=-9, vmax=2; with fix should be ±9.
    """
    from swissclim_evaluations.plots import maps as maps_module

    ds = _make_wind_ds(lo=-9.0, hi=2.0)
    captured = _capture_pcolormesh(monkeypatch)

    maps_module.run(
        ds_target=ds,
        ds_prediction=ds,
        out_root=tmp_path,
        plotting_cfg={"output_mode": "plot", "dpi": 24},
    )

    assert captured, "pcolormesh was never called"
    for kwargs in captured:
        vmin, vmax = kwargs.get("vmin"), kwargs.get("vmax")
        assert vmin is not None and vmax is not None
        assert vmin == pytest.approx(-vmax), (
            f"Colorbar not centred at 0: vmin={vmin}, vmax={vmax}"
        )
