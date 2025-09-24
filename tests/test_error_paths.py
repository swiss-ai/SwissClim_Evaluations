from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import swissclim_evaluations.metrics.vertical_profiles as vprof_mod
from swissclim_evaluations.metrics.vertical_profiles import (
    run as run_vertical_profiles,
)
from swissclim_evaluations.plots.energy_spectra import run as run_energy_spectra


def _ds_no_longitude():
    time = np.array([np.datetime64("2025-01-01T00")])
    lat = np.linspace(-10, 10, 3)
    # Intentionally omit longitude -> expect failure
    data = np.random.default_rng(0).standard_normal((time.size, lat.size))
    return xr.Dataset(
        {"t2m": (["time", "latitude"], data)},
        coords={"time": time, "latitude": lat},
    )


def _ds_vertical_profiles_empty_band():
    # Provide minimal latitude coverage so many 10° bands become empty. Vertical profiles module
    # defines 10-degree bins; using only a narrow slice triggers empty selections gracefully.
    init = np.array([np.datetime64("2025-01-01T00")])
    lead = np.array([np.timedelta64(0, "h")], dtype="timedelta64[h]")
    level = np.array([1000, 850, 500])
    lat = np.linspace(-2, 2, 5)  # narrow subset
    lon = np.linspace(0, 3, 4)
    rng = np.random.default_rng(1)
    data = rng.standard_normal((
        init.size,
        lead.size,
        level.size,
        lat.size,
        lon.size,
    ))
    tgt = xr.Dataset(
        {
            "temperature": (
                ["init_time", "lead_time", "level", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "init_time": init,
            "lead_time": lead.astype("timedelta64[ns]"),
            "level": level,
            "latitude": lat,
            "longitude": lon,
        },
    )
    pred = tgt.copy()
    return tgt, pred


def test_energy_spectra_missing_longitude(tmp_path: Path):
    ds = _ds_no_longitude()
    # Wrap in minimal dataset arguments expected by run (needs prediction dataset too)
    with pytest.raises(ValueError):
        run_energy_spectra(
            ds, ds, out_root=tmp_path / "output", plotting_cfg={}, select_cfg={}
        )


def test_vertical_profiles_empty_band_outputs(tmp_path: Path, monkeypatch):
    tgt, pred = _ds_vertical_profiles_empty_band()
    out = tmp_path / "output"
    # Reduce bands drastically to 2 (half=1) so expectation is a (2,1) axes grid.
    monkeypatch.setattr(vprof_mod, "_lat_bands", lambda: ([-90, 0, 90], 2))

    # Provide a local subplots stub returning a 2x1 ndarray even for half=1.
    class _AX:
        def plot(self, *a, **k):
            return None

        def set_title(self, *a, **k):
            return None

        def set_xlabel(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def invert_yaxis(self):
            return None

        def set_xlim(self, *a, **k):
            return None

    def _stub_subplots(ncols, half, *a, **k):
        import numpy as _np

        axes = _np.empty((ncols, half), dtype=object)
        for i in range(ncols):
            for j in range(half):
                axes[i, j] = _AX()
        return object(), axes

    monkeypatch.setattr(plt, "subplots", _stub_subplots)
    # Should still run and produce a NPZ even if some bands are empty (no exception expected)
    run_vertical_profiles(
        tgt,
        pred,
        out_root=out,
        plotting_cfg={"output_mode": "npz"},
        select_cfg={},
    )
    vp_dir = out / "vertical_profiles"
    assert vp_dir.exists()
    # There should be at least one NPZ (may be more if multiple variables); allow any matching prefix
    assert any(
        f.name.startswith("vprof_nmae_") and f.suffix == ".npz"
        for f in vp_dir.iterdir()
    ), "Expected vprof_nmae*.npz even with empty bands"
