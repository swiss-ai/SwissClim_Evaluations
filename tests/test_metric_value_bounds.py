from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import swissclim_evaluations.metrics.vertical_profiles as vprof_mod
from swissclim_evaluations.metrics.ets import run as run_ets
from swissclim_evaluations.metrics.vertical_profiles import (
    run as run_vertical_profiles,
)
from swissclim_evaluations.plots.energy_spectra import run as run_energy_spectra

# Value range sanity (not strict correctness):
# - LSD >= 0
# - NMAE between 0 and 100
# - ETS in a broad plausible band [-0.5, 1.0]


def _basic_2d():
    time = np.array(
        [
            np.datetime64("2025-01-01T00"),
            np.datetime64("2025-01-01T06"),
        ]
    )
    lat = np.linspace(-10, 10, 5)
    lon = np.linspace(0, 20, 6)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((time.size, lat.size, lon.size))
    ds_t = xr.Dataset(
        {"t2m": (["time", "latitude", "longitude"], data)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds_p = ds_t + 0.1
    return ds_t, ds_p


def _basic_3d():
    init = np.array(
        [
            np.datetime64("2025-01-01T00"),
            np.datetime64("2025-01-01T12"),
        ]
    )
    lead = np.array([np.timedelta64(0, "h")], dtype="timedelta64[h]")
    level = np.array([1000, 850, 500])
    lat = np.linspace(-30, 30, 7)
    lon = np.linspace(0, 10, 8)
    rng = np.random.default_rng(1)
    data = rng.standard_normal(
        (
            init.size,
            lead.size,
            level.size,
            lat.size,
            lon.size,
        )
    )
    ds_t = xr.Dataset(
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
    ds_p = ds_t + 0.05
    return ds_t, ds_p


def test_lsd_non_negative(tmp_path: Path):
    ds_t, ds_p = _basic_2d()
    out = tmp_path / "output"
    run_energy_spectra(
        ds_t,
        ds_p,
        out_root=out,
        plotting_cfg={"output_mode": "npz"},
        select_cfg={},
    )
    # Locate LSD metrics CSV/NPZ presence indirectly by filename pattern; LSD
    # metric arrays are in memory but
    # we rely on absence of exception and plausible derived naming already tested elsewhere.
    # For direct value check, recompute small slice manually via helper:
    # use function from module for LSD.
    from swissclim_evaluations.plots.energy_spectra import (
        calculate_energy_spectra,
        calculate_log_spectral_distance,
    )

    spec_t = calculate_energy_spectra(ds_t["t2m"])
    spec_p = calculate_energy_spectra(ds_p["t2m"])
    spec_t, spec_p = xr.align(spec_t, spec_p, join="inner")
    lsd = calculate_log_spectral_distance(spec_t.values, spec_p.values)
    assert lsd >= 0


def test_vertical_profiles_nmae_bounds(tmp_path: Path, monkeypatch):
    ds_t, ds_p = _basic_3d()
    out = tmp_path / "output"
    # Reduce to 2 bands so plotting expects a (2,1) layout; patch subplots likewise.
    monkeypatch.setattr(vprof_mod, "_lat_bands", lambda: ([-90, 0, 90], 2))

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

        arr = _np.empty((ncols, half), dtype=object)
        for i in range(ncols):
            for j in range(half):
                arr[i, j] = _AX()
        return object(), arr

    monkeypatch.setattr(plt, "subplots", _stub_subplots)
    run_vertical_profiles(
        ds_t,
        ds_p,
        out_root=out,
        plotting_cfg={"output_mode": "npz"},
        select_cfg={},
    )
    npz = next(
        (out / "vertical_profiles").glob("vertical_profiles_nmae_temperature_multi_combined_*.npz")
    )
    data = np.load(npz)
    # Arrays may include zeros for NaN fills; ensure within [0,100]
    for key in ("nmae_neg", "nmae_pos"):
        arr = data[key]
        assert np.nanmin(arr) >= 0 - 1e-6
        assert np.nanmax(arr) <= 100 + 1e-6


def test_ets_plausible_range(tmp_path: Path):
    # Minimal binary event fields: create deterministic threshold exceedances
    lat = np.linspace(0, 2, 3)
    lon = np.linspace(0, 2, 3)
    time = np.array([np.datetime64("2025-01-01T00")])
    obs_vals = np.array([[[0, 1, 0], [1, 1, 0], [0, 0, 1]]], dtype=float)
    pred_vals = obs_vals.copy()
    ds_t = xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], obs_vals)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds_p = xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], pred_vals)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    out = tmp_path / "output"
    # The run_ets signature expects metrics_cfg with nested 'ets' config (mirroring cli usage)
    run_ets(ds_t, ds_p, out_root=out, metrics_cfg={"ets": {"thresholds": [50]}})
    ets_dir = out / "ets"
    csv = next(ets_dir.glob("ets_metrics_*.csv"))
    import pandas as pd

    df = pd.read_csv(csv)
    # Column name pattern 'ETS <threshold>%'
    col = next(c for c in df.columns if c.lower().startswith("ets"))
    series = df[col].dropna()
    # If all NaN (no events), treat as trivially satisfied; else enforce bounds
    if not series.empty:
        assert (series >= -0.5).all() and (series <= 1.0).all()
