from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.plots.maps import run as run_maps


def _make_ensemble():
    time = np.array([np.datetime64("2025-01-01T00")])
    lat = np.linspace(-5, 5, 3)
    lon = np.linspace(0, 10, 4)
    ens = np.arange(2)
    rng = np.random.default_rng(0)
    base = rng.standard_normal((time.size, lat.size, lon.size))
    target = xr.Dataset(
        {"t2m": (["time", "latitude", "longitude"], base)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    pred = xr.Dataset(
        {
            "t2m": (
                ["time", "latitude", "longitude", "ensemble"],
                base[..., None]
                + 0.1
                * rng.standard_normal(
                    (
                        time.size,
                        lat.size,
                        lon.size,
                        ens.size,
                    )
                ),
            )
        },
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon,
            "ensemble": ens,
        },
    )
    return target, pred


def test_maps_ensemble_filenames(tmp_path: Path):
    tgt, pred = _make_ensemble()
    out = tmp_path / "output"
    run_maps(tgt, pred, out_root=out, plotting_cfg={"output_mode": "npz"})
    maps_dir = out / "maps"
    files = list(maps_dir.glob("map_t2m_*.npz"))
    # Expect one file per ensemble member plus ensnone when generation logic
    # repeats (depending on code path produce just ens indices)
    # Safest: ensure at least one ensemble-specific file present.
    assert any("_ens0" in f.name or f.name.endswith("_ens0.npz") for f in files)
    assert any("_ens1" in f.name or f.name.endswith("_ens1.npz") for f in files)
