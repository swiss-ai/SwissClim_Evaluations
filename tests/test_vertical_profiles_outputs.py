from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.metrics.vertical_profiles import (
    run as run_vertical_profiles,
)


def _make_3d_dataset():
    init_time = np.array([
        np.datetime64("2023-01-01T00"),
        np.datetime64("2023-01-01T12"),
    ])
    level = np.array([1000, 850, 500])
    # Cover broad latitude extent so band slicing finds data
    lat = np.linspace(-90, 90, 19)
    lon = np.linspace(0, 6, 6)
    shape = (init_time.size, level.size, lat.size, lon.size)
    rng = np.random.default_rng(42)
    data = rng.standard_normal(shape)
    tgt = xr.Dataset(
        {
            "temperature": (
                ["init_time", "level", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "init_time": init_time,
            "level": level,
            "latitude": lat,
            "longitude": lon,
        },
    )
    # Prediction with slight noise
    pred = tgt + 0.1 * rng.standard_normal(shape)
    return tgt, pred


def test_vertical_profiles_outputs(tmp_path: Path):
    ds_target, ds_prediction = _make_3d_dataset()
    out_root = tmp_path / "out"
    run_vertical_profiles(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        out_root=out_root,
        plotting_cfg={"output_mode": "npz"},  # save NPZ but skip PNG for speed
        select_cfg={},
    )
    vp_dir = out_root / "vertical_profiles"
    assert vp_dir.exists()
    # NPZ combined file should follow standardized filename builder (vprof_nmae_<var>...)
    assert any(
        f.name.startswith("vprof_nmae_temperature_multi_combined_")
        for f in vp_dir.glob("vprof_nmae_*temperature*_combined_*.npz")
    )
