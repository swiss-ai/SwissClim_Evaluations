from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.metrics.ets import run as run_ets


def _make_simple_dataset():
    time = np.array([
        np.datetime64("2023-05-01T00"),
        np.datetime64("2023-05-01T06"),
        np.datetime64("2023-05-01T12"),
    ])
    lat = np.linspace(40, 42, 3)
    lon = np.linspace(7, 9, 4)
    shape = (time.size, lat.size, lon.size)
    rng = np.random.default_rng(7)
    tgt = xr.Dataset(
        {
            "2m_temperature": (
                ["time", "latitude", "longitude"],
                rng.standard_normal(shape),
            )
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    pred = tgt + 0.2 * rng.standard_normal(shape)
    return tgt, pred


def test_ets_outputs(tmp_path: Path):
    ds_target, ds_prediction = _make_simple_dataset()
    out_root = tmp_path / "out"
    run_ets(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        out_root=out_root,
        metrics_cfg={"ets": {"thresholds": [50]}},
    )
    ets_dir = out_root / "ets"
    assert ets_dir.exists()
    # New standardized naming pattern via build_output_filename
    # New schema: no 'multi', 'none', or 'full' placeholders.
    # Expect patterns like: ets_metrics_ensnone.csv OR ets_metrics_averaged_init...._ensnone.csv when init range present.
    assert any(
        f.name.startswith("ets_metrics_ensnone.csv")
        or (
            f.name.startswith("ets_metrics_averaged_init")
            and f.name.endswith("_ensnone.csv")
        )
        for f in ets_dir.glob("ets_metrics*.csv")
    )
