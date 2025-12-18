from __future__ import annotations

import os
from pathlib import Path

from swissclim_evaluations.plots.wd_kde import run as run_wd_kde


def test_wd_kde_ridgeline_evolve_output(tmp_path: Path):
    # Run real wd_kde (disable fast stubs)
    os.environ["SC_PLOTS_FULL"] = "1"
    import numpy as np
    import xarray as xr

    # Build minimal standardized datasets with a lead_time dimension
    init_time = np.array([np.datetime64("2025-01-01T00")])
    lead_time = np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h"), np.timedelta64(12, "h")])
    lat = np.linspace(-10, 10, 3)
    lon = np.linspace(0, 20, 4)
    rng = np.random.default_rng(0)
    targ = xr.Dataset(
        {
            "10m_u_component_of_wind": (
                ["init_time", "lead_time", "latitude", "longitude"],
                rng.standard_normal((init_time.size, lead_time.size, lat.size, lon.size)),
            )
        },
        coords={"init_time": init_time, "lead_time": lead_time, "latitude": lat, "longitude": lon},
    )
    pred = xr.Dataset(
        {
            "10m_u_component_of_wind": (
                ["init_time", "lead_time", "latitude", "longitude"],
                rng.standard_normal((init_time.size, lead_time.size, lat.size, lon.size)) + 0.1,
            )
        },
        coords={"init_time": init_time, "lead_time": lead_time, "latitude": lat, "longitude": lon},
    )
    # Standardized
    tgt_std = (targ - targ.mean()) / targ.std()
    pred_std = (pred - targ.mean()) / targ.std()

    out_root = tmp_path / "output"
    run_wd_kde(
        ds_target=targ,
        ds_prediction=pred,
        ds_target_std=tgt_std,
        ds_prediction_std=pred_std,
        out_root=out_root,
        plotting_cfg={
            "output_mode": "plot",
            "kde_max_samples": 100,
            "wd_kde_include_3d": False,
            "wd_kde_global_evolution": True,
            "wd_kde_per_lat_band": True,
        },
    )

    wd_dir = out_root / "wd_kde"
    assert wd_dir.exists()
