from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.plots.histograms import run as run_histograms
from swissclim_evaluations.plots.maps import run as run_maps
from swissclim_evaluations.plots.wd_kde import run as run_wd_kde


def _make_basic_ds(include_level: bool = False):
    time = np.array(
        [
            np.datetime64("2025-01-01T00"),
            np.datetime64("2025-01-01T06"),
        ]
    )
    lat = np.linspace(-20, 20, 5)
    lon = np.linspace(0, 30, 6)
    rng = np.random.default_rng(0)
    data_2d = rng.standard_normal((time.size, lat.size, lon.size))
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (
                ["time", "latitude", "longitude"],
                data_2d,
            )
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    if include_level:
        level = np.array([1000, 850])
        data_3d = rng.standard_normal(
            (
                time.size,
                level.size,
                lat.size,
                lon.size,
            )
        )
        ds["temperature"] = (
            ["time", "level", "latitude", "longitude"],
            data_3d,
        )
    return ds, ds + 0.1


def test_maps_histograms_wd_kde_suffixed_outputs(tmp_path: Path):
    ds_target, ds_prediction = _make_basic_ds(include_level=True)
    # Standardized copies for wd_kde (simple z-score per variable)
    ds_target_std = (ds_target - ds_target.mean()) / ds_target.std()
    ds_prediction_std = (ds_prediction - ds_target.mean()) / ds_target.std()

    out_root = tmp_path / "out"
    plot_cfg = {
        "output_mode": "npz",
        "random_seed": 1,
        "histograms_include_3d": True,
        "wd_kde_include_3d": True,
    }

    run_maps(ds_target, ds_prediction, out_root=out_root, plotting_cfg=plot_cfg)
    run_histograms(ds_target, ds_prediction, out_root=out_root, plotting_cfg=plot_cfg)
    run_wd_kde(
        ds_target,
        ds_prediction,
        ds_target_std,
        ds_prediction_std,
        out_root=out_root,
        plotting_cfg=plot_cfg,
    )

    maps_dir = out_root / "maps"
    hist_dir = out_root / "histograms"
    kde_dir = out_root / "wd_kde"
    assert maps_dir.exists() and hist_dir.exists() and kde_dir.exists()

    # Standardized naming: hist_* latbands combined, wd_kde_* combined, map_* files
    assert any(
        f.name.startswith("map_") and f.name.endswith(".npz") for f in maps_dir.glob("map_*.npz")
    ), "Expected standardized map npz outputs"
    # New schema: no 'full' token and no placeholder level token. Filenames end with 'combined_ensnone.npz'
    assert any(
        f.name.startswith("hist_") and f.name.endswith("combined_ensnone.npz")
        for f in hist_dir.glob("hist_*combined*.npz")
    ), "Expected hist combined npz outputs with new schema"
    assert any(
        f.name.startswith("wd_kde_") and f.name.endswith("combined_ensnone.npz")
        for f in kde_dir.glob("wd_kde_*combined*.npz")
    ), "Expected wd_kde combined npz outputs with new schema"
