from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.plots.maps import run as run_maps


def _make_basic_ds(include_level: bool = False):
    # Use minimal sizes for speed (still exercises 2D + optional 3D paths)
    time = np.array(
        [
            np.datetime64("2025-01-01T00"),
        ]
    )
    lat = np.linspace(-10, 10, 3)
    lon = np.linspace(0, 20, 4)
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
        level = np.array([1000])  # single level sufficient for coverage
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

    out_root = tmp_path / "output"
    plot_cfg = {
        "output_mode": "npz",  # avoid any real image saving
        "random_seed": 1,
        "histograms_include_3d": True,
        "wd_kde_include_3d": True,
        # Extra limits to reduce inner work where respected
        "histograms_max_levels": 1,
        "kde_max_samples": 200,  # already subsampled in code
    }

    # Run real maps (provides coverage of filename logic there)
    run_maps(ds_target, ds_prediction, out_root=out_root, plotting_cfg=plot_cfg)

    # Import histogram & wd_kde modules AFTER fixtures applied so monkeypatched fast runs are used
    import swissclim_evaluations.plots.histograms as hist_mod
    import swissclim_evaluations.plots.wd_kde as kde_mod

    hist_mod.run(ds_target, ds_prediction, out_root=out_root, plotting_cfg=plot_cfg)
    # Only evolution ridgeline retained; request it explicitly
    kde_mod.run(
        ds_target,
        ds_prediction,
        ds_target_std,
        ds_prediction_std,
        out_root=out_root,
        plotting_cfg={**plot_cfg, "wd_kde_global_evolution": True, "output_mode": "plot"},
    )

    maps_dir = out_root / "maps"
    hist_dir = out_root / "histograms"
    kde_dir = out_root / "wd_kde"
    assert maps_dir.exists() and hist_dir.exists() and kde_dir.exists()

    # Light existence smoke checks (detailed naming covered in golden & output_naming tests)
    assert any(f.name.startswith("map_") for f in maps_dir.glob("map_*.npz"))
    assert any(f.name.startswith("hist_") for f in hist_dir.glob("hist_*.npz"))
    # KDE module now only produces ridgeline plots; under fast plotting stubs we only
    # verify the module ran and the directory exists (image saving is no-op in tests).
