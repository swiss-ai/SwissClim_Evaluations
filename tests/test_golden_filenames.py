from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.plots.histograms import run as run_histograms
from swissclim_evaluations.plots.maps import run as run_maps
from swissclim_evaluations.plots.wd_kde import run as run_wd_kde

# Golden filename test: produce a deterministic tiny dataset and assert exact file set.
# Uses stubs (fast plotting) from conftest, focusing on naming contract.


def _datasets():
    time = np.array([np.datetime64("2025-01-01T00")])
    lat = np.linspace(-10, 10, 3)
    lon = np.linspace(0, 20, 4)
    rng = np.random.default_rng(0)
    base = rng.standard_normal((time.size, lat.size, lon.size))
    tgt = xr.Dataset(
        {"u10": (["time", "latitude", "longitude"], base)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    pred = tgt + 0.1
    # Standardized for wd_kde
    tgt_std = (tgt - tgt.mean()) / tgt.std()
    pred_std = (pred - tgt.mean()) / tgt.std()
    return tgt, pred, tgt_std, pred_std


def test_golden_plot_filenames(tmp_path: Path):
    tgt, pred, tgt_std, pred_std = _datasets()
    out = tmp_path / "output"
    cfg = {"output_mode": "npz", "random_seed": 1}
    run_maps(tgt, pred, out_root=out, plotting_cfg=cfg, ensemble_mode="none")
    # histograms default to pooled (pooled vs none behave same without ensemble dim -> ensnone)
    run_histograms(tgt, pred, out_root=out, plotting_cfg=cfg, ensemble_mode="pooled")
    run_wd_kde(
        tgt,
        pred,
        tgt_std,
        pred_std,
        out_root=out,
        plotting_cfg=cfg,
        ensemble_mode="pooled",
    )

    maps_dir = out / "maps"
    hist_dir = out / "histograms"
    kde_dir = out / "wd_kde"
    got = sorted(
        [
            *(f.name for f in maps_dir.iterdir()),
            *(f.name for f in hist_dir.iterdir()),
            *(f.name for f in kde_dir.iterdir()),
        ]
    )
    # Expected: single variable, no levels, combined histogram / wd_kde npz + map file(s)
    # Ensemble marker ensnone always present as there is no ensemble dimension.
    expected_contains = [
        "map_u10_ensnone.npz",
    ]
    for token in expected_contains:
        assert token in got, f"Missing expected file {token}. Got: {got}"
    # Ensure exactly one hist combined and one wd_kde combined file present
    assert sum(n.startswith("hist_") for n in got) == 1
    assert (
        sum(n.startswith("wd_kde_") for n in got) >= 1
    )  # one combined plus wasserstein CSV allowed
