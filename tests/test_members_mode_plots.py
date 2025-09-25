from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.plots.energy_spectra import run as run_energy
from swissclim_evaluations.plots.maps import run as run_maps
from swissclim_evaluations.plots.wd_kde import run as run_wd_kde


def _make_ensemble_dataset(var2d=True, members=3):
    init = np.array([np.datetime64("2025-01-01T00")])
    lat = np.linspace(-10, 10, 4)
    lon = np.linspace(0, 30, 6)
    ens = np.arange(members)
    rng = np.random.default_rng(0)
    data_shape = (members, init.size, lat.size, lon.size)
    ds_pred = xr.Dataset(
        {
            "u10": (
                ["ensemble", "init_time", "latitude", "longitude"],
                rng.standard_normal(data_shape),
            )
        },
        coords={"ensemble": ens, "init_time": init, "latitude": lat, "longitude": lon},
    )
    ds_tgt = xr.Dataset(
        {
            "u10": (
                ["init_time", "latitude", "longitude"],
                rng.standard_normal((init.size, lat.size, lon.size)),
            )
        },
        coords={"init_time": init, "latitude": lat, "longitude": lon},
    )
    return ds_tgt, ds_pred


def test_maps_members_outputs(tmp_path: Path):
    tgt, pred = _make_ensemble_dataset()
    out = tmp_path / "output"
    run_maps(tgt, pred, out_root=out, plotting_cfg={"output_mode": "npz"}, ensemble_mode="members")
    maps_dir = out / "maps"
    names = {f.name for f in maps_dir.glob("map_u10_*.npz")}
    # Expect at least first two members tokens
    assert any("_ens0" in n for n in names)
    assert any("_ens1" in n for n in names)


def test_energy_spectra_members_outputs(tmp_path: Path):
    # Minimal spectra-friendly grid: need longitude dimension length >=4 already set
    tgt, pred = _make_ensemble_dataset()
    out = tmp_path / "output"
    run_energy(
        tgt,
        pred,
        out_root=out,
        plotting_cfg={"output_mode": "npz"},
        select_cfg={},
        ensemble_mode="members",
    )
    es_dir = out / "energy_spectra"
    # With output_mode=npz we expect NPZ spectrum data files (not necessarily PNG figures)
    names = {f.name for f in es_dir.glob("lsd_*_spectrum_*.npz")}
    assert any("ens0" in n for n in names)
    assert any("ens1" in n for n in names)


def test_wd_kde_members_outputs(tmp_path: Path):
    tgt, pred = _make_ensemble_dataset()
    # Standardized versions
    tgt_std = (tgt - tgt.mean()) / tgt.std()
    pred_std = (pred - tgt.mean()) / tgt.std()
    out = tmp_path / "output"
    run_wd_kde(
        tgt,
        pred,
        tgt_std,
        pred_std,
        out_root=out,
        plotting_cfg={"output_mode": "npz"},
        ensemble_mode="members",
    )
    wd_dir = out / "wd_kde"
    names = {f.name for f in wd_dir.glob("wd_kde_*_combined_*.npz")}
    assert any("ens0" in n for n in names)
    assert any("ens1" in n for n in names)
