from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.metrics import deterministic as det_mod
from swissclim_evaluations.metrics import ets as ets_mod


def _make_multi_member(n_members=3):
    init_time = np.array([np.datetime64("2025-01-01T00"), np.datetime64("2025-01-01T06")])
    lead_time = np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
    lat = np.linspace(-10, 10, 3)
    lon = np.linspace(0, 20, 4)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_members, init_time.size, lead_time.size, lat.size, lon.size))
    ds_pred = xr.Dataset(
        {
            "u10": (["ensemble", "init_time", "lead_time", "latitude", "longitude"], data),
        },
        coords={
            "ensemble": np.arange(n_members),
            "init_time": init_time,
            "lead_time": lead_time,
            "latitude": lat,
            "longitude": lon,
        },
    )
    # target: add small bias
    ds_tgt = xr.Dataset(
        {"u10": (["init_time", "lead_time", "latitude", "longitude"], data.mean(axis=0) + 0.05)},
        coords={"init_time": init_time, "lead_time": lead_time, "latitude": lat, "longitude": lon},
    )
    # standardized versions (simple z-score using combined mean/std)
    mean = ds_tgt["u10"].mean()
    std = ds_tgt["u10"].std()
    ds_tgt_std = (ds_tgt - mean) / std
    ds_pred_std = (ds_pred - mean) / std
    return ds_tgt, ds_pred, ds_tgt_std, ds_pred_std


def test_deterministic_members_outputs(tmp_path: Path):
    ds_tgt, ds_pred, ds_tgt_std, ds_pred_std = _make_multi_member(3)
    cfg_metrics = {
        "deterministic": {
            "include": ["MAE"],
            "standardized_include": ["MAE"],
            "aggregate_members_mean": True,
        }
    }
    plotting = {}
    det_mod.run(
        ds_tgt,
        ds_pred,
        ds_tgt_std,
        ds_pred_std,
        tmp_path,
        plotting,
        cfg_metrics,
        ensemble_mode="members",
    )
    det_dir = tmp_path / "deterministic"
    files = sorted(p.name for p in det_dir.glob("*.csv"))
    # Expect one per member + members_mean aggregated
    assert any(f.endswith("ens0.csv") for f in files)
    assert any(f.endswith("ens1.csv") for f in files)
    assert any(f.endswith("ens2.csv") for f in files)
    assert any("members_mean" in f and "enspooled.csv" in f for f in files)


def test_ets_members_outputs(tmp_path: Path):
    ds_tgt, ds_pred, _, _ = _make_multi_member(2)
    cfg_metrics = {
        "ets": {
            "thresholds": [50],
            "aggregate_members_mean": True,
        }
    }
    ets_mod.run(ds_tgt, ds_pred, tmp_path, cfg_metrics, ensemble_mode="members")
    ets_dir = tmp_path / "ets"
    files = sorted(p.name for p in ets_dir.glob("*.csv"))
    assert any(f.endswith("ens0.csv") for f in files)
    assert any(f.endswith("ens1.csv") for f in files)
    assert any("members_mean" in f and "enspooled.csv" in f for f in files)
