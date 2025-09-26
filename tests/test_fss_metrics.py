from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from swissclim_evaluations.metrics import deterministic as det


def _make_ds(shape=(4, 6), seed=0, dims=("latitude", "longitude")):
    rng = np.random.default_rng(seed)
    data = rng.random(shape)
    return xr.Dataset({"var": (dims, data)})


def test_fss_not_nan_default_dims(tmp_path: Path):
    ds_t = _make_ds()
    # Add slight perturbation so fields differ but share events
    ds_p = ds_t.copy(deep=True)
    ds_p["var"] = ds_p["var"] * 1.05
    cfg = {"deterministic": {"fss": {"quantile": 0.8}}}
    df = det._calculate_all_metrics(  # noqa: SLF001
        ds_t,
        ds_p,
        True,
        ds_t["var"].size,
        include=["FSS"],
        fss_cfg=cfg["deterministic"]["fss"],
    )
    assert not pd.isna(df.loc["var", "FSS"])  # should compute


def test_fss_not_nan_latlon_dims(tmp_path: Path):
    ds_t = _make_ds(dims=("latitude", "longitude"))
    ds_p = ds_t * 0.9
    cfg = {"deterministic": {"fss": {"quantile": 90}}}
    df = det._calculate_all_metrics(  # noqa: SLF001
        ds_t,
        ds_p,
        True,
        ds_t["var"].size,
        include=["FSS"],
        fss_cfg=cfg["deterministic"]["fss"],
    )
    assert not pd.isna(df.loc["var", "FSS"])  # should compute


def test_fss_no_event_defaults_to_one(tmp_path: Path):
    # Both fields constant zeros -> no events; expect default 1.0
    ds_t = xr.Dataset({"var": (("latitude", "longitude"), np.zeros((3, 3)))})
    ds_p = ds_t.copy(deep=True)
    cfg = {"deterministic": {"fss": {"thresholds": {"var": 0.5}}}}
    df = det._calculate_all_metrics(  # noqa: SLF001
        ds_t,
        ds_p,
        True,
        ds_t["var"].size,
        include=["FSS"],
        fss_cfg=cfg["deterministic"]["fss"],
    )
    assert df.loc["var", "FSS"] == 1.0
