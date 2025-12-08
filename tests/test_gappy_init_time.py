from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.cli import prepare_datasets


def _build_continuous_target():
    # ERA5-like target with continuous hourly time
    time = np.array(
        [
            np.datetime64("2023-01-01T00"),
            np.datetime64("2023-01-01T01"),
            np.datetime64("2023-01-01T02"),
            np.datetime64("2023-01-01T03"),
            np.datetime64("2023-01-01T04"),
            np.datetime64("2023-01-01T05"),
            np.datetime64("2023-01-01T06"),
        ],
        dtype="datetime64[ns]",
    )
    lat = np.linspace(46.0, 46.5, 2)
    lon = np.linspace(7.0, 7.5, 2)
    # Add dummy ensemble dimension for strict compliance
    ens = np.array([0])
    data = np.arange(time.size * lat.size * lon.size * ens.size).reshape(
        time.size, lat.size, lon.size, ens.size
    )
    ds = xr.Dataset(
        {
            "2m_temperature": (
                ["time", "latitude", "longitude", "ensemble"],
                data,
            )
        },
        coords={"time": time, "latitude": lat, "longitude": lon, "ensemble": ens},
    )
    return ds


def _build_gappy_predictions():
    # Predictions with non-contiguous init_time: pick a few hours only
    init = np.array(
        [
            np.datetime64("2023-01-01T00"),
            np.datetime64("2023-01-01T03"),
            np.datetime64("2023-01-01T06"),
        ],
        dtype="datetime64[ns]",
    )
    lead = np.array([0], dtype="timedelta64[h]").astype("timedelta64[ns]")
    lat = np.linspace(46.0, 46.5, 2)
    lon = np.linspace(7.0, 7.5, 2)
    ens = np.arange(3)
    shape = (init.size, lead.size, lat.size, lon.size, ens.size)
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "2m_temperature": (
                [
                    "init_time",
                    "lead_time",
                    "latitude",
                    "longitude",
                    "ensemble",
                ],
                rng.standard_normal(shape),
            )
        },
        coords={
            "init_time": init,
            "lead_time": lead,
            "latitude": lat,
            "longitude": lon,
            "ensemble": ens,
        },
    )
    return ds


def test_prepare_handles_gappy_inittime(tmp_path: Path, monkeypatch):
    # Build in-memory datasets and monkeypatch data loaders
    ds_tgt = _build_continuous_target()
    ds_pred = _build_gappy_predictions()

    from swissclim_evaluations import data as data_mod

    monkeypatch.setattr(data_mod, "open_target", lambda path, variables=None: ds_tgt)
    monkeypatch.setattr(data_mod, "open_prediction", lambda path, variables=None: ds_pred)

    cfg = {
        "paths": {
            "target": "unused",
            "prediction": "unused",
            "output_root": str(tmp_path / "output"),
        },
        "modules": {"probabilistic": True},
        "selection": {
            # Intentionally no continuous range; rely on gappy inits in predictions
            "check_missing": True,
        },
    }

    ds_target, ds_prediction, _, _ = prepare_datasets(cfg)

    # Expect alignment to (init_time, lead_time) with exactly the three gappy inits
    assert "init_time" in ds_target.dims and "init_time" in ds_prediction.dims
    assert int(ds_prediction.init_time.size) == 3
    assert np.all(ds_prediction.init_time.values == ds_target.init_time.values)
    # lead_time should exist and be zero
    assert "lead_time" in ds_prediction.dims and int(ds_prediction.lead_time.size) == 1
    assert np.all(
        ds_prediction.lead_time.values
        == np.array([0], dtype="timedelta64[h]").astype("timedelta64[ns]")
    )


def test_multiple_datetime_ranges_selection(tmp_path: Path, monkeypatch):
    # Target continuous, predictions hourly; select two disjoint ranges over
    # init_time and verify alignment
    ds_tgt = _build_continuous_target()
    # Predictions with hourly inits 0..6
    init = np.array(
        [
            np.datetime64("2023-01-01T00"),
            np.datetime64("2023-01-01T01"),
            np.datetime64("2023-01-01T02"),
            np.datetime64("2023-01-01T03"),
            np.datetime64("2023-01-01T04"),
            np.datetime64("2023-01-01T05"),
            np.datetime64("2023-01-01T06"),
        ],
        dtype="datetime64[ns]",
    )
    lead = np.array([0], dtype="timedelta64[h]").astype("timedelta64[ns]")
    lat = np.linspace(46.0, 46.5, 2)
    lon = np.linspace(7.0, 7.5, 2)
    ens = np.arange(2)
    shape = (init.size, lead.size, lat.size, lon.size, ens.size)
    rng = np.random.default_rng(2)
    ds_pred = xr.Dataset(
        {
            "2m_temperature": (
                [
                    "init_time",
                    "lead_time",
                    "latitude",
                    "longitude",
                    "ensemble",
                ],
                rng.standard_normal(shape),
            )
        },
        coords={
            "init_time": init,
            "lead_time": lead,
            "latitude": lat,
            "longitude": lon,
            "ensemble": ens,
        },
    )

    from swissclim_evaluations import data as data_mod

    monkeypatch.setattr(data_mod, "open_target", lambda path, variables=None: ds_tgt)
    monkeypatch.setattr(data_mod, "open_prediction", lambda path, variables=None: ds_pred)

    # Select two disjoint ranges: 00..02 and 05..06
    cfg = {
        "paths": {
            "target": "unused",
            "prediction": "unused",
            "output_root": str(tmp_path / "output"),
        },
        "modules": {"probabilistic": True},
        "selection": {
            "datetimes": [
                "2023-01-01T00:2023-01-01T02",
                "2023-01-01T05:2023-01-01T06",
            ],
        },
    }

    ds_target, ds_prediction, _, _ = prepare_datasets(cfg)
    # Expect retained init_time labels: 00,01,02,05,06
    expected = np.array(
        [
            np.datetime64("2023-01-01T00"),
            np.datetime64("2023-01-01T01"),
            np.datetime64("2023-01-01T02"),
            np.datetime64("2023-01-01T05"),
            np.datetime64("2023-01-01T06"),
        ],
        dtype="datetime64[ns]",
    )
    assert np.all(ds_prediction.init_time.values == expected)
    assert np.all(ds_target.init_time.values == expected)
