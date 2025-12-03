from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swissclim_evaluations.metrics.multivariate import calculate_ssim, run


def test_calculate_ssim_basic():
    # Create two identical images
    # Use size > 11 for default win_size
    img1 = np.random.rand(20, 20)
    img2 = img1.copy()

    # SSIM should be 1.0
    # We need to wrap them in xarray Dataset
    ds1 = xr.Dataset(
        {"var1": (["latitude", "longitude"], img1)},
        coords={"latitude": np.arange(20), "longitude": np.arange(20)},
    )
    ds2 = xr.Dataset(
        {"var1": (["latitude", "longitude"], img2)},
        coords={"latitude": np.arange(20), "longitude": np.arange(20)},
    )

    df = calculate_ssim(ds1, ds2)
    # Index is variable name, column is SSIM
    assert df.loc["var1", "SSIM"] == pytest.approx(1.0)


def test_calculate_ssim_different():
    # Create two different images
    img1 = np.zeros((20, 20))
    img2 = np.ones((20, 20))

    ds1 = xr.Dataset(
        {"var1": (["latitude", "longitude"], img1)},
        coords={"latitude": np.arange(20), "longitude": np.arange(20)},
    )
    ds2 = xr.Dataset(
        {"var1": (["latitude", "longitude"], img2)},
        coords={"latitude": np.arange(20), "longitude": np.arange(20)},
    )

    df = calculate_ssim(ds1, ds2)
    # SSIM should be low
    assert df.loc["var1", "SSIM"] < 1.0
    assert df.loc["var1", "SSIM"] >= -1.0  # SSIM can be negative


def test_calculate_ssim_params():
    # Test with custom parameters
    img1 = np.random.rand(20, 20)
    img2 = img1 + 0.1

    ds1 = xr.Dataset(
        {"var1": (["latitude", "longitude"], img1)},
        coords={"latitude": np.arange(20), "longitude": np.arange(20)},
    )
    ds2 = xr.Dataset(
        {"var1": (["latitude", "longitude"], img2)},
        coords={"latitude": np.arange(20), "longitude": np.arange(20)},
    )

    # Default (gaussian_weights=True)
    df1 = calculate_ssim(ds1, ds2)

    # Custom sigma
    df2 = calculate_ssim(ds1, ds2, sigma=0.5)

    # Results should differ
    assert df1.loc["var1", "SSIM"] != df2.loc["var1", "SSIM"]


def test_run_multivariate(tmp_path: Path):
    # Create dummy datasets
    # Use size > 11 for default win_size
    data = np.random.rand(2, 2, 20, 20)  # init, lead, lat, lon
    coords = {
        "init_time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "lead_time": pd.to_timedelta([0, 6], unit="h"),
        "latitude": np.arange(20),
        "longitude": np.arange(20),
    }
    ds = xr.Dataset(
        {"var1": (["init_time", "lead_time", "latitude", "longitude"], data)},
        coords=coords,
    )

    out_root = tmp_path / "output"

    run(ds, ds, out_root, metrics_cfg=None)

    # Check if output file exists
    out_dir = out_root / "multivariate"
    assert out_dir.exists()
    files = list(out_dir.glob("multivariate_ssim*.csv"))
    assert len(files) > 0

    df = pd.read_csv(files[0])
    # Check content
    assert "variable" in df.columns
    assert "SSIM" in df.columns

    # Check var1
    row_var1 = df[df["variable"] == "var1"]
    assert not row_var1.empty
    assert row_var1["SSIM"].values[0] == pytest.approx(1.0)

    # Check average
    row_avg = df[df["variable"] == "MULTIVARIATE_AVERAGE"]
    assert not row_avg.empty
    assert row_avg["SSIM"].values[0] == pytest.approx(1.0)


def test_run_multivariate_ensemble(tmp_path: Path):
    # Create dummy datasets with ensemble
    # Use size > 11 for default win_size
    data = np.random.rand(2, 2, 3, 20, 20)  # init, lead, ens, lat, lon
    coords = {
        "init_time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "lead_time": pd.to_timedelta([0, 6], unit="h"),
        "ensemble": [0, 1, 2],
        "latitude": np.arange(20),
        "longitude": np.arange(20),
    }
    ds = xr.Dataset(
        {
            "var1": (
                ["init_time", "lead_time", "ensemble", "latitude", "longitude"],
                data,
            )
        },
        coords=coords,
    )

    out_root = tmp_path / "output"

    # Test mean mode (default)
    run(ds, ds, out_root, metrics_cfg=None, ensemble_mode="mean")
    out_dir = out_root / "multivariate"
    files = list(out_dir.glob("multivariate_ssim*ensmean.csv"))
    assert len(files) == 1

    # Test members mode
    run(ds, ds, out_root, metrics_cfg=None, ensemble_mode="members")
    files = list(out_dir.glob("multivariate_ssim*ens0.csv"))
    assert len(files) == 1
