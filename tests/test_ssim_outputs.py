from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swissclim_evaluations.metrics.ssim import calculate_ssim, run as run_ssim

# ── Dataset helpers ───────────────────────────────────────────────────────────


def _make_2d_ds(n_time: int = 3, n_lat: int = 12, n_lon: int = 12) -> tuple[xr.Dataset, xr.Dataset]:
    """2D variable (time × lat × lon)."""
    rng = np.random.default_rng(0)
    lat = np.linspace(45.0, 47.0, n_lat)
    lon = np.linspace(6.0, 9.0, n_lon)
    time = np.array(
        [np.datetime64("2023-01-01T00") + np.timedelta64(h, "h") for h in range(n_time)]
    )
    data = rng.uniform(270, 300, (n_time, n_lat, n_lon))
    coords = {"time": time, "latitude": lat, "longitude": lon}
    ds_t = xr.Dataset({"temperature": (["time", "latitude", "longitude"], data)}, coords=coords)
    ds_p = xr.Dataset(
        {
            "temperature": (
                ["time", "latitude", "longitude"],
                data + 0.5 * rng.standard_normal(data.shape),
            )
        },
        coords=coords,
    )
    return ds_t, ds_p


def _make_mixed_ds(
    n_time: int = 3, n_lat: int = 12, n_lon: int = 12
) -> tuple[xr.Dataset, xr.Dataset]:
    """2D + 3D variables (time × [level ×] lat × lon)."""
    rng = np.random.default_rng(1)
    lat = np.linspace(45.0, 47.0, n_lat)
    lon = np.linspace(6.0, 9.0, n_lon)
    levels = np.array([500, 1000])
    time = np.array(
        [np.datetime64("2023-01-01T00") + np.timedelta64(h, "h") for h in range(n_time)]
    )
    coords = {"time": time, "latitude": lat, "longitude": lon, "level": levels}

    data_2d = rng.uniform(270, 300, (n_time, n_lat, n_lon))
    data_3d = rng.uniform(220, 280, (n_time, len(levels), n_lat, n_lon))

    ds_t = xr.Dataset(
        {
            "temperature_2m": (["time", "latitude", "longitude"], data_2d),
            "temperature_pressure": (["time", "level", "latitude", "longitude"], data_3d),
        },
        coords=coords,
    )
    ds_p = xr.Dataset(
        {
            "temperature_2m": (
                ["time", "latitude", "longitude"],
                data_2d + 0.5 * rng.standard_normal(data_2d.shape),
            ),
            "temperature_pressure": (
                ["time", "level", "latitude", "longitude"],
                data_3d + 0.5 * rng.standard_normal(data_3d.shape),
            ),
        },
        coords=coords,
    )
    return ds_t, ds_p


def _make_lead_ds(
    n_lead: int = 2, n_lat: int = 12, n_lon: int = 12
) -> tuple[xr.Dataset, xr.Dataset]:
    """2D + 3D variables with lead_time instead of time."""
    rng = np.random.default_rng(2)
    lat = np.linspace(45.0, 47.0, n_lat)
    lon = np.linspace(6.0, 9.0, n_lon)
    levels = np.array([500, 1000])
    leads = np.array([np.timedelta64(6 * (i + 1), "h") for i in range(n_lead)])
    coords = {"lead_time": leads, "latitude": lat, "longitude": lon, "level": levels}

    data_2d = rng.uniform(270, 300, (n_lead, n_lat, n_lon))
    data_3d = rng.uniform(220, 280, (n_lead, len(levels), n_lat, n_lon))

    ds_t = xr.Dataset(
        {
            "temperature_2m": (["lead_time", "latitude", "longitude"], data_2d),
            "temperature_pressure": (["lead_time", "level", "latitude", "longitude"], data_3d),
        },
        coords=coords,
    )
    ds_p = xr.Dataset(
        {
            "temperature_2m": (
                ["lead_time", "latitude", "longitude"],
                data_2d + 0.5 * rng.standard_normal(data_2d.shape),
            ),
            "temperature_pressure": (
                ["lead_time", "level", "latitude", "longitude"],
                data_3d + 0.5 * rng.standard_normal(data_3d.shape),
            ),
        },
        coords=coords,
    )
    return ds_t, ds_p


# ── calculate_ssim unit tests ─────────────────────────────────────────────────


def test_calculate_ssim_scalar_result():
    """With no preserve_dims, returns a variable-indexed scalar DataFrame."""
    ds_t, ds_p = _make_2d_ds()
    df = calculate_ssim(ds_t, ds_p)
    assert isinstance(df, pd.DataFrame)
    assert "SSIM" in df.columns
    assert df.index.name == "variable"
    assert "temperature" in df.index
    assert df["SSIM"].between(-1, 1).all()


def test_calculate_ssim_preserve_level():
    """preserve_dims=['level'] returns one row per (variable, level) for 3D vars."""
    ds_t, ds_p = _make_mixed_ds()
    df = calculate_ssim(ds_t, ds_p, preserve_dims=["level"])
    assert "level" in df.columns
    assert "SSIM" in df.columns
    # 3D variable should appear with 2 level rows; 2D variable gets no level column so
    # only appears in rows where level is NaN (from the scalar case in calculate_ssim)
    lvl_rows = df.dropna(subset=["level"])
    assert set(lvl_rows["variable"]) == {"temperature_pressure"}
    assert len(lvl_rows) == 2  # two pressure levels


def test_calculate_ssim_preserve_lead_time():
    """preserve_dims=['lead_time'] returns one row per (variable, lead_time)."""
    ds_t, ds_p = _make_lead_ds()
    df = calculate_ssim(ds_t, ds_p, preserve_dims=["lead_time"])
    assert "lead_time" in df.columns
    lead_rows = df.dropna(subset=["lead_time"])
    assert set(lead_rows["variable"]) == {"temperature_2m", "temperature_pressure"}
    assert len(lead_rows) == 4  # 2 vars × 2 leads


def test_calculate_ssim_empty_on_no_spatial():
    """Returns empty DataFrame when no variable has two spatial dims."""
    ds = xr.Dataset({"x": (["time"], [1.0, 2.0])}, coords={"time": [0, 1]})
    df = calculate_ssim(ds, ds)
    assert df.empty


# ── run() integration tests ───────────────────────────────────────────────────


def test_ssim_run_overall_only(tmp_path: Path):
    """Overall CSV is written with AVERAGE_SSIM row."""
    ds_t, ds_p = _make_2d_ds()
    run_ssim(ds_t, ds_p, tmp_path, metrics_cfg={})
    files = list((tmp_path / "ssim").glob("ssim_ssim_*.csv"))
    # Exclude per_level and by_lead
    overall = [f for f in files if "per_level" not in f.name and "by_lead" not in f.name]
    assert overall, "Expected overall SSIM CSV"
    df = pd.read_csv(overall[0], index_col="variable")
    assert "SSIM" in df.columns
    assert "AVERAGE_SSIM" in df.index
    assert df.loc["temperature", "SSIM"] == pytest.approx(df.loc["temperature", "SSIM"])  # not NaN


def test_ssim_run_per_level_no_nan_levels(tmp_path: Path):
    """Per-level CSV contains only 3D variables and level values are integers."""
    ds_t, ds_p = _make_mixed_ds()
    run_ssim(ds_t, ds_p, tmp_path, metrics_cfg={})
    ssim_dir = tmp_path / "ssim"

    per_level = list(ssim_dir.glob("ssim_ssim_per_level_*.csv"))
    assert per_level, "Expected per-level SSIM CSV"
    df = pd.read_csv(per_level[0])
    assert "level" in df.columns
    assert df["level"].notna().all(), "No NaN levels expected"
    assert df["level"].dtype in (int, np.int64, np.int32)
    assert set(df["variable"]) == {
        "temperature_pressure"
    }, "2D var must not appear in per-level CSV"
    assert sorted(df["level"].unique().tolist()) == [500, 1000]


def test_ssim_run_by_lead(tmp_path: Path):
    """By-lead CSV has lead_time_hours column with integer hours."""
    ds_t, ds_p = _make_lead_ds()
    run_ssim(ds_t, ds_p, tmp_path, metrics_cfg={})
    ssim_dir = tmp_path / "ssim"

    by_lead = list(ssim_dir.glob("ssim_ssim_by_lead_*.csv"))
    assert by_lead, "Expected by-lead SSIM CSV"
    df = pd.read_csv(by_lead[0])
    assert "lead_time_hours" in df.columns
    assert df["lead_time_hours"].notna().all()
    assert sorted(df["lead_time_hours"].unique().tolist()) == [6, 12]


def test_ssim_run_per_level_by_lead(tmp_path: Path):
    """Per-level-by-lead CSV has both level and lead_time_hours columns."""
    ds_t, ds_p = _make_lead_ds()
    run_ssim(ds_t, ds_p, tmp_path, metrics_cfg={})
    ssim_dir = tmp_path / "ssim"

    lvl_lead = list(ssim_dir.glob("ssim_ssim_per_level_by_lead_*.csv"))
    assert lvl_lead, "Expected per-level-by-lead SSIM CSV"
    df = pd.read_csv(lvl_lead[0])
    assert "level" in df.columns
    assert "lead_time_hours" in df.columns
    assert df["level"].notna().all()
    assert set(df["variable"]) == {"temperature_pressure"}
    # 2 levels × 2 leads = 4 rows per variable
    assert len(df) == 4


def test_ssim_run_disabled_outputs(tmp_path: Path):
    """report_per_level=False suppresses per-level CSV."""
    ds_t, ds_p = _make_mixed_ds()
    run_ssim(
        ds_t,
        ds_p,
        tmp_path,
        metrics_cfg={"ssim": {"report_per_level": False, "report_per_lead": False}},
    )
    ssim_dir = tmp_path / "ssim"
    assert not list(ssim_dir.glob("ssim_ssim_per_level_*.csv"))
    assert not list(ssim_dir.glob("ssim_ssim_by_lead_*.csv"))
    assert not list(ssim_dir.glob("ssim_ssim_per_level_by_lead_*.csv"))
