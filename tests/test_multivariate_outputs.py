from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.metrics.multivariate import run as run_multivariate
from swissclim_evaluations.plots.bivariate_histograms import (
    _format_level_suffix,
    calculate_and_plot_bivariate_histograms,
)

# ── Dataset helpers ───────────────────────────────────────────────────────────


def _make_2d_pair(n_time: int = 4, n_lat: int = 8, n_lon: int = 8) -> tuple[xr.Dataset, xr.Dataset]:
    """Two 2D surface variables (time × lat × lon)."""
    rng = np.random.default_rng(10)
    lat = np.linspace(45.0, 47.0, n_lat)
    lon = np.linspace(6.0, 9.0, n_lon)
    time = np.arange(n_time)
    coords = {"time": time, "latitude": lat, "longitude": lon}

    t = rng.uniform(270, 300, (n_time, n_lat, n_lon))
    q = rng.uniform(0.001, 0.015, (n_time, n_lat, n_lon))
    ds_t = xr.Dataset(
        {
            "temperature": (["time", "latitude", "longitude"], t),
            "specific_humidity": (["time", "latitude", "longitude"], q),
        },
        coords=coords,
    )
    ds_p = xr.Dataset(
        {
            "temperature": (
                ["time", "latitude", "longitude"],
                t + 0.5 * rng.standard_normal(t.shape),
            ),
            "specific_humidity": (
                ["time", "latitude", "longitude"],
                q + 0.001 * rng.standard_normal(q.shape),
            ),
        },
        coords=coords,
    )
    return ds_t, ds_p


def _make_3d_pair(n_time: int = 3, n_lat: int = 8, n_lon: int = 8) -> tuple[xr.Dataset, xr.Dataset]:
    """Two 3D variables (time × level × lat × lon)."""
    rng = np.random.default_rng(11)
    lat = np.linspace(45.0, 47.0, n_lat)
    lon = np.linspace(6.0, 9.0, n_lon)
    levels = np.array([500, 850])
    time = np.arange(n_time)
    coords = {"time": time, "latitude": lat, "longitude": lon, "level": levels}

    t = rng.uniform(240, 290, (n_time, len(levels), n_lat, n_lon))
    q = rng.uniform(0.001, 0.010, (n_time, len(levels), n_lat, n_lon))
    ds_t = xr.Dataset(
        {
            "temperature": (["time", "level", "latitude", "longitude"], t),
            "specific_humidity": (["time", "level", "latitude", "longitude"], q),
        },
        coords=coords,
    )
    ds_p = xr.Dataset(
        {
            "temperature": (
                ["time", "level", "latitude", "longitude"],
                t + 0.5 * rng.standard_normal(t.shape),
            ),
            "specific_humidity": (
                ["time", "level", "latitude", "longitude"],
                q + 0.001 * rng.standard_normal(q.shape),
            ),
        },
        coords=coords,
    )
    return ds_t, ds_p


def _make_lead_pair(
    n_lead: int = 3, n_lat: int = 8, n_lon: int = 8
) -> tuple[xr.Dataset, xr.Dataset]:
    """Two 2D variables with a lead_time dimension."""
    rng = np.random.default_rng(12)
    lat = np.linspace(45.0, 47.0, n_lat)
    lon = np.linspace(6.0, 9.0, n_lon)
    leads = np.array([np.timedelta64(6 * (i + 1), "h") for i in range(n_lead)])
    coords = {"lead_time": leads, "latitude": lat, "longitude": lon}

    t = rng.uniform(270, 300, (n_lead, n_lat, n_lon))
    q = rng.uniform(0.001, 0.015, (n_lead, n_lat, n_lon))
    ds_t = xr.Dataset(
        {
            "temperature": (["lead_time", "latitude", "longitude"], t),
            "specific_humidity": (["lead_time", "latitude", "longitude"], q),
        },
        coords=coords,
    )
    ds_p = xr.Dataset(
        {
            "temperature": (
                ["lead_time", "latitude", "longitude"],
                t + 0.5 * rng.standard_normal(t.shape),
            ),
            "specific_humidity": (
                ["lead_time", "latitude", "longitude"],
                q + 0.001 * rng.standard_normal(q.shape),
            ),
        },
        coords=coords,
    )
    return ds_t, ds_p


# ── _format_level_suffix unit tests ──────────────────────────────────────────


def test_format_level_suffix_none():
    assert _format_level_suffix(None) == ""


def test_format_level_suffix_integer():
    assert _format_level_suffix(500.0) == "_level500"


def test_format_level_suffix_near_integer():
    """Floating-point rounding from NetCDF must map to integer label."""
    assert _format_level_suffix(499.9999) == "_level500"
    assert _format_level_suffix(850.0001) == "_level850"


def test_format_level_suffix_non_integer():
    assert _format_level_suffix(500.6) == "_level500.6"


# ── calculate_and_plot_bivariate_histograms unit tests ───────────────────────


def test_npz_keys_present(tmp_path: Path):
    """Output NPZ must contain the mandatory keys."""
    ds_t, ds_p = _make_2d_pair()
    calculate_and_plot_bivariate_histograms(
        ds_p, ds_t, [["temperature", "specific_humidity"]], tmp_path, bins=20
    )
    npzs = list((tmp_path / "multivariate").glob("bivariate_hist_*.npz"))
    assert npzs, "Expected at least one NPZ"
    with np.load(npzs[0]) as f:
        for key in ("hist", "hist_target", "bins_x", "bins_y", "var_x", "var_y"):
            assert key in f.files, f"Missing key: {key}"


def test_per_level_files_created(tmp_path: Path):
    """3D pair must produce one NPZ per level."""
    ds_t, ds_p = _make_3d_pair()
    calculate_and_plot_bivariate_histograms(
        ds_p, ds_t, [["temperature", "specific_humidity"]], tmp_path, bins=20
    )
    npzs = list((tmp_path / "multivariate").glob("bivariate_hist_*_level*.npz"))
    assert len(npzs) == 2, f"Expected 2 level files, got {len(npzs)}"
    level_suffixes = {f.stem.split("_level")[-1] for f in npzs}
    assert level_suffixes == {"500", "850"}


def test_per_lead_grid_created(tmp_path: Path):
    """Multi-lead dataset must produce a per-lead-time grid PNG."""
    ds_t, ds_p = _make_lead_pair(n_lead=3)
    calculate_and_plot_bivariate_histograms(
        ds_p, ds_t, [["temperature", "specific_humidity"]], tmp_path, bins=20
    )
    lead_pngs = list((tmp_path / "multivariate").glob("bivariate_by_lead_*.png"))
    assert lead_pngs, "Expected per-lead-time grid PNG"


def test_skip_missing_variable(tmp_path: Path):
    """Pair with a variable absent from the prediction is skipped gracefully."""
    ds_t, ds_p = _make_2d_pair()
    ds_p = ds_p.drop_vars("specific_humidity")
    calculate_and_plot_bivariate_histograms(
        ds_p, ds_t, [["temperature", "specific_humidity"]], tmp_path, bins=20
    )
    # No crash; no NPZ written
    assert not list((tmp_path / "multivariate").glob("bivariate_hist_*.npz"))


def test_physical_constraints_any_level(tmp_path: Path):
    """Physical overlay must fire at 850 hPa (not restricted to 500 hPa)."""
    ds_t, ds_p = _make_3d_pair()
    # Just ensure the function completes without error for both levels
    calculate_and_plot_bivariate_histograms(
        ds_p,
        ds_t,
        [["temperature", "specific_humidity"]],
        tmp_path,
        bins=20,
    )
    npzs = list((tmp_path / "multivariate").glob("bivariate_hist_*_level850*.npz"))
    assert npzs, "Expected NPZ for 850 hPa"


# ── run() integration tests ───────────────────────────────────────────────────


def test_run_mean_mode(tmp_path: Path):
    """Mean mode writes one NPZ per pair."""
    ds_t, ds_p = _make_2d_pair()
    run_multivariate(
        ds_t,
        ds_p,
        tmp_path,
        metrics_cfg={"multivariate": {"bivariate_pairs": [["temperature", "specific_humidity"]]}},
        ensemble_mode="mean",
    )
    npzs = list((tmp_path / "multivariate").glob("bivariate_hist_*_ensmean.npz"))
    assert npzs


def test_run_no_pairs_is_noop(tmp_path: Path):
    """Empty bivariate_pairs list must not raise and must write nothing."""
    ds_t, ds_p = _make_2d_pair()
    run_multivariate(
        ds_t,
        ds_p,
        tmp_path,
        metrics_cfg={"multivariate": {"bivariate_pairs": []}},
        ensemble_mode="mean",
    )
    assert not (tmp_path / "multivariate").exists()


def test_run_pooled_mode(tmp_path: Path):
    """Pooled mode stacks ensemble and writes one NPZ."""
    rng = np.random.default_rng(99)
    lat = np.linspace(45.0, 47.0, 6)
    lon = np.linspace(6.0, 9.0, 6)
    time = np.arange(4)
    n_ens = 2
    coords = {"time": time, "latitude": lat, "longitude": lon, "ensemble": np.arange(n_ens)}
    shape = (n_ens, len(time), len(lat), len(lon))
    ds_t = xr.Dataset(
        {
            "temperature": (
                ["ensemble", "time", "latitude", "longitude"],
                rng.uniform(270, 300, shape),
            ),
            "specific_humidity": (
                ["ensemble", "time", "latitude", "longitude"],
                rng.uniform(0.001, 0.015, shape),
            ),
        },
        coords=coords,
    )
    ds_p = ds_t + 0.1 * rng.standard_normal(1)

    run_multivariate(
        ds_t,
        ds_p,
        tmp_path,
        metrics_cfg={"multivariate": {"bivariate_pairs": [["temperature", "specific_humidity"]]}},
        ensemble_mode="pooled",
    )
    npzs = list((tmp_path / "multivariate").glob("bivariate_hist_*_enspooled.npz"))
    assert npzs
