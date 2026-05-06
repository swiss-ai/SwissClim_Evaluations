"""Tests for the spherical harmonic energy spectrum implementation."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _global_da(n_lat: int = 16, n_lon: int = 32, seed: int = 0) -> xr.DataArray:
    """Return a synthetic global DataArray with north-to-south latitudes."""
    rng = np.random.default_rng(seed)
    lats = np.linspace(90.0, -90.0 + 180.0 / n_lat, n_lat)  # north → south (DH)
    lons = np.linspace(0.0, 360.0 - 360.0 / n_lon, n_lon)
    return xr.DataArray(
        rng.standard_normal((n_lat, n_lon)).astype(np.float64),
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="test_var",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sph_harm_spectra_output_shape_and_coords():
    """Output shape and coordinate names must match the plan specification."""
    from swissclim_evaluations.plots.energy_spectra import calculate_sph_harm_spectra

    da = _global_da(n_lat=16, n_lon=32)
    # lmax = min(16//2 - 1, 32//4 - 1) = min(7, 7) = 7 → output degrees 1..7
    result = calculate_sph_harm_spectra(da)

    assert result.dims == ("wavenumber",), f"Unexpected dims: {result.dims}"
    assert result.sizes["wavenumber"] == 7, f"Expected 7 degrees, got {result.sizes['wavenumber']}"
    assert "wavenumber" in result.coords, "Missing 'wavenumber' coord"
    assert "wavelength" in result.coords, "Missing 'wavelength' coord"
    assert "sh_degree" in result.coords, "Missing 'sh_degree' coord"


def test_sph_harm_spectra_nonnegative():
    """Power spectrum values must be non-negative."""
    from swissclim_evaluations.plots.energy_spectra import calculate_sph_harm_spectra

    result = calculate_sph_harm_spectra(_global_da())
    assert np.all(result.values >= 0.0), "Power spectrum contains negative values"


def test_sph_harm_spectra_sh_degree_values():
    """sh_degree coordinate must hold integer degrees 1..lmax."""
    from swissclim_evaluations.plots.energy_spectra import calculate_sph_harm_spectra

    result = calculate_sph_harm_spectra(_global_da(n_lat=16, n_lon=32))
    np.testing.assert_array_equal(result["sh_degree"].values, np.arange(1, 8))


def test_sph_harm_spectra_wavenumber_in_cycles_per_km():
    """wavenumber coord must be l / EARTH_CIRCUMFERENCE_KM (effective cycles/km)."""
    from swissclim_evaluations.plots.energy_spectra import (
        EARTH_CIRCUMFERENCE_KM,
        calculate_sph_harm_spectra,
    )

    result = calculate_sph_harm_spectra(_global_da(n_lat=16, n_lon=32))
    expected_k = np.arange(1, 8) / EARTH_CIRCUMFERENCE_KM
    np.testing.assert_allclose(result["wavenumber"].values, expected_k, rtol=1e-12)


def test_sph_harm_spectra_wavelength_consistency():
    """wavelength must equal EARTH_CIRCUMFERENCE_KM / sh_degree."""
    from swissclim_evaluations.plots.energy_spectra import (
        EARTH_CIRCUMFERENCE_KM,
        calculate_sph_harm_spectra,
    )

    result = calculate_sph_harm_spectra(_global_da(n_lat=16, n_lon=32))
    expected_wl = EARTH_CIRCUMFERENCE_KM / result["sh_degree"].values
    np.testing.assert_allclose(result["wavelength"].values, expected_wl, rtol=1e-12)


def test_sph_harm_spectra_extra_dims_preserved():
    """Extra dimensions (e.g. time) must be preserved in the output."""
    from swissclim_evaluations.plots.energy_spectra import calculate_sph_harm_spectra

    n_time = 3
    da_base = _global_da(n_lat=16, n_lon=32)
    da_3d = xr.concat([da_base + float(i) for i in range(n_time)], dim="time")
    da_3d["time"] = np.arange(n_time)

    result = calculate_sph_harm_spectra(da_3d)

    assert "time" in result.dims
    assert "wavenumber" in result.dims
    assert result.sizes["time"] == n_time
    assert result.sizes["wavenumber"] == 7


def test_sph_harm_spectra_average_dims():
    """average_dims should reduce the specified dimension after spectral computation."""
    from swissclim_evaluations.plots.energy_spectra import calculate_sph_harm_spectra

    n_ens = 4
    da_base = _global_da(n_lat=16, n_lon=32)
    da_ens = xr.concat([da_base + float(i) for i in range(n_ens)], dim="ensemble")
    da_ens["ensemble"] = np.arange(n_ens)

    result = calculate_sph_harm_spectra(da_ens, average_dims=["ensemble"])

    assert "ensemble" not in result.dims
    assert result.dims == ("wavenumber",)


def test_sph_harm_spectra_banded_lsd_compatible():
    """_compute_banded_lsd_da must work on SH spectra without error."""
    from swissclim_evaluations.plots.energy_spectra import (
        _compute_banded_lsd_da,
        calculate_sph_harm_spectra,
    )

    # lmax = min(64//2-1, 128//4-1) = 31; l=8 → ~5004 km, l=40 → ~1001 km
    da = _global_da(n_lat=64, n_lon=128)
    spec = calculate_sph_harm_spectra(da)

    result = _compute_banded_lsd_da(spec, spec)

    assert "band" in result.dims
    for val in result.values:
        if not np.isnan(val):
            assert abs(val) < 1e-6, f"LSD(spec, spec) should be 0, got {val}"


def test_sph_harm_spectra_too_small_grid_raises():
    """Grids smaller than 4 lat × 8 lon must raise ValueError."""
    from swissclim_evaluations.plots.energy_spectra import calculate_sph_harm_spectra

    da_small = xr.DataArray(
        np.ones((2, 4)),
        dims=["latitude", "longitude"],
        coords={"latitude": [45.0, 47.0], "longitude": [6.0, 7.0, 8.0, 9.0]},
    )
    with pytest.raises(ValueError, match="too coarse"):
        calculate_sph_harm_spectra(da_small)
