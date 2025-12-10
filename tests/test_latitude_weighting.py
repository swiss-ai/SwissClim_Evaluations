import numpy as np
import pytest
import xarray as xr

from swissclim_evaluations.metrics import deterministic
from swissclim_evaluations.plots import energy_spectra


@pytest.fixture
def weighted_data():
    # Create a grid with two latitudes: Equator (0) and High Lat (60)
    # cos(0) = 1.0, cos(60) = 0.5
    # Weights should be roughly 2:1 ratio.
    lat = np.array([0.0, 60.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])

    # Create datasets
    coords = {
        "init_time": [np.datetime64("2020-01-01")],
        "lead_time": [np.timedelta64(1, "h")],
        "latitude": lat,
        "longitude": lon,
    }

    # Target: All zeros
    data_zeros = np.zeros((1, 1, 2, 4))
    ds_target = xr.Dataset(
        {"var1": (("init_time", "lead_time", "latitude", "longitude"), data_zeros)}, coords=coords
    )

    # Pred 1: Error at Equator (lat index 0)
    data_eq = np.zeros((1, 1, 2, 4))
    data_eq[0, 0, 0, :] = 10.0  # Constant error 10 at equator
    ds_pred_eq = xr.Dataset(
        {"var1": (("init_time", "lead_time", "latitude", "longitude"), data_eq)}, coords=coords
    )

    # Pred 2: Error at High Lat (lat index 1)
    data_hi = np.zeros((1, 1, 2, 4))
    data_hi[0, 0, 1, :] = 10.0  # Constant error 10 at high lat
    ds_pred_hi = xr.Dataset(
        {"var1": (("init_time", "lead_time", "latitude", "longitude"), data_hi)}, coords=coords
    )

    return ds_target, ds_pred_eq, ds_pred_hi


def test_deterministic_weighting(weighted_data):
    ds_target, ds_pred_eq, ds_pred_hi = weighted_data

    # MAE should be weighted.
    # Equator error (weight ~1) should result in higher MAE than High Lat error (weight ~0.5).

    df_eq = deterministic._calculate_all_metrics(
        ds_target, ds_pred_eq, calc_relative=False, n_points=8, include=["MAE"], fss_cfg=None
    )
    mae_eq = df_eq.loc["var1", "MAE"]

    df_hi = deterministic._calculate_all_metrics(
        ds_target, ds_pred_hi, calc_relative=False, n_points=8, include=["MAE"], fss_cfg=None
    )
    mae_hi = df_hi.loc["var1", "MAE"]

    assert (
        mae_eq > mae_hi
    ), f"MAE should be weighted (Equator error {mae_eq} > High Lat error {mae_hi})"


def test_energy_spectra_weighting(weighted_data):
    ds_target, ds_pred_eq, ds_pred_hi = weighted_data

    # Spectra should be WEIGHTED.
    # Signal at Equator should have more power than signal at High Lat.

    # Need variation along longitude for spectral power
    ds_pred_eq["var1"].values[0, 0, 0, :] = [10, -10, 10, -10]
    ds_pred_hi["var1"].values[0, 0, 1, :] = [10, -10, 10, -10]

    spec_eq = energy_spectra.calculate_energy_spectra(ds_pred_eq["var1"])
    power_eq = spec_eq.mean().item()

    spec_hi = energy_spectra.calculate_energy_spectra(ds_pred_hi["var1"])
    power_hi = spec_hi.mean().item()

    assert power_eq > power_hi, "Energy spectra should be weighted"
