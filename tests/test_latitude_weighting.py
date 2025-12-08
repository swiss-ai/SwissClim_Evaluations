import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swissclim_evaluations.aggregations import latitude_weights
from swissclim_evaluations.metrics import deterministic, ets, probabilistic, vertical_profiles
from swissclim_evaluations.plots import energy_spectra


@pytest.fixture
def weighted_data():
    """
    Creates a dataset with 3 latitude points: -90, 0, 90.
    Weights will be roughly [low, high, low].
    """
    lats = np.array([-90.0, 0.0, 90.0])
    lons = np.array([0.0, 90.0, 180.0, 270.0])

    # Create coordinates
    coords = {
        "latitude": lats,
        "longitude": lons,
        "init_time": pd.to_datetime(["2023-01-01"]),
        "lead_time": pd.to_timedelta([0], unit="h"),
        "level": [500],
    }

    # Target: all zeros
    data_target = np.zeros((1, 1, 3, 4))  # init, lead, lat, lon
    ds_target = xr.Dataset(
        {"var1": (("init_time", "lead_time", "latitude", "longitude"), data_target)},
        coords={
            k: v
            for k, v in coords.items()
            if k in ["latitude", "longitude", "init_time", "lead_time"]
        },
    )
    # Add level coord for 3D var simulation
    ds_target_3d = ds_target.expand_dims(level=[500])

    # Prediction 1: Error at Equator (high weight)
    data_pred_eq = np.zeros_like(data_target)
    data_pred_eq[:, :, 1, :] = 10.0  # Error of 10 at equator
    ds_pred_eq = xr.Dataset(
        {"var1": (("init_time", "lead_time", "latitude", "longitude"), data_pred_eq)},
        coords=ds_target.coords,
    )

    # Prediction 2: Error at Poles (low weight)
    data_pred_pole = np.zeros_like(data_target)
    data_pred_pole[:, :, 0, :] = 10.0  # Error of 10 at South Pole
    data_pred_pole[:, :, 2, :] = 10.0  # Error of 10 at North Pole
    ds_pred_pole = xr.Dataset(
        {"var1": (("init_time", "lead_time", "latitude", "longitude"), data_pred_pole)},
        coords=ds_target.coords,
    )

    return ds_target, ds_pred_eq, ds_pred_pole, ds_target_3d


def test_deterministic_weighting(weighted_data):
    ds_target, ds_pred_eq, ds_pred_pole, _ = weighted_data

    # Calculate unweighted metrics
    df_unweighted_eq = deterministic._calculate_all_metrics(
        ds_target,
        ds_pred_eq,
        calc_relative=False,
        n_points=12,
        include=["MAE"],
        latitude_weighting=False,
    )
    mae_unweighted_eq = df_unweighted_eq.loc["var1", "MAE"]

    df_unweighted_pole = deterministic._calculate_all_metrics(
        ds_target,
        ds_pred_pole,
        calc_relative=False,
        n_points=12,
        include=["MAE"],
        latitude_weighting=False,
    )
    mae_unweighted_pole = df_unweighted_pole.loc["var1", "MAE"]

    # Verify unweighted values (simple mean)
    assert np.isclose(mae_unweighted_eq, 10 / 3, atol=0.1)
    assert np.isclose(mae_unweighted_pole, 20 / 3, atol=0.1)

    # Calculate weighted metrics
    df_weighted_eq = deterministic._calculate_all_metrics(
        ds_target,
        ds_pred_eq,
        calc_relative=False,
        n_points=12,
        include=["MAE"],
        latitude_weighting=True,
    )
    mae_weighted_eq = df_weighted_eq.loc["var1", "MAE"]

    df_weighted_pole = deterministic._calculate_all_metrics(
        ds_target,
        ds_pred_pole,
        calc_relative=False,
        n_points=12,
        include=["MAE"],
        latitude_weighting=True,
    )
    mae_weighted_pole = df_weighted_pole.loc["var1", "MAE"]

    # Verify weighting effect: Equator (high weight) -> higher error; Poles (low weight) -> lower
    # error
    assert mae_weighted_eq > mae_unweighted_eq
    assert mae_weighted_pole < mae_unweighted_pole

    # Verify approximate weighted values
    assert np.isclose(mae_weighted_eq, 7.06, atol=0.5)
    assert np.isclose(mae_weighted_pole, 2.93, atol=0.5)


def test_ets_weighting(weighted_data):
    ds_target, ds_pred_eq, ds_pred_pole, _ = weighted_data

    # Setup: Target has event at Equator (20.0). Threshold 90 (value 20).
    ds_target_mixed = ds_target.copy()
    ds_target_mixed["var1"].values[:, :, 1, :] = 20.0

    # Case 1: Perfect Hit at Equator (High weight)
    ds_pred_eq_hit = ds_pred_eq.copy()
    ds_pred_eq_hit["var1"].values[:, :, 1, :] = 20.0

    # Calculate ETS (Weighted vs Unweighted)
    # Note: For perfect forecast, ETS should be 1.0 regardless of weighting
    df_unweighted = ets._calculate_ets_for_thresholds(
        ds_target_mixed, ds_pred_eq_hit, thresholds=[90], latitude_weighting=False
    )

    df_weighted = ets._calculate_ets_for_thresholds(
        ds_target_mixed, ds_pred_eq_hit, thresholds=[90], latitude_weighting=True
    )
    assert np.isclose(df_unweighted.iloc[0, 0], 1.0)
    assert np.isclose(df_weighted.iloc[0, 0], 1.0)

    # Case 2: Miss at Equator (High weight), False Alarm at Pole (Low weight)
    ds_pred_miss = ds_pred_pole.copy()
    ds_pred_miss["var1"].values[:, :, 0, :] = 20.0

    df_unweighted_miss = ets._calculate_ets_for_thresholds(
        ds_target_mixed, ds_pred_miss, thresholds=[90], latitude_weighting=False
    )

    df_weighted_miss = ets._calculate_ets_for_thresholds(
        ds_target_mixed, ds_pred_miss, thresholds=[90], latitude_weighting=True
    )

    # Verify that weighting changes the ETS score for imperfect forecasts
    assert df_unweighted_miss.iloc[0, 0] != df_weighted_miss.iloc[0, 0]


def test_vertical_profiles_weighting(weighted_data):
    ds_target, ds_pred_eq, ds_pred_pole, ds_target_3d = weighted_data

    # Setup: Target with variation along longitude to ensure non-zero NMAE denominator
    ds_target_3d["var1"].values[:] = 0.0
    ds_target_3d["var1"].values[:, :, :, :, 0] = 100.0

    # Expand preds to 3D
    ds_pred_eq_3d = ds_pred_eq.expand_dims(level=[500])
    ds_pred_pole_3d = ds_pred_pole.expand_dims(level=[500])

    weights = latitude_weights(ds_target.latitude)

    # Calculate NMAE for Equator case (High weight)
    nmae_unweighted_eq = vertical_profiles._compute_nmae(
        ds_target_3d["var1"], ds_pred_eq_3d["var1"], slice(None), [500], weights=None
    )
    nmae_weighted_eq = vertical_profiles._compute_nmae(
        ds_target_3d["var1"], ds_pred_eq_3d["var1"], slice(None), [500], weights=weights
    )

    # Calculate NMAE for Pole case (Low weight)
    nmae_unweighted_pole = vertical_profiles._compute_nmae(
        ds_target_3d["var1"], ds_pred_pole_3d["var1"], slice(None), [500], weights=None
    )
    nmae_weighted_pole = vertical_profiles._compute_nmae(
        ds_target_3d["var1"], ds_pred_pole_3d["var1"], slice(None), [500], weights=weights
    )

    # Verify weighting effect
    assert nmae_weighted_eq.item() > nmae_unweighted_eq.item()
    assert nmae_weighted_pole.item() < nmae_unweighted_pole.item()


def test_energy_spectra_weighting(weighted_data):
    ds_target, ds_pred_eq, _, _ = weighted_data

    # Setup: Signal only at Equator (High weight)
    da = ds_pred_eq["var1"].copy()
    da.values[:, :, 1, :] = [10, -10, 10, -10]

    # Calculate spectra
    spec_unweighted = energy_spectra.calculate_energy_spectra(da, latitude_weighting=False)

    spec_weighted = energy_spectra.calculate_energy_spectra(da, latitude_weighting=True)

    # Verify total energy is higher when weighted (since signal is at high-weight equator)
    energy_unweighted = spec_unweighted.sum().item()
    energy_weighted = spec_weighted.sum().item()

    assert energy_weighted > energy_unweighted


def test_probabilistic_weighting(weighted_data):
    ds_target, ds_pred_eq, ds_pred_pole, _ = weighted_data

    # Setup: Create ensemble predictions with spread/error
    # Member 0: Matches target (0)
    # Member 1: Has error (scaled from input)

    # Equator case (High weight)
    ds_pred_eq_ens = ds_pred_eq.expand_dims(ensemble=[0, 1]).copy(deep=True)
    vals = ds_pred_eq_ens["var1"].values
    vals.flags.writeable = True
    vals[0, ...] = ds_pred_eq["var1"].values
    vals[1, ...] = ds_pred_eq["var1"].values * 3
    ds_pred_eq_ens["var1"].values = vals

    # Pole case (Low weight)
    ds_pred_pole_ens = ds_pred_pole.expand_dims(ensemble=[0, 1]).copy(deep=True)
    vals_pole = ds_pred_pole_ens["var1"].values
    vals_pole.flags.writeable = True
    vals_pole[0, ...] = ds_pred_pole["var1"].values
    vals_pole[1, ...] = ds_pred_pole["var1"].values * 3
    ds_pred_pole_ens["var1"].values = vals_pole

    weights = latitude_weights(ds_target.latitude)

    # Calculate CRPS for Equator case
    crps_da_eq = probabilistic.crps_ensemble(ds_target["var1"], ds_pred_eq_ens["var1"])
    crps_unweighted_eq = crps_da_eq.mean().item()
    crps_weighted_eq = crps_da_eq.weighted(weights).mean().item()

    # Calculate CRPS for Pole case
    crps_da_pole = probabilistic.crps_ensemble(ds_target["var1"], ds_pred_pole_ens["var1"])
    crps_unweighted_pole = crps_da_pole.mean().item()
    crps_weighted_pole = crps_da_pole.weighted(weights).mean().item()

    # Verify weighting effect
    assert crps_weighted_eq > crps_unweighted_eq
    assert crps_weighted_pole < crps_unweighted_pole
