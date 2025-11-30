import numpy as np
import xarray as xr

from swissclim_evaluations.data import _ensure_monotonic, _open_many_zarr


def create_dummy_dataset(lat_ascending: bool = True):
    """Create a dummy dataset with latitude either ascending or descending."""
    lats = np.linspace(-90, 90, 10) if lat_ascending else np.linspace(90, -90, 10)

    ds = xr.Dataset(
        {
            "temp": (
                ("init_time", "lead_time", "ensemble", "latitude", "longitude"),
                np.random.rand(1, 1, 2, 10, 20),
            )
        },
        coords={
            "init_time": [np.datetime64("2023-01-01")],
            "lead_time": [np.timedelta64(0, "h")],
            "ensemble": [0, 1],
            "latitude": lats,
            "longitude": np.linspace(0, 360, 20),
        },
    )
    return ds


def test_ensure_monotonic_ascending_input():
    """Test that _ensure_monotonic sorts ascending latitude to descending."""
    ds = create_dummy_dataset(lat_ascending=True)

    # Verify initial state
    assert ds.latitude.values[0] == -90.0
    assert ds.latitude.values[-1] == 90.0

    ds_sorted = _ensure_monotonic(ds)

    # Verify sorted state (should be descending for latitude)
    assert ds_sorted.latitude.values[0] == 90.0
    assert ds_sorted.latitude.values[-1] == -90.0
    # Check monotonicity
    diffs = np.diff(ds_sorted.latitude.values)
    assert np.all(diffs < 0), "Latitude should be strictly descending"


def test_ensure_monotonic_descending_input():
    """Test that _ensure_monotonic keeps descending latitude as is."""
    ds = create_dummy_dataset(lat_ascending=False)

    # Verify initial state
    assert ds.latitude.values[0] == 90.0
    assert ds.latitude.values[-1] == -90.0

    ds_sorted = _ensure_monotonic(ds)

    # Verify sorted state (should remain descending)
    assert ds_sorted.latitude.values[0] == 90.0
    assert ds_sorted.latitude.values[-1] == -90.0
    diffs = np.diff(ds_sorted.latitude.values)
    assert np.all(diffs < 0), "Latitude should be strictly descending"


def test_open_many_zarr_sorting(tmp_path):
    """Test that _open_many_zarr correctly sorts mixed input datasets."""
    # Create two zarr stores, one ascending, one descending
    path_asc = tmp_path / "asc.zarr"
    path_desc = tmp_path / "desc.zarr"

    ds_asc = create_dummy_dataset(lat_ascending=True)
    ds_desc = create_dummy_dataset(lat_ascending=False)

    # Add a time dimension to distinguish them if needed, or just use them as shards
    # For this test, let's give them different init_times so they can be combined
    # ds_asc already has 2023-01-01
    ds_desc = ds_desc.assign_coords(init_time=[np.datetime64("2023-01-02")])

    ds_asc.to_zarr(path_asc)
    ds_desc.to_zarr(path_desc)

    # Open both
    ds_combined = _open_many_zarr([str(path_asc), str(path_desc)])

    # The combined dataset should have descending latitude because _ensure_monotonic
    # is called on each shard before combining.
    assert ds_combined.latitude.values[0] == 90.0
    assert ds_combined.latitude.values[-1] == -90.0
    diffs = np.diff(ds_combined.latitude.values)
    assert np.all(diffs < 0), "Latitude should be strictly descending in combined dataset"


def test_open_single_zarr_sorting(tmp_path):
    """Test that _open_many_zarr (single path) sorts ascending input."""
    path_asc = tmp_path / "single_asc.zarr"
    ds_asc = create_dummy_dataset(lat_ascending=True)
    ds_asc.to_zarr(path_asc)

    ds_opened = _open_many_zarr([str(path_asc)])

    assert ds_opened.latitude.values[0] == 90.0
    assert ds_opened.latitude.values[-1] == -90.0
    diffs = np.diff(ds_opened.latitude.values)
    assert np.all(diffs < 0)
