import numpy as np
import xarray as xr

from swissclim_evaluations.cli import _slice_common


# Helper to create dummy dataset
def create_dataset(lat_descending=True, lon_ascending=True):
    # Lat: 90 to -90 or -90 to 90
    lats = np.arange(90, -91, -10) if lat_descending else np.arange(-90, 91, 10)

    # Lon: 0 to 350
    lons = np.arange(0, 360, 10) if lon_ascending else np.arange(350, -10, -10)

    ds = xr.Dataset(
        {"temp": (("latitude", "longitude"), np.random.rand(len(lats), len(lons)))},
        coords={"latitude": lats, "longitude": lons},
    )
    return ds


class TestSlicingLogic:
    def test_latitude_ordering_in_config(self):
        """Test that [min, max] and [max, min] in config produce same result on descending data."""
        ds = create_dataset(lat_descending=True)

        # Case 1: [10, 30] -> Expect 30, 20, 10
        cfg1 = {"selection": {"latitudes": [10.0, 30.0]}}
        res1 = _slice_common(ds, cfg1)

        # Case 2: [30, 10] -> Expect 30, 20, 10
        cfg2 = {"selection": {"latitudes": [30.0, 10.0]}}
        res2 = _slice_common(ds, cfg2)

        expected = np.array([30, 20, 10])

        np.testing.assert_array_equal(res1.latitude.values, expected)
        np.testing.assert_array_equal(res2.latitude.values, expected)

    def test_longitude_standard_selection(self):
        """Test standard longitude selection [min, max]."""
        ds = create_dataset()
        # Select [10, 30]
        cfg = {"selection": {"longitudes": [10.0, 30.0]}}
        res = _slice_common(ds, cfg)

        expected = np.array([10, 20, 30])
        np.testing.assert_array_equal(res.longitude.values, expected)

    def test_longitude_wrap_around(self):
        """Test wrap-around longitude selection [340, 20]."""
        ds = create_dataset()
        # Select [340, 20] -> 340, 350, 0, 10, 20
        cfg = {"selection": {"longitudes": [340.0, 20.0]}}
        res = _slice_common(ds, cfg)

        expected = np.array([340, 350, 0, 10, 20])
        np.testing.assert_array_equal(res.longitude.values, expected)

    def test_longitude_wrap_around_full(self):
        """Test wrap-around that covers almost everything."""
        ds = create_dataset()
        # Select [10, 350] -> 10..350 (Standard)
        # Select [350, 10] -> 350, 0, 10 (Wrap)

        cfg = {"selection": {"longitudes": [350.0, 10.0]}}
        res = _slice_common(ds, cfg)
        expected = np.array([350, 0, 10])  # Assuming 0, 10, ..., 350 in source
        # Source is 0, 10, ..., 350
        # 350..end -> 350
        # start..10 -> 0, 10
        # Concat -> 350, 0, 10
        np.testing.assert_array_equal(res.longitude.values, expected)

    def test_no_selection(self):
        """Test that no selection returns full dataset."""
        ds = create_dataset()
        cfg = {"selection": {}}
        res = _slice_common(ds, cfg)
        xr.testing.assert_equal(ds, res)

    def test_missing_dims(self):
        """Test that slicing ignores missing dimensions gracefully."""
        ds = xr.Dataset({"temp": (("x"), [1, 2, 3])}, coords={"x": [1, 2, 3]})
        cfg = {"selection": {"latitudes": [10, 20]}}
        # Should return as is because "latitude" not in dims
        res = _slice_common(ds, cfg)
        xr.testing.assert_equal(ds, res)

    def test_single_value_lat(self):
        """Test single value latitude list."""
        ds = create_dataset()
        # [20] -> min=20, max=20. slice(20, 20).
        cfg = {"selection": {"latitudes": [20.0]}}
        res = _slice_common(ds, cfg)
        np.testing.assert_array_equal(res.latitude.values, [20])
