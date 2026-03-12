from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def create_synthetic_zarr(
    path: str,
    n_init: int = 10,
    n_lead: int = 5,
    n_ens: int = 10,
    n_lat: int = 180,
    n_lon: int = 360,
    n_levels: int = 5,
    chunk_size_dict: dict = None,
):
    """
    Creates a synthetic Zarr dataset mimicking SwissClim data structure.
    """
    if chunk_size_dict is None:
        chunk_size_dict = {
            "init_time": 1,
            "lead_time": 1,
            "latitude": 90,
            "longitude": 180,
            "ensemble": 1,
            "level": 1,
        }

    # Coordinates
    init_times = pd.date_range("2024-01-01", periods=n_init, freq="12h")
    lead_times = pd.to_timedelta(np.arange(n_lead), unit="h")
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    ensembles = np.arange(n_ens)
    levels = np.linspace(1000, 100, n_levels)

    # Variables
    # t2m: 2D surface variable (init, lead, ens, lat, lon)
    # z: 3D variable (init, lead, ens, level, lat, lon)

    # Create empty dataset structure first to avoid massive memory usage
    ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "ensemble": ensembles,
            "latitude": lats,
            "longitude": lons,
            "level": levels,
        }
    )

    # Create dummy data lazily using dask
    # Shape: (init, lead, ens, lat, lon)
    shape_2d = (n_init, n_lead, n_ens, n_lat, n_lon)
    chunks_2d = (
        chunk_size_dict.get("init_time", 1),
        chunk_size_dict.get("lead_time", 1),
        chunk_size_dict.get("ensemble", 1),
        chunk_size_dict.get("latitude", 90),
        chunk_size_dict.get("longitude", 180),
    )

    import dask.array as da

    data_2d = da.random.random(shape_2d, chunks=chunks_2d).astype("float32")

    ds["t2m"] = (("init_time", "lead_time", "ensemble", "latitude", "longitude"), data_2d)

    # 3D Variable
    # Shape: (init, lead, ens, level, lat, lon)
    shape_3d = (n_init, n_lead, n_ens, n_levels, n_lat, n_lon)
    chunks_3d = (
        chunk_size_dict.get("init_time", 1),
        chunk_size_dict.get("lead_time", 1),
        chunk_size_dict.get("ensemble", 1),
        chunk_size_dict.get("level", 1),
        chunk_size_dict.get("latitude", 90),
        chunk_size_dict.get("longitude", 180),
    )
    data_3d = da.random.random(shape_3d, chunks=chunks_3d).astype("float32")

    ds["geopotential"] = (
        ("init_time", "lead_time", "ensemble", "level", "latitude", "longitude"),
        data_3d,
    )

    # Attributes
    ds.attrs["institution"] = "CSCS"
    ds.attrs["source"] = "Synthetic Test Data"

    print(f"Writing synthetic Zarr to {path}...")
    ds.to_zarr(path, mode="w", computed=True)
    print("Done.")


if __name__ == "__main__":
    base_dir = Path("data/test_data")
    base_dir.mkdir(parents=True, exist_ok=True)

    target_path = base_dir / "synthetic_target.zarr"
    pred_path = base_dir / "synthetic_prediction.zarr"

    # Target: usually smaller or 1 member (reanalysis), but here we make it 1 member
    create_synthetic_zarr(
        str(target_path), n_init=4, n_lead=3, n_ens=1, n_lat=180, n_lon=360, n_levels=2
    )

    # Prediction: full ensemble
    create_synthetic_zarr(
        str(pred_path), n_init=4, n_lead=3, n_ens=5, n_lat=180, n_lon=360, n_levels=2
    )
