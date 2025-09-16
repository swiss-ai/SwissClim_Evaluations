from __future__ import annotations

import numpy as np
import xarray as xr

VARS_2D = ["2m_temperature", "10m_u_component_of_wind"]


def make_synthetic_datasets(
    with_ensemble: bool = True,
    time: int = 2,
    lat: int = 3,
    lon: int = 4,
    ensemble: int = 5,
) -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(0)
    coords = {
        # Build time coordinate as hourly timedeltas added to a start datetime
        "time": np.datetime64("2021-01-01T00", "h")
        + np.arange(time, dtype="timedelta64[h]"),
        "latitude": np.linspace(45.0, 47.0, lat),
        "longitude": np.linspace(6.0, 9.0, lon),
    }
    targets = xr.Dataset(
        {
            v: (
                ["time", "latitude", "longitude"],
                rng.standard_normal((time, lat, lon)),
            )
            for v in VARS_2D
        },
        coords=coords,
    )

    if with_ensemble:
        coords_ml = coords | {"ensemble": np.arange(ensemble)}
        predictions = xr.Dataset(
            {
                v: (
                    ["time", "latitude", "longitude", "ensemble"],
                    rng.standard_normal((time, lat, lon, ensemble))
                    + targets[v].values[..., None],
                )
                for v in VARS_2D
            },
            coords=coords_ml,
        )
    else:
        predictions = xr.Dataset(
            {
                v: (
                    ["time", "latitude", "longitude"],
                    rng.standard_normal((time, lat, lon)) + targets[v].values,
                )
                for v in VARS_2D
            },
            coords=coords,
        )

    return targets, predictions
