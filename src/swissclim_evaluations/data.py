import numpy as np
import xarray as xr

# Allowed dimension names for all datasets used by the pipeline.
# 'ensemble' is optional and must not be forced when missing.
ALLOWED_DIMS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "level",
    "init_time",
    "lead_time",
    "ensemble",
)


def standardize_dims(ds: xr.Dataset, dataset_name: str) -> xr.Dataset:
    """Standardize dataset dims and coords for this pipeline.

    - Normalize alias names (initial_time->init_time, number/member->ensemble, prediction_timedelta->lead_time, etc.).
    - Convert legacy 'time' -> 'init_time' and add a singleton zero 'lead_time' if absent.
    - Ensure 'lead_time' exists and is timedelta64[ns]; coerce numeric to hours.
    - Ensure spatial dims latitude/longitude exist; add singleton 'level' if missing.
    - Enforce dims set equals either {lat, lon, level, init_time, lead_time} or with optional 'ensemble'.
    """
    # Normalize alias dimension/coordinate names first
    dim_aliases = {
        "initial_time": "init_time",
        "init": "init_time",
        "prediction_timedelta": "lead_time",
        "lead": "lead_time",
        "number": "ensemble",
        "member": "ensemble",
    }
    rename_dims_map = {k: v for k, v in dim_aliases.items() if k in ds.dims}
    if rename_dims_map:
        ds = ds.rename_dims(rename_dims_map)
    rename_coords_map = {k: v for k, v in dim_aliases.items() if k in ds.coords}
    if rename_coords_map:
        ds = ds.rename(rename_coords_map)

    # Forbid any use of valid_time
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        raise ValueError(
            f"Dataset '{dataset_name}' must use ('init_time','lead_time'); 'valid_time' is not allowed."
        )

    # Convert legacy time -> init_time
    if "time" in ds.dims:
        ds = ds.rename_dims({"time": "init_time"})
        if "time" in ds.coords:
            ds = ds.rename({"time": "init_time"})

    # Ensure latitude/longitude present
    for d in ("latitude", "longitude"):
        if d not in ds.dims:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required spatial dim '{d}'."
            )

    # Ensure lead_time exists; add singleton zero if absent when init_time exists
    if "init_time" in ds.dims and "lead_time" not in ds.dims:
        zero_lead = np.array([0], dtype="timedelta64[h]").astype(
            "timedelta64[ns]"
        )
        ds = ds.expand_dims({"lead_time": zero_lead})
    # Coerce lead_time dtype to timedelta64[ns]
    if "lead_time" in ds.dims:
        lt = ds["lead_time"].values
        if not np.issubdtype(lt.dtype, np.timedelta64):
            ds = ds.assign_coords(
                lead_time=np.array(lt, dtype="timedelta64[h]").astype(
                    "timedelta64[ns]"
                )
            )

    # Add singleton dim for level if missing to satisfy IO schema
    if "level" not in ds.dims:
        ds = ds.expand_dims({"level": np.array([0], dtype=np.int32)})

    # Enforce allowed dims only
    bad_dims = [d for d in ds.dims if d not in ALLOWED_DIMS]
    if bad_dims:
        raise ValueError(
            f"Dataset '{dataset_name}' has unsupported dims {bad_dims}. "
            f"Only {ALLOWED_DIMS} are allowed. Please preprocess your data accordingly."
        )

    # Ensure the set of dims matches one of the allowed exact schemas
    required_no_ens = {
        "latitude",
        "longitude",
        "level",
        "init_time",
        "lead_time",
    }
    required_with_ens = required_no_ens | {"ensemble"}
    dims_set = set(ds.dims)
    if dims_set not in (required_no_ens, required_with_ens):
        raise ValueError(
            f"Dataset '{dataset_name}' dims must be exactly {tuple(sorted(required_no_ens))} "
            f"or {tuple(sorted(required_with_ens))}, got {tuple(ds.dims)}."
        )
    return ds


def open_ml(path: str, variables: list[str] | None = None) -> xr.Dataset:
    """Open model dataset from Zarr and optionally subset variables."""
    ds = xr.open_zarr(path, decode_timedelta=True)
    if variables:
        ds = ds[[v for v in variables if v in ds.data_vars]]
    return ds


def era5(path: str, variables: list[str] | None = None) -> xr.Dataset:
    """Open ERA5 dataset from Zarr and optionally subset variables."""
    ds = xr.open_zarr(path, decode_timedelta=True)
    if variables:
        ds = ds[[v for v in variables if v in ds.data_vars]]
    return ds


def land_sea_mask(path: str) -> xr.DataArray:
    """Load land_sea_mask field from a Zarr store without any hardcoded slicing."""
    da = xr.open_zarr(path, decode_timedelta=True)["land_sea_mask"]
    return da
