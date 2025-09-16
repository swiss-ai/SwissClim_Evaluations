import warnings

import numpy as np
import xarray as xr

# Allowed dimension names for all datasets used by the pipeline.
# NOTE: 'level' is optional (only present for genuine 3D variables) and MUST NOT
# be injected artificially. Earlier versions added a singleton level which led
# to downstream misclassification of surface variables. We now treat absence of
# 'level' as a true 2D field. Likewise 'ensemble' is optional.
ALLOWED_DIMS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "level",  # optional
    "init_time",
    "lead_time",
    "ensemble",  # optional
)

# Default chunking policy used across the repository. Values:
#  - 1: chunk size of 1 for that dimension
#  - -1: a single chunk spanning the entire dimension
DESIRED_CHUNKS: dict[str, int] = {
    "init_time": 1,
    "lead_time": 1,
    "level": 1,
    "ensemble": -1,
    "latitude": -1,
    "longitude": -1,
}


def _chunks_match(
    chunks: tuple[int, ...] | None, desired: int, dim_len: int
) -> bool:
    """Return True if an existing chunk pattern matches the desired size.

    Only supports desired sizes of 1 (all chunks size 1) or -1 (single full chunk).
    If chunks is None (un-chunked/numpy), returns False.
    """
    if chunks is None:
        return False
    if desired == 1:
        return all(c == 1 for c in chunks)
    if desired == -1:
        return len(chunks) == 1 and chunks[0] == dim_len
    # For other values (not used in policy), be conservative and require exact matches
    return all(c == desired for c in chunks[:-1]) and (
        chunks[-1] == desired or chunks[-1] <= desired
    )


def enforce_chunking(
    ds: xr.Dataset,
    desired_policy: dict[str, int] | None = None,
    dataset_name: str | None = None,
) -> xr.Dataset:
    """Ensure the dataset uses the desired Dask chunking pattern.

    If existing chunks differ (or dataset is not chunked), rechunk and warn that this may
    increase memory usage and runtime.
    """
    policy = desired_policy or DESIRED_CHUNKS
    # Build chunk map only for dimensions present
    chunk_map: dict[str, int] = {}
    for dim, size in policy.items():
        if dim in ds.dims:
            # Interpret -1 as full dimension length for xarray.chunk()
            chunk_map[dim] = ds.sizes[dim] if size == -1 else size

    # Determine if rechunking is necessary by inspecting existing chunks across variables
    needs_rechunk = False
    # xarray Dataset may not expose a unified chunks mapping; inspect the first variable that has the dim
    for dim, desired in policy.items():
        if dim not in ds.dims:
            continue
        # Find a representative DataArray that contains this dim
        rep = None
        for v in ds.data_vars:
            if dim in ds[v].dims:
                rep = ds[v]
                break
        if rep is None:
            continue
        try:
            var_chunks = rep.chunks  # type: ignore[attr-defined]
            # DataArray.chunks is a tuple of tuples aligned with .dims
            if var_chunks is None:
                needs_rechunk = True
                break
            dim_index = rep.dims.index(dim)
            dim_chunks: tuple[int, ...] | None = var_chunks[dim_index]
        except Exception:
            dim_chunks = None
        if not _chunks_match(dim_chunks, desired, ds.sizes[dim]):
            needs_rechunk = True
            break

    if needs_rechunk:
        name = dataset_name or "dataset"
        warnings.warn(
            f"Rechunking {name} to policy {policy}. This may increase memory usage and runtime.",
            RuntimeWarning,
        )
        return ds.chunk(chunk_map)
    return ds


def standardize_dims(
    ds: xr.Dataset, dataset_name: str, *, first_lead_only: bool | None = None
) -> xr.Dataset:
    """Standardize dataset dims and coords for this pipeline.

    - Normalize alias names (initial_time->init_time, number/member->ensemble, prediction_timedelta->lead_time, etc.).
    - Convert legacy 'time' -> 'init_time' and add a singleton zero 'lead_time' if absent.
    - Ensure 'lead_time' exists and is timedelta64[ns]; coerce numeric to hours.
    - Ensure spatial dims latitude/longitude exist.
    - Do NOT add synthetic 'level' dimension. Only retain if truly present in data.
    - Accept schemas with or without optional 'level' / 'ensemble' dims.
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

    # Convert legacy time -> init_time while keeping an index for label-based selection
    if "time" in ds.dims:
        if "time" in ds.coords:
            # Rename both the dimension and its coordinate in one go
            ds = ds.rename({"time": "init_time"})
            # On newer xarray, rename drops the xindex; restore it if possible
            try:  # xarray>=2024.10
                ds = ds.set_xindex("init_time")
            except Exception:
                # Older versions or if already indexed; safe to ignore
                pass
        else:
            # Fallback: only dimension exists (no coord var)
            ds = ds.rename_dims({"time": "init_time"})

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
        # Optional policy: restrict to first lead_time (no forecasting)
        # If first_lead_only is None, we apply it by default (True) to ensure consistency
        if first_lead_only is None or bool(first_lead_only):
            if int(ds.lead_time.size) > 1:
                # Keep coordinate label and dimension length 1
                lead0 = ds["lead_time"].values[0]
                try:
                    ds = ds.sel(lead_time=[lead0])
                except Exception:
                    ds = ds.isel(lead_time=0, drop=False)

    # If the dataset already has 'level' keep it; absence means purely 2D vars.

    # Enforce allowed dims only
    bad_dims = [d for d in ds.dims if d not in ALLOWED_DIMS]
    if bad_dims:
        raise ValueError(
            f"Dataset '{dataset_name}' has unsupported dims {bad_dims}. "
            f"Only {ALLOWED_DIMS} are allowed. Please preprocess your data accordingly."
        )

    # Relax schema: Required core dims
    core_required = {"latitude", "longitude", "init_time", "lead_time"}
    missing_core = [d for d in core_required if d not in ds.dims]
    if missing_core:
        raise ValueError(
            f"Dataset '{dataset_name}' missing required dims {missing_core}. "
            "Expected at least (latitude, longitude, init_time, lead_time) plus optional level/ensemble."
        )
    # Validate that no unsupported dims remain
    bad_dims = [d for d in ds.dims if d not in ALLOWED_DIMS]
    if bad_dims:
        raise ValueError(
            f"Dataset '{dataset_name}' has unsupported dims {bad_dims}. Allowed: {ALLOWED_DIMS}."
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


def apply_ensemble_policy(
    ds: xr.Dataset,
    ensemble_member: int | None,
    probabilistic_enabled: bool,
) -> xr.Dataset:
    """Apply ensemble selection/aggregation policy.

    Rules:
      - If no 'ensemble' dim: return unchanged.
      - If probabilistic modules are enabled: keep full ensemble regardless of
        `ensemble_member` (probabilistic metrics will use it).
      - Else if `ensemble_member` is provided: select that member and drop dim.
      - Else (probabilistic disabled and no member specified): use mean over ensemble.
    """
    if "ensemble" not in ds.dims:
        return ds
    # Probabilistic path: always keep full ensemble, ignore requested member.
    if probabilistic_enabled:
        return ds
    # Deterministic path with explicit member selection
    if ensemble_member is not None:
        return ds.isel(ensemble=int(ensemble_member), drop=True)
    # Deterministic path without member: use ensemble mean
    return ds.mean(dim="ensemble", keep_attrs=True)
