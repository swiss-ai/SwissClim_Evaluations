import contextlib
import logging
from collections.abc import Sequence
from typing import cast

import xarray as xr

from . import console as c, customizations as custom

# Allowed dimension names for all datasets used by the pipeline.
# NOTE: 'level' is optional (only present for genuine 3D variables) and MUST NOT
# be injected artificially. 'ensemble' is mandatory.
ALLOWED_DIMS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "level",  # optional
    "init_time",
    "lead_time",
    "ensemble",  # mandatory
)


def validate_dataset_structure(ds: xr.Dataset, name: str) -> None:
    """Strictly validate dataset structure against requirements.

    Requirements:
    1. 'ensemble' must be a dimension and a coordinate.
    2. All data variables must have 'ensemble' as a dimension.
    3. 'level' dimension must NOT be present if the dataset contains only 2D variables.
       If mixed, 2D variables must NOT have 'level' dimension.
    """
    errors = []

    # 1. Ensemble Check
    if "ensemble" not in ds.dims:
        errors.append(f"Dataset '{name}' is missing required dimension 'ensemble'.")
    if "ensemble" not in ds.coords:
        errors.append(f"Dataset '{name}' is missing required coordinate 'ensemble'.")

    # 2. Data Variables Check
    for var_name in ds.data_vars:
        var = ds[var_name]
        if "ensemble" not in var.dims:
            errors.append(f"Variable '{var_name}' in '{name}' is missing 'ensemble' dimension.")

    # 3. Level Check
    # Identify 2D vs 3D variables.
    has_level_dim = "level" in ds.dims

    # Check if any variable actually uses level
    vars_with_level = [v for v in ds.data_vars if "level" in ds[v].dims]

    if not vars_with_level and has_level_dim:
        errors.append(
            f"Dataset '{name}' has 'level' dimension but no variables use it (purely 2D dataset). "
            "'level' must not be present."
        )

    # 4. Core Dimensions Check
    required_dims = ["latitude", "longitude", "init_time", "lead_time"]
    for dim in required_dims:
        if dim not in ds.dims:
            errors.append(f"Dataset '{name}' is missing required dimension '{dim}'.")
        if dim not in ds.coords:
            errors.append(f"Dataset '{name}' is missing required coordinate '{dim}'.")

    # 5. Allowed Dimensions Check
    for dim in ds.dims:
        if dim not in ALLOWED_DIMS:
            errors.append(
                f"Dataset '{name}' has forbidden dimension '{dim}'. Allowed: {ALLOWED_DIMS}"
            )

    if errors:
        c.console.print(f"[bold red]Data Validation Failed for {name}[/bold red]")
        for error in errors:
            c.console.print(f"[red] - {error}[/red]")
        raise ValueError(f"Dataset '{name}' does not meet strict format requirements.")


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

logger = logging.getLogger(__name__)


def _chunks_match(chunks: tuple[int, ...] | None, desired: int, dim_len: int) -> bool:
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
    # xarray Dataset may not expose a unified chunks mapping; inspect first variable with the dim
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
            var_chunks = rep.chunks
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
        # Downgrade to log message to avoid noisy warnings during tests; users can enable
        # logging to see this information. Rechunking remains functional.
        logger.info(
            "Rechunking %s to policy %s. This may increase memory usage and runtime.",
            name,
            policy,
        )
        return ds.chunk(chunk_map)
    return ds


def _ensure_monotonic(ds: xr.Dataset) -> xr.Dataset:
    """Sort common coordinate dims to a consistent monotonic order.

    This avoids xarray.combine_by_coords errors when shards have different
    coordinate ordering (e.g., 'level' ascending vs descending). Sorting is
    lazy with Dask-backed arrays and preserves graph structure.
    """
    sort_prefs: dict[str, bool] = {
        "level": True,  # ascending (e.g., 50, 100, ..., 1000)
        "latitude": False,  # ERA5 typically descending 90 -> -90
        "longitude": True,  # 0..360 ascending
        "init_time": True,  # chronological
        "time": True,  # chronological
        "lead_time": True,  # increasing timedelta
    }
    for dim, asc in sort_prefs.items():
        if dim in ds.dims:
            with contextlib.suppress(Exception):
                ds = ds.sortby(dim, ascending=asc)
    return ds


def _open_many_zarr(paths: Sequence[str], variables: list[str] | None = None) -> xr.Dataset:
    """Open multiple Zarr stores and combine lazily by coordinates.

    - Preserves Dask laziness: each store remains a separate dask graph branch.
    - Requires matching variable schemas (user guarantees same format).
    - Uses combine_by_coords to concatenate/merge along labeled coords (e.g., time/init_time).
    """
    dsets: list[xr.Dataset] = []

    for p in paths:
        ds_i = xr.open_zarr(p, decode_timedelta=True)
        if variables:
            # Subselect only variables present in this shard to reduce metadata
            keep = [v for v in variables if v in ds_i.data_vars]
            if keep:
                ds_i = ds_i[keep]
        ds_i = _ensure_monotonic(ds_i)
        # Custom interaction based on the zarr file
        ds_i = custom.modify_ds(ds_i, p)
        validate_dataset_structure(ds_i, p)
        dsets.append(ds_i)

    # Harmonize 'level' coordinate across shards to a canonical sorted union to avoid
    # non-monotonic global indexes during combine_by_coords.
    try:
        levels_list = [
            ds.coords["level"].values
            for ds in dsets
            if ("level" in ds.coords and ds.level.ndim == 1 and ds.level.size > 0)
        ]
        if levels_list:
            import numpy as _np

            canon_level = _np.unique(_np.concatenate(levels_list))
            canon_level.sort()
            dsets = [(ds.reindex(level=canon_level) if "level" in ds.dims else ds) for ds in dsets]
    except Exception:
        # Best-effort; if harmonization fails, proceed and let combine raise a clearer error
        pass
    if not dsets:
        raise ValueError("No Zarr paths provided.")
    if len(dsets) == 1:
        return dsets[0]
    # Combine strictly by coords; override attrs to avoid conflicts. Join outer to allow
    # non-overlapping time ranges.
    combined: xr.Dataset = cast(
        xr.Dataset,
        xr.combine_by_coords(
            dsets,
            combine_attrs="override",
            data_vars="all",
            join="outer",
            compat="no_conflicts",
        ),
    )
    return combined


def open_ml(path: str | Sequence[str], variables: list[str] | None = None) -> xr.Dataset:
    """Open model dataset(s) from Zarr and optionally subset variables.

    Accepts a single path or a list/sequence of paths. When multiple paths are given,
    they are combined lazily by coordinates without materializing data.
    """
    if isinstance(path, (list | tuple)):
        return _open_many_zarr(list(path), variables)
    ds = xr.open_zarr(path, decode_timedelta=True)
    if variables:
        ds = ds[[v for v in variables if v in ds.data_vars]]

    ds = _ensure_monotonic(ds)

    # Custom interaction based on the zarr file
    ds = custom.modify_ds(ds, cast(str, path))
    validate_dataset_structure(ds, cast(str, path))
    return ds


def era5(path: str | Sequence[str], variables: list[str] | None = None) -> xr.Dataset:
    """Open ERA5 dataset(s) from Zarr and optionally subset variables.

    Accepts a single path or a list/sequence of paths. When multiple paths are given,
    they are combined lazily by coordinates without materializing data.
    """
    if isinstance(path, (list | tuple)):
        return _open_many_zarr(list(path), variables)
    ds = xr.open_zarr(path, decode_timedelta=True)
    if variables:
        ds = ds[[v for v in variables if v in ds.data_vars]]

    ds = _ensure_monotonic(ds)

    # Custom interaction based on the zarr file
    ds = custom.modify_ds(ds, cast(str, path))
    validate_dataset_structure(ds, cast(str, path))
    return ds


def land_sea_mask(path: str) -> xr.DataArray:
    """Load land_sea_mask field from a Zarr store without any hardcoded slicing."""
    da = xr.open_zarr(path, decode_timedelta=True)["land_sea_mask"]
    return da


def apply_ensemble_policy(
    ds: xr.Dataset,
    ensemble_members: int | list[int] | None = None,
) -> xr.Dataset:
    """Apply ensemble selection/aggregation policy.

    - ensemble_members can be:
          * None: keep all members.
          * int: select that single member (keeping 'ensemble' dimension).
          * list[int]: subset to those members (keeping 'ensemble' dimension).
    """
    if "ensemble" not in ds.dims:
        return ds

    # Normalize list vs int
    indices_list: list[int] | None
    if ensemble_members is None:
        indices_list = None
    elif isinstance(ensemble_members, list):
        indices_list = [int(i) for i in ensemble_members]
    else:
        indices_list = [int(ensemble_members)]

    if indices_list is not None:
        # subset but keep ensemble dimension
        return ds.isel(ensemble=indices_list)

    # No explicit selection: keep full ensemble
    return ds
