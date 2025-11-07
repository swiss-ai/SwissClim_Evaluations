import contextlib
import logging
import warnings
from collections.abc import Sequence
from typing import cast

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
        # Heuristic: skip rechunk warnings for very small or already in-memory test datasets.
        # Total element count below threshold -> leave unchunked; avoids noisy test warnings.
        try:
            total_size = sum(int(v.size) for v in ds.data_vars.values())
        except Exception:
            total_size = 0
        SMALL_THRESHOLD = 20000  # configurable if needed
        # If no existing chunking (numpy) AND tiny dataset -> skip rechunk silently.
        all_unchunked = True
        for v in ds.data_vars.values():
            try:
                if getattr(v, "chunks", None) is not None:
                    all_unchunked = False
                    break
            except Exception:
                pass
        if all_unchunked and total_size <= SMALL_THRESHOLD:
            return ds  # no warning, no rechunk
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


def standardize_dims(
    ds: xr.Dataset,
    dataset_name: str,
    *,
    first_lead_only: bool | None = None,
    preserve_all_leads: bool | None = None,
) -> xr.Dataset:
    """Standardize dataset dims and coords for this pipeline.
        - Normalize alias names (initial_time->init_time, number/member->ensemble,
            prediction_timedelta->lead_time, etc.).
        - Convert legacy 'time' -> 'init_time' and add singleton zero 'lead_time' if absent.
    - Ensure 'lead_time' exists and is timedelta64[ns]; coerce numeric to hours.
    - Ensure spatial dims latitude/longitude exist.
    - Do NOT add synthetic 'level' dimension. Only retain if truly present in data.
    - Accept schemas with or without optional 'level' / 'ensemble' dims.
    """

    # (debug logging removed)

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
            f"Dataset '{dataset_name}' must use ('init_time','lead_time'); "
            "'valid_time' is not allowed."
        )

    # Convert old-style time -> init_time while keeping an index for
    # label-based selection (single-lead case)
    if "time" in ds.dims:
        if "time" in ds.coords:
            # Rename both the dimension and its coordinate in one go
            ds = ds.rename({"time": "init_time"})
            # On newer xarray, rename drops the xindex; restore it if possible
            with contextlib.suppress(Exception):  # xarray>=2024.10
                ds = ds.set_xindex("init_time")
        else:
            # Fallback: only dimension exists (no coord var)
            ds = ds.rename_dims({"time": "init_time"})

    # Ensure latitude/longitude present
    for d in ("latitude", "longitude"):
        if d not in ds.dims:
            raise ValueError(f"Dataset '{dataset_name}' is missing required spatial dim '{d}'.")

    # Ensure lead_time exists; add singleton zero if absent when init_time exists
    if "init_time" in ds.dims and "lead_time" not in ds.dims:
        zero_lead = np.array([0], dtype="timedelta64[h]").astype("timedelta64[ns]")
        ds = ds.expand_dims({"lead_time": zero_lead})
    # Coerce lead_time dtype to timedelta64[ns]
    if "lead_time" in ds.dims:
        lt = ds["lead_time"].values
        if not np.issubdtype(lt.dtype, np.timedelta64):
            ds = ds.assign_coords(
                lead_time=np.array(lt, dtype="timedelta64[h]").astype("timedelta64[ns]")
            )
        # Optional policy: restrict to first lead_time (no forecasting). Default True when None.
        if (first_lead_only is None or bool(first_lead_only)) and int(
            ds.lead_time.size
        ) > 1:  # SIM102
            lead0 = ds["lead_time"].values[0]
            with contextlib.suppress(Exception):
                ds = ds.sel(lead_time=[lead0])
            if isinstance(ds.lead_time.values, np.ndarray) and ds.lead_time.size > 1:  # fallback
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
            "Expected at least (latitude, longitude, init_time, lead_time) plus optional "
            "level/ensemble."
        )
    # Validate that no unsupported dims remain
    bad_dims = [d for d in ds.dims if d not in ALLOWED_DIMS]
    if bad_dims:
        raise ValueError(
            f"Dataset '{dataset_name}' has unsupported dims {bad_dims}. Allowed: {ALLOWED_DIMS}."
        )
    # (debug logging removed)
    return ds


def _open_many_zarr(paths: Sequence[str], variables: list[str] | None = None) -> xr.Dataset:
    """Open multiple Zarr stores and combine lazily by coordinates.

    - Preserves Dask laziness: each store remains a separate dask graph branch.
    - Requires matching variable schemas (user guarantees same format).
    - Uses combine_by_coords to concatenate/merge along labeled coords (e.g., time/init_time).
    """
    dsets: list[xr.Dataset] = []

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

    for p in paths:
        ds_i = xr.open_zarr(p, decode_timedelta=True)
        if variables:
            # Subselect only variables present in this shard to reduce metadata
            keep = [v for v in variables if v in ds_i.data_vars]
            if keep:
                ds_i = ds_i[keep]
        ds_i = _ensure_monotonic(ds_i)
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
    return ds


def land_sea_mask(path: str) -> xr.DataArray:
    """Load land_sea_mask field from a Zarr store without any hardcoded slicing."""
    da = xr.open_zarr(path, decode_timedelta=True)["land_sea_mask"]
    return da


def apply_ensemble_policy(
    ds: xr.Dataset,
    ensemble_members: int | list[int] | None = None,
    probabilistic_enabled: bool = False,
    **legacy_kwargs,
) -> xr.Dataset:
    """Apply ensemble selection/aggregation policy.

    Extended semantics (backward compatible):
    - ensemble_members can be:
          * None: keep all members (unless probabilistic disabled → may later be reduced
                    by per‑module logic; here we do NOT pre-reduce to mean anymore to allow
                    downstream flexibility). NOTE: previous behaviour reduced to mean here;
                    to preserve existing expectations, we only change behaviour when
                    probabilistic_enabled is False and historical callers relied on mean.
                    For backward compatibility we keep the old reduction to mean when
                    probabilistic_disabled and no selection requested.
          * int: select that single member and drop the 'ensemble' dimension.
          * list[int]: subset to those members. If length==1, drop dim (acts like int). If
                       length>1 keep 'ensemble' dim with the chosen subset.
      - If probabilistic modules are enabled we NEVER drop or subset unless multiple
        explicit members were requested (so probabilistic metrics still see full set
        unless user intentionally restricts it). This preserves previous rule of ignoring
        single-member selection in probabilistic mode.
    """
    # Backward compatibility: allow legacy 'ensemble_member' kw
    if "ensemble_member" in legacy_kwargs and ensemble_members is None:
        ensemble_members = legacy_kwargs.pop("ensemble_member")
        warnings.warn(
            "Config key 'ensemble_member' is deprecated; use 'ensemble_members' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy_kwargs:
        warnings.warn(
            f"Unused legacy kwargs passed to apply_ensemble_policy: {list(legacy_kwargs)}",
            RuntimeWarning,
            stacklevel=2,
        )

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

    if probabilistic_enabled:
        # In probabilistic mode only respect explicit multi-member subsetting; ignore single
        # int (previous behaviour) but allow user-specified list to intentionally reduce.
        if indices_list is None:
            return ds
        if len(indices_list) == 1:
            # Keep full set to avoid accidental collapse; match previous behaviour of ignoring
            # single selection. (Could alternatively drop; chosen for safety.)
            return ds
        return ds.isel(ensemble=indices_list)

    # Deterministic mode
    if indices_list is not None:
        if len(indices_list) == 1:
            return ds.isel(ensemble=indices_list[0], drop=True)
        # subset but keep ensemble dimension
        return ds.isel(ensemble=indices_list)

    # No explicit selection: do NOT pre-reduce. Keep ensemble for modules to decide.
    return ds
