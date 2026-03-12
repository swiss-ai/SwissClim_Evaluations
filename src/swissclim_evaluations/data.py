import contextlib
import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import xarray as xr

from . import console as c, customizations as custom

# Allowed dimension names for all datasets used by the pipeline.
# NOTE: 'level' is optional (only present for genuine 3D variables) and MUST NOT
# be injected artificially. 'ensemble' is mandatory (will be auto-created if missing).
ALLOWED_DIMS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "level",  # optional
    "init_time",
    "lead_time",
    "ensemble",  # mandatory
)


def ensure_ensemble_dim(ds: xr.Dataset) -> xr.Dataset:
    """Ensure 'ensemble' dimension exists.

    If missing, inject a dummy ensemble dimension of size 1 with coordinate value 0.
    This allows deterministic datasets to pass validation and be processed uniformly.
    """
    if "ensemble" not in ds.dims:
        # Expand dims injects the dimension and coordinate
        ds = ds.expand_dims(ensemble=[0])
    return ds


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
    # We allow either (init_time, lead_time) OR (time) to support raw WeatherBench format
    has_time = "time" in ds.dims

    has_init_lead = "init_time" in ds.dims and "lead_time" in ds.dims

    if not has_time and not has_init_lead:
        errors.append(
            f"Dataset '{name}' is missing required dimensions: "
            "either ('init_time', 'lead_time') or ('time')."
        )

    required_dims = ["latitude", "longitude"]
    if has_init_lead:
        required_dims.extend(["init_time", "lead_time"])
    elif has_time:
        required_dims.append("time")

    for dim in required_dims:
        if dim not in ds.dims:
            errors.append(f"Dataset '{name}' is missing required dimension '{dim}'.")
        if dim not in ds.coords:
            errors.append(f"Dataset '{name}' is missing required coordinate '{dim}'.")

    # 5. Allowed Dimensions Check
    allowed = set(ALLOWED_DIMS)
    if has_time:
        allowed.add("time")

    for dim in ds.dims:
        if dim not in allowed:
            errors.append(
                f"Dataset '{name}' has forbidden dimension '{dim}'. Allowed: {sorted(allowed)}"
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
        # Heuristic: skip rechunk warnings for very small or already in-memory test datasets.
        # Total element count below threshold -> leave unchunked; avoids noisy test warnings.
        try:
            total_size = sum(int(v.size) for v in ds.data_vars.values())
        except Exception:
            total_size = 0
        SMALL_THRESHOLD = 20000  # configurable if needed
        # If no existing chunking (numpy) AND tiny dataset -> skip rechunk silently.
        all_unchunked = True
        for v_da in ds.data_vars.values():
            try:
                if getattr(v_da, "chunks", None) is not None:
                    all_unchunked = False
                    break
            except Exception as e:
                logger.debug(
                    "Exception while checking 'chunks' attribute for variable %r: %s",
                    getattr(v_da, "name", None),
                    e,
                )
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
            # On newer xarray, rename may drop the xindex; restore it only if not already indexed.
            try:
                existing_indexes = getattr(ds, "xindexes", {}) or getattr(ds, "_indexes", {})
                if "init_time" not in existing_indexes:
                    ds = ds.set_xindex("init_time")
            except Exception:
                # Deterministic fallback: if re-indexing fails we proceed without raising here.
                pass
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
            ds = ds.sel(lead_time=[lead0])

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
        ds_i = ensure_ensemble_dim(ds_i)
        validate_dataset_structure(ds_i, p)
        dsets.append(ds_i)

    # Harmonize 'level' coordinate across shards to a canonical sorted union to avoid
    # non-monotonic global indexes during combine_by_coords.
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
    if not dsets:
        raise ValueError("No Zarr paths provided.")
    if len(dsets) == 1:
        return dsets[0]
    combined: xr.Dataset = cast(
        xr.Dataset,
        xr.combine_by_coords(
            dsets,
            combine_attrs="override",
            coords="minimal",
            data_vars="all",
            join="outer",
            compat="override",
        ),
    )
    return combined


def open_prediction(path: str | Sequence[str], variables: list[str] | None = None) -> xr.Dataset:
    """Open prediction/model dataset(s) from Zarr and optionally subset variables.

    Accepts a single path or a list/sequence of paths. When multiple paths are given,
    they are combined lazily by coordinates without materializing data.
    """
    if isinstance(path, list | tuple):
        return _open_many_zarr(list(path), variables)
    ds = xr.open_zarr(path, decode_timedelta=True)
    if variables:
        ds = ds[[v for v in variables if v in ds.data_vars]]

    ds = _ensure_monotonic(ds)

    # Custom interaction based on the zarr file
    ds = custom.modify_ds(ds, cast(str, path))
    ds = ensure_ensemble_dim(ds)
    validate_dataset_structure(ds, cast(str, path))
    return ds


def open_target(path: str | Sequence[str], variables: list[str] | None = None) -> xr.Dataset:
    """Open target dataset(s) from Zarr and optionally subset variables.

    Accepts a single path or a list/sequence of paths. When multiple paths are given,
    they are combined lazily by coordinates without materializing data.
    """
    if isinstance(path, list | tuple):
        return _open_many_zarr(list(path), variables)
    ds = xr.open_zarr(path, decode_timedelta=True)
    if variables:
        ds = ds[[v for v in variables if v in ds.data_vars]]

    ds = _ensure_monotonic(ds)

    # Custom interaction based on the zarr file
    ds = custom.modify_ds(ds, cast(str, path))
    ds = ensure_ensemble_dim(ds)
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
        return ds.isel(ensemble=indices_list, drop=False)

    # No explicit selection: do NOT pre-reduce. Keep ensemble for modules to decide.
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Derived-variable machinery
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def _wind_speed(ds: xr.Dataset, u_var: str, v_var: str) -> xr.DataArray:
    """Lazy wind speed magnitude: sqrt(U² + V²).  Units: m s⁻¹."""
    da = np.sqrt(ds[u_var] ** 2 + ds[v_var] ** 2)
    da.attrs["units"] = "m s**-1"
    da.attrs["long_name"] = "Wind Speed"
    return da


# Maps recipe name → callable(ds, u_var, v_var) → xr.DataArray
# Available recipes that a user may reference via ``kind:`` in config:
#   wind_speed  — sqrt(U² + V²), m s⁻¹, suitable for all modules
_DERIVED_RECIPES: dict[str, Any] = {
    "wind_speed": _wind_speed,
}


def _parse_derived_cfg(
    derived_cfg: dict[str, Any],
) -> list[tuple[str, str, str, str]]:
    """Parse the ``derived_variables`` config block.

    Returns a list of ``(output_name, kind, u_var, v_var)`` tuples.

    Each sub-key becomes the output variable name::

        10m_wind_speed:
          kind: wind_speed                    # see _DERIVED_RECIPES for available kinds
          u: 10m_u_component_of_wind
          v: 10m_v_component_of_wind
    """
    entries: list[tuple[str, str, str, str]] = []

    for key, block in derived_cfg.items():
        if not isinstance(block, dict):
            c.warn(f"derived_variables.{key}: expected a mapping, skipping.")
            continue

        if "kind" not in block:
            c.warn(
                f"derived_variables.{key}: missing required 'kind' key. "
                f"Available kinds: {sorted(_DERIVED_RECIPES)}. Skipping."
            )
            continue

        kind = str(block["kind"])
        if kind not in _DERIVED_RECIPES:
            c.warn(
                f"derived_variables.{key}: unknown kind '{kind}'. "
                f"Available kinds: {sorted(_DERIVED_RECIPES)}."
            )
            continue

        u_var = str(block.get("u") or block.get("u_var") or "")
        v_var = str(block.get("v") or block.get("v_var") or "")
        if not u_var or not v_var:
            c.warn(f"derived_variables.{key}: requires 'u' and 'v' keys. Skipping.")
            continue

        entries.append((key, kind, u_var, v_var))

    return entries


def add_derived_variables(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    derived_cfg: dict[str, Any],
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute derived variables and append them to both datasets.

    Currently supported recipes (see ``_DERIVED_RECIPES``):

    * ``wind_speed`` — ``sqrt(U² + V²)`` (m s⁻¹)

    **Inner-join guard**: a derived variable is only added when *both* source
    components are present in *both* the target and prediction datasets.  If a
    component is missing from either dataset the variable is skipped with a
    warning.

    All computations are **lazy** — no Dask graphs are evaluated here.

    .. note::
        Not all evaluation metrics are physically or statistically meaningful
        for every variable — derived or otherwise.  The user is responsible for
        choosing sensible module combinations.  See the README for guidance.

    Parameters
    ----------
    ds_target, ds_prediction : xr.Dataset
        Datasets returned by :func:`data_selection.prepare_datasets`.
    derived_cfg : dict
        The ``derived_variables`` config block.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Updated ``(ds_target, ds_prediction)`` with derived variables appended.
    """
    if not derived_cfg:
        return ds_target, ds_prediction

    entries = _parse_derived_cfg(derived_cfg)
    added: list[str] = []
    skipped: list[str] = []

    for out_name, kind, u_var, v_var in entries:
        # Inner-join guard
        missing_target = {u_var, v_var} - set(ds_target.data_vars)
        missing_pred = {u_var, v_var} - set(ds_prediction.data_vars)
        if missing_target or missing_pred:
            parts: list[str] = []
            if missing_target:
                parts.append(f"target missing {sorted(missing_target)}")
            if missing_pred:
                parts.append(f"prediction missing {sorted(missing_pred)}")
            c.warn(f"Skipping derived '{out_name}' ({kind}): {'; '.join(parts)}.")
            skipped.append(out_name)
            continue

        recipe = _DERIVED_RECIPES[kind]
        ds_target = ds_target.assign({out_name: recipe(ds_target, u_var, v_var)})
        ds_prediction = ds_prediction.assign({out_name: recipe(ds_prediction, u_var, v_var)})
        added.append(out_name)
        logger.info("Derived variable '%s' (%s) added to both datasets.", out_name, kind)

    if added:
        c.info(f"Derived variables added to both datasets: {added}")
    if skipped:
        c.warn(f"Derived variables skipped (missing source components): {skipped}")

    return ds_target, ds_prediction
