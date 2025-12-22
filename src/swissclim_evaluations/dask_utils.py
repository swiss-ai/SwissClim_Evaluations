from collections.abc import Callable
from typing import Any

import dask
import dask.array as dsa
import numpy as np

from . import console as c


def to_finite_array(arr: Any) -> np.ndarray:
    """Converts input to a 1D numpy array containing only finite values."""
    a = np.asarray(arr).ravel()
    return a[np.isfinite(a)]


def as_float_array(arr: Any) -> np.ndarray:
    """Converts input to a numpy array of floats."""
    return np.asarray(arr).astype(float)


def compute_jobs(
    jobs: list[dict[str, Any]],
    key_map: dict[str, str],
    post_process: dict[str, Callable[[Any], Any]] | None = None,
    chunk_size: int = 20,
    optimize_graph: bool | None = None,
) -> None:
    """
    Batch compute dask lazy objects inside a list of job dictionaries.

    Args:
        jobs: List of dictionaries containing lazy objects.
        key_map: Mapping from lazy key name to result key name.
                 e.g. {"sub_t_lazy": "sub_t"}
        post_process: Optional mapping from result key name to a function
                      that processes the computed result.
                      e.g. {"sub_t": to_finite_array}
        chunk_size: Number of jobs to compute in one batch.
                    Default is 20.
    """
    if not jobs:
        return

    num_batches = (len(jobs) + chunk_size - 1) // chunk_size
    c.print(
        f"[Dask] Processing {len(jobs)} jobs in {num_batches} batches (chunk_size={chunk_size})..."
    )

    # Chunk the jobs directly
    for i in range(0, len(jobs), chunk_size):
        job_chunk = jobs[i : i + chunk_size]

        # Collect lazy objects for this chunk of jobs
        lazy_flat = []
        # We need to track where each lazy object came from to assign results back
        # Structure: (job_index_in_chunk, lazy_key, result_key)
        metadata = []

        for j, job in enumerate(job_chunk):
            for lazy_key, res_key in key_map.items():
                if lazy_key in job:
                    val = job[lazy_key]
                    if val is not None:
                        lazy_flat.append(val)
                        metadata.append((j, lazy_key, res_key))

        if not lazy_flat:
            _fill_defaults(job_chunk, key_map, post_process)
            continue

        # Compute this chunk
        # Auto-tune graph optimization: for small batches the rewrite cost
        # can dominate, so skip it when the batch is tiny.
        do_optimize = optimize_graph
        if do_optimize is None:
            do_optimize = len(lazy_flat) > 4

        results = dask.compute(*lazy_flat, optimize_graph=do_optimize)

        # Distribute results
        for (j, _, res_key), res in zip(metadata, results, strict=False):
            job = job_chunk[j]

            if post_process and res_key in post_process:
                job[res_key] = post_process[res_key](res)
            else:
                job[res_key] = res

        # Handle Nones (items that were None in the job)
        _fill_defaults(job_chunk, key_map, post_process, skip_existing=True)


def _fill_defaults(
    jobs: list[dict[str, Any]],
    key_map: dict[str, str],
    post_process: dict[str, Callable[[Any], Any]] | None,
    skip_existing: bool = False,
) -> None:
    """Helper to fill default values for missing/None lazy keys."""
    for job in jobs:
        for lazy_key, res_key in key_map.items():
            if skip_existing and res_key in job:
                continue

            if lazy_key in job:  # Only if the job was supposed to have this key
                # If the lazy value was None, we provide a default based on post_process
                if post_process and res_key in post_process:
                    if (
                        post_process[res_key] == to_finite_array
                        or post_process[res_key] == as_float_array
                    ):
                        job[res_key] = np.array([], dtype=float)
                    else:
                        job[res_key] = None
                else:
                    job[res_key] = None


def dask_histogram(
    da: Any, bins: np.ndarray | int, range: tuple[float, float] | None = None
) -> Any:
    """
    Compute histogram lazily using dask.

    Args:
        da: Input dask array or xarray DataArray.
        bins: Bin edges or number of bins.
        range: (min, max) range for histogram if bins is an integer.

    Returns:
        dask array representing the histogram counts.
    """
    # Handle xarray DataArray
    data = getattr(da, "data", da)
    darr = dsa.asarray(data)
    darr = darr.ravel()
    darr = darr[~dsa.isnan(darr)]

    counts, _ = dsa.histogram(darr, bins=bins, range=range)
    return counts


def compute_global_quantile(
    da_var: Any,
    q: float | list[float] | np.ndarray,
    skipna: bool = True,
    bins: int = 100000,
) -> Any:
    """
    Compute global quantile(s) of a DataArray using subsampling and xarray's quantile method.
    """
    import xarray as xr

    # Ensure input is a DataArray
    if not isinstance(da_var, xr.DataArray):
        da_var = xr.DataArray(da_var)

    # Flatten to 1D using dask.array.ravel or np.ravel to avoid expensive xarray stack
    data = da_var.data
    flat = data.ravel() if hasattr(data, "ravel") else np.ravel(data)

    # Subsample if large
    MAX_SAMPLES = 1_000_000
    if flat.size > MAX_SAMPLES:
        stride = int(np.ceil(flat.size / MAX_SAMPLES))
        flat = flat[::stride]

    # Rechunk to single chunk for exact quantile computation
    if hasattr(flat, "rechunk"):
        flat = flat.rechunk(-1)

    # Compute quantile
    da_sub = xr.DataArray(flat, dims="sample")
    return da_sub.quantile(q, dim="sample", skipna=skipna)


def compute_quantile_preserving(da: Any, q: list[float], preserve_dims: list[str] | None) -> Any:
    import xarray as xr

    if not preserve_dims:
        return compute_global_quantile(da, q, skipna=True)

    # Ensure preserve_dims are in da
    preserve_dims = [d for d in preserve_dims if d in da.dims]
    if not preserve_dims:
        return compute_global_quantile(da, q, skipna=True)

    reduce_dims = [d for d in da.dims if d not in preserve_dims]
    if not reduce_dims:
        return da.quantile(q, dim=[], skipna=True)

    # Transpose so preserve_dims are first
    ordered_dims = preserve_dims + reduce_dims
    da_transposed = da.transpose(*ordered_dims)

    data = da_transposed.data

    # Calculate shapes
    preserve_shape = tuple(da.sizes[d] for d in preserve_dims)
    reduce_shape = tuple(da.sizes[d] for d in reduce_dims)
    reduce_size = np.prod(reduce_shape)

    target_shape = preserve_shape + (reduce_size,)

    # Reshape to (*preserve, reduce_flat)
    if hasattr(data, "reshape"):
        flat_reduced = data.reshape(target_shape)
    else:
        flat_reduced = np.reshape(data, target_shape)

    # Subsample along the last dimension
    MAX_SAMPLES = 5_000_000
    if reduce_size > MAX_SAMPLES:
        stride = int(np.ceil(reduce_size / MAX_SAMPLES))
        slices = (slice(None),) * len(preserve_dims) + (slice(None, None, stride),)
        flat_reduced = flat_reduced[slices]

    # Rechunk last dimension to -1 if dask
    if hasattr(flat_reduced, "rechunk"):
        flat_reduced = flat_reduced.rechunk({-1: -1})

    # Wrap back in DataArray
    coords = {d: da[d] for d in preserve_dims}
    dims = preserve_dims + ["sample"]
    da_sub = xr.DataArray(flat_reduced, coords=coords, dims=dims)

    return da_sub.quantile(q, dim="sample", skipna=True)


def calculate_dynamic_chunk_size(
    n_points: int | None = None,
    num_vars: int | None = None,
    config_chunk_size: int | str | None = None,
    safe_points_per_batch: int = 10**9,
    ds: Any | None = None,
) -> int:
    """
    Calculate a dynamic chunk size for dask computations to avoid OOM.
    Uses a fast approximation if n_points is not provided.
    """
    # 1. Check for config override
    if config_chunk_size is not None:
        if isinstance(config_chunk_size, str) and config_chunk_size.lower() == "no-chunk":
            return 10**9
        try:
            return int(config_chunk_size)
        except (ValueError, TypeError):
            pass

    # 2. Estimate n_points if not provided (Fast Approximation)
    if (n_points is None or num_vars is None) and ds is not None:
        try:
            # Fast approximation: use size of first variable * num_vars
            # Avoid creating a list of all data variables as it can be slow
            num_vars = len(ds.data_vars)
            if num_vars > 0:
                first_var_name = next(iter(ds.data_vars))
                n_points = int(ds[first_var_name].size * num_vars)
            else:
                n_points = 0
                num_vars = 0
        except Exception:
            n_points = 0
            num_vars = 0

    # 3. Heuristic
    dynamic_chunk = 20
    if num_vars and n_points and num_vars > 0 and n_points > 0:
        avg_points = n_points / num_vars
        if avg_points > 0:
            dynamic_chunk = int(safe_points_per_batch / avg_points)
            dynamic_chunk = max(1, min(dynamic_chunk, 100))

    return dynamic_chunk
