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
    batch_size: int | None = None,
    chunk_size: int | None = None,
    optimize_graph: bool | None = None,
    desc: str | None = None,
    batch_callback: Callable[[list[dict[str, Any]]], None] | None = None,
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
        batch_size: Number of jobs to compute in one batch.
                Default is 20.
        chunk_size: Deprecated alias for `batch_size`.
        optimize_graph: Whether to optimize the dask graph.
        desc: Optional description to prepend to the log message.
        batch_callback: Optional function to call with the list of jobs
                        in the current batch after results have been computed
                        and assigned. This is useful for processing results
                        immediately (e.g. saving to disk) and clearing memory.
    """
    if not jobs:
        return

    if batch_size is None:
        batch_size = chunk_size if chunk_size is not None else 20
    batch_size = max(1, int(batch_size))

    num_batches = (len(jobs) + batch_size - 1) // batch_size
    msg = f"Processing {len(jobs)} jobs in {num_batches} batches (batch_size={batch_size})..."
    msg = f"[Dask] {desc}: {msg}" if desc else f"[Dask] {msg}"
    c.print(msg)

    # Chunk the jobs directly
    for i in range(0, len(jobs), batch_size):
        job_chunk = jobs[i : i + batch_size]

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

        if batch_callback:
            batch_callback(job_chunk)


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


def calculate_dynamic_batch_size(
    n_points: int | None = None,
    num_vars: int | None = None,
    config_batch_size: int | str | None = None,
    safe_points_per_batch: int | None = None,
    max_dynamic_batch_size: int | None = None,
    ds: Any | None = None,
) -> int:
    """
    Resolve batch size used for Dask job batching.

    Precedence:
    1) `config_batch_size` is set and valid -> use it directly (manual mode).
       - If set to "no-chunk", returns a very large value to effectively disable batching.
    2) Otherwise derive automatically from dataset/point heuristics (auto mode).

    Uses a fast approximation if n_points is not provided.
    """
    # 1. Check for config override
    if config_batch_size is not None:
        if isinstance(config_batch_size, str) and config_batch_size.lower() == "no-chunk":
            return 10**9
        try:
            return max(1, int(config_batch_size))
        except (ValueError, TypeError):
            pass

    # 2. Estimate n_points if not provided (Fast Approximation)
    if (n_points is None or num_vars is None) and ds is not None:
        n_points, num_vars = _estimate_dataset_points(ds)

    # 3. Heuristic
    #    If tuning values are omitted, derive conservative defaults from dataset footprint.
    dynamic_batch = 12
    if num_vars and n_points and num_vars > 0 and n_points > 0:
        avg_points = n_points / num_vars
        if avg_points > 0:
            if max_dynamic_batch_size is None:
                # Dataset-driven conservative cap: larger per-variable chunk footprints
                # -> fewer jobs per batch. Tuned for safety-first behavior.
                auto_cap = int(np.sqrt(64_000_000 / avg_points))
                cap = max(1, min(auto_cap, 8))
            else:
                cap = max(1, int(max_dynamic_batch_size))

            if safe_points_per_batch is None:
                # Default target corresponds to the chosen cap at the estimated avg_points.
                safe_points = int(max(avg_points * cap, avg_points))
            else:
                safe_points = max(1, int(safe_points_per_batch))

            dynamic_batch = int(safe_points / avg_points)
            dynamic_batch = max(1, min(dynamic_batch, cap))

    return dynamic_batch


def calculate_dynamic_chunk_size(
    n_points: int | None = None,
    num_vars: int | None = None,
    config_chunk_size: int | str | None = None,
    safe_points_per_batch: int | None = None,
    max_dynamic_chunk_size: int | None = None,
    ds: Any | None = None,
) -> int:
    """Backward-compatible alias for `calculate_dynamic_batch_size`."""
    return calculate_dynamic_batch_size(
        n_points=n_points,
        num_vars=num_vars,
        config_batch_size=config_chunk_size,
        safe_points_per_batch=safe_points_per_batch,
        max_dynamic_batch_size=max_dynamic_chunk_size,
        ds=ds,
    )


def _estimate_dataset_points(ds: Any | None) -> tuple[int, int]:
    """Estimate total points and variable count from dataset chunk footprint."""
    if ds is None:
        return 0, 0
    try:
        num_vars = len(ds.data_vars)
        if num_vars <= 0:
            return 0, 0

        first_var_name = next(iter(ds.data_vars))
        first_da = ds[first_var_name]
        first_var_points: int | None = None

        data = getattr(first_da, "data", first_da)
        chunks = getattr(data, "chunks", None)
        if chunks:
            try:
                first_var_points = int(
                    np.prod(
                        [max(1, int(dim_chunks[0])) for dim_chunks in chunks if len(dim_chunks) > 0]
                    )
                )
            except Exception:
                first_var_points = None

        if first_var_points is None:
            first_var_points = int(first_da.size)

        return int(first_var_points * num_vars), int(num_vars)
    except Exception:
        return 0, 0


def resolve_dynamic_batch_size(
    performance_cfg: dict[str, Any] | None,
    *,
    ds: Any | None = None,
    n_points: int | None = None,
    num_vars: int | None = None,
    safe_points_per_batch: int | None = None,
    max_dynamic_batch_size: int | None = None,
) -> int:
    """Resolve batch size from shared `performance.*` settings.

    `performance.batch_size` takes precedence over auto-tuning.
    Auto-tuning is used only when `batch_size` is omitted or invalid.

    Legacy `performance.chunk_size` and `performance.max_dynamic_chunk_size`
    are still accepted for backward compatibility.
    """
    perf = performance_cfg or {}
    batch_size_cfg = perf.get("batch_size", perf.get("chunk_size"))

    safe_points_cfg = perf.get("safe_points_per_batch", safe_points_per_batch)
    if isinstance(safe_points_cfg, str) and safe_points_cfg.lower() == "auto":
        safe_points_cfg = None

    max_batch_cfg = perf.get(
        "max_dynamic_batch_size",
        perf.get("max_dynamic_chunk_size", max_dynamic_batch_size),
    )
    if isinstance(max_batch_cfg, str) and max_batch_cfg.lower() == "auto":
        max_batch_cfg = None

    return calculate_dynamic_batch_size(
        n_points=n_points,
        num_vars=num_vars,
        config_batch_size=batch_size_cfg,
        safe_points_per_batch=(None if safe_points_cfg is None else int(safe_points_cfg)),
        max_dynamic_batch_size=(None if max_batch_cfg is None else int(max_batch_cfg)),
        ds=ds,
    )


def resolve_dynamic_chunk_size(
    performance_cfg: dict[str, Any] | None,
    *,
    ds: Any | None = None,
    n_points: int | None = None,
    num_vars: int | None = None,
    safe_points_per_batch: int | None = None,
    max_dynamic_chunk_size: int | None = None,
) -> int:
    """Backward-compatible alias for `resolve_dynamic_batch_size`."""
    return resolve_dynamic_batch_size(
        performance_cfg,
        ds=ds,
        n_points=n_points,
        num_vars=num_vars,
        safe_points_per_batch=safe_points_per_batch,
        max_dynamic_batch_size=max_dynamic_chunk_size,
    )


def describe_batch_size_mode(performance_cfg: dict[str, Any] | None) -> str:
    """Return human-readable batch-size mode used by shared resolver."""
    perf = performance_cfg or {}
    cfg_val = perf.get("batch_size", perf.get("chunk_size"))
    if cfg_val is None:
        return "auto"
    if isinstance(cfg_val, str) and cfg_val.lower() == "no-chunk":
        return "manual (no-chunk)"
    try:
        int(cfg_val)
        return "manual"
    except Exception:
        return "auto (invalid batch_size fallback)"


def describe_chunk_size_mode(performance_cfg: dict[str, Any] | None) -> str:
    """Backward-compatible alias for `describe_batch_size_mode`."""
    return describe_batch_size_mode(performance_cfg)


def resolve_dynamic_batch_details(
    performance_cfg: dict[str, Any] | None,
    *,
    ds: Any | None = None,
    n_points: int | None = None,
    num_vars: int | None = None,
    safe_points_per_batch: int | None = None,
    max_dynamic_batch_size: int | None = None,
) -> dict[str, Any]:
    """Return resolved batch details for logging/debugging."""
    perf = performance_cfg or {}
    mode = describe_batch_size_mode(performance_cfg)
    resolved_batch = resolve_dynamic_batch_size(
        performance_cfg,
        ds=ds,
        n_points=n_points,
        num_vars=num_vars,
        safe_points_per_batch=safe_points_per_batch,
        max_dynamic_batch_size=max_dynamic_batch_size,
    )

    out: dict[str, Any] = {
        "mode": mode,
        "batch_size": int(resolved_batch),
        "chunk_size": int(resolved_batch),
    }
    if not mode.startswith("auto"):
        return out

    if n_points is None or num_vars is None:
        n_points, num_vars = _estimate_dataset_points(ds)

    if not num_vars or not n_points:
        return out

    avg_points = float(n_points) / float(num_vars)

    safe_cfg = perf.get("safe_points_per_batch", safe_points_per_batch)
    if isinstance(safe_cfg, str) and safe_cfg.lower() == "auto":
        safe_cfg = None

    max_cfg = perf.get(
        "max_dynamic_batch_size", perf.get("max_dynamic_chunk_size", max_dynamic_batch_size)
    )
    if isinstance(max_cfg, str) and max_cfg.lower() == "auto":
        max_cfg = None

    if max_cfg is None:
        auto_cap = int(np.sqrt(64_000_000 / avg_points)) if avg_points > 0 else 4
        cap = max(1, min(auto_cap, 8))
    else:
        cap = max(1, int(max_cfg))

    if safe_cfg is None:
        safe_points = int(max(avg_points * cap, avg_points))
    else:
        safe_points = max(1, int(safe_cfg))

    out.update(
        {
            "num_vars": int(num_vars),
            "avg_points_per_var": int(avg_points),
            "effective_cap": int(cap),
            "effective_safe_points_per_batch": int(safe_points),
        }
    )
    return out


def resolve_dynamic_chunk_details(
    performance_cfg: dict[str, Any] | None,
    *,
    ds: Any | None = None,
    n_points: int | None = None,
    num_vars: int | None = None,
    safe_points_per_batch: int | None = None,
    max_dynamic_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for `resolve_dynamic_batch_details`."""
    return resolve_dynamic_batch_details(
        performance_cfg,
        ds=ds,
        n_points=n_points,
        num_vars=num_vars,
        safe_points_per_batch=safe_points_per_batch,
        max_dynamic_batch_size=max_dynamic_chunk_size,
    )


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        out = int(value)
        return out if out > 0 else default
    except Exception:
        return default


def _profile_default_block_size(perf: dict[str, Any], fallback_default: int) -> int:
    """Return profile-aware default split block size.

    This applies only when the explicit `lead_time_block_size` / `init_time_block_size`
    keys are omitted (or invalid) in `performance`.

    Profile targets are tuned to reduce tiny-job overhead on large nodes while still
    preserving conservative behavior for `safe`.

    To avoid overriding module-specific larger defaults, profile uplift is applied
    only for legacy/small fallback defaults (<= 12).
    """
    profile = str(perf.get("dask_profile", "safe") or "safe").strip().lower()
    profile_default = {
        "safe": 128,
        "balanced": 256,
        "fast": 512,
    }.get(profile)
    if profile_default is None:
        return fallback_default
    if int(fallback_default) > 12:
        return int(fallback_default)
    return max(int(fallback_default), int(profile_default))


def resolve_module_batching_options(
    performance_cfg: dict[str, Any] | None,
    *,
    default_split_level: bool = True,
    default_split_lead_time: bool = True,
    default_split_init_time: bool = True,
    default_lead_time_block_size: int = 12,
    default_init_time_block_size: int = 12,
) -> dict[str, Any]:
    """Resolve module batching options from config.

    Shared behavior contract:
    - Batching/splitting is always applied across data variables (data_vars).
    - Additional splits across level / lead_time / init_time are optional.
    - Defaults are enabled for lead_time and init_time split flags.

        Preferred keys live under `performance.*`:
    - `split_3d_by_level`
    - `split_lead_time`
    - `split_init_time`
    - `lead_time_block_size`
    - `init_time_block_size`

        Notes:
        - These split/block options control how each variable is partitioned across
            `level`, `lead_time`, and `init_time`.
        - They are orthogonal to `batch_size`, which controls how many prepared jobs
            are executed per Dask batch.
    """
    perf = performance_cfg or {}

    split_level = _coerce_bool(
        perf.get("split_3d_by_level"),
        default_split_level,
    )
    split_lead_time = _coerce_bool(
        perf.get("split_lead_time"),
        default_split_lead_time,
    )
    split_init_time = _coerce_bool(
        perf.get("split_init_time"),
        default_split_init_time,
    )

    lead_default = _profile_default_block_size(perf, default_lead_time_block_size)
    init_default = _profile_default_block_size(perf, default_init_time_block_size)
    lead_time_block_size = _coerce_positive_int(perf.get("lead_time_block_size"), lead_default)
    init_time_block_size = _coerce_positive_int(perf.get("init_time_block_size"), init_default)

    return {
        "split_level": split_level,
        "split_lead_time": split_lead_time,
        "split_init_time": split_init_time,
        "lead_time_block_size": lead_time_block_size,
        "init_time_block_size": init_time_block_size,
    }


def build_variable_level_lead_splits(
    ds: Any,
    variables: list[str] | None = None,
    split_level: bool = True,
    split_lead_time: bool = True,
    lead_time_block_size: int | None = None,
    split_init_time: bool = False,
    init_time_block_size: int | None = None,
) -> list[dict[str, Any]]:
    """Build common split specs across variable × level × lead_time × init_time blocks.

    Returns a list of specs with keys:
      - variable: str
      - level: Any | None
      - lead_slice: slice
      - lead_start: int
    - lead_len: int
            - init_slice: slice
            - init_start: int
            - init_len: int
    """
    vars_to_use = [str(v) for v in (variables or list(getattr(ds, "data_vars", [])))]
    splits: list[dict[str, Any]] = []

    for variable in vars_to_use:
        if variable not in ds.data_vars:
            continue
        da_var = ds[variable]

        level_values: list[Any] = [None]
        if split_level and ("level" in da_var.dims):
            level_values = [v.item() if hasattr(v, "item") else v for v in da_var["level"].values]

        for level_val in level_values:
            lead_slices = [slice(None)]
            if split_lead_time and ("lead_time" in da_var.dims):
                lead_count = int(da_var.sizes.get("lead_time", 0))
                if (
                    lead_time_block_size is not None
                    and lead_time_block_size > 0
                    and lead_count > lead_time_block_size
                ):
                    lead_slices = [
                        slice(i, min(i + int(lead_time_block_size), lead_count))
                        for i in range(0, lead_count, int(lead_time_block_size))
                    ]

            init_slices = [slice(None)]
            if split_init_time and ("init_time" in da_var.dims):
                init_count = int(da_var.sizes.get("init_time", 0))
                if (
                    init_time_block_size is not None
                    and init_time_block_size > 0
                    and init_count > init_time_block_size
                ):
                    init_slices = [
                        slice(i, min(i + int(init_time_block_size), init_count))
                        for i in range(0, init_count, int(init_time_block_size))
                    ]

            for lead_slice in lead_slices:
                lead_start = 0 if lead_slice.start is None else int(lead_slice.start)
                lead_stop = (
                    int(da_var.sizes.get("lead_time", 0))
                    if lead_slice.stop is None
                    else int(lead_slice.stop)
                )
                lead_len = max(1, lead_stop - lead_start)
                for init_slice in init_slices:
                    init_start = 0 if init_slice.start is None else int(init_slice.start)
                    init_stop = (
                        int(da_var.sizes.get("init_time", 0))
                        if init_slice.stop is None
                        else int(init_slice.stop)
                    )
                    init_len = max(1, init_stop - init_start)
                    splits.append(
                        {
                            "variable": variable,
                            "level": level_val,
                            "lead_slice": lead_slice,
                            "lead_start": lead_start,
                            "lead_len": lead_len,
                            "init_slice": init_slice,
                            "init_start": init_start,
                            "init_len": init_len,
                        }
                    )

    return splits


def apply_split_to_dataarray(
    da: Any,
    level: Any | None = None,
    lead_slice: slice | None = None,
    init_slice: slice | None = None,
) -> Any:
    """Apply a common split spec to a DataArray-like object."""
    out = da
    if (level is not None) and ("level" in out.dims):
        out = out.sel(level=level, drop=True)
    if (lead_slice is not None) and (lead_slice != slice(None)) and ("lead_time" in out.dims):
        out = out.isel(lead_time=lead_slice)
    if (init_slice is not None) and (init_slice != slice(None)) and ("init_time" in out.dims):
        out = out.isel(init_time=init_slice)
    return out
