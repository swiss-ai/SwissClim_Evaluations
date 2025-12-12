from collections.abc import Callable
from typing import Any

import dask
import dask.array as dsa
import numpy as np


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
    """
    if not jobs:
        return

    # Collect lazy objects
    lazy_flat = []
    # We need to track where each lazy object came from to assign results back
    # Structure: (job_index, lazy_key, result_key)
    metadata = []

    for i, job in enumerate(jobs):
        for lazy_key, res_key in key_map.items():
            if lazy_key in job:
                val = job[lazy_key]
                if val is not None:
                    lazy_flat.append(val)
                    metadata.append((i, lazy_key, res_key))
                else:
                    # If None, we handle it after compute loop or set default now.
                    # We'll handle it in the distribution phase to ensure consistency.
                    pass

    if not lazy_flat:
        # Handle case where all are None or keys missing
        _fill_defaults(jobs, key_map, post_process)
        return

    results = dask.compute(*lazy_flat)

    # Distribute results
    for (job_idx, _, res_key), res in zip(metadata, results, strict=False):
        job = jobs[job_idx]

        if post_process and res_key in post_process:
            job[res_key] = post_process[res_key](res)
        else:
            job[res_key] = res

    # Handle Nones (items that were None in the job)
    _fill_defaults(jobs, key_map, post_process, skip_existing=True)


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
