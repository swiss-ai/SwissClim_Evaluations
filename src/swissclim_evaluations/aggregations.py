import dask.array as da
import numpy as np
from pandas.core.indexes.interval import Interval 
import xarray as xr


def histogram(ds: xr.Dataset, bins: int | np.ndarray, dims: list[str] | None=None, bindim: str | None = None) -> xr.Dataset:
    """
    Compute the histogram of the data along the given dimension.

    Notes
    -----
    `da.histogram` must be used instead of `np.histogram` to support chunked arrays.
    """
    bins = bins if isinstance(bins, (list, np.ndarray)) else np.linspace(0, 1, bins + 1)
    other_dims = list(set(ds.dims) - set(dims))
    other_dims = [dim for dim in ds.dims if dim in other_dims]
    stackdims = {"core_dims": dims, "other_dims": other_dims}
    bincoord = np.array([Interval(bins[i], bins[i+1]) for i in range(len(bins)-1)])
    bindim = bindim or "bin"

    binidx = xr.apply_ufunc(
        da.digitize,
        ds.stack(**stackdims).chunk({"other_dims": "auto"}),
        kwargs={"bins": bins[1:], "right": True},
        input_core_dims=[["core_dims"]],
        output_core_dims=[["core_dims"]],
        dask="allowed",
    )

    bincount: xr.Dataset = xr.apply_ufunc(
        lambda x: da.apply_along_axis(da.bincount, 1, x, minlength=len(bins)-1),
        binidx,
        input_core_dims=[["other_dims", "core_dims"]],
        output_core_dims=[["other_dims", bindim]],
        dask="allowed",
    )

    return bincount.unstack("other_dims").assign_coords({bindim: bincoord}).transpose(..., bindim)
