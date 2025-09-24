import dask.array as da
import numpy as np
import xarray as xr
from pandas.core.indexes.interval import Interval


def histogram(
    ds: xr.Dataset, bins: int | np.ndarray, dims: list[str] | None = None, bindim: str | None = None
) -> xr.Dataset:
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
    bincoord = np.array([Interval(bins[i], bins[i + 1]) for i in range(len(bins) - 1)])
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
        lambda x: da.apply_along_axis(da.bincount, 1, x, minlength=len(bins) - 1),
        binidx,
        input_core_dims=[["other_dims", "core_dims"]],
        output_core_dims=[["other_dims", bindim]],
        dask="allowed",
    )

    return bincount.unstack("other_dims").assign_coords({bindim: bincoord}).transpose(..., bindim)


def latitude_weights(latitudes: xr.DataArray, normalize: bool = True) -> xr.DataArray:
    """
    Computes weights proportional to the area represented by latitude bands.
    """
    if latitudes.ndim != 1:
        raise ValueError("Latitude array must be 1-dimensional.")

    lat_data = latitudes.data
    reversed_order = False
    if not np.all(np.diff(lat_data) > 0):
        if np.all(np.diff(lat_data) < 0):
            reversed_order = True
            lat_data = lat_data[::-1]
        else:
            raise ValueError("Latitude array must be strictly monotonic.")

    lat_rad = np.deg2rad(lat_data)

    delta = np.diff(lat_rad)
    delta = np.concatenate(([delta[0]], delta))
    bounds = np.concatenate(([lat_rad[0] - delta[0] / 2], lat_rad + delta / 2))

    bounds[0] = max(bounds[0], -np.pi / 2)
    bounds[-1] = min(bounds[-1], np.pi / 2)

    weights = np.sin(bounds[1:]) - np.sin(bounds[:-1])

    if reversed_order:
        weights = weights[::-1]

    weights_da = xr.DataArray(
        weights,
        dims=latitudes.dims,
        coords={latitudes.dims[0]: latitudes.data},
    )

    if normalize:
        weights_da = weights_da / weights_da.mean()

    return weights_da
