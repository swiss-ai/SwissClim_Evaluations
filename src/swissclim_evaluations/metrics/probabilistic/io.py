from __future__ import annotations

from pathlib import Path
from typing import Any

import dask
import numpy as np
import xarray as xr

from ...helpers import build_output_filename, save_data


def save_pit_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    out_root: Path,
    variable: str,
    level: Any,
    init_range: tuple | None,
    lead_range: tuple | None,
    ens_token: str,
    save_npz: bool,
):
    """Save PIT histogram data to NPZ."""
    if not save_npz:
        return

    width = np.diff(edges)
    total = counts.sum()
    density = counts / (total * width.mean()) if total > 0 else counts

    pit_npz = out_root / build_output_filename(
        metric="pit_hist",
        variable=str(variable),
        level=level,
        qualifier=f"level{level}" if level is not None else None,
        init_time_range=init_range,
        lead_time_range=lead_range,
        ensemble=ens_token,
        ext="npz",
    )

    save_data(
        pit_npz,
        counts=density,
        edges=edges,
        module="probabilistic",
    )


def save_npz_with_coords(path: Path, da: xr.DataArray, module: str | None = None, **kwargs):
    """Save DataArray to NPZ with compact coordinate payload."""
    coords: dict[str, Any] = {}
    for dim_name in da.dims:
        if dim_name in da.coords:
            coords[dim_name] = np.asarray(da.coords[dim_name].values)

    coords.update(kwargs)

    data_obj = da.data
    data_arr = (
        np.asarray(dask.compute(data_obj, optimize_graph=False)[0])
        if hasattr(data_obj, "compute")
        else np.asarray(da.values)
    )
    save_data(path, module=module, data=data_arr, **coords)
