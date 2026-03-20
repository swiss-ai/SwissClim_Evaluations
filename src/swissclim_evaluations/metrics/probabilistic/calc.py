from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dask
import numpy as np
import xarray as xr
from weatherbenchX.metrics import base
from weatherbenchX.metrics.probabilistic import (
    EnsembleVariance,
    UnbiasedEnsembleMeanSquaredError,
    UnbiasedSpreadSkillRatio,
)

from ...dask_utils import dask_histogram


class RobustUnbiasedEnsembleMeanSquaredError(UnbiasedEnsembleMeanSquaredError):
    """Unbiased MSE that filters out negative estimates (statistical artifacts)."""

    def _compute_per_variable(
        self,
        predictions: xr.DataArray,
        targets: xr.DataArray,
    ) -> xr.DataArray:
        if self._ensemble_dim in targets.dims and targets.sizes[self._ensemble_dim] == 1:
            targets = targets.squeeze(self._ensemble_dim, drop=True)

        val = super()._compute_per_variable(predictions, targets)
        return val.where(val >= 0)


class RobustUnbiasedSpreadSkillRatio(UnbiasedSpreadSkillRatio):
    """SSR that uses RobustUnbiasedEnsembleMeanSquaredError to avoid NaNs."""

    @property
    def statistics(self) -> Mapping[str, base.Statistic]:
        return {
            "EnsembleVariance": EnsembleVariance(
                ensemble_dim=self._ensemble_dim,
                skipna_ensemble=self._skipna_ensemble,
            ),
            "UnbiasedEnsembleMeanSquaredError": RobustUnbiasedEnsembleMeanSquaredError(
                ensemble_dim=self._ensemble_dim,
                skipna_ensemble=self._skipna_ensemble,
            ),
        }

    def _values_from_mean_statistics_per_variable(
        self,
        statistic_values: Mapping[str, xr.DataArray],
    ) -> xr.DataArray:
        variance = statistic_values["EnsembleVariance"]
        mse = statistic_values["UnbiasedEnsembleMeanSquaredError"]
        ratio = variance / mse
        return np.sqrt(ratio)


def _pit(da_target, da_prediction):
    return np.mean(da_prediction < da_target[..., None], axis=-1)


def probability_integral_transform(
    da_target, da_prediction, ensemble_dim="ensemble", name_prefix: str | None = "PIT"
):
    """Compute the probability integral transform for ensemble predictions vs targets."""
    res = xr.apply_ufunc(
        _pit,
        da_target,
        da_prediction,
        input_core_dims=[[], [ensemble_dim]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float],
    )
    return _add_metric_prefix(res, name_prefix) if name_prefix else res


def _add_metric_prefix(da_or_ds: xr.Dataset | xr.DataArray, prefix: str):
    if isinstance(da_or_ds, xr.DataArray):
        name = da_or_ds.name or "value"
        return da_or_ds.rename(f"{prefix}.{name}")
    else:
        return da_or_ds.rename({var: f"{prefix}.{var}" for var in da_or_ds.data_vars})


def pit_histogram_dask_lazy(da: xr.DataArray, bins: int = 50) -> tuple[Any, np.ndarray]:
    """Compute PIT histogram lazily using dask_histogram."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = dask_histogram(da, bins=edges)
    return counts, edges


def pit_histogram_dask(
    da: xr.DataArray, bins: int = 50, density: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PIT histogram using dask."""
    counts_lazy, edges = pit_histogram_dask_lazy(da, bins)
    counts = np.asarray(dask.compute(counts_lazy, optimize_graph=False)[0]).astype(np.float64)

    if density:
        total = counts.sum()
        if total > 0:
            bin_width = 1.0 / bins
            counts = counts / (total * bin_width)
    return counts, edges
