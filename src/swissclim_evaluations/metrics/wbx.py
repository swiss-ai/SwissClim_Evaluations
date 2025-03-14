from weatherbenchX.metrics.base import PerVariableMetric, PerVariableStatistic
from weatherbenchX.metrics.wrappers import WrappedStatistic, EnsembleMean
from weatherbenchX.metrics.deterministic import SquaredError
import numpy as np
import xarray as xr

from swissclim_evaluations.metrics.probabilistic import crps_e1, crps_e2, probability_integral_transform



class ProbabilityIntegralTransform(PerVariableMetric):
    """Compute the PIT for ensemble forecasts."""
    
    def __init__(self, ensemble_dim="ensemble"):
        self.ensemble_dim = ensemble_dim
    
    def _compute_per_variable(self, predictions, targets):
        return probability_integral_transform(
            targets, predictions, ensemble_dim=self.ensemble_dim, name_prefix=None
        )

class EnsembleVariance(PerVariableStatistic):
    """Compute the ensemble variance."""
    
    def __init__(self, ensemble_dim="ensemble"):
        self.ensemble_dim = ensemble_dim
    
    def _compute_per_variable(self, predictions, targets):
        return xr.apply_ufunc(
            lambda x: np.var(x, axis=-1),
            predictions,
            input_core_dims=[[self.ensemble_dim]],
            output_core_dims=[[]],
            dask="parallelized",
        )

class SpreadSkillRatio(PerVariableMetric):
    """Computes the (biased) spread-skill ratio.
    """

    def __init__(self, ensemble_dim: str = "ensemble"):
        self.ensemble_dim = ensemble_dim

    @property
    def statistics(self):
        return {
            "EnsembleVariance": EnsembleVariance(ensemble_dim=self.ensemble_dim),
            "EnsembleMeanSquaredError": WrappedStatistic(
                SquaredError(),
                EnsembleMean(
                    which="predictions",
                    ensemble_dim=self.ensemble_dim,
                ),
            ),
        }

    def _values_from_mean_statistics_per_variable(self, statistic_values) -> xr.DataArray:
        """Computes metrics from aggregated statistics."""
        return np.sqrt(
            statistic_values['EnsembleVariance'] / 
            statistic_values['EnsembleMeanSquaredError']
        ).compute(scheduler="threads", num_workers=8)
    
class CRPSAccuracyTerm(PerVariableStatistic):
    """Compute the CRPS accuracy term E|y - f|."""
    
    def __init__(self, ensemble_dim="ensemble"):
        self.ensemble_dim = ensemble_dim
    
    def _compute_per_variable(self, predictions, targets):
        return crps_e1(targets, predictions, ensemble_dim=self.ensemble_dim)
    
class CRPSSpreadTerm(PerVariableStatistic):
    """Compute the CRPS spread term E|f - f"|."""

    def __init__(self, ensemble_dim="ensemble"):
        self.ensemble_dim = ensemble_dim
    
    def _compute_per_variable(self, predictions, targets):
        return crps_e2(predictions, ensemble_dim=self.ensemble_dim)
    
class CRPSEnsemble(PerVariableMetric):
    """Compute the CRPS for ensemble forecasts."""

    def __init__(self, ensemble_dim="ensemble"):
        self.ensemble_dim = ensemble_dim

    @property
    def statistics(self):
        return {
            "CRPSAccuracyTerm": CRPSAccuracyTerm(self.ensemble_dim),
            "CRPSSpreadTerm": CRPSSpreadTerm(self.ensemble_dim),
        }
    
    def _values_from_mean_statistics_per_variable(self, statistic_values):
        return (statistic_values["CRPSAccuracyTerm"] - 0.5 * statistic_values["CRPSSpreadTerm"]).compute(scheduler="threads", num_workers=8)