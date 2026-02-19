from __future__ import annotations

import numpy as np
import xarray as xr

from swissclim_evaluations.metrics.probabilistic import wbx


def _tiny_dataset() -> xr.Dataset:
    return xr.Dataset({"var": (("x",), np.array([1.0], dtype=float))})


def test_compute_wbx_metric_dataset_uses_legacy_aggregator_method() -> None:
    metrics = {"CRPS": object()}

    class LegacyAggregator:
        def compute_metric_values(self, ds_prediction, ds_target):
            assert "var" in ds_prediction
            assert "var" in ds_target
            return xr.Dataset({"CRPS.var": (("x",), np.array([0.5], dtype=float))})

    out = wbx._compute_wbx_metric_dataset(
        LegacyAggregator(), metrics, _tiny_dataset(), _tiny_dataset()
    )

    assert "CRPS.var" in out.data_vars


def test_compute_wbx_metric_dataset_uses_current_aggregation_state(monkeypatch) -> None:
    metrics = {"CRPS": object()}
    expected_stats = {
        "dummy_stat": {"var": xr.DataArray(np.array([1.0], dtype=float), dims=("x",))}
    }

    def _fake_compute_unique_statistics_for_all_metrics(*, metrics, predictions, targets):
        assert "var" in predictions
        assert "var" in targets
        return expected_stats

    monkeypatch.setattr(
        wbx.metrics_base,
        "compute_unique_statistics_for_all_metrics",
        _fake_compute_unique_statistics_for_all_metrics,
    )

    class FakeState:
        def metric_values(self, got_metrics):
            assert got_metrics is metrics
            return xr.Dataset({"CRPS.var": (("x",), np.array([0.25], dtype=float))})

    class CurrentAggregator:
        def aggregate_statistics(self, statistics):
            assert statistics is expected_stats
            return FakeState()

    out = wbx._compute_wbx_metric_dataset(
        CurrentAggregator(), metrics, _tiny_dataset(), _tiny_dataset()
    )

    assert "CRPS.var" in out.data_vars
