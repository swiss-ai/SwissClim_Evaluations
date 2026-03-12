from __future__ import annotations

import numpy as np
import xarray as xr

from swissclim_evaluations.dask_utils import (
    compute_jobs,
    resolve_dynamic_batch_size,
    resolve_module_batching_options,
)
from swissclim_evaluations.metrics import deterministic as det_mod


def _tiny_dataset(val: float) -> xr.Dataset:
    data = np.full((2, 2, 2, 2), val, dtype=np.float32)
    return xr.Dataset(
        {
            "2m_temperature": (
                ("init_time", "lead_time", "latitude", "longitude"),
                data,
            )
        },
        coords={
            "init_time": np.array(["2023-01-01T00", "2023-01-01T06"], dtype="datetime64[h]"),
            "lead_time": np.array([0, 6], dtype="timedelta64[h]"),
            "latitude": np.array([46.0, 47.0], dtype=float),
            "longitude": np.array([7.0, 8.0], dtype=float),
        },
    )


def test_resolve_dynamic_batch_size_clamps_non_positive_manual_value() -> None:
    assert resolve_dynamic_batch_size({"batch_size": 0}) == 1
    assert resolve_dynamic_batch_size({"chunk_size": 0}) == 1


def test_deterministic_compute_path_uses_direct_dask_compute(monkeypatch) -> None:
    compute_calls: list[int] = []

    def _fake_dask_compute(*args, **kwargs):
        compute_calls.append(len(args))
        return tuple(0.0 for _ in args)

    ds_target = _tiny_dataset(1.0)
    ds_prediction = _tiny_dataset(2.0)

    monkeypatch.setattr(
        "swissclim_evaluations.metrics.deterministic.calc.dask.compute",
        _fake_dask_compute,
    )

    metrics_df = det_mod.calc.calculate_all_metrics(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        calc_relative=False,
        include=["MAE"],
        compute=True,
        performance_cfg={"batch_size": 1},
    )

    assert not metrics_df.empty
    assert compute_calls
    assert all(call_size >= 1 for call_size in compute_calls)


def test_resolve_module_batching_options_defaults_to_split_level_true() -> None:
    opts = resolve_module_batching_options(performance_cfg={})
    assert opts["split_level"] is True


def test_resolve_module_batching_options_respects_split_level_override() -> None:
    opts = resolve_module_batching_options(performance_cfg={"split_3d_by_level": False})
    assert opts["split_level"] is False


def test_resolve_module_batching_options_ignores_legacy_lead_init_keys() -> None:
    opts = resolve_module_batching_options(
        performance_cfg={
            "split_lead_time": True,
            "split_init_time": True,
            "lead_time_block_size": 9,
            "init_time_block_size": 7,
        },
        default_split_level=False,
    )

    assert opts == {"split_level": False}


def test_compute_jobs_uses_stable_optimize_graph_default(monkeypatch) -> None:
    calls: list[bool] = []

    def _fake_dask_compute(*args, optimize_graph=None, **kwargs):
        calls.append(bool(optimize_graph))
        return tuple(1.0 for _ in args)

    monkeypatch.setattr("swissclim_evaluations.dask_utils.dask.compute", _fake_dask_compute)

    jobs = [{"lazy": object()}, {"lazy": object()}, {"lazy": object()}]
    compute_jobs(jobs, key_map={"lazy": "res"}, batch_size=2)
    assert calls == [True, True]

    calls.clear()
    jobs2 = [{"lazy": object()}]
    compute_jobs(jobs2, key_map={"lazy": "res"}, batch_size=1, optimize_graph=False)
    assert calls == [False]
