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


def test_deterministic_compute_path_respects_manual_batch_size(monkeypatch) -> None:
    batch_sizes: list[int] = []

    def _fake_compute_jobs(
        jobs,
        key_map,
        post_process=None,
        batch_size=None,
        chunk_size=None,
        optimize_graph=None,
        desc=None,
        batch_callback=None,
    ):
        effective_batch_size = batch_size if batch_size is not None else chunk_size
        batch_sizes.append(int(effective_batch_size))
        for job in jobs:
            for lazy_key, res_key in key_map.items():
                if lazy_key in job and job[lazy_key] is not None:
                    job[res_key] = 0.0
                elif lazy_key in job:
                    job[res_key] = None
        if batch_callback:
            batch_callback(jobs)

    monkeypatch.setattr(det_mod, "compute_jobs", _fake_compute_jobs)

    ds_target = _tiny_dataset(1.0)
    ds_prediction = _tiny_dataset(0.9)

    metrics_df = det_mod._calculate_all_metrics(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        calc_relative=False,
        include=["MAE"],
        compute=True,
        performance_cfg={"batch_size": 1},
    )

    assert not metrics_df.empty
    assert batch_sizes
    assert all(bs == 1 for bs in batch_sizes)


def test_resolve_module_batching_options_uses_profile_defaults_when_omitted() -> None:
    safe_opts = resolve_module_batching_options(
        performance_cfg={"dask_profile": "safe"},
        default_lead_time_block_size=6,
        default_init_time_block_size=6,
    )
    balanced_opts = resolve_module_batching_options(
        performance_cfg={"dask_profile": "balanced"},
        default_lead_time_block_size=6,
        default_init_time_block_size=6,
    )
    fast_opts = resolve_module_batching_options(
        performance_cfg={"dask_profile": "fast"},
        default_lead_time_block_size=6,
        default_init_time_block_size=6,
    )

    assert safe_opts["lead_time_block_size"] == 4
    assert safe_opts["init_time_block_size"] == 8
    assert balanced_opts["lead_time_block_size"] == 4
    assert balanced_opts["init_time_block_size"] == 8
    assert fast_opts["lead_time_block_size"] == 4
    assert fast_opts["init_time_block_size"] == 8


def test_resolve_module_batching_options_prefers_explicit_block_sizes() -> None:
    opts = resolve_module_batching_options(
        performance_cfg={
            "dask_profile": "fast",
            "lead_time_block_size": 9,
            "init_time_block_size": 7,
        },
        default_lead_time_block_size=6,
        default_init_time_block_size=6,
    )

    assert opts["lead_time_block_size"] == 9
    assert opts["init_time_block_size"] == 7


def test_resolve_module_batching_options_respects_larger_module_defaults() -> None:
    opts = resolve_module_batching_options(
        performance_cfg={"dask_profile": "safe"},
        default_lead_time_block_size=20,
        default_init_time_block_size=16,
    )

    assert opts["lead_time_block_size"] == 20
    assert opts["init_time_block_size"] == 16


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
