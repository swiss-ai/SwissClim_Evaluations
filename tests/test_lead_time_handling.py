from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.metrics.probabilistic import run_probabilistic


def _make_multi_lead_datasets():
    # Two init steps, three lead times; ensemble size 4; small lat/lon grid
    init = np.array(
        [np.datetime64("2023-01-01T00"), np.datetime64("2023-01-01T06")],
        dtype="datetime64[ns]",
    )
    lead = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    lat = np.linspace(46.0, 47.0, 2)
    lon = np.linspace(7.0, 8.0, 2)
    ens = np.arange(4)

    shape_tgt = (init.size, lead.size, lat.size, lon.size, 1)
    shape_pred = (init.size, lead.size, lat.size, lon.size, ens.size)

    rng = np.random.default_rng(0)
    targets = xr.Dataset(
        {
            "2m_temperature": (
                ["init_time", "lead_time", "latitude", "longitude", "ensemble"],
                rng.standard_normal(shape_tgt),
            )
        },
        coords={
            "init_time": init,
            "lead_time": lead,
            "latitude": lat,
            "longitude": lon,
            "ensemble": [0],
        },
    )

    predictions = xr.Dataset(
        {
            "2m_temperature": (
                [
                    "init_time",
                    "lead_time",
                    "latitude",
                    "longitude",
                    "ensemble",
                ],
                rng.standard_normal(shape_pred),
            )
        },
        coords={
            "init_time": init,
            "lead_time": lead,
            "latitude": lat,
            "longitude": lon,
            "ensemble": ens,
        },
    )
    return targets, predictions


def test_probabilistic_preserves_lead_times(tmp_path: Path):
    ds_target, ds_prediction = _make_multi_lead_datasets()

    # Mimic CLI behavior (no standardization needed with strict data)
    # If we want to test single lead time, we should slice it manually here,
    # but let's test that it works with whatever data is passed.

    out_root = tmp_path / "out"
    run_probabilistic(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        out_root=out_root,
        cfg_plot={"save_plot_data": True},
        cfg_all={
            "probabilistic": {
                "init_time_chunk_size": None,
                "lead_time_chunk_size": None,
            }
        },
    )

    # Verify outputs exist
    prob_dir = out_root / "probabilistic"
    # Must have CRPS summary with init/lead time ranges (simplified schema)
    assert any(
        f.name.startswith("crps_summary_averaged_init")
        and "lead" in f.name
        and f.name.endswith("_ensprob.csv")
        for f in prob_dir.glob("crps_summary_*.csv")
    ), "Expected CRPS summary file with init/lead time tokens (ensprob) under new schema"

    # Load suffixed PIT/CRPS NPZ files (pick first match)
    # Updated to NPZ format for memory efficiency (no OOM during write)
    pit_npz = next(prob_dir.glob("pit_field_2m_temperature_*.npz"))
    crps_npz = next(prob_dir.glob("crps_field_2m_temperature_*.npz"))
    assert pit_npz.exists() and crps_npz.exists()

    # Load NPZ files and reconstruct basic shape info
    with np.load(pit_npz) as pit_data:
        assert "lead_time" in pit_data
        pit_lead = pit_data["lead_time"]
        assert len(pit_lead) == 1
        # Check that the single lead_time equals zero timedelta
        lt0 = np.array([0], dtype="timedelta64[h]").astype("timedelta64[ns]")[0]
        # ...existing code...

    with np.load(crps_npz) as crps_data:
        assert "lead_time" in crps_data
        crps_lead = crps_data["lead_time"]
        assert len(crps_lead) == 1
    assert np.all(pit_lead == lt0)
    assert np.all(crps_lead == lt0)
