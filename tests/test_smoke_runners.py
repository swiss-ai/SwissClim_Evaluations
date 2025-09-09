from __future__ import annotations

from pathlib import Path

from swissclim_evaluations.metrics.deterministic import run as run_deterministic
from swissclim_evaluations.metrics.probabilistic import run_probabilistic

from ._smoke_data import make_synthetic_datasets


def test_deterministic_smoke(tmp_path: Path):
    obs, ml = make_synthetic_datasets(with_ensemble=False)
    obs_std = (obs - obs.mean()) / obs.std()
    ml_std = (ml - obs.mean()) / obs.std()

    out_root = tmp_path / "out"
    run_deterministic(
        ds=obs,
        ds_ml=ml,
        ds_std=obs_std,
        ds_ml_std=ml_std,
        out_root=out_root,
        plotting_cfg={},
        metrics_cfg={
            "deterministic": {
                "include": ["MAE"],
                "standardized_include": ["MAE"],
            }
        },
    )

    det_dir = out_root / "deterministic"
    assert (det_dir / "metrics.csv").exists()
    assert (det_dir / "metrics_standardized.csv").exists()


def test_probabilistic_smoke(tmp_path: Path):
    obs, ml = make_synthetic_datasets(with_ensemble=True)

    out_root = tmp_path / "out"
    run_probabilistic(
        ds=obs,
        ds_ml=ml,
        out_root=out_root,
        cfg_plot={"save_plot_data": True},
        cfg_all={
            "probabilistic": {
                "init_time_chunk_size": None,
                "lead_time_chunk_size": None,
            }
        },
    )

    prob_dir = out_root / "probabilistic"
    # at least one pit histogram and crps summary should exist
    assert (prob_dir / "crps_summary.csv").exists()
    pit_files = list(prob_dir.glob("*_pit_hist.npz"))
    assert len(pit_files) >= 1
