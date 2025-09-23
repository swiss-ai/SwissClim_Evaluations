from __future__ import annotations

from pathlib import Path

from swissclim_evaluations.metrics.deterministic import run as run_deterministic
from swissclim_evaluations.metrics.probabilistic import run_probabilistic

from ._smoke_data import make_synthetic_datasets


def test_deterministic_smoke(tmp_path: Path):
    targets, predictions = make_synthetic_datasets(with_ensemble=False)
    targets_std = (targets - targets.mean()) / targets.std()
    predictions_std = (predictions - targets.mean()) / targets.std()

    out_root = tmp_path / "out"
    run_deterministic(
        ds_target=targets,
        ds_prediction=predictions,
        ds_target_std=targets_std,
        ds_prediction_std=predictions_std,
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
    # New schema: no 'multi' or 'full' tokens; expect plain deterministic_metrics_*.csv with ensemble token
    assert any(
        f.name == "deterministic_metrics_ensnone.csv"
        or f.name == "deterministic_metrics_standardized_ensnone.csv"
        for f in det_dir.glob("deterministic_metrics*.csv")
    )


def test_probabilistic_smoke(tmp_path: Path):
    targets, predictions = make_synthetic_datasets(with_ensemble=True)

    out_root = tmp_path / "out"
    run_probabilistic(
        ds_target=targets,
        ds_prediction=predictions,
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
    # New schema: summary may include init/lead tokens if ranges present else just crps_summary_ensnone.csv
    assert any(
        f.name == "crps_summary_ensnone.csv"
        or (
            f.name.startswith("crps_summary_averaged_init")
            and f.name.endswith("_ensnone.csv")
        )
        for f in prob_dir.glob("crps_summary*.csv")
    )
    pit_files = list(prob_dir.glob("pit_hist_*_ensnone.npz"))
    assert len(pit_files) >= 1
