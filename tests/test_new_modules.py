from __future__ import annotations

from pathlib import Path

from swissclim_evaluations.metrics.multivariate import run as run_multivariate
from swissclim_evaluations.metrics.ssim import run as run_ssim

from ._smoke_data import make_synthetic_datasets


def test_ssim_smoke(tmp_path: Path):
    # SSIM requires at least 11x11 window (default sigma=1.5 -> win_size=11)
    targets, predictions = make_synthetic_datasets(with_ensemble=False, lat=20, lon=20)
    out_root = tmp_path / "output"

    run_ssim(
        ds_target=targets,
        ds_prediction=predictions,
        out_root=out_root,
        metrics_cfg={"ssim": {"sigma": 1.5}},
        ensemble_mode="mean",
    )

    ssim_dir = out_root / "ssim"
    files = list(ssim_dir.glob("ssim_ssim_*.csv"))
    assert len(files) > 0
    assert "ssim_ssim_ensmean.csv" in [f.name for f in files]


def test_multivariate_smoke(tmp_path: Path):
    targets, predictions = make_synthetic_datasets(with_ensemble=False, lat=10, lon=10)
    out_root = tmp_path / "output"

    # Select a deterministic bivariate pair: sort data_vars before picking
    var_names = sorted(targets.data_vars)
    if len(var_names) < 2:
        return  # Skip if not enough vars

    pair = [var_names[0], var_names[1]]

    run_multivariate(
        ds_target=targets,
        ds_prediction=predictions,
        out_root=out_root,
        metrics_cfg={"multivariate": {"bivariate_pairs": [pair]}},
        ensemble_mode="mean",
    )

    multi_dir = out_root / "multivariate"
    files = list(multi_dir.glob("bivariate_hist_*.npz"))
    assert len(files) > 0
    assert f"bivariate_hist_{pair[0]}_{pair[1]}_ensmean.npz" in [f.name for f in files]
