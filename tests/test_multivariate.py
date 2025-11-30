from __future__ import annotations

from pathlib import Path

import pandas as pd

from swissclim_evaluations.metrics.multivariate import run as run_multivariate

try:
    from ._smoke_data import make_synthetic_datasets
except ImportError:
    from tests._smoke_data import make_synthetic_datasets


def test_multivariate_smoke(tmp_path: Path):
    # SSIM requires at least 7x7 images by default
    targets, predictions = make_synthetic_datasets(with_ensemble=False, lat=10, lon=10)

    out_root = tmp_path / "output"
    run_multivariate(
        ds_target=targets,
        ds_prediction=predictions,
        ds_target_std=targets,  # Not used
        ds_prediction_std=predictions,  # Not used
        out_root=out_root,
        plotting_cfg={},
        metrics_cfg={},
    )

    multi_dir = out_root / "multivariate"
    assert multi_dir.exists()
    files = list(multi_dir.glob("*.csv"))
    assert len(files) > 0

    # Check content
    df = pd.read_csv(files[0])
    assert "SSIM" in df.columns
    assert "MULTIVARIATE_AVERAGE" in df["variable"].values
