from __future__ import annotations

from pathlib import Path

from swissclim_evaluations.plots.wd_kde import run as run_wd_kde

from ._smoke_data import make_synthetic_datasets


def test_wd_kde_wasserstein_exports(tmp_path: Path):
    # Use minimal synthetic dataset (1 time) to exercise code path quickly
    ds_target, ds_prediction = make_synthetic_datasets(
        with_ensemble=False, time=1, lat=3, lon=4
    )
    tgt_std = (ds_target - ds_target.mean()) / ds_target.std()
    pred_std = (ds_prediction - ds_target.mean()) / ds_target.std()

    out_root = tmp_path / "out"
    run_wd_kde(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        ds_target_std=tgt_std,
        ds_prediction_std=pred_std,
        out_root=out_root,
        plotting_cfg={
            "output_mode": "npz",
            "kde_max_samples": 100,
            "wd_kde_include_3d": False,  # only 2D variables here anyway
        },
    )

    wd_dir = out_root / "wd_kde"
    assert wd_dir.exists()
    # Wasserstein CSV naming may change; just assert some csv present
    csvs = list(wd_dir.glob("wd_kde_wasserstein_*.csv"))
    assert csvs, "Expected Wasserstein CSV outputs not found"

    # Basic content checks
    import pandas as pd

    df_any = pd.read_csv(csvs[0])
    assert "variable" in df_any.columns
