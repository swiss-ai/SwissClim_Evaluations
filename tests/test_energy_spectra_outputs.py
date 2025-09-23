from __future__ import annotations

from pathlib import Path

from swissclim_evaluations.plots.energy_spectra import run as run_energy_spectra

from ._smoke_data import make_synthetic_datasets


def test_energy_spectra_creates_suffixed_csvs(tmp_path: Path):
    # Reuse 2D synthetic data (time, lat, lon)
    ds_target, ds_prediction = make_synthetic_datasets(
        with_ensemble=False, time=3
    )

    out_root = tmp_path / "out"
    run_energy_spectra(
        ds_target=ds_target,
        ds_prediction=ds_prediction,
        out_root=out_root,
        plotting_cfg={},  # no figures for speed
        select_cfg={},
    )

    spec_dir = out_root / "energy_spectra"
    assert spec_dir.exists()
    # New standardized naming: expect a single averaged file using helper naming
    summary = list(spec_dir.glob("lsd_2d_metrics_averaged_*.csv"))
    assert summary, "Expected simplified averaged LSD CSV not found"
