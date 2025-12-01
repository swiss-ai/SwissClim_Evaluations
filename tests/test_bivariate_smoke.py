import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

# Import the modules to test
import swissclim_evaluations.cli as cli_mod
from swissclim_evaluations.intercompare import run_from_config


def _synthetic_prepared_multivariate():
    """
    Create synthetic datasets with u10m, v10m, t2m for bivariate testing.
    """
    init_time = np.array([np.datetime64("2025-01-01T00")])
    lead_time = np.array([np.timedelta64(0, "h")], dtype="timedelta64[h]")
    lat = np.linspace(-1, 1, 20)
    lon = np.linspace(0, 2, 20)

    # Create random data
    data_shape = (init_time.size, lead_time.size, lat.size, lon.size)
    u10m = np.random.randn(*data_shape)
    v10m = np.random.randn(*data_shape)
    t2m = np.random.randn(*data_shape)

    ds = xr.Dataset(
        {
            "u10m": (["init_time", "lead_time", "latitude", "longitude"], u10m),
            "v10m": (["init_time", "lead_time", "latitude", "longitude"], v10m),
            "t2m": (["init_time", "lead_time", "latitude", "longitude"], t2m),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time.astype("timedelta64[ns]"),
            "latitude": lat,
            "longitude": lon,
        },
    )
    # Return same dataset for target and prediction for simplicity
    return ds, ds, ds, ds


def test_bivariate_workflow(monkeypatch, tmp_path: Path):
    """
    Test the full workflow for bivariate histograms:
    1. CLI run (metrics generation) -> produces .npz
    2. Intercomparison run (plotting) -> produces .png
    """

    # 1. Mock dataset preparation
    monkeypatch.setattr(cli_mod, "prepare_datasets", lambda cfg: _synthetic_prepared_multivariate())

    # 2. Setup directories
    model_dir_a = tmp_path / "output" / "modelA"
    model_dir_a.mkdir(parents=True)
    model_dir_b = tmp_path / "output" / "modelB"
    model_dir_b.mkdir(parents=True)

    # 3. Create CLI config for Model A
    cli_cfg_a = {
        "paths": {"nwp": "dummy.zarr", "ml": "dummy.zarr", "output_root": str(model_dir_a)},
        "modules": {"multivariate": True},
        "metrics": {"multivariate": {"bivariate_pairs": [["u10m", "v10m"], ["t2m", "u10m"]]}},
        "plotting": {"output_mode": "both"},
    }

    cli_cfg_file_a = tmp_path / "cli_config_a.yaml"
    with open(cli_cfg_file_a, "w") as f:
        yaml.dump(cli_cfg_a, f)

    # 4. Run CLI for Model A
    monkeypatch.setattr(sys, "argv", ["swissclim-evaluations", "--config", str(cli_cfg_file_a)])
    cli_mod.main()

    # 5. Run CLI for Model B (reuse config structure but change output)
    cli_cfg_b = cli_cfg_a.copy()
    cli_cfg_b["paths"]["output_root"] = str(model_dir_b)
    cli_cfg_file_b = tmp_path / "cli_config_b.yaml"
    with open(cli_cfg_file_b, "w") as f:
        yaml.dump(cli_cfg_b, f)

    monkeypatch.setattr(sys, "argv", ["swissclim-evaluations", "--config", str(cli_cfg_file_b)])
    cli_mod.main()

    # 6. Verify CLI output (.npz files)
    npz_dir_a = model_dir_a / "multivariate"
    assert (npz_dir_a / "bivariate_hist_u10m_v10m.npz").exists()
    assert (npz_dir_a / "bivariate_hist_t2m_u10m.npz").exists()

    # 7. Setup Intercomparison config
    inter_out_root = tmp_path / "output" / "intercomparison"
    inter_cfg = {
        "models": [str(model_dir_a), str(model_dir_b)],
        "labels": ["Model A", "Model B"],
        "output_root": str(inter_out_root),
        "modules": ["bivariate"],
        "bivariate_pairs": [["u10m", "v10m"], ["t2m", "u10m"]],
    }

    # 8. Run Intercomparison
    # Note: run_from_config takes the dict directly
    run_from_config(inter_cfg)

    # 9. Verify Intercomparison output (.png files)
    plot_dir = inter_out_root / "bivariate"
    assert plot_dir.exists()
    # The naming convention in intercompare.py is bivariate_{var_x}_{var_y}_{lab}_vs_{ref}.png
    # Labels are "Model A" and "Model B", so safe labels are "Model_A" and "Model_B"
    assert (plot_dir / "bivariate_u10m_v10m_Model_B_vs_Model_A.png").exists()
    assert (plot_dir / "bivariate_t2m_u10m_Model_B_vs_Model_A.png").exists()
