import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

# Import the modules to test
import swissclim_evaluations.cli as cli_mod


def _synthetic_prepared_multivariate():
    """
    Create synthetic datasets with 10m_u_component_of_wind, 10m_v_component_of_wind, 2m_temperature
    for bivariate testing.
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
            "10m_u_component_of_wind": (["init_time", "lead_time", "latitude", "longitude"], u10m),
            "10m_v_component_of_wind": (["init_time", "lead_time", "latitude", "longitude"], v10m),
            "2m_temperature": (["init_time", "lead_time", "latitude", "longitude"], t2m),
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
        "metrics": {
            "multivariate": {
                "bivariate_pairs": [
                    ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                    ["2m_temperature", "10m_u_component_of_wind"],
                ]
            }
        },
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

    # 6. Verify CLI output (.npz files AND .png plots)
    npz_dir_a = model_dir_a / "multivariate"
    assert (
        npz_dir_a / "bivariate_hist_10m_u_component_of_wind_10m_v_component_of_wind.npz"
    ).exists()
    assert (npz_dir_a / "bivariate_hist_2m_temperature_10m_u_component_of_wind.npz").exists()

    # Verify that plots are generated directly by the CLI
    assert (npz_dir_a / "bivariate_10m_u_component_of_wind_10m_v_component_of_wind.png").exists()
    assert (npz_dir_a / "bivariate_2m_temperature_10m_u_component_of_wind.png").exists()

    # Intercomparison step for bivariate plots is no longer needed/supported
    # as plots are generated per-model.
