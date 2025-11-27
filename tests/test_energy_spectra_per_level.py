import numpy as np
import xarray as xr

from swissclim_evaluations.plots.energy_spectra import run as run_energy_spectra


def test_energy_spectra_per_level_flag(tmp_path):
    # Create 3D dataset
    coords = {
        "init_time": [np.datetime64("2023-01-01")],
        "lead_time": [np.timedelta64(0, "h")],
        "level": [500, 850],
        "latitude": np.linspace(-90, 90, 10),
        "longitude": np.linspace(0, 360, 20),
    }

    data = np.random.rand(1, 1, 2, 10, 20)
    ds = xr.Dataset(
        {
            "u_component_of_wind": (
                ("init_time", "lead_time", "level", "latitude", "longitude"),
                data,
            )
        },
        coords=coords,
    )

    # Case 1: report_per_level = True (default)
    out_root_true = tmp_path / "output_true"
    cfg_true = {"metrics": {"energy_spectra": {"report_per_level": True}}}

    run_energy_spectra(
        ds_target=ds,
        ds_prediction=ds,
        out_root=out_root_true,
        plotting_cfg={},
        select_cfg={"levels": [500, 850]},
        cfg=cfg_true,
    )

    assert (out_root_true / "energy_spectra" / "lsd_3d_metrics_per_level_ensmean.csv").exists()
    assert (
        out_root_true / "energy_spectra" / "lsd_bands_3d_metrics_per_level_ensmean.csv"
    ).exists()

    # Case 2: report_per_level = False
    out_root_false = tmp_path / "output_false"
    cfg_false = {"metrics": {"energy_spectra": {"report_per_level": False}}}

    run_energy_spectra(
        ds_target=ds,
        ds_prediction=ds,
        out_root=out_root_false,
        plotting_cfg={},
        select_cfg={"levels": [500, 850]},
        cfg=cfg_false,
    )

    assert not (out_root_false / "energy_spectra" / "lsd_3d_metrics_per_level_ensmean.csv").exists()
    assert not (
        out_root_false / "energy_spectra" / "lsd_bands_3d_metrics_per_level_ensmean.csv"
    ).exists()
    # Averaged should still exist
    assert (out_root_false / "energy_spectra" / "lsd_3d_metrics_averaged_ensmean.csv").exists()
