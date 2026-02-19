from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

import swissclim_evaluations.cli as cli_mod
import swissclim_evaluations.core.data_selection as data_selection_mod

# Revised CLI smoke: the real parser only accepts --config. We create a minimal
# YAML config file and monkeypatch prepare_datasets so we avoid any I/O on zarr stores.


def _synthetic_prepared():  # returns (ds_target, ds_prediction, ds_target_std, ds_prediction_std)
    init_time = np.array([np.datetime64("2025-01-01T00")])
    lead_time = np.array([np.timedelta64(0, "h")], dtype="timedelta64[h]")
    lat = np.linspace(-1, 1, 3)
    lon = np.linspace(0, 2, 3)
    data = np.zeros((init_time.size, lead_time.size, lat.size, lon.size))
    ds = xr.Dataset(
        {"t2m": (["init_time", "lead_time", "latitude", "longitude"], data)},
        coords={
            "init_time": init_time,
            "lead_time": lead_time.astype("timedelta64[ns]"),
            "latitude": lat,
            "longitude": lon,
        },
    )
    # Provide standardized versions (identical but suffices for stubs)
    return ds, ds, ds, ds


def test_cli_main_smoke(monkeypatch, tmp_path: Path):
    # Monkeypatch the heavy dataset preparation to return tiny in-memory datasets.
    monkeypatch.setattr(data_selection_mod, "prepare_datasets", lambda cfg: _synthetic_prepared())
    # Create minimal config enabling only maps (which is stubbed by tests/conftest.py)
    cfg_text = (
        "paths:\n"  # paths are ignored due to monkeypatch
        "  target: dummy.zarr\n"
        "  prediction: dummy.zarr\n"
        f"  output_root: {tmp_path / 'output'}\n"
        "modules:\n"
        "  maps: true\n"
        "plotting:\n"
        "  output_mode: npz\n"
    )
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    argv = ["swissclim-evaluations", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", argv)
    # The CLI calls argparse which raises SystemExit; accept code 0.
    try:
        cli_mod.main()
    except SystemExit as e:
        assert e.code == 0
    # Ensure stub produced maps directory (smoke success)
    assert (tmp_path / "output").exists()
    assert (tmp_path / "output" / "maps").exists()
    # The used config should have been copied into the output root
    assert (tmp_path / "output" / cfg_file.name).exists()


def test_cli_main_smoke_output_mode_none_skips_maps(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(data_selection_mod, "prepare_datasets", lambda cfg: _synthetic_prepared())
    cfg_text = (
        "paths:\n"
        "  target: dummy.zarr\n"
        "  prediction: dummy.zarr\n"
        f"  output_root: {tmp_path / 'output'}\n"
        "modules:\n"
        "  maps: true\n"
        "plotting:\n"
        "  output_mode: none\n"
    )
    cfg_file = tmp_path / "config_none.yaml"
    cfg_file.write_text(cfg_text)
    argv = ["swissclim-evaluations", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", argv)

    try:
        cli_mod.main()
    except SystemExit as e:
        assert e.code == 0

    assert (tmp_path / "output").exists()
    assert not (tmp_path / "output" / "maps").exists()
    assert (tmp_path / "output" / cfg_file.name).exists()
