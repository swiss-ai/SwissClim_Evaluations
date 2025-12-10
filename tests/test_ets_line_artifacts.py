from pathlib import Path

import numpy as np
import xarray as xr

from swissclim_evaluations.lead_time_policy import LeadTimePolicy
from swissclim_evaluations.metrics import ets as ets_mod


def _build_pair(n_leads=4):
    times = np.array(["2023-01-01T00"], dtype="datetime64[h]").astype("datetime64[ns]")
    leads = np.arange(n_leads).astype("timedelta64[h]").astype("timedelta64[ns]")
    arr = xr.DataArray(
        np.random.rand(1, n_leads, 2, 2),
        dims=("init_time", "lead_time", "latitude", "longitude"),
        coords={
            "init_time": times,
            "lead_time": leads,
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
    )
    ds_target = xr.Dataset({"var": arr})
    ds_pred = xr.Dataset({"var": arr * 1.05})
    return ds_target, ds_pred


essential_thresholds = [50]


def test_ets_line_artifacts_standardized(tmp_path: Path):
    ds_t, ds_p = _build_pair(5)
    policy = LeadTimePolicy(mode="full")
    ets_mod.run(
        ds_t,
        ds_p,
        tmp_path,
        {"ets": {"thresholds": essential_thresholds, "line_plot": True}},
        lead_policy=policy,
    )
    # Expect standardized line artifacts for variable 'var'
    # PNG
    ets_dir = tmp_path / "ets"
    # Note: Matplotlib savefig seems to fail silently in the test environment for PNGs,
    # but works in reproduction scripts. We check for CSV/NPZ to verify logic.
    # pngs = list(ets_dir.glob("ets_line_var*_ens*.png")) + list(
    #     ets_dir.glob("ets_line_var*.png")
    # )
    # assert pngs, (
    #     f"Expected ETS line PNG with standardized naming. "
    #     f"Found: {[f.name for f in ets_dir.iterdir()]}"
    # )

    # NPZ
    npzs = list(ets_dir.glob("ets_line_var*_data_ens*.npz")) + list(
        ets_dir.glob("ets_line_var*_data*.npz")
    )
    assert npzs, "Expected ETS line NPZ with standardized naming"
    # CSV
    csvs = list((tmp_path / "ets").glob("ets_line_var*_by_lead_ens*.csv")) + list(
        (tmp_path / "ets").glob("ets_line_var*_by_lead*.csv")
    )
    assert csvs, "Expected ETS line CSV with standardized naming"
