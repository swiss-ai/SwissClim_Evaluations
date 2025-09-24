"""
Ensure the local package under this workspace's `src/` is imported first during tests.

This avoids accidentally picking up another installed copy of
`swissclim_evaluations` elsewhere on PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert the workspace's src/ at the beginning of sys.path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Pre-import modules to guarantee monkeypatch will target same objects referenced in tests
from swissclim_evaluations.plots import histograms as _hist_pre
from swissclim_evaluations.plots import wd_kde as _kde_pre

import pytest
import numpy as _np
import os


# Fast algorithm stubs (applied unless SC_PLOTS_FULL=1) ---------------------------------
@pytest.fixture(autouse=True)
def _fast_plot_algorithms(monkeypatch):
    if os.environ.get("SC_PLOTS_FULL", "0") == "1":
        # Allow opting out to run full heavy routines manually.
        yield
        return
    from pathlib import Path
    import numpy as np
    from swissclim_evaluations.helpers import build_output_filename
    import xarray as xr

    def _fast_histograms(ds_target: xr.Dataset, ds_prediction: xr.Dataset, out_root: Path, plotting_cfg):
        section = out_root / "histograms"
        section.mkdir(parents=True, exist_ok=True)
        for var in ds_target.data_vars:
            out_npz = section / build_output_filename(
                metric="hist",
                variable=var,
                level="",  # mimic real empty skip
                qualifier="latbands_combined",
                init_time_range=None,
                lead_time_range=None,
                ensemble=None,
                ext="npz",
            )
            np.savez(out_npz, dummy=np.array([1, 2, 3]))

    def _fast_wd_kde(
        ds_target,
        ds_prediction,
        ds_target_std,
        ds_prediction_std,
        out_root: Path,
        plotting_cfg,
    ):
        section = out_root / "wd_kde"
        section.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        rows = []
        for var in ds_target.data_vars:
            out_npz = section / build_output_filename(
                metric="wd_kde",
                variable=var,
                level="",
                qualifier="combined",
                init_time_range=None,
                lead_time_range=None,
                ensemble=None,
                ext="npz",
            )
            np.savez(out_npz, mean_w=np.array([0.0]))
            rows.append({"variable": var, "hemisphere": "both", "lat_min": -90, "lat_max": 90, "wasserstein": 0.0})
        if rows:
            csv_path = section / build_output_filename(
                metric="wd_kde_wasserstein",
                variable=None,
                level=None,
                qualifier="averaged",
                init_time_range=None,
                lead_time_range=None,
                ensemble=None,
                ext="csv",
            )
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    import swissclim_evaluations.plots.histograms as _hist
    import swissclim_evaluations.plots.wd_kde as _kde
    monkeypatch.setattr(_hist, "run", _fast_histograms)
    monkeypatch.setattr(_kde, "run", _fast_wd_kde)
    yield

# ----------------------------------------------------------------------------------------

# Existing fast plotting surface simplifications -----------------------------------------
class _DummyImage:
    def __init__(self):
        self.cmap = None

class _DummyAxis:
    def __init__(self):
        self._title = ""
    def pcolormesh(self, *a, **k):
        return _DummyImage()
    def add_feature(self, *a, **k):
        return None
    def coastlines(self, *a, **k):
        return None
    def set_title(self, t):
        self._title = t
    def bar(self, *a, **k):
        return None
    def plot(self, *a, **k):
        return None

class _DummyFig:
    def __init__(self, axes):
        self._axes = axes
    def add_axes(self, *a, **k):
        return _DummyAxis()
    def colorbar(self, *a, **k):
        return None
    def savefig(self, *a, **k):
        return None
    def subplots_adjust(self, *a, **k):
        return None
    def tight_layout(self, *a, **k):
        return None

@pytest.fixture(autouse=True)
def _fast_plots(monkeypatch):
    import matplotlib.pyplot as plt

    def _fast_subplots(nrows=1, ncols=1, *a, squeeze=True, **k):
        if nrows == 1 and ncols == 1:
            axes = _DummyAxis()
        elif nrows == 1:
            axes = [_DummyAxis() for _ in range(ncols)]
        else:
            arr = _np.empty((nrows, ncols), dtype=object)
            for i in range(nrows):
                for j in range(ncols):
                    arr[i, j] = _DummyAxis()
            axes = arr
        return _DummyFig(axes), axes

    monkeypatch.setattr(plt, "subplots", _fast_subplots)
    monkeypatch.setattr(plt, "figure", lambda *a, **k: _DummyFig(_np.empty((1,1),dtype=object)))
    monkeypatch.setattr(plt, "gcf", lambda: _DummyFig(_np.empty((1,1),dtype=object)))
    monkeypatch.setattr(plt, "colorbar", lambda *a, **k: None)
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    class _DummyCRS:
        def _as_mpl_axes(self):
            from matplotlib.axes import Axes
            return Axes, {}
    try:
        import cartopy.crs as ccrs  # type: ignore
        monkeypatch.setattr(ccrs, "PlateCarree", lambda *a, **k: _DummyCRS())
    except Exception:
        pass

    try:
        import cartopy.feature as cfeature
        monkeypatch.setattr(cfeature, "BORDERS", object())
    except Exception:
        pass

    yield
