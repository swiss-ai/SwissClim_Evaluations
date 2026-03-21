from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import matplotlib

# Force non-interactive backend for tests
matplotlib.use("Agg")

import numpy as _np
import pytest

# Insert the workspace's src/ at the beginning of sys.path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Fast algorithm stubs (applied unless SC_PLOTS_FULL=1) ---------------------------------
@pytest.fixture(autouse=True)
def _fast_plot_algorithms(monkeypatch):
    if os.environ.get("SC_PLOTS_FULL", "0") == "1":
        # Allow opting out to run full heavy routines manually.
        yield
        return
    import numpy as np
    import xarray as xr

    from swissclim_evaluations.helpers import build_output_filename

    def _fast_histograms(
        ds_target: xr.Dataset,
        ds_prediction: xr.Dataset,
        out_root: Path,
        plotting_cfg,
    ):
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
            rows.append(
                {
                    "variable": var,
                    "hemisphere": "both",
                    "lat_min": -90,
                    "lat_max": 90,
                    "wasserstein": 0.0,
                }
            )
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
class _DummyLine:
    """Minimal Line2D stand-in for tests that unpack ax.plot() results."""

    pass


class _DummyImage:
    def __init__(self):
        self.cmap = None


class _DummyAxis:
    def __init__(self):
        self._title = ""
        self.transAxes = object()
        self._xlim = (0, 1)
        self._ylim = (0, 1)

    def pcolormesh(self, *a, **k):
        return _DummyImage()

    def add_feature(self, *a, **k):
        return None

    def coastlines(self, *a, **k):
        return None

    def set_title(self, t, *a, **k):
        self._title = t
        return None

    def bar(self, *a, **k):
        return None

    def plot(self, *a, **k):
        return [_DummyLine()]

    def fill_between(self, *a, **k):
        return None

    def fill_betweenx(self, *a, **k):
        return None

    def loglog(self, *a, **k):
        return None

    def set_xlabel(self, *a, **k):
        return None

    def set_ylabel(self, *a, **k):
        return None

    def set_yticks(self, *a, **k):
        return None

    def set_yticklabels(self, *a, **k):
        return None

    def legend(self, *a, **k):
        return None

    def set_xlim(self, a, b=None, *args, **kwargs):
        if b is None and hasattr(a, "__len__") and len(a) == 2:
            self._xlim = (a[0], a[1])
        else:
            self._xlim = (a, b)
        return None

    def set_ylim(self, a, b=None, *args, **kwargs):
        if b is None and hasattr(a, "__len__") and len(a) == 2:
            self._ylim = (a[0], a[1])
        else:
            self._ylim = (a, b)
        return None

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def text(self, *a, **k):
        return None

    def twiny(self, *a, **k):
        return _DummyAxis()

    def set_xscale(self, *a, **k):
        return None

    def set_yscale(self, *a, **k):
        return None

    def set_aspect(self, *a, **k):
        return None

    def set_xticks(self, *a, **k):
        return None

    def set_xticklabels(self, *a, **k):
        return None

    def grid(self, *a, **k):
        return None

    def tick_params(self, *a, **k):
        return None

    def invert_yaxis(self):
        return None

    def axhline(self, *a, **k):
        return None

    def axvline(self, *a, **k):
        return None

    def semilogx(self, *a, **k):
        return None

    def get_lines(self):
        return []

    def get_xticklabels(self):
        return []

    def contourf(self, *a, **k):
        return _DummyImage()

    def contour(self, *a, **k):
        return _DummyImage()

    def get_figure(self):
        return _DummyFig(self)


class _DummyXAxis:
    def set_major_locator(self, *a, **k):
        return None

    def set_major_formatter(self, *a, **k):
        return None


class _DummyColorbarAx:
    xaxis = _DummyXAxis()
    yaxis = _DummyXAxis()


class _DummyColorbar:
    ax = _DummyColorbarAx()

    def set_label(self, *a, **k):
        return None

    def add_lines(self, *a, **k):
        return None


class _DummyFig:
    def __init__(self, axes):
        self._axes = axes

    def add_axes(self, *a, **k):
        return _DummyAxis()

    def add_subplot(self, *a, **k):
        return _DummyAxis()

    def colorbar(self, *a, **k):
        return _DummyColorbar()

    def savefig(self, *a, **k):
        if a and isinstance(a[0], str | Path):
            Path(a[0]).touch()
        return None

    def subplots_adjust(self, *a, **k):
        return None

    def tight_layout(self, *a, **k):
        return None

    def suptitle(self, *a, **k):
        return None


# Re-define fixture with improved axes shape logic
@pytest.fixture(autouse=True)
def _fast_plots(monkeypatch):
    import matplotlib.pyplot as plt

    def _fast_subplots(nrows=1, ncols=1, *a, squeeze=True, **k):
        if not squeeze:
            # Always return 2D array if squeeze=False
            arr = _np.empty((nrows, ncols), dtype=object)
            for i in range(nrows):
                for j in range(ncols):
                    arr[i, j] = _DummyAxis()
            axes = arr
            return _DummyFig(axes), axes

        if nrows == 1 and ncols == 1:
            axes = _DummyAxis()
        elif nrows == 1:  # return 1D list like real matplotlib for single row multi-col
            axes = [_DummyAxis() for _ in range(ncols)]
        elif ncols == 1:  # multi-row single column -> 1D list
            axes = [_DummyAxis() for _ in range(nrows)]
        else:  # true 2D grid
            arr = _np.empty((nrows, ncols), dtype=object)
            for i in range(nrows):
                for j in range(ncols):
                    arr[i, j] = _DummyAxis()
            axes = arr
        return _DummyFig(axes), axes

    monkeypatch.setattr(plt, "subplots", _fast_subplots)
    monkeypatch.setattr(
        plt,
        "figure",
        lambda *a, **k: _DummyFig(_np.empty((1, 1), dtype=object)),
    )
    monkeypatch.setattr(plt, "gcf", lambda: _DummyFig(_np.empty((1, 1), dtype=object)))
    monkeypatch.setattr(plt, "colorbar", lambda *a, **k: None)

    def _mock_savefig(*args, **kwargs):
        if args and isinstance(args[0], str | Path):
            Path(args[0]).touch()

    monkeypatch.setattr(plt, "savefig", _mock_savefig)
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
    try:
        import swissclim_evaluations.plots.histograms as _hist
        import swissclim_evaluations.plots.wd_kde as _kde

        monkeypatch.setattr(_hist, "_lat_bands", lambda: ([-90, -30, 0, 30, 90], 4, 2))
        monkeypatch.setattr(_kde, "_lat_bands", lambda: ([-90, -30, 0, 30, 90], 4, 2))
    except Exception:
        pass
    try:
        import swissclim_evaluations.metrics.vertical_profiles as _vp

        monkeypatch.setattr(_vp, "_lat_bands", lambda: ([-90, 0, 90], 2))
    except Exception:
        pass
    dummy_for_gca = _DummyAxis()
    monkeypatch.setattr(plt, "gca", lambda *a, **k: dummy_for_gca)
    yield


# ----------------------------------------------------------------------------------------
# Ensure default output directories never persist in repo root during tests


@pytest.fixture(scope="session", autouse=True)
def _isolate_default_output_root():
    """Redirect any use of the CLI's default output path ("output/verification_esfm")
    into a pytest-managed temporary directory and clean everything afterwards.

    This guarantees that running the test suite leaves no persistent artifacts
    in the project working tree (especially accidental creation of ./output/...).
    """
    import shutil
    import tempfile

    from swissclim_evaluations import cli as _cli

    base = Path(tempfile.mkdtemp(prefix="sc_eval_outputs_"))

    def _redirecting_ensure_output(path: str | os.PathLike[str]):  # type: ignore[name-defined]
        p = Path(path)
        # Force relative paths (like the default 'output/verification_esfm') under the temp base
        if not p.is_absolute():
            p = base / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Monkeypatch by direct assignment (session scope)
    with contextlib.suppress(Exception):
        _cli._ensure_output = _redirecting_ensure_output  # type: ignore[attr-defined]

    yield

    # Cleanup temp base
    with contextlib.suppress(Exception):
        shutil.rmtree(base, ignore_errors=True)
