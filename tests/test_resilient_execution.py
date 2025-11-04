from __future__ import annotations

from pathlib import Path

import swissclim_evaluations.cli as cli

from ._smoke_data import make_synthetic_datasets


def test_module_failure_is_captured(monkeypatch, tmp_path: Path, capsys):
    # Prepare minimal config enabling deterministic and ets (we will force deterministic to raise)
    cfg = {
        "paths": {},
        "modules": {
            "deterministic": True,
            "ets": True,
        },
        "plotting": {},
    }

    # Monkeypatch deterministic.run to raise
    import swissclim_evaluations.metrics.deterministic as det_mod

    def boom(*args, **kwargs):  # noqa: D401
        raise RuntimeError("boom for testing")

    monkeypatch.setattr(det_mod, "run", boom)

    # Monkeypatch data loading to use synthetic datasets
    import swissclim_evaluations.data as data_mod

    t, p = make_synthetic_datasets(with_ensemble=False)

    monkeypatch.setattr(data_mod, "era5", lambda *a, **k: t)
    monkeypatch.setattr(data_mod, "open_ml", lambda *a, **k: p)

    # Run
    cli.run_selected(cfg)

    out = capsys.readouterr().out
    # Expect error line and ets success line
    assert "deterministic failed" in out
    # Module results summary present
    assert "Module Results" in out
    # The other module (ets) should have succeeded (its run prints saved messages or timing summary)
    assert "ets" in out
