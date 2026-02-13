from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from swissclim_evaluations.intercomparison.core import (
    as_paths,
    ensure_dir,
    model_label,
)
from swissclim_evaluations.intercomparison.modules.deterministic import (
    intercompare_deterministic_metrics,
)
from swissclim_evaluations.intercomparison.modules.energy_spectra import (
    intercompare_energy_spectra,
)
from swissclim_evaluations.intercomparison.modules.ets import intercompare_ets_metrics
from swissclim_evaluations.intercomparison.modules.histograms import (
    intercompare_histograms,
)
from swissclim_evaluations.intercomparison.modules.maps import intercompare_maps
from swissclim_evaluations.intercomparison.modules.probabilistic import (
    intercompare_probabilistic,
)
from swissclim_evaluations.intercomparison.modules.vertical_profiles import (
    intercompare_vertical_profiles,
)
from swissclim_evaluations.intercomparison.modules.wd_kde import intercompare_wd_kde

from . import console as c


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SwissClim Evaluations — Intercomparison runner (YAML-configured)"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to intercomparison YAML config (models, labels, output_root, modules)",
    )
    return p


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


MODULE_ALIASES: dict[str, str] = {
    # canonical
    "maps": "maps",
    "hist": "hist",
    "kde": "kde",
    "spectra": "spectra",
    "vprof": "vprof",
    "metrics": "metrics",
    "ets": "ets",
    "prob": "prob",
    # long-form aliases
    "histograms": "hist",
    "wd_kde": "kde",
    "energy_spectra": "spectra",
    "vertical_profiles": "vprof",
    "deterministic": "metrics",
    "probabilistic": "prob",
}


def _normalize_modules(modules: list[str]) -> set[str]:
    """Normalize module names from config to canonical keys used by dispatch."""
    normalized: set[str] = set()
    unknown: list[str] = []
    for module_name in modules:
        key = str(module_name).lower().strip()
        mapped = MODULE_ALIASES.get(key)
        if mapped is None:
            unknown.append(str(module_name))
            continue
        normalized.add(mapped)

    if unknown:
        c.warn(f"[intercompare] Unknown modules ignored: {', '.join(sorted(unknown))}")
    return normalized


def run_from_config(cfg: dict) -> None:
    # Resolve models
    model_strs = cfg.get("models") or []
    if not isinstance(model_strs, list) or len(model_strs) < 2:
        raise ValueError("config.models must be a list with at least two model directories")
    models = as_paths(model_strs)
    # Labels
    labels = cfg.get("labels") or []
    if not labels or len(labels) != len(models):
        labels = [model_label(p) for p in models]
    # Output root
    out_root = ensure_dir(Path(cfg.get("output_root", "output/intercomparison")).resolve())
    c.info(f"Output directory: {out_root}")
    # Modules
    modules = cfg.get("modules") or [
        "spectra",
        "hist",
        "kde",
        "maps",
        "metrics",
        "ets",
        "prob",
        "vprof",
    ]
    modules = [str(m).lower() for m in modules]
    # Other options
    max_map_panels = int(cfg.get("max_map_panels", 4))
    max_crps_map_panels = int(cfg.get("max_crps_map_panels", 4))

    # Light validation: warn on missing model dirs
    for m in models:
        if not m.exists():
            c.print(f"[intercompare] WARNING: model folder does not exist: {m}")

    mods = _normalize_modules(modules)
    if "maps" in mods:
        intercompare_maps(models, labels, out_root, max_panels=max_map_panels)
    if "hist" in mods:
        intercompare_histograms(models, labels, out_root)
    if "kde" in mods:
        intercompare_wd_kde(models, labels, out_root)
    if "spectra" in mods:
        intercompare_energy_spectra(models, labels, out_root)
    if "vprof" in mods:
        intercompare_vertical_profiles(models, labels, out_root)
    if "metrics" in mods:
        intercompare_deterministic_metrics(models, labels, out_root)
    if "ets" in mods:
        intercompare_ets_metrics(models, labels, out_root)
    if "prob" in mods:
        intercompare_probabilistic(
            models, labels, out_root, max_crps_map_panels=max_crps_map_panels
        )

    c.success("Intercomparison finished.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    run_from_config(cfg)


if __name__ == "__main__":
    main()
