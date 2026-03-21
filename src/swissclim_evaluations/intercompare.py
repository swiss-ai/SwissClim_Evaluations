from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
from swissclim_evaluations.intercomparison.modules.ssim import intercompare_ssim
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
    # additional aliases used in config/docs
    "deterministic_metrics": "metrics",
    "energy": "spectra",
    "vertical": "vprof",
}


MODULE_INPUT_PATTERNS: dict[str, tuple[str, ...]] = {
    "maps": ("maps/map_*.npz",),
    "hist": ("histograms/hist_*.npz",),
    "kde": (
        "wd_kde/wd_kde_*.npz",
        "wd_kde/wd_kde_evolve_*_ridgeline_data*.npz",
        "wd_kde/wd_kde_wasserstein_averaged_*.csv",
    ),
    "spectra": (
        "energy_spectra/energy_spectrum_*.npz",
        "energy_spectra/energy_spectra_per_lead_*.npz",
        "energy_spectra/energy_spectra_per_lead_*_bundle*.npz",
        "energy_spectra/energy_ratios_*.csv",
        "energy_spectra/energy_ratios_3d_*.csv",
    ),
    "vprof": ("vertical_profiles/vertical_profiles_nmae_*.npz",),
    "metrics": ("deterministic/deterministic_metrics*.csv",),
    "ets": (
        "ets/ets_metrics*.csv",
        "ets/ets_line*by_lead*.csv",
    ),
    "prob": (
        "probabilistic/crps_summary*.csv",
        "probabilistic/crps_line*.csv",
        "probabilistic/ssr*.csv",
        "probabilistic/ssr_line*.csv",
        "probabilistic/pit_hist*.npz",
        "probabilistic/crps_map_*.npz",
    ),
}


def _module_has_inputs(module: str, models: list[Path]) -> bool:
    patterns = MODULE_INPUT_PATTERNS.get(module, ())
    if not patterns:
        return True
    for model_root in models:
        for pattern in patterns:
            if any(model_root.glob(pattern)):
                return True
    return False


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


def _compact_cfg_value(value: Any, max_len: int = 120) -> str:
    text = "default" if value is None else repr(value)
    text = text.replace("\n", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _module_metric_threshold_summary(module: str, cfg: dict[str, Any]) -> tuple[str, str]:
    metrics_cfg = (cfg.get("metrics", {}) or {}) if isinstance(cfg, dict) else {}

    if module == "metrics":
        det_cfg = metrics_cfg.get("deterministic", {}) if isinstance(metrics_cfg, dict) else {}
        include = det_cfg.get("include") if isinstance(det_cfg, dict) else None
        std_include = det_cfg.get("standardized_include") if isinstance(det_cfg, dict) else None
        report_per_level = (
            bool(det_cfg.get("report_per_level", True)) if isinstance(det_cfg, dict) else True
        )
        if include is None and std_include is None:
            metrics_sel = f"default; report_per_level={report_per_level}"
        else:
            metrics_sel = (
                f"include={_compact_cfg_value(include)}; "
                f"standardized_include={_compact_cfg_value(std_include)}; "
                f"report_per_level={report_per_level}"
            )

        fss_cfg = det_cfg.get("fss", {}) if isinstance(det_cfg, dict) else {}
        if isinstance(fss_cfg, dict):
            if "thresholds" in fss_cfg:
                thresholds_sel = f"fss.thresholds={_compact_cfg_value(fss_cfg.get('thresholds'))}"
            elif "threshold" in fss_cfg:
                thresholds_sel = f"fss.threshold={_compact_cfg_value(fss_cfg.get('threshold'))}"
            elif "quantile" in fss_cfg:
                thresholds_sel = f"fss.quantile={_compact_cfg_value(fss_cfg.get('quantile'))}"
            else:
                thresholds_sel = "default"
        else:
            thresholds_sel = "default"
        return metrics_sel, thresholds_sel

    if module == "ets":
        ets_cfg = metrics_cfg.get("ets", {}) if isinstance(metrics_cfg, dict) else {}
        report_per_level = (
            bool(ets_cfg.get("report_per_level", True)) if isinstance(ets_cfg, dict) else True
        )
        metrics_sel = f"ETS; report_per_level={report_per_level}"
        if isinstance(ets_cfg, dict) and "thresholds" in ets_cfg:
            thresholds_sel = _compact_cfg_value(ets_cfg.get("thresholds"))
        else:
            thresholds_sel = "default"
        return metrics_sel, thresholds_sel

    if module == "prob":
        prob_cfg = metrics_cfg.get("probabilistic", {}) if isinstance(metrics_cfg, dict) else {}
        report_per_level = (
            bool(prob_cfg.get("report_per_level", True)) if isinstance(prob_cfg, dict) else True
        )
        return f"CRPS, PIT, SSR; report_per_level={report_per_level}", "n/a"
    if module == "vprof":
        return "NMAE", "n/a"
    if module == "spectra":
        spec_cfg = metrics_cfg.get("energy_spectra", {}) if isinstance(metrics_cfg, dict) else {}
        report_per_level = (
            bool(spec_cfg.get("report_per_level", True)) if isinstance(spec_cfg, dict) else True
        )
        individual_plots = (
            bool(spec_cfg.get("individual_plots", False)) if isinstance(spec_cfg, dict) else False
        )
        return (
            f"LSD (+ Δ spectrogram compare); report_per_level={report_per_level}; "
            f"individual_plots={individual_plots}",
            "n/a",
        )
    return "n/a", "n/a"


def _print_module_config_summary(mods: set[str], cfg: dict[str, Any]) -> None:
    module_order = ["maps", "hist", "kde", "spectra", "vprof", "metrics", "ets", "prob"]
    c.section("Configured Metrics/Thresholds")
    for module in module_order:
        enabled = module in mods
        metrics_sel, thresholds_sel = _module_metric_threshold_summary(module, cfg)
        c.info(
            f"[{module}] enabled={enabled} | metrics={metrics_sel} | thresholds={thresholds_sel}"
        )


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

    # Energy spectra individual plots flag (default: False — only spectrograms for multi-lead)
    spectra_cfg = (cfg.get("metrics") or {}).get("energy_spectra") or {}
    energy_spectra_individual_plots = bool(spectra_cfg.get("individual_plots", False))

    # Light validation: warn on missing model dirs
    for m in models:
        if not m.exists():
            c.print(f"[intercompare] WARNING: model folder does not exist: {m}")

    mods = _normalize_modules(modules)
    requested_mods = set(mods)
    missing_inputs = sorted([m for m in requested_mods if not _module_has_inputs(m, models)])
    if missing_inputs:
        c.warn(
            "[intercompare] Skipping modules with no matching source artifacts: "
            + ", ".join(missing_inputs)
        )
        mods = requested_mods.difference(missing_inputs)

    _print_module_config_summary(mods, cfg)
    if "maps" in mods:
        intercompare_maps(models, labels, out_root, max_panels=max_map_panels)
    if "hist" in mods:
        intercompare_histograms(models, labels, out_root)
    if "kde" in mods:
        intercompare_wd_kde(models, labels, out_root)
    if "spectra" in mods:
        intercompare_energy_spectra(
            models, labels, out_root, individual_plots=energy_spectra_individual_plots
        )
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
    if "ssim" in mods:
        intercompare_ssim(models, labels, out_root)

    c.success("Intercomparison finished.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    run_from_config(cfg)


if __name__ == "__main__":
    main()
