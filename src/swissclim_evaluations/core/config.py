from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from .. import console as c


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML file into a dict, returning an empty dict on empty files.

    Kept local to this module to avoid cross-imports so the CLI can run in isolation
    during tests where other modules are monkeypatched.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Ensure we return a plain dict for downstream mutation
    if not isinstance(data, dict):
        return {}
    return data


def ensure_output_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_config_to_output(cfg: dict[str, Any], out_root: Path) -> None:
    """If the CLI provided a config file path, copy it into the output folder.

    - Uses the original basename (e.g., config.yaml) to aid reproducibility.
    - Overwrites any existing file with the same name to reflect the last run.
    - Silently skips if the path is missing or invalid.
    """
    try:
        src_path = cfg.get("_config_path")
        if not src_path:
            return
        src = Path(str(src_path))
        if not src.exists() or not src.is_file():
            return
        dst = out_root / src.name
        try:
            # Avoid SameFileError if output_root is same folder as config
            if dst.resolve() == src.resolve():
                return
        except Exception:
            # If resolve fails (e.g., permissions), proceed with copy best-effort
            pass
        shutil.copy2(src, dst)
    except Exception:
        # Best-effort only; do not fail the run because of a copy issue
        pass


def _compact_cfg_value(value: Any, max_len: int = 120) -> str:
    text = "default" if value is None else repr(value)
    text = text.replace("\n", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _module_metric_threshold_summary(module: str, cfg: dict[str, Any]) -> tuple[str, str]:
    metrics_cfg = (cfg.get("metrics", {}) or {}) if isinstance(cfg, dict) else {}
    module_cfg = metrics_cfg.get(module, {}) if isinstance(metrics_cfg, dict) else {}

    if module == "deterministic":
        include = module_cfg.get("include") if isinstance(module_cfg, dict) else None
        std_include = (
            module_cfg.get("standardized_include") if isinstance(module_cfg, dict) else None
        )
        if include is None and std_include is None:
            metrics_sel = "default"
        else:
            metrics_sel = (
                f"include={_compact_cfg_value(include)}; "
                f"standardized_include={_compact_cfg_value(std_include)}"
            )

        fss_cfg = module_cfg.get("fss", {}) if isinstance(module_cfg, dict) else {}
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
        metrics_sel = "ETS"
        if isinstance(module_cfg, dict) and "thresholds" in module_cfg:
            thresholds_sel = _compact_cfg_value(module_cfg.get("thresholds"))
        else:
            thresholds_sel = "default"
        return metrics_sel, thresholds_sel

    if module == "probabilistic":
        return "CRPS, PIT, SSR", "n/a"

    if module == "vertical_profiles":
        return "NMAE", "n/a"

    if module == "energy_spectra":
        return "LSD", "n/a"

    if module == "multivariate":
        module_cfg = cfg.get("metrics", {}).get("multivariate", {})
        pairs = module_cfg.get("bivariate_pairs", [])
        n_pairs = len(pairs) if isinstance(pairs, list) else 0
        return f"Bivariate Histograms ({n_pairs} pairs)", "n/a"

    return "n/a", "n/a"


def print_module_config_summary(cfg: dict[str, Any], chapter_flags: dict[str, Any]) -> None:
    module_order = [
        "maps",
        "histograms",
        "wd_kde",
        "energy_spectra",
        "vertical_profiles",
        "deterministic",
        "ets",
        "probabilistic",
        "multivariate",
    ]
    c.section("Configured Metrics/Thresholds")
    for module in module_order:
        enabled = bool(chapter_flags.get(module))
        metrics_sel, thresholds_sel = _module_metric_threshold_summary(module, cfg)
        c.info(
            f"[{module}] enabled={enabled} | metrics={metrics_sel} | thresholds={thresholds_sel}"
        )


def resolve_dask_performance_report(performance_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve optional Dask performance-report settings from performance config."""

    def _as_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "no", "n", "off"}:
                return False
        try:
            return bool(value)
        except Exception:
            return default

    enabled = _as_bool(performance_cfg.get("dask_performance_report"), False)
    path_template = str(
        performance_cfg.get(
            "dask_performance_report_path",
            "logs/dask_performance_{job_id}_{timestamp}.html",
        )
    )

    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    try:
        path = path_template.format(job_id=job_id, timestamp=timestamp)
    except Exception:
        path = path_template

    return {
        "enabled": enabled,
        "path": path,
    }


def resolve_dask_profile(performance_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve Dask worker/client settings from performance config.

    Safety-first defaults are used unless explicitly overridden in YAML.
    """

    def _as_int(value: Any, default: int) -> int:
        try:
            out = int(value)
            return out if out > 0 else default
        except Exception:
            return default

    def _as_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "no", "n", "off"}:
                return False
        try:
            return bool(value)
        except Exception:
            return default

    cpu_count = max(1, int(os.cpu_count() or 1))
    is_gh200_class = cpu_count >= 192
    profile = str(performance_cfg.get("dask_profile", "safe")).strip().lower()

    if is_gh200_class:
        safe_workers = max(1, min(cpu_count, 16))
        balanced_workers = max(1, min(cpu_count, 32))
        fast_workers = max(1, min(cpu_count, 64))
        gh200_defaults = {
            "safe": {
                "profile": "safe",
                "n_workers": safe_workers,
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": "24GiB",
            },
            "balanced": {
                "profile": "balanced",
                "n_workers": balanced_workers,
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": "12GiB",
            },
            "fast": {
                "profile": "fast",
                "n_workers": fast_workers,
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": "8GiB",
            },
        }
        defaults = gh200_defaults.get(profile, gh200_defaults["safe"])
    else:
        if profile == "fast":
            defaults = {
                "profile": "fast",
                "n_workers": max(1, min(cpu_count, 64)),
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": "8GiB",
            }
        elif profile == "balanced":
            defaults = {
                "profile": "balanced",
                "n_workers": max(1, min(cpu_count, 32)),
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": "12GiB",
            }
        else:
            defaults = {
                "profile": "safe",
                "n_workers": max(1, min(cpu_count, 16)),
                "threads_per_worker": 1,
                "processes": True,
                "memory_limit": "24GiB",
            }

    default_n_workers = _as_int(defaults.get("n_workers"), 1)
    default_threads_per_worker = _as_int(defaults.get("threads_per_worker"), 1)
    default_processes = _as_bool(defaults.get("processes"), False)

    n_workers = _as_int(performance_cfg.get("dask_n_workers"), default_n_workers)
    threads_per_worker = _as_int(
        performance_cfg.get("dask_threads_per_worker"),
        default_threads_per_worker,
    )
    processes = _as_bool(performance_cfg.get("dask_processes"), default_processes)
    memory_limit = performance_cfg.get("dask_memory_limit", defaults["memory_limit"])

    # -----------------------------------------------------------------------
    # Honour the per-eval memory budget injected by launchscript_multi.sh.
    #
    # The launchscript exports two env vars to prevent OOM when multiple evals
    # run in parallel on the same node:
    #   SWISSCLIM_DASK_MEMORY_BUDGET_GIB   – total GiB assigned to this eval
    #   SWISSCLIM_DASK_MEMORY_BUDGET_FRACTION – fraction given to Dask workers
    #
    # If set, derive per-worker memory limit instead of using hardcoded
    # profile defaults.  SLURM_CPUS_PER_TASK also caps the worker count so
    # we never spawn more workers than CPUs actually allocated to this task.
    # -----------------------------------------------------------------------
    budget_gib_env = os.environ.get("SWISSCLIM_DASK_MEMORY_BUDGET_GIB", "")
    budget_fraction_env = os.environ.get("SWISSCLIM_DASK_MEMORY_BUDGET_FRACTION", "")
    slurm_cpus_env = os.environ.get("SLURM_CPUS_PER_TASK", "")

    if budget_gib_env:
        try:
            budget_gib = float(budget_gib_env)
            fraction = float(budget_fraction_env) if budget_fraction_env else 0.9
            fraction = max(0.1, min(1.0, fraction))
            total_dask_gib = budget_gib * fraction

            # Cap worker count to CPUs available to this Slurm task
            if slurm_cpus_env:
                slurm_cpus = max(1, int(slurm_cpus_env))
                n_workers = min(n_workers, slurm_cpus)

            # Ensure at least 1 worker; derive per-worker limit
            n_workers = max(1, n_workers)
            per_worker_gib = total_dask_gib / n_workers
            # Keep at least 1 GiB so Dask doesn't complain
            per_worker_gib = max(1.0, per_worker_gib)
            # Use config override if explicitly provided, else use derived value
            if not performance_cfg.get("dask_memory_limit"):
                memory_limit = f"{per_worker_gib:.1f}GiB"
        except (ValueError, ZeroDivisionError):
            pass  # Fall back to unmodified defaults

    return {
        "profile": defaults["profile"],
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "processes": processes,
        "memory_limit": memory_limit,
    }
