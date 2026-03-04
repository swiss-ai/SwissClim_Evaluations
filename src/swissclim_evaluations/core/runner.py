from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import console as c
from ..dask_utils import (
    resolve_module_batching_options,
)
from ..data import add_derived_variables
from ..helpers import (
    format_ensemble_log,
    resolve_ensemble_mode,
    validate_and_normalize_ensemble_config,
)
from ..lead_time_policy import LeadTimePolicy
from . import config as config_mod, data_selection


def setup_dask_logging(log_file: str = "logs/dask_distributed.log") -> None:
    """Redirects dask distributed logs to a file and suppresses them from stderr."""
    import logging

    try:
        # Append SLURM_JOB_ID and SLURM_PROCID to filename if available to avoid collisions
        job_id = os.environ.get("SLURM_JOB_ID")
        proc_id = os.environ.get("SLURM_PROCID")

        if job_id:
            p = Path(log_file)
            suffix = f"_{job_id}"
            if proc_id:
                suffix += f"_{proc_id}"
            log_file = str(p.parent / f"{p.stem}{suffix}{p.suffix}")

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        dask_logger = logging.getLogger("distributed")
        dask_logger.propagate = False

        # Clear existing handlers
        if dask_logger.hasHandlers():
            dask_logger.handlers.clear()

        fh = logging.FileHandler(log_path, mode="w")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        dask_logger.addHandler(fh)

        # Capture standard warnings (like those from contextlib or distributed.client)
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.propagate = False
        if warnings_logger.hasHandlers():
            warnings_logger.handlers.clear()
        warnings_logger.addHandler(fh)
    except Exception as e:
        c.print(f"Failed to setup dask logging: {e}")


def run_selected(cfg: dict[str, Any]) -> None:
    c.header("SwissClim Evaluations")

    out_root = config_mod.ensure_output_dir(
        cfg.get("paths", {}).get("output_root", "output/verification_esfm")
    )
    # Persist the exact configuration used for this run into the output directory
    config_mod.copy_config_to_output(cfg, out_root)

    t0 = time.time()
    module_timings: list[tuple[str, float]] = []
    # Track per-module outcomes: name, status(success|failed|skipped), seconds, optional error
    module_results: list[dict[str, Any]] = []
    # Prepare datasets (this also parses and attaches __lead_time_policy onto cfg)
    (
        ds_target,
        ds_prediction,
        ds_target_std,
        ds_prediction_std,
    ) = data_selection.prepare_datasets(cfg)

    # Apply user-defined derived variables (e.g. wind speed/direction from U+V).
    # Runs before any module so all modules see the derived variables automatically.
    derived_cfg = cfg.get("derived_variables") or {}
    if derived_cfg:
        ds_target, ds_prediction = add_derived_variables(ds_target, ds_prediction, derived_cfg)

    # Retrieve the parsed lead time policy AFTER preparation
    lead_policy = cfg.get("__lead_time_policy")

    try:
        plotting_block = cfg.setdefault("plotting", {})
        if "lead_panel_hours" in plotting_block:
            del plotting_block["lead_panel_hours"]
    except Exception:
        # Ignore errors when removing legacy config keys; not critical for pipeline execution.
        pass

    # Derive per-plot datasets if a specific plot datetime is requested
    ds_target_plot, ds_prediction_plot = data_selection.select_plot_datetime(
        ds_target, ds_prediction, cfg
    )
    # For maps only: optionally subset ensemble members and/or a single datetime
    # Other modules use full datasets (no plot-time/ensemble filtering)
    ds_prediction_plot, _ = data_selection.select_plot_ensemble(
        ds_prediction_plot, ds_prediction_std, cfg
    )

    chapter_flags = cfg.get("modules", {})
    config_mod.print_module_config_summary(cfg, chapter_flags)
    plotting = cfg.get("plotting", {})
    mode = str(plotting.get("output_mode", "plot")).lower()
    if mode not in {"plot", "npz", "both", "none"}:
        c.warn(
            f"Unsupported plotting.output_mode='{mode}'. Falling back to 'plot'. "
            "Allowed: plot|npz|both|none."
        )
        mode = "plot"
        plotting["output_mode"] = mode
    # Validate / normalize ensemble block early to surface typos (e.g. 'member').
    # Support ensemble block under either top-level or selection (example_config uses selection).
    raw_ensemble_top = cfg.get("ensemble", {}) or {}
    raw_ensemble_sel = (cfg.get("selection", {}) or {}).get("ensemble", {}) or {}
    if raw_ensemble_top and raw_ensemble_sel:
        # Merge giving precedence to top-level definitions.
        merged = {**raw_ensemble_sel, **raw_ensemble_top}
        c.warn(
            "Both top-level 'ensemble' and 'selection.ensemble' blocks present; "
            "top-level keys take precedence where duplicated."
        )
        raw_ensemble_cfg = merged
    elif raw_ensemble_top:
        raw_ensemble_cfg = raw_ensemble_top
    else:
        raw_ensemble_cfg = raw_ensemble_sel
    has_ens = "ensemble" in ds_prediction.dims
    ensemble_cfg, ens_warnings = validate_and_normalize_ensemble_config(raw_ensemble_cfg, has_ens)
    # Defer printing of warnings until after dataset summary so all ensemble info appears together.
    fallback_notes = [w for w in ens_warnings if w.startswith("[ensemble-fallback]")]
    other_notes = [w for w in ens_warnings if w not in fallback_notes]
    # Compute resolved modes (post-normalization) using resolver for transparency.
    module_names = [
        "maps",
        "histograms",
        "wd_kde",
        "energy_spectra",
        "vertical_profiles",
        "deterministic",
        "ets",
        "probabilistic",
        "ssim",
        "multivariate",
    ]
    resolved_modes: dict[str, str] = {}
    for _m in module_names:
        req = ensemble_cfg.get(_m)
        try:
            resolved_modes[_m] = resolve_ensemble_mode(_m, req, ds_target, ds_prediction)
        except Exception:
            # Fallback; shouldn't happen
            resolved_modes[_m] = req or "mean"
    # We'll show resolved modes later with fallbacks & summary

    # Basic overview
    all_vars = list(ds_target.data_vars)
    # Classify variables: check if 'level' is in dims, regardless of size
    if "level" in ds_target.dims:
        vars_3d = [v for v in all_vars if "level" in ds_target[v].dims]
        vars_2d = [v for v in all_vars if v not in vars_3d]
    else:
        vars_3d = []
        vars_2d = all_vars
    c.panel(
        (
            f"Output: [bold]{out_root}[/]"
            f"\nPlotting Mode: [bold]{mode}[/]"
            f"\nVariables → 2D: [bold]{len(vars_2d)}[/], 3D: [bold]{len(vars_3d)}[/]"
        ),
        title="Run Overview",
        style="cyan",
    )

    # Show the prepared model dataset and describe ensemble handling
    c.section("Model dataset (prepared)")
    # printing the Dataset object provides a concise summary (dims/coords/vars)
    try:
        from ..console import (
            USE_RICH,
            console as _rc,
        )

        if USE_RICH:
            from rich.pretty import Pretty

            _rc.print(Pretty(ds_prediction))
        else:
            c.print(ds_prediction)
    except Exception:
        c.print(ds_prediction)
    # Consolidated ensemble information (fallbacks + resolved modes + high-level message)
    try:
        ens_msg = data_selection._ensemble_handling_message(ds_prediction, cfg, resolved_modes)
        blocks: list[str] = []
        if fallback_notes:
            blocks.append("Fallbacks:\n" + "\n".join(fallback_notes))
        if other_notes:
            blocks.append("Notes:\n" + "\n".join(other_notes))
        blocks.append(
            "Resolved Modes:\n" + "\n".join(f"{m}: {resolved_modes[m]}" for m in module_names)
        )
        blocks.append("ℹ️  Summary:\n" + ens_msg)
        c.panel(
            "\n\n".join(blocks),
            title="Ensemble Configuration",
            style="blue",
        )
    except Exception:
        pass

    # Lead time policy summary (multi-lead visibility)
    if lead_policy is not None:
        try:
            if isinstance(lead_policy, LeadTimePolicy):  # runtime guard
                hours = []
                if "lead_time" in ds_prediction.dims:
                    hrs = (ds_prediction["lead_time"].values // np.timedelta64(1, "h")).astype(int)
                    hours = hrs.tolist()
                mode = lead_policy.mode
                details: list[str] = [f"mode={mode}"]
                if lead_policy.stride_hours:
                    details.append(f"stride={lead_policy.stride_hours}h")
                if lead_policy.subset_hours:
                    details.append(f"subset={lead_policy.subset_hours}")
                if lead_policy.max_hour is not None:
                    details.append(f"max_hour={lead_policy.max_hour}")
                # 'bins' mode removed
                policy_text = "Lead time policy → " + ", ".join(details)
                if USE_RICH:
                    c.print(f"[magenta][bold]{policy_text}[/]")
                else:
                    c.print(policy_text)

                # Compact lead-time snapshot (moved down near policy)
                try:
                    audit = cfg.get("__lead_time_audit") or []

                    def _fmt(step_key: str, label: str) -> str | None:
                        for e in audit:
                            if e.get("step") == step_key and "prediction" in e:
                                pred = e["prediction"]
                                sample = pred.get("sample", [])
                                return (
                                    f"Available lead hours {label}: count={pred.get('count', 0)} "
                                    f"sample={sample[:8]}"
                                )
                        return None

                    lines = []
                    s1 = _fmt("after open", "before selection")
                    if s1:
                        lines.append(s1)
                    s2 = _fmt("after selection", "after selection")
                    if s2:
                        lines.append(s2)

                    for ln in lines:
                        if USE_RICH:
                            c.print(f"[cyan][bold]{ln}[/]")
                        else:
                            c.print(ln)
                except Exception as exc:
                    c.warn(f"Failed to log lead-time audit information: {exc}")
                # Extra visibility: warn if multi-lead requested but only a single lead remains
                try:
                    if isinstance(lead_policy, LeadTimePolicy):
                        multi_requested = lead_policy.mode != "first"
                    else:
                        multi_requested = False
                except Exception:
                    multi_requested = False
                if (
                    multi_requested
                    and ("lead_time" in ds_prediction.dims)
                    and int(ds_prediction.lead_time.size) == 1
                ):
                    c.warn(
                        "Multi-lead policy requested but only a single lead is present after "
                        "selection/alignment. This typically means either your prediction store "
                        "has a single lead, your date window clips most valid times, or "
                        "target/prediction time alignment leaves only one overlapping offset. "
                        "Consider widening 'selection.datetimes', setting lead_time.mode=full "
                        "temporarily, or inspecting the prediction Zarr."
                    )
        except Exception as ex:
            c.warn(f"Exception occurred during lead time policy handling: {ex}")

    # Get performance configuration
    performance_cfg = cfg.get("performance", {}) or {}

    # Log effective Dask execution mode used by modules
    quiet_dask_logs = bool(performance_cfg.get("quiet_dask_logs", False))
    batch_opts = resolve_module_batching_options(
        performance_cfg=performance_cfg,
        default_split_level=True,
    )
    msg = (
        "Dask Execution: direct dask.compute graph execution "
        f"(split_3d_by_level={bool(batch_opts['split_level'])})"
    )
    if not quiet_dask_logs:
        if USE_RICH:
            c.print(f"[cyan][bold]{msg}[/]")
        else:
            c.print(msg)

    # Import lazily to avoid import time if not needed
    if chapter_flags.get("maps"):
        from ..plots import maps as maps_mod

        if mode == "none":
            c.module_status("maps", "skip", "output_mode=none")
            module_results.append({
                "name": "maps",
                "status": "skipped",
                "seconds": 0.0,
                "error": "output_mode=none",
            })
        else:
            c.module_status("maps", "run", f"vars_2d={len(vars_2d)}, vars_3d={len(vars_3d)}")
            if "ensemble" in ds_prediction.dims:
                ens_full = int(ds_prediction.sizes.get("ensemble", 0))
                ens_plot = int(ds_prediction_plot.sizes.get("ensemble", ens_full))
                use_mode = resolved_modes.get("maps", "members")
                msg = format_ensemble_log(
                    "maps",
                    use_mode,
                    ens_full,
                    None if ens_plot == ens_full else f"selected {ens_plot} of {ens_full}",
                )
                c.info(msg)
            else:
                c.info("No ensemble dimension → deterministic inputs.")
            _t = time.time()
            try:
                maps_mod.run(
                    ds_target_plot,
                    ds_prediction_plot,
                    out_root,
                    plotting,
                    ensemble_mode=ensemble_cfg.get("maps"),
                )
                dt = time.time() - _t
                module_timings.append(("maps", dt))
                module_results.append({
                    "name": "maps",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                })
            except Exception as ex:  # pragma: no cover - robustness
                dt = time.time() - _t
                c.error(f"maps failed: {ex}")
                module_results.append({
                    "name": "maps",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                })

    if chapter_flags.get("histograms"):
        from ..plots import histograms as hist_mod

        if mode == "none":
            c.module_status("histograms", "skip", "output_mode=none")
            module_results.append({
                "name": "histograms",
                "status": "skipped",
                "seconds": 0.0,
                "error": "output_mode=none",
            })
        else:
            c.module_status("histograms", "run", f"vars_2d={len(vars_2d)}")
            if "ensemble" in ds_prediction.dims:
                ens_size = int(ds_prediction.sizes.get("ensemble", 0))
                c.info(
                    format_ensemble_log(
                        "histograms", resolved_modes.get("histograms", "pooled"), ens_size
                    )
                )
            else:
                c.info("No ensemble dimension → deterministic inputs.")
            _t = time.time()
            try:
                hist_mod.run(
                    ds_target,
                    ds_prediction,
                    out_root,
                    plotting,
                    ensemble_mode=ensemble_cfg.get("histograms"),
                    performance_cfg=performance_cfg,
                )
                dt = time.time() - _t
                module_timings.append(("histograms", dt))
                module_results.append({
                    "name": "histograms",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                })
            except Exception as ex:  # pragma: no cover
                dt = time.time() - _t
                c.error(f"histograms failed: {ex}")
                module_results.append({
                    "name": "histograms",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                })

    if chapter_flags.get("wd_kde"):
        from ..plots import wd_kde as wd_mod

        c.module_status("wd_kde", "run", f"vars_2d={len(vars_2d)} (standardized)")
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            c.info(format_ensemble_log("wd_kde", resolved_modes.get("wd_kde", "pooled"), ens_size))
        else:
            c.info("No ensemble dimension → deterministic standardized inputs.")
        _t = time.time()
        try:
            wd_mod.run(
                ds_target,
                ds_prediction,
                ds_target_std,
                ds_prediction_std,
                out_root,
                plotting,
                ensemble_mode=ensemble_cfg.get("wd_kde"),
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("wd_kde", dt))
            module_results.append({
                "name": "wd_kde",
                "status": "success",
                "seconds": dt,
                "error": None,
            })
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"wd_kde failed: {ex}")
            module_results.append({
                "name": "wd_kde",
                "status": "failed",
                "seconds": dt,
                "error": str(ex),
            })

    if chapter_flags.get("energy_spectra"):
        from ..plots import energy_spectra as es_mod

        c.module_status("energy_spectra", "run", f"vars_2d={len(vars_2d)}")
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            c.info(
                format_ensemble_log(
                    "energy_spectra", resolved_modes.get("energy_spectra", "mean"), ens_size
                )
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            es_mod.run(
                ds_target,
                ds_prediction,
                out_root,
                plotting,
                cfg.get("selection", {}),
                ensemble_mode=ensemble_cfg.get("energy_spectra"),
                cfg=cfg,
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("energy_spectra", dt))
            module_results.append({
                "name": "energy_spectra",
                "status": "success",
                "seconds": dt,
                "error": None,
            })
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"energy_spectra failed: {ex}")
            module_results.append({
                "name": "energy_spectra",
                "status": "failed",
                "seconds": dt,
                "error": str(ex),
            })

    if chapter_flags.get("vertical_profiles"):
        from ..metrics import vertical_profiles as vp_mod

        if mode == "none":
            c.module_status("vertical_profiles", "skip", "output_mode=none")
            module_results.append({
                "name": "vertical_profiles",
                "status": "skipped",
                "seconds": 0.0,
                "error": "output_mode=none",
            })
        else:
            c.module_status("vertical_profiles", "run", f"vars_3d={len(vars_3d)}")
            if "ensemble" in ds_prediction.dims:
                ens_size = int(ds_prediction.sizes.get("ensemble", 0))
                c.info(
                    format_ensemble_log(
                        "vertical_profiles",
                        resolved_modes.get("vertical_profiles", "mean"),
                        ens_size,
                    )
                )
            else:
                c.info("No ensemble dimension → deterministic inputs.")
            _t = time.time()
            try:
                vp_mod.run(
                    ds_target,
                    ds_prediction,
                    out_root,
                    plotting,
                    cfg.get("selection", {}),
                    ensemble_mode=ensemble_cfg.get("vertical_profiles"),
                    metrics_cfg=cfg.get("metrics", {}),
                    performance_cfg=performance_cfg,
                )
                dt = time.time() - _t
                module_timings.append(("vertical_profiles", dt))
                module_results.append({
                    "name": "vertical_profiles",
                    "status": "success",
                    "seconds": dt,
                    "error": None,
                })
            except Exception as ex:  # pragma: no cover
                dt = time.time() - _t
                c.error(f"vertical_profiles failed: {ex}")
                module_results.append({
                    "name": "vertical_profiles",
                    "status": "failed",
                    "seconds": dt,
                    "error": str(ex),
                })

    # Deterministic (previously called objective metrics)
    if chapter_flags.get("deterministic"):
        from ..metrics import deterministic as det_mod

        c.module_status("deterministic", "run", f"variables={len(all_vars)}")
        if "ensemble" in ds_prediction.dims:
            ens_size_det: int = int(ds_prediction.sizes.get("ensemble", 0))
            use_mode = resolved_modes.get("deterministic", "mean")
            c.info(format_ensemble_log("deterministic", use_mode, ens_size_det))
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            # Pass lead_policy through so per-lead deterministic artifacts are generated
            det_mod.run(
                ds_target,
                ds_prediction,
                ds_target_std,
                ds_prediction_std,
                out_root,
                plotting,
                cfg.get("metrics", {}),
                ensemble_mode=ensemble_cfg.get("deterministic"),
                lead_policy=lead_policy,
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("deterministic", dt))
            module_results.append({
                "name": "deterministic",
                "status": "success",
                "seconds": dt,
                "error": None,
            })
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"deterministic failed: {ex}")
            module_results.append({
                "name": "deterministic",
                "status": "failed",
                "seconds": dt,
                "error": str(ex),
            })

    if chapter_flags.get("ets"):
        from ..metrics import ets as ets_mod

        c.module_status("ets", "run", f"variables={len(all_vars)}")
        if "ensemble" in ds_prediction.dims:
            ens_size_ets = int(ds_prediction.sizes.get("ensemble", 0))
            use_mode = resolved_modes.get("ets", "mean")
            c.info(format_ensemble_log("ets", use_mode, ens_size_ets))
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            # Provide lead_policy to unlock by-lead ETS artifacts
            ets_mod.run(
                ds_target,
                ds_prediction,
                out_root,
                cfg.get("metrics", {}),
                plotting_cfg=plotting,
                ensemble_mode=ensemble_cfg.get("ets"),
                lead_policy=lead_policy,
                performance_cfg=performance_cfg,
            )
            dt = time.time() - _t
            module_timings.append(("ets", dt))
            module_results.append({
                "name": "ets",
                "status": "success",
                "seconds": dt,
                "error": None,
            })
        except Exception as ex:  # pragma: no cover
            dt = time.time() - _t
            c.error(f"ets failed: {ex}")
            module_results.append({
                "name": "ets",
                "status": "failed",
                "seconds": dt,
                "error": str(ex),
            })

    # Combined probabilistic: run PIT diagnostics plus WBX CRPS/SSR when enabled
    if chapter_flags.get("probabilistic"):
        from ..metrics.probabilistic import (
            driver as orchestrator,
        )

        c.module_status(
            "probabilistic",
            "run",
            "PIT + WBX CRPS/SSR",
        )
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            if ens_size < 2:
                c.warn(
                    "Ensemble size="
                    f"{ens_size} <2 → skipping probabilistic metrics (CRPS/PIT + WBX require >=2)."
                )
                # Register skipped modules
                module_results.append({
                    "name": "probabilistic",
                    "status": "skipped",
                    "seconds": 0.0,
                    "error": "ensemble size <2",
                })
                # Continue to completion without executing probabilistic submodules
                pass
            else:
                c.info(format_ensemble_log("probabilistic", "prob", ens_size))
                # Combined probabilistic module
                _t = time.time()
                try:
                    # 1. PIT diagnostics (computation + plotting)
                    # The orchestrator currently keeps compute and plotting as split calls.

                    if mode != "none":
                        c.info("[probabilistic] Stage 1/2: PIT metric computation + plotting")
                        orchestrator.run_probabilistic(
                            ds_target,
                            ds_prediction,
                            out_root,
                            cfg_plot=plotting,
                            cfg_all=cfg,
                            ensemble_mode=ensemble_cfg.get("probabilistic"),
                            performance_cfg=performance_cfg,
                            include_wbx_outputs=False,
                        )
                    else:
                        c.info(
                            "[probabilistic] Stage 1/2: PIT artifact computation skipped "
                            "(output_mode=none)"
                        )

                    # 2. WBX CRPS/SSR aggregation and outputs
                    c.info("[probabilistic] Stage 2/2: WBX CRPS/SSR aggregation and outputs")
                    orchestrator.run_probabilistic_wbx(
                        ds_target,
                        ds_prediction,
                        out_root,
                        plotting,
                        cfg,
                        performance_cfg=performance_cfg,
                    )

                    dt = time.time() - _t
                    module_timings.append(("probabilistic", dt))
                    module_results.append({
                        "name": "probabilistic",
                        "status": "success",
                        "seconds": dt,
                        "error": None,
                    })
                except Exception as ex:  # pragma: no cover
                    dt = time.time() - _t
                    c.error(f"probabilistic failed: {ex}")
                    module_results.append({
                        "name": "probabilistic",
                        "status": "failed",
                        "seconds": dt,
                        "error": str(ex),
                    })
        else:
            c.warn("No ensemble dimension → skipping probabilistic metrics (requires 'ensemble').")
            module_results.append({
                "name": "probabilistic",
                "status": "skipped",
                "seconds": 0.0,
                "error": "no ensemble dimension",
            })

    # SSIM (Structural Similarity Index)
    if chapter_flags.get("ssim"):
        from ..metrics.ssim import run as run_ssim

        c.module_status("ssim", "run", "Structural Similarity Index")
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            c.info(format_ensemble_log("ssim", ensemble_cfg.get("ssim", "mean"), ens_size))
        else:
            c.info("No ensemble dimension \u2192 deterministic inputs.")
        _t = time.time()
        try:
            run_ssim(
                ds_target,
                ds_prediction,
                out_root,
                cfg.get("metrics", {}),
                ensemble_mode=ensemble_cfg.get("ssim"),
            )
            dt = time.time() - _t
            module_timings.append(("ssim", dt))
            module_results.append({
                "name": "ssim",
                "status": "success",
                "seconds": dt,
                "error": None,
            })
        except Exception as ex:
            dt = time.time() - _t
            c.error(f"ssim failed: {ex}")
            module_results.append({
                "name": "ssim",
                "status": "failed",
                "seconds": dt,
                "error": str(ex),
            })

    # Multivariate (Bivariate Histograms)
    if chapter_flags.get("multivariate"):
        from ..metrics.multivariate import run as run_multivariate

        c.module_status("multivariate", "run", "Bivariate Histograms")
        if "ensemble" in ds_prediction.dims:
            ens_size = int(ds_prediction.sizes.get("ensemble", 0))
            c.info(
                format_ensemble_log(
                    "multivariate", ensemble_cfg.get("multivariate", "mean"), ens_size
                )
            )
        else:
            c.info("No ensemble dimension → deterministic inputs.")
        _t = time.time()
        try:
            run_multivariate(
                ds_target,
                ds_prediction,
                out_root,
                cfg.get("metrics", {}),
                ensemble_mode=ensemble_cfg.get("multivariate"),
            )
            dt = time.time() - _t
            module_timings.append(("multivariate", dt))
            module_results.append({
                "name": "multivariate",
                "status": "success",
                "seconds": dt,
                "error": None,
            })
        except Exception as ex:
            dt = time.time() - _t
            c.error(f"multivariate failed: {ex}")
            module_results.append({
                "name": "multivariate",
                "status": "failed",
                "seconds": dt,
                "error": str(ex),
            })

    # Final completion message + timings summary + module results summary (pass/fail)
    elapsed = time.time() - t0
    try:
        if "module_results" in locals() and module_results:
            # Build a text table (avoid adding dependency). Use rich if available.
            from ..console import (
                USE_RICH,
                console as _rc,
            )

            successes = sum(1 for r in module_results if r["status"] == "success")
            failures = [r for r in module_results if r["status"] == "failed"]
            skipped = [r for r in module_results if r["status"] == "skipped"]
            if USE_RICH:
                try:  # pragma: no cover
                    from rich import box
                    from rich.table import Table

                    tbl = Table(title="Module Results", box=box.SIMPLE_HEAVY)
                    tbl.add_column("Module", style="bold")
                    tbl.add_column("Status")
                    tbl.add_column("Time (s)", justify="right")
                    tbl.add_column("Error")
                    for r in module_results:
                        status = r["status"]
                        style = {
                            "success": "green",
                            "failed": "red",
                            "skipped": "yellow",
                        }.get(status, "white")
                        err = r.get("error") or ""
                        if len(err) > 80:
                            err = err[:77] + "..."
                        tbl.add_row(
                            r["name"],
                            f"[{style}]{status}[/]",
                            f"{r['seconds']:.2f}",
                            err,
                        )
                    _rc.print(tbl)
                except Exception:
                    c.print("Module Results:")
                    for r in module_results:
                        c.print(
                            f" - {r['name']}: {r['status']} ({r['seconds']:.2f}s)"
                            + (f" error={r['error']}" if r["error"] else "")
                        )
            else:
                c.print("Module Results:")
                for r in module_results:
                    c.print(
                        f" - {r['name']}: {r['status']} ({r['seconds']:.2f}s)"
                        + (f" error={r['error']}" if r["error"] else "")
                    )
            if failures:
                from .. import console as c2

                c2.warn(
                    f"{len(failures)} module(s) failed: {', '.join(r['name'] for r in failures)}"
                )
            else:
                from .. import console as c2

                c2.success(f"All {successes} module(s) succeeded.")

    except Exception:
        pass
    # Always print a plain-text completion line so logs are readable without Rich
    c.print(
        f"FINISHED: duration={elapsed:,.1f}s • outputs={out_root}",
    )
    # If Rich is enabled, also show a styled panel
    try:
        from ..console import USE_RICH

        if USE_RICH:
            c.panel(
                (
                    f"Completed in [bold]{elapsed:,.1f}[/] seconds"
                    f"\nOutputs written to: [bold]{out_root}[/]"
                ),
                title="✅ Finished",
                style="green",
            )
    except Exception:
        pass
