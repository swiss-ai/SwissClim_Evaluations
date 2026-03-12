from __future__ import annotations

import argparse
import contextlib
import socket
import sys
from pathlib import Path
from typing import Any

from . import console as c
from .core import config as config_mod, data_selection, runner as runner_mod


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SwissClim Evaluations runner")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p


def main(argv: list[str] | None = None) -> None:
    # Try to enforce line-buffered stdout/stderr so Slurm logs update promptly
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    # In non-interactive (no TTY) environments like Slurm, force plain output
    try:
        from .console import set_color_mode

        is_tty = False
        try:
            is_tty = bool(sys.stdout.isatty())
        except Exception:
            is_tty = False
        if not is_tty:
            set_color_mode("never")
    except Exception:
        pass

    args = build_parser().parse_args(argv)
    cfg = config_mod.load_config(args.config)
    # Record the original config path for reproducibility actions (not part of user schema)
    with contextlib.suppress(Exception):
        cfg["_config_path"] = args.config

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

    # Check if user wants to use distributed scheduler (default: True for backwards compat)
    performance_cfg = cfg.get("performance", {}) or {}
    quiet_dask_logs = _as_bool(performance_cfg.get("quiet_dask_logs", False), False)
    callback_stride_default = 1
    try:
        batch_callback_stride = max(
            1,
            int(performance_cfg.get("batch_callback_stride", callback_stride_default)),
        )
    except Exception:
        batch_callback_stride = callback_stride_default

    use_distributed = performance_cfg.get("dask_scheduler", "distributed").lower() != "threaded"
    dask_profile = config_mod.resolve_dask_profile(performance_cfg)
    dask_perf_report = config_mod.resolve_dask_performance_report(performance_cfg)

    if use_distributed:
        # Initialize Dask Client if available to enable spillover and distributed scheduling
        try:
            import dask.config
            from dask.distributed import Client, performance_report

            # Disable worker profiling to avoid AttributeError: '_AllCompletedWaiter' object has no
            # attribute 'f_back' on Python 3.11+ with recent distributed versions.
            # Also set timeouts to avoid hanging clients when workers die.
            c.print("Configuring Dask Client with timeouts and retry limits...")
            dask.config.set(
                {
                    "swissclim.quiet_dask_logs": bool(quiet_dask_logs),
                    "swissclim.batch_callback_stride": int(batch_callback_stride),
                    "distributed.worker.profile.enabled": False,
                    "distributed.comm.timeouts.connect": "30s",
                    "distributed.comm.timeouts.tcp": "30s",
                    "distributed.comm.retry.count": 3,
                    "distributed.scheduler.allowed-failures": 3,
                    "distributed.worker.memory.target": 0.60,
                    "distributed.worker.memory.spill": 0.70,
                    "distributed.worker.memory.pause": 0.80,
                    "distributed.worker.memory.terminate": 0.95,
                }
            )

            runner_mod.setup_dask_logging()

            try:
                _ = Client.current()
                runner_mod.run_selected(cfg)
            except (ValueError, OSError):
                c.print("Initializing Dask Client for distributed scheduling and spillover...")
                c.print(
                    "Dask profile: "
                    f"{dask_profile['profile']} "
                    f"(workers={dask_profile['n_workers']}, "
                    f"threads/worker={dask_profile['threads_per_worker']}, "
                    f"processes={dask_profile['processes']}, "
                    f"memory_limit={dask_profile['memory_limit']})"
                )

                # Check performance report first
                report_enabled = bool(dask_perf_report["enabled"])
                report_path = str(dask_perf_report["path"]) if report_enabled else None

                if report_enabled and report_path:
                    with contextlib.suppress(Exception):
                        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                    c.print(f"Dask performance report enabled: {report_path}")

                with Client(
                    n_workers=int(dask_profile["n_workers"]),
                    threads_per_worker=int(dask_profile["threads_per_worker"]),
                    processes=bool(dask_profile["processes"]),
                    memory_limit=dask_profile["memory_limit"],
                ) as client:
                    # Retrieve dashboard port and hostname for tunneling instructions
                    try:
                        dask_info = client.scheduler_info()
                        dashboard_port = dask_info["services"]["dashboard"]
                        host = socket.gethostname()
                        c.print(f"Dask Dashboard running on node: {host}")
                        c.print(f"Dashboard URL: {client.dashboard_link}")
                        c.print(
                            "To view the dashboard in VS Code, "
                            "add these entries to your SSH config:"
                        )
                        c.print("'AddKeysToAgent yes'")
                        c.print("'ForwardAgent yes'")
                        c.print("1. Open a new terminal in VS Code")
                        c.print(
                            f"2. Run: ssh -N -L {dashboard_port}:127.0.0.1:"
                            f"{dashboard_port} {host} &"
                        )
                        c.print(
                            f"3. Open 'Simple Browser' in VS Code and go to: http://localhost:{dashboard_port}/status"
                        )
                    except Exception:
                        c.print(f"Dask dashboard available at: {client.dashboard_link}")

                    report_ctx = (
                        performance_report(filename=report_path)
                        if (report_enabled and report_path)
                        else contextlib.nullcontext()
                    )
                    with report_ctx:
                        runner_mod.run_selected(cfg)

                    if report_enabled:
                        c.print(f"Dask performance report written to: {report_path}")
                c.print("Evaluation finished, closing Dask Client...")
        except ImportError:
            # Fallback to threaded if dask.distributed is missing
            c.print(
                "dask.distributed not found, falling back to threaded scheduler "
                "(no distributed client)."
            )
            # Reuse threaded logic logic
            threaded_workers = max(1, int(dask_profile["n_workers"]))
            try:
                import dask.config

                dask.config.set(
                    {
                        "swissclim.quiet_dask_logs": bool(quiet_dask_logs),
                        "swissclim.batch_callback_stride": int(batch_callback_stride),
                    }
                )
                dask.config.set(
                    scheduler="threads",
                    num_workers=threaded_workers,
                )
            except Exception:
                pass
            runner_mod.run_selected(cfg)

    else:
        # Use threaded scheduler with conservative worker count by default
        threaded_workers = max(1, int(dask_profile["n_workers"]))
        try:
            import dask.config

            dask.config.set(
                {
                    "swissclim.quiet_dask_logs": bool(quiet_dask_logs),
                    "swissclim.batch_callback_stride": int(batch_callback_stride),
                }
            )
            dask.config.set(
                scheduler="threads",
                num_workers=threaded_workers,
            )
        except Exception:
            pass
        c.print(
            "Using threaded Dask scheduler (no distributed client), "
            f"num_workers={threaded_workers}."
        )
        runner_mod.run_selected(cfg)


def run_selected(cfg: dict[str, Any]) -> None:
    """Compatibility shim for historical callers/tests importing cli.run_selected."""
    runner_mod.run_selected(cfg)


def _load_yaml(path: str) -> dict[str, Any]:
    """Compatibility shim for historical notebook imports."""
    return config_mod.load_config(path)


def prepare_datasets(
    cfg: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    """Compatibility shim for historical notebook imports."""
    return data_selection.prepare_datasets(cfg)


def _select_plot_datetime(
    ds_target: Any,
    ds_prediction: Any,
    cfg: dict[str, Any],
) -> tuple[Any, Any]:
    """Compatibility shim for historical notebook imports."""
    return data_selection.select_plot_datetime(ds_target, ds_prediction, cfg)


def _select_plot_ensemble(
    ds_prediction: Any,
    ds_prediction_std: Any,
    cfg: dict[str, Any],
) -> tuple[Any, Any]:
    """Compatibility shim for historical notebook imports."""
    return data_selection.select_plot_ensemble(ds_prediction, ds_prediction_std, cfg)


if __name__ == "__main__":
    main()
