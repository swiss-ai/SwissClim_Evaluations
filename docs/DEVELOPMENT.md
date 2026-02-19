# Development

```text
- Python: >=3.11
- Key libs: xarray, numpy, scipy, pandas, matplotlib, cartopy, scores, weatherbenchX
```

### Debugging

This repository includes ready-to-use debug launch configurations under `.vscode/launch.json`.

Tip: If you need to debug only one module to save time, temporarily disable others in the YAML (`modules: { ... }`). The debug configuration just passes the config file; everything else is controlled by the YAML content.

Possible workflow:

1. Start an interactive session (container or env activated).
2. Open a VS Code tunnel (`code tunnel`), connect from your workstation.
3. Open the repository folder and use the provided debug configuration. No additional adapter setup required (uses `debugpy`). -> `F5` to start debugging.

Run tests:

```bash
pytest -q
```

Contributions welcome — keep changes chunk-aware (xarray/dask friendly) and small.

### Performance tuning knobs (YAML)

Use the `performance` section in your run config to control Dask runtime behavior:

- `dask_scheduler`: `distributed` or `threaded`.
- `dask_profile`: `safe` (default), `balanced`, or `fast`.
- `dask_n_workers`: explicit worker override (optional).
- `dask_threads_per_worker`: explicit thread override (optional).
- `dask_processes`: process-based workers toggle (optional).
- `dask_memory_limit`: per-worker memory limit (optional).
- `dask_performance_report`: enable Dask HTML performance report generation (default `false`).
- `dask_performance_report_path`: output path template for the HTML report.
- `quiet_dask_logs`: suppress high-frequency Dask progress prints (optional).
- `split_3d_by_level`: global level split toggle for 3D module job partitioning.

Execution strategy (current standard):

- Modules build lazy xarray/dask graphs and execute via direct `dask.compute(...)`.
- Manual per-module job batching is intentionally not used in `metrics` and `plots`.
- Memory and throughput are controlled by chunking policy, scheduler/profile, and worker limits.
- `plotting.output_mode` supports `plot`, `npz`, `both`, `none`.

Optional legacy/override keys:

- `batch_size`, `safe_points_per_batch`, and `max_dynamic_batch_size` remain accepted in config for compatibility and future tuning hooks, but they are not part of the primary execution path for current `metrics`/`plots` modules.

### Dev environment (linting & formatting)

This project uses Ruff for both linting and formatting, managed via an optional "dev" extra
and pre-commit hooks.

Install dev extra with uv:

```bash
uv sync --extra dev
```

or with uv pip:

```bash
uv pip install -e '.[dev]'
```

#### Pre-commit hooks

Enable once (installs the Git hooks):

```bash
pre-commit install
```

and then run manually on all files:

```bash
pre-commit run --all-files
```

### Naming conventions

- target: the ground-truth/reference dataset (e.g., ERA5). Public APIs now consistently use the parameter name `target`.
- prediction: the model outputs to be evaluated (e.g., ML). Public APIs now consistently use the parameter name `prediction`.
