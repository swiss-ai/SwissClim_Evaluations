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

Use the `performance` section in your run config to control batching/splitting behavior:

- `dask_scheduler`: `distributed` or `threaded`.
- `dask_profile`: `safe` (default), `balanced`, or `fast`.
- `dask_n_workers`: explicit worker override (optional).
- `dask_threads_per_worker`: explicit thread override (optional).
- `dask_processes`: process-based workers toggle (optional).
- `dask_memory_limit`: per-worker memory limit (optional).
- `dask_performance_report`: enable Dask HTML performance report generation (default `false`).
- `dask_performance_report_path`: output path template for the HTML report.
- `quiet_dask_logs`: suppress high-frequency Dask progress prints (optional).
- `batch_callback_stride`: coalesce writer callbacks across multiple compute batches (optional).
- `batch_size`: global Dask batch size (`"no-chunk"` disables batching).
- `split_3d_by_level`: global level split toggle.
- `split_lead_time`: global lead-time split toggle.
- `split_init_time`: global init-time split toggle.
- `lead_time_block_size`: global lead-time block size.
- `init_time_block_size`: global init-time block size.

Behavior and precedence:

- Runtime worker profile and worker overrides control task parallelism and memory pressure.
- When `dask_performance_report: true`, the distributed execution is wrapped in `performance_report(...)` and writes one HTML file per run.
- `dask_performance_report_path` supports placeholders `{job_id}` and `{timestamp}` (for SLURM-aware, collision-resistant filenames).
- In `threaded` mode, distributed-only knobs (`dask_threads_per_worker`, `dask_processes`, `dask_memory_limit`) are not applied to a `Client`; the resolved profile still drives `num_workers` and batching/splitting defaults.
- Profile-default batching is used when `batch_size` is omitted and no auto-tuning knobs are set.
- If `batch_size` is set, it supersedes auto tuning.
- Auto batch tuning is used when `batch_size` is omitted and either
	`safe_points_per_batch` or `max_dynamic_batch_size` is set.
- `batch_size` controls how many prepared jobs are computed per Dask batch.
- `quiet_dask_logs` reduces print overhead from per-batch messages.
- `batch_callback_stride` can reduce write/callback overhead by flushing every N batches.
- `split_*` flags and `*_block_size` control how those jobs are partitioned by dimensions first.
- In practice: split settings determine number/shape of jobs; `batch_size` determines how many run together.
- When `*_block_size` is omitted, defaults are profile-aware (`safe`/`balanced`/`fast`) to avoid too many tiny jobs.
- `plotting.output_mode` supports `plot`, `npz`, `both`, `none`.
- In `npz`, figure/map preparation is skipped where applicable and only NPZ artifacts are written.
- In `none`, PNG/NPZ artifacts are not generated and artifact-only modules are skipped.

Safety-first defaults:

- `dask_profile` defaults to `safe`.
- On GH200-class nodes (`>=192` CPUs), profile defaults are tuned for the node:
	- `safe`: 96 workers × 2 threads, `4GiB` per worker
	- `balanced`: 128 workers × 2 threads, `3GiB` per worker
	- `fast`: 128 workers × 2 threads, `3GiB` per worker
- Profile-default batch sizes (when `batch_size` is omitted and no auto knobs are set):
	- `safe`: `batch_size=32`
	- `balanced`: `batch_size=32`
	- `fast`: `batch_size=48`
- Split block size defaults are also profile-aware (applied only when omitted):
	- `safe`: `lead_time_block_size=4`, `init_time_block_size=8`
	- `balanced`: `lead_time_block_size=4`, `init_time_block_size=8`
	- `fast`: `lead_time_block_size=4`, `init_time_block_size=8`
- Explicit `performance.lead_time_block_size` / `performance.init_time_block_size` always override these defaults.

Optional advanced tuning:

- `safe_points_per_batch` (default `auto`, dataset-driven)
- `max_dynamic_batch_size` (default `auto`, dataset-driven)
- `plotting.kde_max_samples` (`auto` recommended; `null` disables subsampling)
- `plotting.histogram_max_samples` (`auto` recommended; `null` disables subsampling)

Auto mode is intentionally conservative (favors more batches / fewer jobs per batch).

Throughput-oriented full-node preset (GH200-class):

- `dask_profile: fast`
- `quiet_dask_logs: true`
- `batch_callback_stride: 4`
- `split_lead_time: true`, `split_init_time: true`
- (optional explicit override) `batch_size: 64` then benchmark against `96`

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
