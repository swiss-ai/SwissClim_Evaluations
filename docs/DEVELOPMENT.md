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
- `chunk_size`: global Dask batch size (`"no-chunk"` disables batching).
- `split_3d_by_level`: global level split toggle.
- `split_lead_time`: global lead-time split toggle.
- `split_init_time`: global init-time split toggle.
- `lead_time_block_size`: global lead-time block size.
- `init_time_block_size`: global init-time block size.

Behavior and precedence:

- Auto chunk tuning is ON when `chunk_size` is omitted.
- If `chunk_size` is set, it supersedes auto tuning.
- `chunk_size` controls how many prepared jobs are computed per Dask batch.
- `split_*` flags and `*_block_size` control how those jobs are partitioned by dimensions first.
- In practice: split settings determine number/shape of jobs; `chunk_size` determines how many run together.

Optional advanced tuning:

- `safe_points_per_batch` (default `200000000`)
- `max_dynamic_chunk_size` (default `64`)

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
