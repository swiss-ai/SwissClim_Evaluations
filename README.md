# SwissClim Evaluations

Fast, reproducible evaluation of weather/climate model outputs against ERA5 (or other references). Compute deterministic and probabilistic scores, spectral metrics, and helpful diagnostic plots — all driven by a single YAML config.

## Quickstart (default: container with Podman + Enroot)

We recommend the container workflow for fastest, reproducible setup.

1. Build the container (Podman) at repo root int interactive session:

```bash
srun --container-writable -t 01:00:00 -A a122 -p debug --pty bash
podman build -t swissclim-eval .
```

2. (CSCS Alps) Export to Enroot SQuashFS and set up EDF once:

```bash
rm -f tools/swissclim-eval.sqsh
enroot import -x mount -o tools/swissclim-eval.sqsh podman://swissclim-eval
exit # exit the interactive build session
mkdir -p ~/.edf
sed "s/{{username}}/$USER/g" tools/edf_template.toml > ~/.edf/swissclim-eval.toml
```

3. Review and edit the example config:

The project ships with a commented config that explains every key and valid
values. Copy it and adjust the paths and selections as needed.

```bash
cp config/example_config.yaml config/my_run.yaml

4. (CSCS Alps) Launch an interactive session using the container:

```bash
srun --container-writable --environment=swissclim-eval -A a122 -t 01:30:00 -p debug --pty /bin/bash
```

You are now inside the container with all dependencies installed.
For a richter debugging experience we consider using `code tunnel`.

5. Run:

```bash
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

Outputs appear under paths.output_root (one sub-folder per module).

> Prefer a plain virtual environment? Use one of the alternatives below.

<details>
<summary>Install with uv (fast Python)</summary>

```bash
bash tools/setup_env_uv.sh
# Activates .venv and installs deps via uv
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

</details>

<details>
<summary>Install with conda</summary>

```bash
bash tools/setup_env_conda.sh
conda activate swissclim-eval
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

</details>

## Configure

The YAML is the single source of truth. Key sections:

- paths
  - ml: path to your model zarr
  - nwp: path to ERA5 (WeatherBench2 style) or another reference
  - output_root: where results go
- selection
  - variables_2d, variables_3d: variable names present in both datasets
  - levels: pressure levels for 3D variables
  - datetimes: [ISO start, ISO end] for slicing
  - latitudes, longitudes: [min, max]
  - temporal_resolution_hours: downsample along time to speed up
  - ensemble_member: pick one ensemble member from ML if desired
- plotting
  - output_mode: plot | npz | both
    - plot: save PNGs only
    - npz: write numeric arrays only
    - both: do both
  - dpi, random_seed, plot_datetime
  - map_variable: optional variable name to use for CRPS/PIT maps (defaults to first common variable)
- modules
  - Toggle what to run: maps, histograms, wd_kde, energy_spectra, vertical_profiles, deterministic, ets, probabilistic (combined xarray + WBX)
- metrics
  - deterministic.include / .standardized_include (optional filters)
  - ets.thresholds (percentiles)

- probabilistic (optional)
  - init_time_chunk_size: chunk size for iterating over init_time (WBX and xarray runners)
  - lead_time_chunk_size: chunk size for iterating over lead_time (WBX and xarray runners)
  - lead_times_ns: optional explicit list of numpy timedelta64[ns] lead times to override dataset values (WBX)

Notes

- Time alignment: ERA5 uses time while the ML dataset uses init_time + lead_time. The CLI aligns them by valid_time = init_time + lead_time so both sides compare the same moments.
- Ensemble handling: If your ML data has an ensemble dim, you can:
  - selection.ensemble_member: pick a specific member, or
  - leave unset and the CLI will take the ensemble mean when probabilistic modules are off. If probabilistic is on, the ensemble is kept.
- Plotting control: output_mode is the main switch for figures vs NPZ for most modules. Energy spectra NPZ and probabilistic CRPS/PIT artifacts are always saved regardless of this mode.
  - Optional: plotting.plot_datetime lets you choose a specific init_time to plot (must lie within selection.datetimes and exist in predictions).
  - Default: when plot_datetime is not set, plotting uses the first available init_time.

## Dataset Requirements

This verification is based on xarray Datasets with the following structure.
Currently the dataloader expects the a zarr archive of the following shape:

```bash
<xarray.Dataset> Size: 297GB
Dimensions:                  (init_time: 28, ensemble: 8, lead_time: 40,
                              latitude: 720, longitude: 1440, level: 37)
Coordinates:
  * ensemble                 (ensemble) int64 64B 0 1 2 3 4 5 6 7
  * init_time                (init_time) datetime64[ns] 224B 2023-01-02 ... 2...
  * latitude                 (latitude) float32 3kB 90.0 89.75 ... -89.5 -89.75
  * lead_time                (lead_time) timedelta64[ns] 320B 0 days 06:00:00...
  * level                    (level) int64 296B 1 2 3 5 ... 950 975 1000
  * longitude                (longitude) float32 6kB 0.0 0.25 ... 359.5 359.8
Data variables:
    10m_u_component_of_wind  (init_time, lead_time, ensemble, latitude, longitude) float32 37GB dask.array<chunksize=(1, 1, 8, 720, 1440), meta=np.ndarray>
    10m_v_component_of_wind  (init_time, lead_time, ensemble, latitude, longitude) float32 37GB dask.array<chunksize=(1, 1, 8, 720, 1440), meta=np.ndarray>
    2m_temperature           (init_time, lead_time, ensemble, latitude, longitude) float32 37GB dask.array<chunksize=(1, 1, 8, 720, 1440), meta=np.ndarray>
    global_CO2               (init_time, lead_time, ensemble, latitude, longitude) float32 37GB dask.array<chunksize=(1, 1, 8, 720, 1440), meta=np.ndarray>
    u_component_of_wind      (init_time, lead_time, level, ensemble, latitude, longitude) float32 1TB dask.array<chunksize=(1, 1, 1, 8, 720, 1440), meta=np.ndarray>
Attributes:
    model:    model_ckpt-step=7300-loss_train=0.07.ckpt
```

### Chunking policy (xarray/dask)

- The repository enforces a default Dask chunking policy in code:
  - init_time: 1, lead_time: 1, level: 1, latitude: -1, longitude: -1, ensemble: -1 (-1 = no chunking)
- This ensures apply_ufunc metrics that use the ensemble as a core dimension work without errors and keeps memory usage predictable. If a dataset deviates, it will be rechunked automatically with a warning.

## Output Overview

The evaluation generates organized results for each enabled module:

### Deterministic Metrics

- CSV summaries: `deterministic/metrics.csv` and `deterministic/metrics_standardized.csv`
- Terminal preview of key statistics

### Extreme Threshold Statistics (ETS)

- Metrics file: `ets/ets_metrics.csv`
- Terminal summary preview

### Energy Spectra Analysis

- Spectral plots: `energy_spectra/*_energy.png`
- Metrics summary: `energy_spectra/lsd_metrics.csv`
- Raw data: accompanying `.npz` files with spectral arrays

### Distribution Analysis

- Histograms by latitude: `histograms/{var}_sfc_latbands.png`
- KDE + Wasserstein distance: `wd_kde/{var}_sfc_latbands_norm.png`
- Supporting data: combined NPZ files per variable

### Vertical Structure (3D variables only)

- Profile plots: `vertical_profiles/{var}_pl_rel_error.png`
- Raw data: combined NPZ files per variable

### Spatial Maps

- Surface maps: `maps/{timestamp}_{var}_sfc.png`
- Pressure level maps: `maps/{timestamp}_{var}_pl.png`
- Raw arrays: NPZ dumps for each plot

### Probabilistic Verification (combined)

- Xarray-based (per-variable fields and plots):
  - CRPS summary: `probabilistic/crps_summary.csv`
  - PIT histogram NPZ: `probabilistic/{var}_pit_hist.npz`
  - PIT/CRPS fields: `probabilistic/{var}_pit.nc`, `probabilistic/{var}_crps.nc`
  - Optional figures (when `plot` or `both`): `probabilistic/crps_map_{var}.png`, `probabilistic/pit_hist_{var}.png`
- WeatherBenchX-based (summaries and aggregates):
  - CSV summaries: `probabilistic_wbx/spread_skill_ratio.csv`, `probabilistic_wbx/crps_ensemble.csv`
  - Temporal aggregates (NetCDF): `probabilistic_wbx/probabilistic_metrics_temporal.nc`
  - Spatial/regional aggregates (NetCDF): `probabilistic_wbx/probabilistic_metrics_spatial.nc`
  - Optional CRPS map: `probabilistic_wbx/crps_map_{var}.png`

### Details for probabilistic outputs

- CRPS and PIT are computed per variable using the ensemble along the `ensemble` dimension.
- CRPS returned by the library functions is a DataArray (not a Dataset). In notebooks, use the DataArray directly and then reduce over time-like dims to make maps.
- PIT histograms are stored as NPZ (counts, edges) for reproducibility; corresponding PIT fields are also written to NetCDF.

All modules print concise progress like:

- [swissclim] Module: deterministic — variables=5
- [histograms] variable: 10m_u_component_of_wind
- [energy_spectra] saved output/verification_esfm/energy_spectra/u_component_of_wind_500hPa_energy.png

## Tips and best practices

- Reduce size
  - selection.temporal_resolution_hours: 1, 3, 6 …
  - plotting.plot_datetime: pick a single init_time to plot (default already selects the first).
- Keep the same variable names between ERA5 and ML; only overlapping variables are processed.
- Maps require Cartopy; use npz mode on headless systems if you only need data exports.
- For reproducibility in plots, prefer plotting.output_mode: npz or both, which writes the exact arrays used to render figures.

## Notebooks

You can explore the outputs interactively using the provided notebooks:

- notebooks/deterministic_verification.ipynb (classic deterministic metrics, maps, histograms, spectra, profiles)
- notebooks/probabilistic_verification.ipynb (classic CRPS/PIT using our xarray-based implementation)
- notebooks/probabilistic_verification_wbx.ipynb (WeatherBenchX Spread–Skill Ratio and CRPS summaries)

Notebook tips

- Use the YAML config and `prepare_datasets` to ensure alignment, selection, and chunking are consistent with the CLI.

## Development

```text
- Python: 3.11
- Key libs: xarray, numpy, scipy, pandas, matplotlib, cartopy, scores, weatherbenchX
```

Run tests:

```bash
pytest -q
```

Contributions welcome — keep changes chunk-aware (xarray/dask friendly) and small.

## Naming conventions

- targets: the ground-truth/reference dataset (e.g., ERA5). Public APIs now consistently use the parameter name `targets`.
- predictions: the model outputs to be evaluated (e.g., ML). Public APIs now consistently use the parameter name `predictions`.

In notebooks and internal code we also favor the explicit names `ds_targets` and `ds_predictions` for clarity.
