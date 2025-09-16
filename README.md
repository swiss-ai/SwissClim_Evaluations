# SwissClim Evaluations

Fast, reproducible evaluation of weather/climate model outputs against ERA5 (or other references). Compute deterministic and probabilistic scores, spectral metrics, and helpful diagnostic plots — all driven by a single YAML config.

## Quickstart

1. Create an environment (pick one):

- UV virtualenv
  - Run: bash tools/setup_env_uv.sh
  - Creates .venv with Python 3.11 via uv and installs deps.

- Conda
  - Run: bash tools/setup_env_conda.sh
  - Creates env (Python 3.11) from tools/environment.yml and installs deps.

- Container (CSCS Enroot/Podman)
  - See tools/swissai_container.toml and tools/swissai.dockerfile

1. Review and edit the example config:

The project ships with a commented config that explains every key and valid values. Copy it and adjust the paths and selections as needed.

Example:

- cp config/example_config.yaml config/my_run.yaml
- Edit config/my_run.yaml (see inline comments)

1. Run:

```bash
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

Outputs appear under paths.output_root (one sub-folder per module).

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
  - dpi, random_seed, time_subsamples
- modules
  - Toggle what to run: maps, histograms, wd_kde, energy_spectra, vertical_profiles, deterministic, ets, probabilistic, probabilistic_wbx
- metrics
  - deterministic.include / .standardized_include (optional filters)
  - ets.thresholds (percentiles)

Notes

- Time alignment: ERA5 uses time while the ML dataset uses init_time + lead_time. The CLI aligns them by valid_time = init_time + lead_time so both sides compare the same moments.
- Ensemble handling: If your ML data has an ensemble dim, you can:
  - selection.ensemble_member: pick a specific member, or
  - leave unset and the CLI will take the ensemble mean when probabilistic modules are off. If probabilistic is on, the ensemble is kept.
- Plotting control: output_mode is the only switch for figures vs NPZ for most modules. Energy spectra NPZ and probabilistic CRPS/PIT artifacts are always saved regardless of this mode.

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

## What you get

- Deterministic metrics
  - deterministic/metrics.csv and deterministic/metrics_standardized.csv
  - Summary previews printed to the terminal
- ETS
  - ets/ets_metrics.csv and a terminal preview
- Energy spectra
  - energy_spectra/*_energy.png, energy_spectra/lsd_metrics.csv, and always an accompanying .npz with spectra arrays
- Histograms by latitude bands (2D only)
  - histograms/{var}_sfc_latbands.png (+ combined NPZ per var)
- KDE + Wasserstein by bands (standardized)
  - wd_kde/{var}_sfc_latbands_norm.png (+ combined NPZ per var, mean Wasserstein)
- Vertical profiles (3D)
  - vertical_profiles/{var}_pl_rel_error.png (+ combined NPZ per var)
- Maps
  - maps/{timestamp}_{var}_sfc.png and maps/{timestamp}_{var}_pl.png (+ NPZ dumps of arrays)
- Probabilistic
  - probabilistic/crps_summary.csv, probabilistic/{var}_pit_hist.npz, and always {var}_pit.nc and {var}_crps.nc
- Probabilistic (WBX)
  - probabilistic_wbx/spread_skill_ratio.csv, probabilistic_wbx/crps_ensemble.csv

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
  - plotting.time_subsamples: limit number of init_time samples for plots
- Keep the same variable names between ERA5 and ML; only overlapping variables are processed.
- Maps require Cartopy; use npz mode on headless systems if you only need data exports.
- For reproducibility in plots, prefer plotting.output_mode: npz or both, which writes the exact arrays used to render figures.

## Notebooks

For the probabilistic modules, you can explore the outputs interactively using the provided notebooks (legacy support):

- notebooks/probabilistic_verification.ipynb (classic CRPS/PIT using our xarray-based implementation)
- notebooks/probabilistic_verification_wbx.ipynb (WeatherBenchX Spread–Skill Ratio and CRPS summaries)

Notebook tips

- Use the YAML config and `prepare_datasets` to ensure alignment, selection, and chunking are consistent with the CLI.
- For CRPS maps in the notebook: compute `crps_ensemble(targets, predictions, ensemble_dim="ensemble")` and then average over `["time", "init_time", "lead_time", "ensemble"]` where present to produce a lat/lon field for plotting.

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

Backwards compatibility: older code using keyword arguments like `ds=` and `ds_ml=` has been updated in this repo. If you have downstream code pinned to those names, update calls to use `targets=` and `predictions=` respectively.
