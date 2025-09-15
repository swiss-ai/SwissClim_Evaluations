# SwissClim Evaluations

Fast, reproducible evaluation of weather/climate model outputs against ERA5 (or other references). Compute deterministic and probabilistic scores, spectral metrics, and helpful diagnostic plots — all driven by a single YAML config.

## Quickstart

1. Create an environment (pick one):

- UV virtualenv
  - Run: bash tools/setup_env_uv.sh
  - Creates .venv with Python 3.11 via uv and installs deps (+ clones ../weatherbenchX for WBX metrics).

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

- notebooks/probabilistic_verification.ipynb
- notebooks/probabilistic_verification_wbx.ipynb

## Development

```
- Python: 3.11
- Key libs: xarray, numpy, scipy, pandas, matplotlib, cartopy, scores, weatherbenchX
```

Run tests:

```bash
pytest -q
```

Contributions welcome — keep changes chunk-aware (xarray/dask friendly) and small.
