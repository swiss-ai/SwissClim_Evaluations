# SwissClim Evaluations

Evaluate weather and climate model output against reference datasets (e.g., ERA5) with deterministic and probabilistic metrics, spectral analysis, and diagnostic plots. Built on Xarray, NumPy/SciPy, Dask, and Cartopy (for maps).

## Installation

Pick one workflow:

- UV virtualenv (recommended)
  - Use the provided script which also clones WBX next to the repo:
    - bash tools/setup_env_uv.sh
  - This sets up a `.venv` with Python 3.12 (via uv), installs deps, and clones `../weatherbenchX` for WBX metrics.

- Conda environment
  - Create and activate env from `tools/environment.yml` (Python 3.11):
    - bash tools/setup_env_conda.sh
  - Then activate it later with: `conda activate swissai-eval`

- Container (CSCS Enroot/Sarus)
  - See `tools/swissai_container.toml` and `tools/swissai.dockerfile` for building/importing a Python 3.11 image that clones WBX and installs this project.

Notes:

- WeatherBench-X (WBX) is used for additional probabilistic metrics. The project is configured to source WBX locally (see `[tool.uv.sources]` in `pyproject.toml`). If you plan to run WBX modules, ensure `../weatherbenchX` exists. The UV and container scripts handle this automatically.
- Cartopy is required for map plots and is included as a dependency.

1. Create a example_config YAML config (e.g., `config/example_config.yaml`):

```yaml
paths:
  nwp: "/path/to/era5.zarr"           # ERA5 or similar reference
  ml:  "/path/to/model_output.zarr"   # Your model predictions
  output_root: "./output/verification"

selection:
  variables_2d: ["2m_temperature"]    # optional: variables_3d, levels, latitudes, longitudes, datetimes
  temporal_resolution_hours: 6          # optional downsampling
  # ensemble_member: 0                  # pick a member if ML has an ensemble dim

modules:
  deterministic: true
  probabilistic: true
  maps: false
  histograms: false
  wd_kde: false
  energy_spectra: false
  vertical_profiles: false
  ets: false
  probabilistic_wbx: false

plotting:
  outputs: ["file"]                   # ["file", "cell", "cell-first"]
  dpi: 48
  random_seed: 42
  time_subsamples: 4
  save_plot_data: false

metrics:
  deterministic:
    include: ["MAE", "RMSE", "Pearson R", "FSS", "Wasserstein"]   # optional
    standardized_include: ["MAE", "RMSE", "Wasserstein"]            # optional
  ets:
    thresholds: [90, 95, 99]     # quantile thresholds in %
    save_csv: true
```

1. Run modules using the module entry point:

```bash
python -m swissclim_evaluations.cli --config config/example_config.yaml
# Or select modules explicitly (preferred flag):
python -m swissclim_evaluations.cli --config config/example_config.yaml --modules deterministic probabilistic
```

Outputs are written under `paths.output_root`, one subfolder per module.

## Data requirements

- Xarray Datasets with standard dims:
  - time (optional), latitude/lat, longitude/lon
  - level for 3D variables (pressure level)
  - ensemble (optional)
- CLI expects:
  - paths.nwp: reference dataset (e.g., ERA5)
  - paths.ml: model predictions (e.g., Zarr)
- Optional selection keys: variables_2d, variables_3d, levels, latitudes, longitudes, datetimes, temporal_resolution_hours, ensemble_member
- Time alignment and optional subsampling are handled in the CLI.

## Modules

Enable via YAML (`modules.*`) or `--modules` flag:

- maps: Global maps for 2D and per-level 3D fields (Cartopy)
- histograms: Distributions by latitude bands
- wd_kde: KDEs by latitude band with Wasserstein distances (standardized)
- energy_spectra: Zonal energy spectra and Log Spectral Distance (LSD)
- vertical_profiles: Relative error vertical profiles per latitude band
- deterministic: MAE, RMSE, MSE, Pearson R, FSS, Wasserstein; plus standardized metrics
- ets: Equitable Threat Score across quantile thresholds
- probabilistic: CRPS, PIT, ensemble statistics
- probabilistic_wbx: WeatherBench-X compatible probabilistic metrics

## Outputs

Examples under `output_root`:

- `deterministic/metrics.csv`, `deterministic/metrics_standardized.csv`
- `histograms/*.png` (+ optional `*.npz`)
- `wd_kde/*.png` (+ optional `*.npz`)
- `energy_spectra/*_energy.png`, `energy_spectra/lsd_metrics.csv` (+ optional `*.npz`)
- `vertical_profiles/*_pl_rel_error.png` (+ optional `*_pl_rel_error.nc`)
- `maps/*_sfc.png` and `maps/*_pl.png` (+ optional NetCDF dumps)
- `probabilistic/crps_summary.csv`, `probabilistic/{var}_pit_hist.npz` (+ optional `probabilistic/{var}_pit.nc`, `probabilistic/{var}_crps.nc`)
- `probabilistic_wbx/spread_skill_ratio.csv`, `probabilistic_wbx/crps_ensemble.csv`

Standardization: Some comparisons also run on standardized pairs (combined mean/std over ds and ds_ml) and are saved separately.

## Tips

- Large data: use `selection.temporal_resolution_hours` and `plotting.time_subsamples` to reduce work.
- Ensemble data: set `selection.ensemble_member` to pick a single member from ML data when needed.
- Reproducibility: `plotting.save_plot_data` writes arrays used to create figures.

## Notebooks

- notebooks/probabilistic_verification.ipynb: Probabilistic verification
- notebooks/probabilistic_verification_wbx.ipynb: WeatherBench-X workflow

Launch with your environment active:

```bash
python -m ipykernel install --user --name swissclim-evals --display-name "Python (swissclim-evals)"
python -m jupyter notebook notebooks/
```

## Dependencies

- xarray, numpy, scipy, pandas, dask
- matplotlib, seaborn, bokeh
- cartopy (maps)
- scores (verification metrics)
- weatherbenchX (WBX-compatible metrics; local source expected at `../weatherbenchX`)

## Contributing

Contributions are welcome. Please follow Xarray-friendly patterns and keep computations chunk-aware where practical.
