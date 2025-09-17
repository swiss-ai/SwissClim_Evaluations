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
```

4. (CSCS Alps) Launch an interactive session using the container:

```bash
srun --container-writable --environment=swissclim-eval -A a122 -t 01:30:00 -p debug --pty /bin/bash
```

You are now inside the container with all dependencies installed.
For a richer debugging experience we recommend using `code tunnel`.

5. Run:

```bash
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

Outputs appear under paths.output_root (one sub-folder per module).

6. Or submit a batch job (CSCS Alps):

```bash
sbatch launchscript.sh
```

Don't forget to adjust the path to your `config/my_run.yaml` in `launchscript.sh`.

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

The YAML is the single source of truth. Use the commented example directly:

<!-- markdownlint-disable MD033 -->
<details>
<summary><strong>Example config (click to expand)</strong></summary>

```yaml
# Global configuration for SwissClim Evaluations
#
# Tip: All paths should be absolute or relative to the repository root.
# Python: 3.11 (uv, conda, and container setups)

paths:
  # Path to your model predictions in Zarr format (root directory).
  # Must contain variables with dims (init_time, [lead_time], latitude, longitude[, level][, ensemble]).
  ml: "/capstor/store/cscs/swissai/a122/ESFM_Results/esfm_precipERA5_8tails_6h/rollout_predensemble_ci-co2-sst_2023-01-02x28_40steps.zarr"

  # Path to the reference dataset (ERA5 in WeatherBench2 layout).
  # Expected dims: (time or init_time+lead_time), latitude, longitude[, level].
  nwp: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr"

  # Where outputs are written; a subfolder per module will be created here.
  output_root: "output/verification_esfm"

selection:
  # Pressure levels (hPa) for 3D variables. Only levels present in the data are kept.
  # Common choices: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  levels: [100, 500, 1000]

  # Optional time downsampling in hours. Applied as a stride along lead_time if present,
  # otherwise along init_time. Set to null to disable.
  temporal_resolution_hours: null

  # Time selection (one of the following):
  #   1) Single range [start, end]:
  #      datetimes: ["2023-01-02T12", "2023-01-03T00"]
  #   2) Multiple ranges:
  #      datetimes: ["2023-01-02T00:2023-01-02T06", "2023-01-03T12:2023-01-03T18"]
  #      # or as pairs: datetimes: [["2023-01-02T00","2023-01-02T06"], ["2023-01-03T12","2023-01-03T18"]]
  #   3) Explicit non-contiguous timestamps:
  #      datetimes_list: ["2023-01-10T00","2023-01-10T06","2023-04-10T00","2023-07-10T00","2023-10-10T00", ...]
  # For ERA5 (time) and ML (init_time+lead_time), the CLI aligns by valid_time = init_time + lead_time.
  datetimes: ["2023-01-02T12", "2023-01-03T00"]

  # Latitude slice [north, south] in degrees. ERA5 uses descending latitudes (90 → -90),
  # so [90, -89.75] is typical for Aurora. Adjust to match your grid extents.
  latitudes: [90.0, -89.75]

  # Longitude slice [west, east] in degrees east. ERA5/WeatherBench2 uses 0..360.
  # Aurora/ESFM has a polar cutoff, so [0, 359.75] is typical.
  longitudes: [0.0, 359.75]

  # Variables without a level dimension present in BOTH datasets.
  variables_2d: [
    "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "2m_temperature",
    # "mean_sea_level_pressure",
  ]

  # Variables with a level dimension present in BOTH datasets (pressure levels).
  variables_3d: [
    # "geopotential",
    # "specific_humidity",
    # "temperature",
    # "u_component_of_wind",
    # "v_component_of_wind",
  ]

  # For ML datasets with an 'ensemble' dimension:
  #   - Set to an integer index to select a single member for deterministic runs.
  #   - Leave as null/omit to use the ensemble mean when probabilistic modules are disabled.
  #   - When probabilistic modules are enabled, ensemble is kept regardless.
  ensemble_member: 0

  # If true, raise an error when requested pressure levels are missing from the data.
  check_missing: false

plotting:
  # Random seed used for reproducible time subsampling in plots.
  random_seed: 42

  # Plot a specific init_time. Must lie within selection.datetimes and exist in predictions.
  # Example: "2023-01-03T12". Leave null/omit to plot the first available init_time.
  plot_datetime: null

  # Plot specific ensemble members (indices)
  # Provide a list of integers. Leave null/omit to plot all ensemble members present.
  # Example: [0, 3, 7]
  plot_ensemble_members: null

  # Base DPI for plots.
  dpi: 48

  # Maximum number of samples per latitude band used for KDE/Wasserstein in wd_kde.
  # Larger values improve smoothness but increase compute and memory. Typical: 50_000–200_000.
  # Note: Provide as a plain integer (no underscores) in YAML.
  kde_max_samples: 200000

  # Unified plotting output mode:
  #   - plot: save PNG figures only
  #   - npz: export numeric data files only (NPZ) for supported modules
  #   - both: save PNGs and export NPZ
  # Notes:
  #   - Energy spectra NPZ and probabilistic CRPS/PIT artifacts are always saved regardless of this mode.
  output_mode: both

modules:
  # Toggle individual modules on/off. The CLI runs based on these flags only.
  maps: true                 # Global and per-level maps (Cartopy)
  histograms: true           # Distributions by latitude bands (2D variables)
  wd_kde: true               # KDE by latitude band on standardized fields; also reports mean Wasserstein
  energy_spectra: true       # Zonal energy spectra + LSD table; NPZ always saved
  vertical_profiles: true    # Relative error vertical profiles per latitude band (3D)
  deterministic: true        # Deterministic metrics (MAE, RMSE, etc.) incl. standardized variants
  ets: true                  # Equitable Threat Score across quantile thresholds
  probabilistic: true       # Combined probabilistic (xarray CRPS/PIT + WBX SSR/CRPS)

metrics:
  # Deterministic metrics configuration. If lists are omitted, a default set is computed.
  # Available metric names:
  #   "MAE", "RMSE", "MSE", "Relative MAE", "Pearson R",
  #   "FSS", "Wasserstein", "Relative L1", "Relative L2"
  deterministic:
    include: ["MAE", "RMSE", "MSE", "Relative MAE", "Relative L1", "Relative L2", "Pearson R", "FSS"]
  # Subset to compute on standardized pairs (combined mean/std across targets+predictions).
    standardized_include: ["MAE", "RMSE", "MSE", "Relative MAE"]

    # Fractional Skill Score (FSS) configuration
    # - quantile: threshold is computed as this quantile of the observed sample in [0,1]
    # - window_size: integer (square window) or two-element list [height, width]
    # Typical defaults for global verification: quantile=0.90, window_size=9 (i.e., 9x9)
    # Optional: use absolute thresholds instead of quantiles:
    #   - threshold: single float applied to all variables (units must match each variable)
    #   - thresholds: mapping of variable name -> float (overrides 'threshold' for those vars)
    fss:
      quantile: 0.95
      window_size: 9            # Example: square window of 9x9
      # threshold: 10.0          # Example: global absolute threshold (e.g., 10 m/s or 1 mm)
      # thresholds:              # Example: per-variable thresholds (units must match dataset)
      #   "total_precipitation": 1.0     # e.g., 1 mm over accumulation period
      #   "10m_u_component_of_wind": 10.0  # e.g., 10 m/s event

  # ETS thresholds in percentiles (0–100). Each variable’s threshold is computed from the
  # observed distribution at the given percentile.
  ets:
    thresholds: [50, 60, 70, 80, 90]
```

</details>
<!-- markdownlint-enable MD033 -->

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

- All probabilistic outputs (xarray + WeatherBenchX) are written into the same folder: `probabilistic/`
  - Xarray-based (per-variable fields and plots):
    - CRPS summary: `probabilistic/crps_summary.csv`
    - PIT histogram NPZ: `probabilistic/{var}_pit_hist.npz`
    - PIT/CRPS fields: `probabilistic/{var}_pit.nc`, `probabilistic/{var}_crps.nc`
    - Optional figures (when `plot` or `both`): `probabilistic/crps_map_{var}.png`, `probabilistic/pit_hist_{var}.png`
  - WeatherBenchX-based (summaries and aggregates):
    - CSV summaries: `probabilistic/spread_skill_ratio.csv`, `probabilistic/crps_ensemble.csv`
    - Temporal aggregates (NetCDF): `probabilistic/probabilistic_metrics_temporal.nc`
    - Spatial/regional aggregates (NetCDF): `probabilistic/probabilistic_metrics_spatial.nc`
    - Optional CRPS map (WBX): `probabilistic/crps_map_wbx_{var}.png`

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

## Intercomparison of Saved Artifacts

This repo includes a lightweight CLI to combine plots and CSVs from multiple model runs that wrote artifacts (NPZ/CSV) to disk. It reuses the saved outputs under each model's output folder and generates combined visualizations for quick model-vs-model comparisons.

Expected structure per model (created by the main runner):

- output/modelA/
  - maps/*.npz
  - histograms/*_sfc_latbands_combined.npz
  - wd_kde/*_sfc_latbands_kde_combined.npz
  - energy_spectra/*.npz, lsd_2d_metrics.csv
  - deterministic/metrics.csv, metrics_standardized.csv
  - ets/ets_metrics.csv
  - vertical_profiles/*_pl_rel_error_combined.npz
  - probabilistic/
    - crps_summary.csv, spread_skill_ratio.csv, crps_ensemble.csv
    - {var}_pit_hist.npz,
    - crps_map_{var}.png (xarray) and/or crps_map_wbx_{var}.png
    - temporal_metrics.nc, spatial_metrics.nc (WBX)

Run the intercomparison:

```bash
python -m swissclim_evaluations.intercompare output/modelA output/modelB \
  --labels ModelA ModelB \
  --out output/intercomparison \
  --modules spectra hist kde maps metrics prob \
  --max-map-panels 4
```

Outputs are written under `output/intercomparison/` mirroring the module folders. The tool is read-only on the source folders and will only generate figures/CSVs by loading the existing artifacts.

What gets combined:

- energy_spectra: overlays of DS baseline + model spectra per variable (and per level), plus `lsd_2d_metrics_combined.csv`.
- histograms: per-latitude band distributions (DS line + model lines) using saved combined NPZs.
- wd_kde: standardized KDE overlays by latitude band (DS + models) using saved NPZs.
- maps: panel maps with DS in the first column and each model as subsequent columns.
- deterministic: merged CSVs (`metrics_combined.csv`, `metrics_standardized_combined.csv`) and simple bar charts for MAE/RMSE/FSS when data is present.
- ets: merged CSV (`ets_metrics_combined.csv`).
- probabilistic: merged CSVs (`crps_summary_combined.csv`, `spread_skill_ratio_combined.csv`, `crps_ensemble_combined.csv`), PIT histogram overlays, and CRPS map panels when NPZ map exports exist.
  - Additionally merges WBX spatial and temporal aggregates (`spatial_metrics_combined.csv`, `temporal_metrics_combined.csv`), with simple region-wise bar charts and time-bin line plots if the corresponding dimensions are present.

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

### Naming conventions

- targets: the ground-truth/reference dataset (e.g., ERA5). Public APIs now consistently use the parameter name `targets`.
- predictions: the model outputs to be evaluated (e.g., ML). Public APIs now consistently use the parameter name `predictions`.
