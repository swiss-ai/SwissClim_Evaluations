# SwissClim Evaluations

Fast, reproducible evaluation of weather/climate model outputs against ERA5 (or
other references). Compute deterministic and probabilistic scores, spectral
metrics, and helpful diagnostic plots — all driven by a single YAML config.

## Quickstart (default: container with Podman + Enroot)

1. Clone into the expected CSCS path (important)

Several paths in the provided `tools/edf_template.toml` as well as example YAML configs assume the repository lives under:

```bash
/capstor/store/cscs/swissai/a122/$USER/SwissClim_Evaluations
```

If you place the repo elsewhere, you must adapt those absolute paths (image, workdir, mounts) in your generated EDF TOML and any YAML configs that reference model/data locations. To follow the convention used by collaborators, clone like this:

```bash
mkdir -p /capstor/store/cscs/swissai/a122/$USER
git clone git@github.com:swiss-ai/SwissClim_Evaluations.git /capstor/store/cscs/swissai/a122/$USER/SwissClim_Evaluations
cd /capstor/store/cscs/swissai/a122/$USER/SwissClim_Evaluations
```

We recommend the container workflow for fastest, reproducible setup.

1. Build the container (Podman) at repo root in an interactive session:

```bash
srun --container-writable -t 01:00:00 -A a122 -p debug --pty bash
podman build -t swissclim-eval .
```

1. (CSCS Alps) Export to Enroot SQuashFS and set up EDF once:

```bash
rm -f tools/swissclim-eval.sqsh
enroot import -x mount -o tools/swissclim-eval.sqsh podman://swissclim-eval
exit # exit the interactive build session
mkdir -p ~/.edf
sed "s/{{username}}/$USER/g" tools/edf_template.toml > ~/.edf/swissclim-eval.toml
```

1. Review and edit the example config:

The project ships with a commented config that explains every key and valid
values. Copy it and adjust the paths and selections as needed.

```bash
cp config/example_config.yaml config/my_run.yaml
```

1. (CSCS Alps) Launch an interactive session using the container:

```bash
srun --container-writable --environment=swissclim-eval -A a122 -t 01:30:00 -p debug --pty /bin/bash
```

You are now inside the container with all dependencies installed. For a richer
debugging experience we recommend using `code tunnel`.

1. Run:

```bash
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

Outputs appear under paths.output_root (one sub-folder per module).

Note: For reproducibility, the CLI copies the exact YAML config you pass with `--config`
into the output_root directory at the start of the run (using the original filename).

1. Or submit a batch job (CSCS Alps):

```bash
sbatch launchscript.sh
```

Or directly with a single command:

```bash
srun --job-name=swissclim-eval --time=00:30:00 --account=a122 \
    --partition=normal --nodes=1 --ntasks=1 --cpu-bind=cores \
    --container-writable --environment=/users/$USER/.edf/swissclim-eval.toml \
    python -u -m swissclim_evaluations.cli --config config/my_run.yaml
```

Don't forget to adjust the path to your `config/my_run.yaml` in
`launchscript.sh`.

1. Here is a one-liner with `srun` instead of the `launchscript`:

```bash
srun --job-name=swissclim-eval --time=01:30:00 --account=a122 --partition=normal --container-writable --environment=swissclim-eval /bin/bash -c 'export PYTHONUNBUFFERED=1 && python -u -m swissclim_evaluations.cli --config config/my_run.yaml'
```

> Prefer a plain virtual environment? Use one of the alternatives below.

## Quick debug: single-lead run

Use the provided debug config to run a very small, single-lead evaluation end-to-end. This keeps all modules enabled while trimming data volume to finish fast.

- Config: `config/debug_single_lead.yaml`
- Output folder: `output/debug_single_lead/`

Run inside the container (recommended):

```bash
srun --container-writable --environment=/users/$USER/.edf/swissclim-eval.toml \
  -A a122 -t 00:15:00 -p debug --pty \
  python -u -m swissclim_evaluations.cli --config config/debug_single_lead.yaml
```

Or submit as a batch job by temporarily setting `CONFIG_PATH_IN_REPO="config/debug_single_lead.yaml"` in `launchscript.sh` and then:

```bash
sbatch launchscript.sh
```

## Install with uenv + uv

```bash
bash tools/setup_env_uenv.sh # activates uenv and exits
bash tools/setup_env_uenv.sh # installs deps with uv
# Activates .venv and installs deps via uv
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

## Install with conda + uv

```bash
bash tools/setup_env_conda.sh
conda activate swissclim-eval
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

## Configure

The YAML is the single source of truth. Use the commented example directly:

<!-- markdownlint-disable MD033 -->
<details>
<summary><strong>Example config (click to expand)</strong></summary>

```yaml
# Source: config/example_config.yaml
# Global configuration for SwissClim Evaluations
#
# Tip: All paths should be absolute or relative to the repository root.

paths:
  # Path(s) to your model predictions in Zarr format (root directory).
  # Accepts a single string or a list of strings. When a list is provided, the
  # archives are combined lazily by coordinates (no data is materialized; Dask
  # graphs remain intact). All stores must follow the same variable/dimension schema.
  ml: "/capstor/store/cscs/swissai/a122/ESFM_Results/aurora_small_6h/rollout_pred_2023-01-02x112_2steps.zarr" # has 3D vars

  # Path(s) to the reference dataset (ERA5 in WeatherBench2 layout).
  # Accepts a single string or a list of strings. When multiple stores are provided,
  # they are combined lazily by coordinates.
  nwp: [
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr"
  ]

  # Where outputs are written; a subfolder per module will be created here.
  output_root: "output/verification_esfm"

selection:
  # Pressure levels (hPa) for 3D variables. Only levels present in the data are kept.
  # Common choices: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  levels: [100, 500, 1000]

  # Optional coarse temporal downsampling in hours. Applied as a stride along lead_time if present,
  # otherwise along init_time. This is an early reduction step and is independent of the
  # multi-lead selection policy below. Leave null if you plan to control lead selection
  # exclusively via the lead_time policy.
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
  datetimes: [
    "2023-01-02T00:2023-01-08T23",
    "2023-04-02T00:2023-04-08T23",
    "2023-07-02T00:2023-07-08T23",
    "2023-10-02T00:2023-10-08T23",
    "2024-01-02T00:2024-01-08T23",
    "2024-04-02T00:2024-04-08T23",
    "2024-07-02T00:2024-07-08T23",
    "2024-10-02T00:2024-10-08T23"
  ]

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
    "u_component_of_wind",
    # "v_component_of_wind",
  ]

  # Optional pre-selection of ensemble members BEFORE any module logic runs.
  #   Accepts:
  #     - null  : keep all members (recommended) and let per‑module ensemble modes decide.
  #     - int   : select exactly one member; 'ensemble' dimension is dropped and ALL per-module
  #               ensemble modes (mean/members/pooled/prob) become invalid except probabilistic
  #               which will still use the single deterministic path.
  #     - list[int]: select multiple members; the dataset is subset to those members but the
  #               'ensemble' dimension is preserved (length = len(list)). Per‑module ensemble
  #               modes still function but operate only over the retained subset.
  #   Notes:
  #     - Providing a single-element list behaves like the int form (dimension is dropped).
  #     - Indices are zero-based.
  #     - Use plotting.plot_ensemble_members for restricting WHICH members are plotted in
  #       members-capable modules without discarding others for metrics.
  ensemble_members: null

  # Per-module ensemble modes (override defaults if needed):
  #   mean     → reduce to mean (ensmean)
  #   pooled   → pooled sample (enspooled)
  #   prob     → probabilistic semantics (ensprob; probabilistic module only)
  #   members  → per-member outputs (ens0..ensN)
  #   none     → no ensemble handling (only if dataset lacks ensemble dim)
  ensemble:
    maps: members            # members | mean | none*
    histograms: pooled       # pooled | members | mean | none*
    wd_kde: pooled           # pooled | members | mean | none*
    energy_spectra: mean     # mean | members | pooled | none*
    vertical_profiles: mean  # mean | members | none*
    deterministic: mean      # mean | members | none*
    ets: mean                # mean | members | none*
    probabilistic: prob      # prob (only valid choice; module forces prob)
    # * 'none' only if dataset truly has no ensemble dimension.

  # If true, raise an error when requested pressure levels are missing from the data.
  check_missing: false

plotting:
  # Random seed used for reproducible time subsampling in plots.
  random_seed: 42

  # Plot a specific init_time. Must lie within selection.datetimes and exist in predictions.
  # Example: "2023-01-03T12". Leave null/omit to plot the first available init_time.
  plot_datetime: null

  # Plot specific ensemble members (indices) for figure/NPZ exports in members-capable modules (e.g. maps, spectra).
  # Provide a list of integers to restrict outputs to that subset; leave null/omit to include all members.
  # Ignored when the module's ensemble mode is not 'members' (e.g. maps set to mean).
  # Example: [0, 3, 7]
  plot_ensemble_members: null

  # Base DPI for plots.
  dpi: 48

  # Maximum number of samples per latitude band used for KDE/Wasserstein in wd_kde.
  # Larger values improve smoothness but increase compute and memory. Typical: 50_000–200_000.
  # Note: Provide as a plain integer (no underscores) in YAML.
  kde_max_samples: 200000

  # Maximum number of samples per latitude band for histograms.
  # When provided, histograms are built from a deterministic subsample (mirrors wd_kde logic)
  # rather than the full arrays. Set to null/omit to use full data. Typical: 50000–200000.
  histogram_max_samples: 200000


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
  histograms: true           # Distributions by latitude bands (2D + optional per-level 3D variables)
  wd_kde: true               # KDE by latitude band on standardized fields (2D + optional per-level 3D); reports mean Wasserstein
  energy_spectra: true       # Zonal energy spectra + LSD table; NPZ always saved
  vertical_profiles: true    # Normalized MAE (NMAE) vertical profiles per latitude band (3D)
  deterministic: true        # Deterministic metrics (MAE, RMSE, etc.) incl. standardized variants
  ets: true                  # Equitable Threat Score across quantile thresholds
  probabilistic: true        # Combined probabilistic (xarray CRPS/PIT + WBX SSR/CRPS)

# Multi-lead evaluation policy
# modes:
#   first  - only first lead time (single-lead evaluation mode)
#   full   - use all available lead hours
#   subset - explicit list of lead hours to keep (provide subset_hours)
#   stride - keep every Nth hour (provide stride_hours)
# panel.* controls which lead hours are used for multi-panel plots (maps, histograms, spectra, etc.)
lead_time:
  mode: stride            # first | full | subset | stride
  # Prefer nested style for stride/subset; flat keys remain supported
  stride:
    hours: 12
  max_hour: 120
  # Subset example (only if mode=subset)
  # subset:
  #   hours: [0, 6, 12, 24]
  # Legacy: subset_hours: [0, 6, 12, 24]
  # Panel (multi-figure subset) configuration has been removed; all modules now
  # use the same retained set of lead_time hours after applying this policy.


metrics:
  # Deterministic metrics configuration. If lists are omitted, a default set is computed.
  # Available metric names:
  #   "MAE", "RMSE", "MSE", "Relative MAE", "Pearson R",
  #   "FSS", "Relative L1", "Relative L2"
  # Ensemble behaviour now controlled by ensemble.deterministic (mean|members).
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
      quantile: 0.90
      window_size: 9            # Example: square window of 9x9
      # threshold: 10.0          # Example: global absolute threshold (e.g., 10 m/s or 1 mm)
      # thresholds:              # Example: per-variable thresholds (units must match dataset)
      #   "total_precipitation": 1.0     # e.g., 1 mm over accumulation period
      #   "10m_u_component_of_wind": 10.0  # e.g., 10 m/s event

  # ETS thresholds in percentiles (0–100). Each variable’s threshold is computed from the
  # observed distribution at the given percentile.
  # Ensemble behaviour now controlled by ensemble.ets (mean|members).
  ets:
    thresholds: [50, 70, 90]
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
  - `init_time: 1, lead_time: 1, level: 1, latitude: -1, longitude: -1, ensemble:
    -1 (-1 = no chunking)`
- This ensures apply_ufunc metrics that use the ensemble as a core dimension
  work without errors and keeps memory usage predictable. If a dataset deviates,
  it will be rechunked automatically with a warning.

## Output Overview

The evaluation generates organized results for each enabled module:

### Deterministic Metrics

Filenames encode only information that is actually present:

- Metric family (e.g. `deterministic_metrics`)
- Optional qualifier (`averaged`, `init_time`, `standardized`, combinations thereof)
- Optional time range tokens if an init and/or lead range exists: `initYYYYMMDDHH-YYYYMMDDHH` and `leadXXXh-YYYh`
- Ensemble token (always; see "Ensemble Tokens" below)

Examples (mean reduction vs members):

```text
deterministic_metrics_ensmean.csv
deterministic_metrics_averaged_init2023010200-2023010412_lead000h-036h_ensmean.csv
deterministic_metrics_standardized_ensmean.csv
deterministic_metrics_ens0.csv            # members mode example (member 0)
```

### Extreme Threshold Statistics (ETS)

Outputs include single-file summaries and multi-lead artifacts with standardized filenames built by the shared helper (time-range and ensemble tokens appended when present):

- Summary CSVs
  - `ets/ets_metrics_ens*.csv`
  - `ets/ets_metrics_averaged_[initYYYYMMDDHH-YYYYMMDDHH]_[leadXXXh-YYYh]_ens*.csv`
  - In members mode, one file per member (e.g., `ens0`, `ens1`, …).

- Multi-lead wide table (when lead_time policy keeps >1 hour)
  - `ets/ets_metrics_by_lead_wide.csv`
  - Columns: `lead_time_hours` + one column per (variable × threshold), e.g., `var_ETS 50%`.

- Line artifacts by variable (thresholds vs lead_time)
  - PNG: `ets/ets_line_<variable>_[init...]_[lead...]_ens*.png`
  - NPZ: `ets/ets_line_<variable>_data_[init...]_[lead...]_ens*.npz`
  - CSV (tidy): `ets/ets_line_<variable>_by_lead_[init...]_[lead...]_ens*.csv`

Notes

- Thresholds are computed as percentiles of the observed distribution per variable.
- Ensemble handling follows `ensemble.ets` in the YAML (mean | members). The filename token will be `ensmean`, `ens<i>`, `enspooled` if applicable, or omitted when no ensemble dim exists (defaults to `ensnone`).

### Energy Spectra Analysis

Per-variable (and per-level) energy spectra are computed retaining time structure; the Log Spectral Distance (LSD)
is exported per init_time/lead_time and summarized. Outputs:

- Figures / NPZ (subset init_time for plotting) : `energy_spectra/lsd_<variable>[_<level>]_spectrum[_init...][_lead...]_ens*.{png|npz}`
- LSD per-time (2D): `energy_spectra/lsd_2d_metrics_per_init_time_<range>.csv` or `per_lead_time` depending on dims
- LSD averaged (2D mean): `energy_spectra/lsd_2d_metrics_averaged_<range>.csv`
- LSD init_time (2D): `energy_spectra/lsd_2d_metrics_init_time_<range>.csv` (mean over other time dims, retaining init_time)
- LSD per-time (3D): `energy_spectra/lsd_3d_metrics_per_init_time_<range>.csv` (or per_lead_time)
- LSD averaged (3D levels wide): `energy_spectra/lsd_3d_metrics_averaged_<range>.csv` (rows=levels, columns=variables)
- LSD init_time (3D): `energy_spectra/lsd_3d_metrics_init_time_<range>.csv`
- LSD (banded by wavelength) — new: `energy_spectra/lsd_bands_2d_metrics_*` and `lsd_bands_3d_metrics_*` variants for detailed, averaged, and init_time summaries.

### Distribution Analysis (Histograms & KDE / Wasserstein)

Histogram and KDE outputs encode:

- Prefix: `hist_` or `wd_kde_`
- Variable + optional level token
- Latitudinal aggregation token (`latbands` or `_latbands_combined` / `_latbands_kde_combined`)
- Optional time-range tokens
- Ensemble token

Examples (pooled vs members):

```text
hist_2m_temperature_latbands_enspooled.npz
hist_temperature_500_latbands_combined_enspooled.npz
wd_kde_2m_temperature_combined_enspooled.npz
wd_kde_wasserstein_averaged_enspooled.csv
wd_kde_2m_temperature_latbands_kde_combined_ens3.npz   # per-member (members mode)
```

Time ranges (if present) appear just before the ensemble token: `..._init2023010200-2023010412_lead000h-036h_enspooled.npz`.

Multi-lead figure artifacts (PNG + NPZ):

- Global histograms grid by lead_time
  - PNG: `hist_global_<variable>_grid_*.png`
  - NPZ bundle: `hist_global_<variable>_grid_data_*.npz` containing per-panel edges and densities for ground truth and model, plus lead_hours.
- WD-KDE ridgeline (global standardized KDE evolving along lead_time)
  - PNG: `wd_kde_evolve_ridgeline_*.png`
  - NPZ bundle: `wd_kde_evolve_ridgeline_data_*.npz` with `lead_hours`, `y_eval`, `density_target`, `density_model`.

### Vertical Structure (3D variables only)

Outputs (standardized naming):

- Plot: `vertical_profiles/vprof_nmae_<variable>_multi_plot[_init...][_lead...]_ens*.png`
- Combined band data (NPZ): `vertical_profiles/vprof_nmae_<variable>_multi_combined[_init...][_lead...]_ens*.npz`
- Lead-time evolution overlays:
  - PNG: `vertical_profiles/vprof_nmae_<variable>_multi_evolve*.png`
  - NPZ: `vertical_profiles/vprof_nmae_<variable>_multi_evolve_data*.npz` (profiles per lead)
  - Heatmap PNG: `vertical_profiles/vprof_nmae_<variable>_multi_evolve_heatmap*.png`
  - Heatmap NPZ: `vertical_profiles/vprof_nmae_<variable>_multi_evolve_heatmap_data*.npz`
- Summaries (CSV) may be produced by intercomparison rather than the module itself.

### Spatial Maps

Maps include the selected (or single) init/lead span and ensemble token. In members mode one PNG (and/or NPZ if `output_mode` includes it) per member is produced.

```text
map_10m_u_component_of_wind_init2023010200-2023010412_ens0.png   # member 0
map_temperature_500_init2023010200-2023010412_ensmean.png        # mean reduction
map_10m_u_component_of_wind_init2023010200-2023010412_ens3.npz   # NPZ export (output_mode=npz/both)
```

- All probabilistic outputs (xarray + WeatherBenchX) are written into the same
  folder: `probabilistic/`
  - Xarray-based (per-variable fields and plots):
    - CRPS summary: `probabilistic/crps_summary.csv`
    - PIT histogram NPZ: `probabilistic/{var}_pit_hist.npz`
    - PIT/CRPS fields: `probabilistic/{var}_pit.nc`,
      `probabilistic/{var}_crps.nc`
    - Optional figures (when `plot` or `both`):
      `probabilistic/crps_map_{var}.png`, `probabilistic/pit_hist_{var}.png`
  - WeatherBenchX-based (summaries and aggregates):
    - CSV summaries: `probabilistic/spread_skill_ratio.csv`,
      `probabilistic/crps_ensemble.csv`
    - Temporal aggregates (NetCDF):
      `probabilistic/probabilistic_metrics_temporal.nc`
    - Spatial/regional aggregates (NetCDF):
      `probabilistic/probabilistic_metrics_spatial.nc`
    - Optional CRPS map (WBX): `probabilistic/crps_map_wbx_{var}.png`

### Probabilistic Verification (combined xarray + WeatherBenchX)

All probabilistic artifacts use the dedicated token `ensprob` (never `ensmean` / `enspooled`). This distinguishes probabilistic semantics (ensemble retained for PIT/CRPS computation) from deterministic or pooled reductions.

Per-variable artifacts:

```text
pit_hist_2m_temperature_ensprob.npz
pit_field_2m_temperature_ensprob.nc
crps_field_2m_temperature_ensprob.nc
crps_map_2m_temperature_ensprob.png        # optional map (if plotting enabled)
crps_map_wbx_2m_temperature_ensprob.png    # WeatherBenchX CRPS map

Additional multi-lead lines and tables:

- CRPS line vs lead_time per variable
  - PNG: `probabilistic/crps_line_<variable>*.png`
  - CSV: `probabilistic/crps_line_<variable>_by_lead*.csv`
  - NPZ: `probabilistic/crps_line_<variable>_by_lead_data*.npz`
- PIT histogram uniform deviation summary vs lead_time
  - CSV: `probabilistic/pit_hist_<variable>_uniform_diff_by_lead*.csv`
  - NPZ: `probabilistic/pit_hist_<variable>_uniform_diff_by_lead_data*.npz`
```

WBX summary tables / fields:

```text
spread_skill_ratio_ensprob.csv
crps_ensemble_ensprob.csv
prob_metrics_temporal_ensprob.nc
prob_metrics_spatial_ensprob.nc
```

Aggregated CRPS summaries (xarray based):

```text
crps_summary_ensprob.csv
crps_summary_averaged_init2023010200-2023010412_lead000h-024h_ensprob.csv
```

### Details for probabilistic outputs

- CRPS and PIT are computed per variable using the ensemble along the `ensemble`
  dimension.
- CRPS returned by the library functions is a DataArray (not a Dataset). In
  notebooks, use the DataArray directly and then reduce over time-like dims to
  make maps.
- PIT histograms are stored as NPZ (counts, edges) for reproducibility;
  corresponding PIT fields are also written to NetCDF.
- When multiple lead_time hours are present, CRPS grid maps are produced across all hours; for tabular review, we also export a `crps_line` CSV per variable listing the mean CRPS by lead_time (and a matching NPZ).
- Likewise, we export a compact PIT uniform-deviation CSV (difference from a uniform histogram) by lead_time for quick diagnostics (and a matching NPZ).

All modules print concise progress like:

- [swissclim] Module: deterministic — variables=5
- [histograms] variable: 10m_u_component_of_wind
- [energy_spectra] saved
  output/verification_esfm/energy_spectra/u_component_of_wind_500hPa_energy.png
- [energy_spectra] saved output/verification_esfm/energy_spectra/u_component_of_wind_500hPa_spectrum.png

## Tips and best practices

- Reduce size
  - selection.temporal_resolution_hours: 1, 3, 6 …
  - plotting.plot_datetime: pick a single init_time to plot (default already
    selects the first).
- Keep the same variable names between ERA5 and ML; only overlapping variables
  are processed.
- Maps require Cartopy; use npz mode on headless systems if you only need data
  exports.
- For reproducibility in plots, prefer plotting.output_mode: npz or both, which
  writes the exact arrays used to render figures.

## Notebooks

You can explore the outputs interactively using the provided notebooks:

- notebooks/deterministic_verification.ipynb (classic deterministic metrics,
  maps, histograms, spectra, profiles)
- notebooks/probabilistic_verification.ipynb (classic CRPS/PIT using our
  xarray-based implementation)
- notebooks/probabilistic_verification_wbx.ipynb (WeatherBenchX Spread–Skill
  Ratio and CRPS summaries)

Notebook tips

- Use the YAML config and `prepare_datasets` to ensure alignment, selection, and
  chunking are consistent with the CLI.

## Multi-lead Time Evaluation

Multi-lead (forecast horizon) evaluation computes metrics and plots for every retained `lead_time` rather than only the first (t+0). It is now a first-class feature.

### Policy Overview

Add a `lead_time` block to the YAML (see embedded example config). Supported modes:

| Mode | Description |
|------|-------------|
| first  | Keep only the first lead (single-lead evaluation mode). |
| full   | Keep all leads (optionally capped by `max_hour`). |
| subset | Keep only hours listed in `subset_hours`. |
| stride | Keep every Nth hour defined by `stride_hours`. |

Additional field: `max_hour` (inclusive cap). The previous `panel.*` keys and
`plotting.lead_panel_hours` have been deprecated; every module now visualizes
the complete retained lead_time set.

At runtime the CLI prints a “Lead Time Policy” panel summarizing the resolved mode, stride/subset, and the final list of lead hours after selection.

### Produced Artifacts

| Module | Additional Files (multi-lead) | Single-lead Summary File (averaged across leads or first lead) |
|--------|-------------------------------|---------------------------------------------------------------|
| deterministic | metrics_by_lead_long.csv, metrics_by_lead_wide.csv, per-variable det_line PNG + NPZ + CSV | metrics.csv / metrics_standardized.csv |
| ets | ets_metrics_by_lead_wide.csv, per-variable ets_line PNG + NPZ + CSV | ets_metrics.csv |
| probabilistic | crps_map grid PNGs; crps_line per-variable PNG + NPZ + CSV; PIT grid PNGs; PIT uniform_diff_by_lead CSV + NPZ | crps_summary.csv |
| energy_spectra | spectrogram target/model/diff PNGs + NPZ bundle; lsd_2d_by_lead CSV (long+wide) | lsd_2d_metrics.csv |
| plots (maps, histograms, wd_kde) | Maps grid PNG (+ NPZ grid_data); hist_global grid PNG (+ NPZ grid_data); KDE ridgeline PNG (+ NPZ) | — |

If no `lead_time` section is provided behavior falls back to `first` (single lead) for backward compatibility.

### Tips

- Use `stride_hours` with `max_hour` to bound compute when models produce many leads.
- Use `panel.strategy: evenly_spaced` for balanced visual coverage.
- Keep `subset_hours` small (< ~10) if you plan to enable many plotting modules simultaneously.

### Multi-lead plotting features

Enable these in the `plotting:` block to visualize evolution across lead_time:

- `histograms_per_lead`: one histogram page per selected lead_time (2D vars)
- `maps_per_lead_grid`: grid panels for all retained lead_time hours
- `wd_kde_global_evolution`: 3D perspective of global standardized KDE evolving along lead_time
- `energy_spectra_spectrogram`: 2D image with x=lead_time, y=wavenumber, color=log10(energy)
- `probabilistic_line_plots`: CRPS vs lead_time line per variable (requires ensemble size >=2)
- `vertical_profiles_evolve_lead`: overlay vertical NMAE profiles for all retained lead times (3D vars)

ETS line plot (thresholds vs lead_time) can be enabled via `metrics.ets.line_plot: true`.

### Migration Notes (Panel Removal)

Prior versions supported a separate "panel" subset (`panel.strategy`, `panel.count`,
`plotting.lead_panel_hours`) to reduce the number of lead times shown in multi-panel
figures. This complexity was removed:

- All modules (maps, histograms, wd_kde, energy_spectra spectrograms, vertical profiles, probabilistic CRPS/PIT) now plot the full retained lead_time set after applying the selection policy.
- Any existing `panel:` block or `plotting.lead_panel_hours` key in configs is ignored safely.
- To limit the number of leads in both metrics and plots, adjust the main policy using `mode: stride`, `stride.hours`, or provide a concise `subset.hours` list, optionally with `max_hour`.

Configs referencing deprecated keys do not need immediate cleanup; they simply no longer affect output. Remove them gradually to avoid confusion.

## Intercomparison of Saved Artifacts

This repo includes a lightweight CLI to combine plots and CSVs from multiple
model runs that wrote artifacts (NPZ/CSV) to disk. It reuses the saved outputs
under each model's output folder and generates combined visualizations for quick
model-vs-model comparisons.

Expected structure per model (created by the main runner):

- output/modelA/
  - maps/*.npz
  - histograms/*_sfc_latbands_combined.npz, *_pl*_latbands_combined.npz
  - wd_kde/*_sfc_latbands_kde_combined.npz, *_pl*_latbands_kde_combined.npz
  - energy_spectra/*.npz, lsd_2d_metrics.csv
  - deterministic/metrics.csv, metrics_standardized.csv
  - ets/ets_metrics.csv
  - vertical_profiles/*_pl_nmae_combined.npz
  - probabilistic/
    - crps_summary.csv, spread_skill_ratio.csv, crps_ensemble.csv
    - {var}_pit_hist.npz,
    - crps_map_{var}.png (xarray) and/or crps_map_wbx_{var}.png
    - temporal_metrics.nc, spatial_metrics.nc (WBX)
This repo includes a lightweight CLI to combine plots and CSVs from multiple model runs that wrote artifacts (NPZ/CSV) to disk. It reuses the saved outputs under each model's output folder and generates combined visualizations for quick model-vs-model comparisons. A separate config is available for intercomparison.

Run the intercomparison:

```bash
python -m swissclim_evaluations.intercompare --config config/intercomparison.yaml
```

Outputs are written under `output/intercomparison/` mirroring the module
folders. The tool is read-only on the source folders and will only generate
figures/CSVs by loading the existing artifacts.

What gets combined:

- energy_spectra: overlays of DS baseline + model spectra per variable (and per
  level), plus `lsd_2d_metrics_combined.csv`.
- histograms: per-latitude band distributions (DS line + model lines) using
  saved combined NPZs.
- wd_kde: standardized KDE overlays by latitude band (DS + models) using saved
  NPZs.
- maps: panel maps with DS in the first column and each model as subsequent
  columns.
- deterministic: merged CSVs (`metrics_combined.csv`,
  `metrics_standardized_combined.csv`) and simple bar charts for MAE/RMSE/FSS
  when data is present.
- ets: merged CSV (`ets_metrics_combined.csv`).
- probabilistic: merged CSVs (`crps_summary_combined.csv`,
  `spread_skill_ratio_combined.csv`, `crps_ensemble_combined.csv`), PIT
  histogram overlays, and CRPS map panels when NPZ map exports exist.
  - Additionally merges WBX spatial and temporal aggregates
    (`spatial_metrics_combined.csv`, `temporal_metrics_combined.csv`), with
    simple region-wise bar charts and time-bin line plots if the corresponding
    dimensions are present.
- vertical profiles (vprof): overlay plots per variable of latitude-band vertical NMAE across models — saved as `vertical_profiles/{var}_pl_nmae_combined_compare.png` — plus per-variable summary tables (`{var}_pl_nmae_combined_summary.csv`) listing mean metric by band, hemisphere, and model.
- energy_spectra: overlays of DS baseline + model spectra per variable (and per level), plus `lsd_2d_metrics_combined.csv` and `lsd_bands_2d_metrics_averaged_combined.csv` when available.
- histograms: per-latitude band distributions (DS line + model lines) using saved combined NPZs.
- wd_kde: standardized KDE overlays by latitude band (DS + models) using saved NPZs.
- maps: panel maps with DS in the first column and each model as subsequent columns.
- deterministic: merged CSVs (`metrics_combined.csv`, `metrics_standardized_combined.csv`) and simple bar charts for MAE/RMSE/FSS when data is present.
- ets: merged CSV (`ets_metrics_combined.csv`).
- probabilistic: merged CSVs (`crps_summary_combined.csv`, `spread_skill_ratio_combined.csv`, `crps_ensemble_combined.csv`), PIT histogram overlays, and CRPS map panels when NPZ map exports exist.
  - Additionally merges WBX spatial and temporal aggregates from `prob_metrics_{spatial,temporal}_*.nc` (or legacy names) into (`spatial_metrics_combined.csv`, `temporal_metrics_combined.csv`), with simple region-wise bar charts and time-bin line plots if the corresponding dimensions are present.
  - A single availability panel covers all probabilistic artifacts (PIT, CRPS maps, spatial/temporal WBX).
- vertical profiles (vprof): overlay plots per variable of latitude-band vertical NMAE across models — saved as `vertical_profiles/vprof_nmae_<variable>_multi_combined_compare.png` — plus per-variable summary tables (`vprof_nmae_<variable>_multi_combined_summary.csv`) listing mean metric by band, hemisphere, and model. Legacy `*_pl_nmae_combined*` files are still supported as input.

## Development

```text
- Python: 3.11
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

Contributions welcome — keep changes chunk-aware (xarray/dask friendly) and
small.

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

- targets: the ground-truth/reference dataset (e.g., ERA5). Public APIs now
  consistently use the parameter name `targets`.
- predictions: the model outputs to be evaluated (e.g., ML). Public APIs now
  consistently use the parameter name `predictions`.
