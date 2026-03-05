# Output Overview

This document explains the naming conventions of the output files, but the user should not need to care about this in general use cases. Files are read automatically by the notebooks and this is the main interaction the user has with outputs.

`plotting.output_mode` supports `plot`, `npz`, `both`, and `none`.
With `none`, PNG/NPZ artifacts are not generated; only CSV outputs listed here are emitted.
In `none` mode, artifact-only modules (`maps`, `histograms`, `vertical_profiles`) are skipped and
the run log prints explicit skip messages.

The evaluation generates organized results for each enabled module.

## Derived Variables

The `derived_variables` config block computes new variables lazily from existing
data_vars **before any module runs**. Derived variables therefore appear as ordinary variables in
all module outputs — outputs are named after the derived variable exactly as they would be for any
raw variable.

Currently available recipes:

| Recipe | Formula | Units | Inputs |
|---|---|---|---|
| `wind_speed` | `sqrt(U²+V²)` | m s⁻¹ | `u` + `v` |
| `geopotential_height` | `geopotential / 9.80665` | m | `u` (or `source`) only || `geopotential_height_gradient` | `sqrt((∂Z/∂y)²+(∂Z/∂x)²)` — horizontal gradient magnitude \|∇Z\| | m m⁻¹ | `u` (or `source`) only; supply `geopotential_height` variable |
See `README.md § Derived Variables` for the config syntax and [WIND_UV_ASSESSMENT.md](WIND_UV_ASSESSMENT.md) for a per-module impact analysis of wind variables.

### Physical constraint overlays in Bivariate Histograms

The multivariate bivariate-histogram plots automatically add physical constraint overlays for
recognised variable pairs:

| Pair | Constraint | Overlay |
|---|---|---|
| `temperature` × `specific_humidity` | Clausius–Clapeyron saturation curve (Bolton 1980, 500 hPa) | dashed red saturation line; supersaturated region hatched red; `q < 0` region hatched dark-red |
| `geopotential_height` × `wind_speed` | Wind speed is a magnitude ≥ 0 by definition | dashed red line at wind speed = 0; below-zero region hatched red |
| `geopotential_height_gradient` × `wind_speed` | Geostrophic balance: U_g = (g/f)\|∇Z\| with mid-latitude f = 10⁻⁴ s⁻¹ | diagonal dashed orange reference line through the origin; dashed red line at wind speed = 0 |

Constraints are drawn in both orientations (either variable may be on either axis).

## Ensemble handling: modes and filename tokens

Datasets may include an `ensemble` dimension. You can pre‑select members (`selection.ensemble_members`) and set per‑module modes under `ensemble`. Invalid combinations are rejected early.

Modes → tokens:

- `mean` → `_ensmean`
- `pooled` → `_enspooled`
- `members` → `_ens<i>`
- `prob` → `_ensprob`
- `none` → only when no ensemble dim (still names `_ensmean`)

**Allowed ensemble modes by module:**

If the dataset contains multiple ensembles, metrics can be computed on the mean of all ensembles (mean setting below), individually per ensemble (members setting below), or pooled across all members (pooled setting below).

- **maps**: mean, members
- **histograms**: mean, pooled, members
- **wd_kde**: mean, pooled, members
- **energy_spectra**: mean, pooled, members
- **vertical_profiles**: mean, pooled, members
- **deterministic**: mean, pooled, members
- **ets**: mean, pooled, members
- **probabilistic**: prob only

The ensemble dimension is always present (size 1 for deterministic datasets). For such datasets, the ensemble mean is identical to the single member. Output filenames will reflect the configured mode (e.g., `_ensmean` for mean, `_ens0` for members). Legacy `_ensnone` tokens are also accepted by the intercomparison tool.

Notes: Members mode may include mean aggregates in some summaries (e.g., energy spectra LSD tables).

For modules where 3D variables are applicable, outputs are reported per level by default.

### Deterministic Metrics

Filenames encode only information that is actually present:

- Metric family (e.g. `deterministic_metrics`)
- Optional qualifier (`averaged`, `init_time`, `standardized`, `per_lead_time`, combinations thereof). Note: `averaged` implies scalar mean over all dimensions (including levels for 3D variables).
- Optional time range tokens if an init and/or lead range exists: `initYYYYMMDDHH-YYYYMMDDHH` and `leadXXXh-YYYh`
- Ensemble token (always; see "Ensemble Tokens" below)

Examples (mean reduction vs members):

```text
deterministic_metrics_ensmean.csv
deterministic_metrics_averaged_init2023010200-2023010412_lead000h-036h_ensmean.csv
deterministic_metrics_standardized_ensmean.csv
deterministic_metrics_per_level_ensmean.csv
deterministic_metrics_per_lead_time_ensmean.csv
deterministic_metrics_standardized_per_level_ensmean.csv
deterministic_metrics_ens0.csv            # members mode example (member 0)
deterministic_metrics_members_mean_enspooled.csv
deterministic_metrics_by_lead_long_ensmean.csv
deterministic_metrics_by_lead_wide_ensmean.csv
det_line_2m_temperature_MAE_by_lead_ensmean.csv
det_line_2m_temperature_MAE_ensmean.png
det_line_2m_temperature_MAE_data_ensmean.npz
det_line_temperature_500_MAE_by_lead_ensmean.csv   # 3D variable with level token
```

#### Spatial Metric Maps (MAE / RMSE / Bias)

When `output_mode` is `plot`, `npz`, or `both`, the deterministic module also produces
spatial-field metric maps for every selected variable (2-D and 3-D at each pressure level).
Only metrics present in `deterministic.include` are generated (by default all three).
For multi-lead runs each lead time is shown as a separate row in the figure.

```text
det_mae_map_2m_temperature_ensmean.png
det_mae_map_2m_temperature_ensmean.npz
det_rmse_map_2m_temperature_ensmean.png
det_bias_map_2m_temperature_lead000h-072h_ensmean.png   # multi-lead
det_mae_map_temperature_500_init2023010200-2023010412_ensmean.png   # 3D per-level
det_mae_map_temperature_500_init2023010200-2023010412_ensmean.npz
```

NPZ keys: `mae` (or `rmse`/`bias`) — mean spatial field, `latitude`, `longitude`,
`variable`, `units`, and optionally `level`, `<metric>_per_lead` (3-D stack), `lead_labels`.

### Spread-Skill Ratio (SSR)

SSR filenames follow a similar pattern:

```text
ssr_summary_averaged_init2023010200-2023010412_ensprob.csv
ssr_line_2m_temperature_by_lead_ensprob.csv
ssr_line_temperature_500_by_lead_ensprob.csv   # 3D variable with level token
```

### Extreme Threshold Statistics (ETS)

ETS filenames follow the same minimal pattern as deterministic metrics and include `ensmean` or `ens<i>` tokens depending on ensemble mode:

```text
ets_metrics_ensmean.csv
ets_metrics_averaged_init2023010200-2023010412_ensmean.csv
ets_metrics_per_level_ensmean.csv
ets_metrics_init_time_ens0.csv   # members mode per-member file
ets_line_2m_temperature_by_lead_ensmean.csv
ets_line_2m_temperature_ensmean.png
ets_line_2m_temperature_data_ensmean.npz
```

### SSIM (Structural Similarity Index)

SSIM filenames follow the same minimal pattern as deterministic metrics and include `ensmean` or `ens<i>` tokens depending on ensemble mode:

```text
ssim/ssim_ssim_ensmean.csv
ssim/ssim_ssim_ens0.csv   # members mode per-member file
```

Each CSV contains per-variable SSIM scores plus an `AVERAGE_SSIM` summary row. The metric is calculated per 2-D spatial slice (lat/lon) and averaged over all remaining dimensions.

**Configuration** — edit `metrics.ssim` in your YAML:

```yaml
metrics:
  ssim:
    sigma: 1.5        # Gaussian kernel sigma (default 1.5)
    # K1: 0.01        # luminance constant
    # K2: 0.03        # contrast constant
```

### Energy Spectra Analysis

Per-variable (and per-level) energy spectra are computed retaining time structure; the Log Spectral Distance (LSD)
is exported per init_time/lead_time and summarized. Outputs are split into single-lead (standard) and per-lead variants.

**Standard (Single Lead or averaged):**

- Figures / NPZ: `energy_spectra/energy_spectrum_<variable>[_<level>][_init<start>-<end>]...`
- LSD averaged (2D): `energy_spectra/energy_ratios_averaged_<range>.csv`
- LSD lead_time (2D): `energy_spectra/energy_ratios_lead_time_<range>.csv`
- LSD plot_datetime (2D): `energy_spectra/energy_ratios_plot_datetime_<range>.csv`
- LSD averaged (3D): `energy_spectra/energy_ratios_3d_averaged_<range>.csv`
- LSD per-level (3D): `energy_spectra/energy_ratios_3d_per_level_<range>.csv`
- LSD lead_time (3D): `energy_spectra/energy_ratios_3d_lead_time_<range>.csv`
- LSD plot_datetime (3D): `energy_spectra/energy_ratios_3d_plot_datetime_<range>.csv`
- LSD Banded: `energy_spectra/energy_ratios_bands_averaged_<range>.csv` (and 3D variants)

**Per Lead (Multi Lead):**

- Spectrograms: `energy_spectra/energy_spectra_per_lead_<variable>...`
- LSD per-lead (2D): `energy_spectra/energy_ratios_per_lead_by_lead_long_<range>.csv` (and wide format)
- LSD Banded per-lead: `energy_spectra/energy_ratios_bands_per_lead_per_lead_time_<range>.csv`
- LSD Line Plot: `energy_spectra/energy_ratios_line_per_lead_<variable>...`

### Distribution Analysis (Histograms & KDE / Wasserstein)

Histogram and KDE outputs encode:

- Prefix: `hist_` or `wd_kde_`
- Variable + optional level token
- Latitudinal aggregation token (`latbands` or `global`)
- Optional time-range tokens
- Ensemble token

By default, histograms and KDEs are computed globally (`*_global.npz`). To also generate them per latitude band, set `plotting.histograms_per_lat_band: true` or `plotting.wd_kde_per_lat_band: true` in your config.

Examples (pooled vs members):

```text
hist_2m_temperature_surface_global_enspooled.npz
hist_2m_temperature_surface_latbands_enspooled.npz
wd_kde_2m_temperature_surface_global_enspooled.npz
wd_kde_2m_temperature_surface_latbands_enspooled.npz
wd_kde_wasserstein_averaged_enspooled.csv
```

Time ranges (if present) appear just before the ensemble token: `..._init2023010200-2023010412_lead000h-036h_enspooled.npz`.

### Vertical Structure (3D variables only)

Outputs (standardized naming):

- Plot: `vertical_profiles/vertical_profiles_nmae_<variable>_multi_plot[_init...][_lead...]_ens*.png`
- Plot data (CSV): `vertical_profiles/vertical_profiles_nmae_<variable>_multi_plot[_init...][_lead...]_ens*.csv`
- Combined band data (NPZ): `vertical_profiles/vertical_profiles_nmae_<variable>_multi_combined[_init...][_lead...]_ens*.npz`
- Optional evolution/global bundles (NPZ/PNG): `..._evolve*`, `..._global_profile*`, `..._all_leads*`
- Summaries (CSV) may be produced by intercomparison rather than the module itself.

### Spatial Maps

Maps include the selected (or single) init/lead span and ensemble token. In members mode one PNG (and/or NPZ if `output_mode` includes it) per member is produced.

```text
map_10m_u_component_of_wind_init2023010200-2023010412_ens0.png   # member 0
map_temperature_500_init2023010200-2023010412_ensmean.png        # mean reduction
map_10m_u_component_of_wind_init2023010200-2023010412_ens3.npz   # NPZ export (output_mode=npz/both)
```

**3D variables and multi-lead NPZ files**: For 3D atmospheric variables evaluated with more than one lead time, the NPZ files are saved **per pressure level** (one file per level) with the full lead-time stack preserved:

```text
map_temperature_500_init2023010200-2023010412_lead000h-072h_ens0.npz  # level 500 hPa, shape (n_leads, lat, lon)
map_temperature_850_init2023010200-2023010412_lead000h-072h_ens0.npz  # level 850 hPa
```

For single-lead or purely single-init 3D runs, a combined NPZ is written instead (all levels in one file). The intercomparison tool automatically prefers per-level files and ignores the combined `_to_`-range equivalent when it finds the individual level files.

### Bivariate Histograms (`multivariate` module)

Bivariate histograms compare the joint distribution of two variables between prediction and ground truth using log-scale 2-D density contour plots. Outputs are written to a `multivariate/` sub-folder under `output_root`.

**Outputs per pair `[var_x, var_y]`:**

- Plot (PNG): `multivariate/bivariate_<var_x>_<var_y><_ens*>.png` — filled colour contours for the ground truth overlaid with greyscale contours for the prediction.
- Data (NPZ): `multivariate/bivariate_hist_<var_x>_<var_y><_ens*>.npz` — raw histogram counts and bin edges (`hist`, `hist_target`, `bins_x`, `bins_y`).

The ensemble token appended to the filename depends on the resolved mode:

| Mode     | Token            | Behaviour                             |
|----------|------------------|---------------------------------------|
| `mean`   | `ensmean`        | Ensemble is reduced to mean first.    |
| `pooled` | `enspooled`      | All members' values are pooled.       |
| `members`| `ens0`, `ens1`, … | One plot/NPZ pair per member.         |

**Example filenames:**

```text
multivariate/bivariate_10m_u_component_of_wind_10m_v_component_of_wind_ensmean.png
multivariate/bivariate_hist_10m_u_component_of_wind_10m_v_component_of_wind_ensmean.npz
multivariate/bivariate_u_component_of_wind_v_component_of_wind_ens0.png   # members mode
```

**Configuring pairs** — edit `metrics.multivariate.bivariate_pairs` in your YAML:

```yaml
metrics:
  multivariate:
    bivariate_pairs:
      - ["10m_u_component_of_wind", "10m_v_component_of_wind"]
      - ["u_component_of_wind", "v_component_of_wind"]
      - ["temperature", "specific_humidity"]   # requires both vars in dataset
    # bins: 100   # optional, histogram resolution (default 100)
```

Pairs whose variables are absent from either dataset are skipped with a warning. Only variables present in **both** the prediction and target datasets can be used.

### Probabilistic Verification (combined xarray + WeatherBenchX)

All probabilistic artifacts use the dedicated token `ensprob` (never `ensmean` / `enspooled`). This distinguishes probabilistic semantics (ensemble retained for PIT/CRPS computation) from deterministic or pooled reductions.

Per-variable artifacts (NPZ/CSV/PNG):

```text
pit_hist_2m_temperature_ensprob.npz                  # single-lead histogram
pit_hist_2m_temperature_grid_ensprob.png              # per-lead-time grid (multi-lead only)
pit_hist_2m_temperature_grid_ensprob.npz              # per-lead-time grid data (multi-lead only)
pit_hist_temperature_500_ensprob.npz                  # 3D variable with level token (single-lead)
pit_hist_temperature_500_grid_ensprob.png             # 3D variable per-lead grid
crps_map_2m_temperature_ensprob.png                   # optional (if plotting enabled)
crps_line_2m_temperature_by_lead_ensprob.csv
crps_line_2m_temperature_ensprob.png
crps_line_2m_temperature_data_ensprob.npz             # optional (if output_mode includes npz)
crps_line_temperature_500_by_lead_ensprob.csv         # 3D variable with level token
ssr_line_2m_temperature_by_lead_ensprob.csv
ssr_line_temperature_500_by_lead_ensprob.csv          # 3D variable with level token
temporal_probabilistic_metrics_2m_temperature_per_lead_time_ensprob.csv  # legacy-compatible alias
ssr_temporal_2m_temperature_ensprob.png               # optional (SSR temporal plot)
ssr_map_2m_temperature_ensprob.png                    # optional (SSR map plot)
ssr_regions_2m_temperature_ensprob.png                # optional (SSR regional plot)
spaghetti_2m_temperature_init2023010200-2023010200_lead000h-072h_ensprob.png  # spaghetti timeseries
spaghetti_2m_temperature_init2023010200-2023010200_lead000h-072h_ensprob.npz  # spaghetti data (for intercomparison)
spaghetti_temperature_500_init2023010200-2023010200_lead000h-072h_ensprob.png # 3D spaghetti per level
spaghetti_temperature_500_init2023010200-2023010200_lead000h-072h_ensprob.npz # 3D spaghetti data per level
```

Note: For multi-lead runs, only the per-lead-time grid is produced (no separate averaged/global PIT
histogram). For single-lead runs, the individual histogram is emitted. 3D variables emit one
histogram per pressure level.

WeatherBenchX per-variable spatial aggregations (NPZ format):

```text
crps_spatial_2m_temperature_ensprob.npz
crps_spatial_temperature_500_ensprob.npz
ssr_spatial_2m_temperature_ensprob.npz
ssr_spatial_temperature_500_ensprob.npz

```

Summary tables:

```text
ssr_summary_ensprob.csv
crps_summary_ensprob.csv
crps_summary_averaged_init2023010200-2023010412_lead000h-024h_ensprob.csv
```

### Details for probabilistic outputs

- CRPS and PIT are computed per variable using the ensemble along the `ensemble` dimension.
- WBX CRPS/SSR fields are computed once per variable batch and reused for summary and by-lead line outputs.
- `crps_line_*_by_lead*.csv` and `ssr_line_*_by_lead*.csv` are written for multi-lead runs independent of plot mode.
- Legacy CRPS aliases `temporal_probabilistic_metrics_*_per_lead_time*.csv` are also written to preserve older analysis scripts.
- Spatial NPZ artifacts (`*_spatial_*.npz`) are emitted when `plotting.output_mode` includes `npz` (`npz` or `both`).
- CRPS returned by the library functions is a DataArray (not a Dataset). In notebooks, use the DataArray directly and then reduce over time-like dims to make maps.
- PIT histograms are stored as NPZ (counts, edges) for reproducibility.
- For multi-lead runs, PIT histograms are produced only as a per-lead-time grid (no
  averaged/global histogram). Single-lead runs emit a single histogram per variable.
- For 3D variables, PIT histograms are computed and stored per pressure level.
- Full PIT/CRPS fields are not written to keep output size manageable.
- For 3D variables, level-resolved outputs are produced by default where supported.
- **Spaghetti time-series** (`spaghetti_*.png` / `spaghetti_*.npz`): spatially averaged line plots
  with one line per ensemble member (thin, vermilion) and the target/ground truth (thick, black).
  X-axis shows lead time in hours; y-axis shows the spatial mean of the variable. By default the
  first available `init_time` is used; override with `plotting.plot_datetime`. Requires ensemble
  size >= 2 and multiple lead times. For 3D variables, one plot per pressure level is generated.
  Controlled by `metrics.probabilistic.spaghetti` (default: `true`). PNG emitted when
  `output_mode` includes `"plot"` (`"plot"` or `"both"`); NPZ data artifacts emitted when
  `output_mode` includes `"npz"` (`"npz"` or `"both"`). NPZ keys: `lead_hours`, `member_values`
  (shape `n_members × n_leads`), `target_values`, `variable`, `units`, and optionally `level`.

All modules print concise progress like:

- [swissclim] Module: deterministic — variables=5
- [histograms] variable: 10m_u_component_of_wind
- [energy_spectra] saved output/verification_esfm/energy_spectra/u_component_of_wind_500hPa_spectrum.png

## Intercomparison Outputs

Intercomparison outputs are written under `output/intercomparison/<module>/` and are consumed by `notebooks/model_intercomparison.ipynb` via folder-level display helpers.

All intercomparison plots use **consistent model colours**: each label in `intercomparison.yaml → labels` is assigned a fixed colour from the `tab10` palette in the order it appears in the list. This colour is shared across every module (bar charts, line plots, histograms, KDE, spectra, PIT, CRPS).

- **maps**
	- `maps/map_<var>[_<level>]_compare.png`: Target + per-model panels. Multi-lead 2D variables produce a row per lead time labelled `(+Xh)` on the left; multi-lead 3D variables produce one gridded figure per pressure level.
	- `maps/det_mae_map_*_compare.png`: MAE spatial map comparison (panels per model).
	- `maps/det_mae_map_*_per_lead_compare.png`: MAE per-lead gridded comparison with `(+Xh)` row labels.
	- `maps/det_rmse_map_*_compare.png`: RMSE spatial map comparison.
	- `maps/det_rmse_map_*_per_lead_compare.png`: RMSE per-lead gridded comparison.
	- `maps/det_bias_map_*_compare.png`: Bias spatial map comparison.
	- `maps/det_bias_map_*_per_lead_compare.png`: Bias per-lead gridded comparison.
- **histograms**
	- `histograms/*_compare.png`
- **wd_kde**
	- `wd_kde/*_compare.png`
	- `wd_kde/*_ridgeline_compare.png`
	- `wd_kde/wd_kde_wasserstein_averaged_combined.csv`
- **energy_spectra**
	- `energy_spectra/*_compare.png`
	- `energy_spectra/*_ratio_compare.png`
	- `energy_spectra/*_spectrogram_delta_compare.png`
	- `energy_spectra/lsd_metrics_averaged_combined.csv`
	- `energy_spectra/lsd_metrics_banded_averaged_combined.csv`
	- `energy_spectra/lsd_metrics_per_level_combined.csv`
	- `energy_spectra/lsd_metrics_banded_per_level_combined.csv`
	- `energy_spectra/lsd_metrics_3d_averaged_combined.csv`
	- `energy_spectra/lsd_metrics_lead_time_combined.csv`
	- `energy_spectra/lsd_metrics_3d_lead_time_combined.csv`
	- `energy_spectra/lsd_metrics_banded_lead_time_combined.csv`
- **vertical_profiles**
	- `vertical_profiles/*_compare.png`
	- `vertical_profiles/*_summary.csv`
- **deterministic**
	- `deterministic/metrics_combined.csv`
	- `deterministic/metrics_standardized_combined.csv`
	- `deterministic/metrics_per_level_combined.csv`
	- `deterministic/metrics_standardized_per_level_combined.csv`
	- `deterministic/temporal_metrics_combined.csv`
	- `deterministic/temporal_*_compare.png`
- **ets**
	- `ets/ets_metrics_combined.csv`
	- `ets/ets_*_compare.png`
- **probabilistic**
	- `probabilistic/crps_summary_combined.csv`
	- `probabilistic/crps_summary_per_level_combined.csv`
	- `probabilistic/ssr_combined.csv`
	- `probabilistic/ssr_per_level_combined.csv`
	- `probabilistic/crps_ensemble_combined.csv`
	- `probabilistic/temporal_metrics_combined.csv`
	- `probabilistic/temporal_*_compare.png`
	- `probabilistic/pit_hist_*_compare.png`
	- `probabilistic/crps_map_*_compare.png`
	- `probabilistic/crps_spatial_*_per_lead_compare.png`
	- `probabilistic/spaghetti_*_compare.png`: side-by-side spaghetti timeseries (one panel per model, ensemble members as thin lines, shared target as thick black line)
