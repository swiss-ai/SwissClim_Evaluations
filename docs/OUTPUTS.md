# Output Overview

This document explains the naming conventions of the output files, but the user should not need to care about this in general use cases. Files are read automatically by the notebooks and this is the main interaction the user has with outputs.

The evaluation generates organized results for each enabled module.

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
```

### Extreme Threshold Statistics (ETS)

ETS filenames follow the same minimal pattern as deterministic metrics and include `ensmean` or `ens<i>` tokens depending on ensemble mode:

```text
ets_metrics_ensmean.csv
ets_metrics_averaged_init2023010200-2023010412_ensmean.csv
ets_metrics_per_level_ensmean.csv
ets_metrics_init_time_ens0.csv   # members mode per-member file
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
- Combined band data (NPZ): `vertical_profiles/vertical_profiles_nmae_<variable>_multi_combined[_init...][_lead...]_ens*.npz`
- Summaries (CSV) may be produced by intercomparison rather than the module itself.

### Spatial Maps

Maps include the selected (or single) init/lead span and ensemble token. In members mode one PNG (and/or NPZ if `output_mode` includes it) per member is produced.

```text
map_10m_u_component_of_wind_init2023010200-2023010412_ens0.png   # member 0
map_temperature_500_init2023010200-2023010412_ensmean.png        # mean reduction
map_10m_u_component_of_wind_init2023010200-2023010412_ens3.npz   # NPZ export (output_mode=npz/both)
```

### Probabilistic Verification (combined xarray + WeatherBenchX)

All probabilistic artifacts use the dedicated token `ensprob` (never `ensmean` / `enspooled`). This distinguishes probabilistic semantics (ensemble retained for PIT/CRPS computation) from deterministic or pooled reductions.

Per-variable artifacts (NPZ/CSV/PNG):

```text
pit_hist_2m_temperature_ensprob.npz
pit_evolution_2m_temperature_ensprob.png
pit_evolution_2m_temperature_data_ensprob.npz
pit_field_2m_temperature_ensprob.npz
crps_field_2m_temperature_ensprob.npz
crps_map_2m_temperature_ensprob.png        # optional map (if plotting enabled)
probabilistic_metrics_2m_temperature_per_lead_time_ensprob.csv
probabilistic_metrics_2m_temperature_per_lead_time_ensprob.png
```

WeatherBenchX per-variable temporal/spatial aggregations (NPZ format):

```text
crps_temporal_wbx_2m_temperature_ensprob.npz
crps_spatial_wbx_2m_temperature_ensprob.npz
ssr_temporal_wbx_2m_temperature_ensprob.npz
ssr_spatial_wbx_2m_temperature_ensprob.npz
wbx_temporal_SSR.2m_temperature_ensprob.png # SSR temporal plot
wbx_spatial_SSR.2m_temperature_ensprob.png  # SSR spatial plot
```

Summary tables:

```text
spread_skill_ratio_ensprob.csv
crps_summary_ensprob.csv
crps_summary_averaged_init2023010200-2023010412_lead000h-024h_ensprob.csv
crps_summary_per_level_ensprob.csv
```

### Details for probabilistic outputs

- CRPS and PIT are computed per variable using the ensemble along the `ensemble` dimension.
- CRPS returned by the library functions is a DataArray (not a Dataset). In notebooks, use the DataArray directly and then reduce over time-like dims to make maps.
- PIT histograms are stored as NPZ (counts, edges) for reproducibility; corresponding PIT fields are also written to NPZ.

All modules print concise progress like:

- [swissclim] Module: deterministic — variables=5
- [histograms] variable: 10m_u_component_of_wind
- [energy_spectra] saved output/verification_esfm/energy_spectra/u_component_of_wind_500hPa_spectrum.png

## Intercomparison Outputs

Intercomparison outputs are written under `output/intercomparison/<module>/` and are consumed by `notebooks/model_intercomparison.ipynb` via folder-level display helpers.

- **maps**
	- `maps/*_compare.png`
- **histograms**
	- `histograms/*_compare.png`
- **wd_kde**
	- `wd_kde/*_compare.png`
	- `wd_kde/wd_kde_wasserstein_averaged_combined.csv`
- **energy_spectra**
	- `energy_spectra/*_compare.png`
	- `energy_spectra/*_ratio_compare.png`
	- `energy_spectra/lsd_metrics_averaged_combined.csv`
	- `energy_spectra/lsd_metrics_banded_averaged_combined.csv`
	- `energy_spectra/lsd_metrics_per_level_combined.csv`
	- `energy_spectra/lsd_metrics_banded_per_level_combined.csv`
	- `energy_spectra/lsd_metrics_3d_averaged_combined.csv`
	- `energy_spectra/lsd_metrics_lead_time_combined.csv`
	- `energy_spectra/lsd_metrics_3d_lead_time_combined.csv`
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
	- `probabilistic/spread_skill_ratio_combined.csv`
	- `probabilistic/crps_ensemble_combined.csv`
	- `probabilistic/temporal_metrics_combined.csv`
	- `probabilistic/temporal_*_compare.png`
	- `probabilistic/pit_hist_*_compare.png`
	- `probabilistic/crps_map_*_compare.png`
