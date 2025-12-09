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
- **vertical_profiles**: mean, pooled, members
- **histograms**: mean, pooled, members
- **wd_kde**: mean, pooled, members
- **energy_spectra**: mean, pooled, members
- **deterministic**: mean, pooled, members
- **ets**: mean, pooled, members
- **probabilistic**: prob only

The ensemble dimension is always present (size 1 for deterministic datasets). For such datasets, the ensemble mean is identical to the single member. Output filenames will reflect the configured mode (e.g., `_ensmean` for mean, `_ens0` for members). Legacy `_ensnone` tokens are also accepted by the intercomparison tool.

Notes: Members mode may include mean aggregates in some summaries (e.g., energy spectra LSD tables).

### Deterministic Metrics

Filenames encode only information that is actually present:

- Metric family (e.g. `deterministic_metrics`)
- Optional qualifier (`averaged`, `init_time`, `standardized`, combinations thereof). Note: `averaged` implies scalar mean over all dimensions (including levels for 3D variables).
- Optional time range tokens if an init and/or lead range exists: `initYYYYMMDDHH-YYYYMMDDHH` and `leadXXXh-YYYh`
- Ensemble token (always; see "Ensemble Tokens" below)

Examples (mean reduction vs members):

```text
deterministic_metrics_ensmean.csv
deterministic_metrics_averaged_init2023010200-2023010412_lead000h-036h_ensmean.csv
deterministic_metrics_standardized_ensmean.csv
deterministic_metrics_per_level_ensmean.csv
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

### Multivariate Metrics

Multivariate metrics (SSIM) filenames follow the standard pattern:

```text
multivariate_ssim_ensmean.csv
multivariate_ssim_ens0.csv
```

Additionally, bivariate density plots (histograms) are generated for specified variable pairs (e.g., u10m vs v10m). These plots visualize the joint distribution of two variables, comparing the prediction (grey contours) against the reference (filled plasma contours).

**Recommended Bivariate Pairs:**

To evaluate physical consistency, we recommend plotting pairs that capture fundamental relationships:

1.  **Surface Dynamics (Wind Vector):** `10m_u_component_of_wind` vs `10m_v_component_of_wind`.
    *   *Suggested Level:* Surface (10m).
    *   *Why:* Visualizes the surface circulation and wind variability. Biases in the spread indicate errors in surface friction or storm intensity.
2.  **Upper-level Dynamics (Wind Vector):** `u_component_of_wind` vs `v_component_of_wind`.
    *   *Suggested Level:* 250 hPa (Jet Stream).
    *   *Why:* Captures the jet stream structure and synoptic variability. Biases in the spread indicate errors in the storm track or mean flow.
3.  **Thermodynamics (Clausius-Clapeyron):** `temperature` vs `specific_humidity`.
    *   *Suggested Level:* 850 hPa (Lower Troposphere).
    *   *Why:* Warmer air holds more moisture. The distribution should show a sharp cutoff at the saturation curve, which follows the Clausius-Clapeyron relation. Points beyond this curve indicate unphysical supersaturation.
4.  **Hydrostatics:** `geopotential` vs `temperature`.
    *   *Suggested Level:* 500 hPa (Mid-troposphere).
    *   *Why:* Relates to hydrostatic balance. Low geopotential heights (troughs) are typically associated with cold air (cold-core cyclones).
5.  **Precipitation Physics:** `total_precipitation` vs `total_column_water_vapor`.
    *   *Suggested Level:* Surface / Column-integrated.
    *   *Why:* Checks precipitation efficiency. Deep convection requires a moist column. Look for a "hockey stick" relationship where precipitation increases rapidly above a critical $TCWV$ threshold (Bretherton et al. 2004).
6.  **Vertical Motion:** `vertical_velocity` vs `temperature`.
    *   *Suggested Level:* 500 hPa (Mid-troposphere).
    *   *Why:* Evaluates convective processes. Upward motion (negative $\omega$) is often driven by buoyancy (warm anomalies) in convective systems. Look for asymmetries between strong narrow updrafts and broad downdrafts.

Outputs:
- `bivariate_hist_<var1>_<var2>.npz`: Saved histogram counts and bin edges.
- `bivariate_<var1>_<var2>.png`: The generated density plot comparing the current model (Prediction) against the reference.

### Energy Spectra Analysis

Per-variable (and per-level) energy spectra are computed retaining time structure; the Log Spectral Distance (LSD)
is exported per init_time/lead_time and summarized. Outputs:

- Figures / NPZ (subset init_time for plotting) : `energy_spectra/lsd_<variable>[_<level>]_spectrum[_init...][_lead...]_ens*.{png|npz}`
- LSD per-time (2D): `energy_spectra/lsd_2d_metrics_per_init_time_<range>.csv` or `per_lead_time` depending on dims
- LSD averaged (2D mean): `energy_spectra/lsd_2d_metrics_averaged_<range>.csv`
- LSD init_time (2D): `energy_spectra/lsd_2d_metrics_init_time_<range>.csv` (mean over other time dims, retaining init_time)
- LSD per-time (3D): `energy_spectra/lsd_3d_metrics_per_init_time_<range>.csv` (or per_lead_time)
- LSD averaged (3D): `energy_spectra/lsd_3d_metrics_averaged_<range>.csv` (scalar mean over levels and time)
- LSD per-level (3D): `energy_spectra/lsd_3d_metrics_per_level_<range>.csv` (only if `report_per_level=true`)
- LSD init_time (3D): `energy_spectra/lsd_3d_metrics_init_time_<range>.csv`
- LSD (banded by wavelength) — new: `energy_spectra/lsd_bands_2d_metrics_*` and `lsd_bands_3d_metrics_*` variants for detailed, averaged (scalar), per-level, and init_time summaries.

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

### Vertical Structure (3D variables only)

Outputs (standardized naming):

- Plot: `vertical_profiles/vprof_nmae_<variable>_multi_plot[_init...][_lead...]_ens*.png`
- Combined band data (NPZ): `vertical_profiles/vprof_nmae_<variable>_multi_combined[_init...][_lead...]_ens*.npz`
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

Per-variable artifacts:

```text
pit_hist_2m_temperature_ensprob.npz
pit_field_2m_temperature_ensprob.nc
crps_field_2m_temperature_ensprob.nc
crps_map_2m_temperature_ensprob.png
```

WBX summary tables / fields:

```text
spread_skill_ratio_ensprob.csv
prob_metrics_temporal_ensprob.nc
prob_metrics_spatial_ensprob.nc
prob_metrics_map_ensprob.nc
```

Aggregated CRPS summaries (xarray based):

```text
crps_summary_ensprob.csv
crps_summary_averaged_init2023010200-2023010412_lead000h-024h_ensprob.csv
crps_summary_per_level_ensprob.csv
```

### Details for probabilistic outputs

- CRPS and PIT are computed per variable using the ensemble along the `ensemble` dimension.
- CRPS returned by the library functions is a DataArray (not a Dataset). In notebooks, use the DataArray directly and then reduce over time-like dims to make maps.
- PIT histograms are stored as NPZ (counts, edges) for reproducibility; corresponding PIT fields are also written to NetCDF.

All modules print concise progress like:

- [swissclim] Module: deterministic — variables=5
- [histograms] variable: 10m_u_component_of_wind
- [energy_spectra] saved output/verification_esfm/energy_spectra/u_component_of_wind_500hPa_spectrum.png
