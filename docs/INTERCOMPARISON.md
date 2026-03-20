# Intercomparison of Saved Artifacts

This repo includes a lightweight CLI to combine plots and CSVs from multiple model runs that wrote artifacts (NPZ/CSV) to disk. It reuses the saved outputs under each model's output folder and generates combined visualizations for quick model-vs-model comparisons. A separate config is available for intercomparison.

Run the intercomparison:

```bash
python -m swissclim_evaluations.intercompare --config config/intercomparison.yaml
```

Or if installed:

```bash
swissclim-evaluations-compare --config config/intercomparison.yaml
```

Outputs are written under `output/intercomparison/` mirroring the module folders. The tool is read-only on the source folders and will only generate figures/CSVs by loading the existing artifacts.

Module selection notes:

- `modules` accepts both short and long aliases:
    - `maps`
    - `hist` / `histograms`
    - `kde` / `wd_kde`
    - `spectra` / `energy_spectra` / `energy`
    - `vprof` / `vertical_profiles` / `vertical`
    - `metrics` / `deterministic` / `deterministic_metrics`
    - `ets`
    - `prob` / `probabilistic`
    - `ssim`
    - `multivariate`
- Unknown module names are ignored with a warning.
- If a requested module has no matching source artifacts across all model folders, it is skipped automatically with a warning.

This is especially useful for runs created with `plotting.output_mode: none`, where artifact-only modules (for example `maps`, `histograms`, `vertical_profiles`, `multivariate`) may not produce files for intercomparison.

What gets combined:

*   **Maps**
    *   **Plots**:
        *   `map_<var>[_level<lvl>]_compare.png`: Panel maps with DS in the first column and each model as subsequent columns.
        *   `det_mae_map_<var>[_<level>]_compare.png`: MAE spatial map comparison (one panel per model).
        *   `det_rmse_map_<var>[_<level>]_compare.png`: RMSE spatial map comparison.
        *   `det_bias_map_<var>[_<level>]_compare.png`: Bias spatial map comparison (diverging colour scale).

*   **Histograms**
    *   **Plots**:
        *   `hist_*global*_compare.png`: Global histogram comparison (log scale).
        *   `hist_*latbands*_compare.png`: Per-latitude band distributions (DS line + model lines).

*   **WD KDE**
    *   **Plots**:
        *   `wd_kde_*global*_compare.png`: Global normalized KDE comparison.
        *   `wd_kde_*latbands*_compare.png`: Standardized KDE overlays by latitude band (DS + models).
        *   `wd_kde_evolve_*_ridgeline_compare.png`: Multi-model ridgeline KDE evolution compare (target + models).
    *   **CSVs**:
        *   `wd_kde_wasserstein_averaged_combined.csv`: Averaged Wasserstein distance metrics.

*   **Energy Spectra**
    *   **Plots**:
        *   `*_compare.png`: Overlays of DS baseline + model spectra per variable (and per level).
        *   `*_compare_ratio.png`: Ratio of Model/Target energy density vs wavenumber.
        *   `energy_spectrum_<var>_lead<hhh>h_compare.png`: Per-lead time energy spectra comparison (when `individual_plots: true`).
        *   `energy_spectra_per_lead_*_bundle*_spectrogram_delta_compare.png`: Multi-model Δlog10(model-target) spectrogram compare
        *   `lsd_banded_lead_time_<var>_compare.png`: LSD vs lead time by spectral band (requires `individual_plots: false` multi-lead run).
    *   **CSVs**:
        *   `lsd_metrics_averaged_combined.csv`: Global averaged Log Spectral Distance (LSD).
        *   `lsd_metrics_lead_time_combined.csv`: LSD metrics per lead time.
        *   `lsd_metrics_plot_datetime_combined.csv`: LSD metrics per plot datetime.
        *   `lsd_metrics_3d_averaged_combined.csv`: 3D averaged LSD metrics.
        *   `lsd_metrics_3d_lead_time_combined.csv`: 3D LSD metrics per lead time.
        *   `lsd_metrics_3d_plot_datetime_combined.csv`: 3D LSD metrics per plot datetime.
        *   `lsd_metrics_banded_averaged_combined.csv`: Banded averaged LSD metrics.
        *   `lsd_metrics_per_level_combined.csv`: Per-level LSD metrics.
        *   `lsd_metrics_banded_per_level_combined.csv`: Banded per-level LSD metrics.
        *   `lsd_metrics_banded_lead_time_combined.csv`: Banded LSD metrics per lead time.

*   **Vertical Profiles**
    *   **Plots**:
        *   `vertical_profiles_nmae_<var>_compare.png`: Overlay plots per variable of latitude-band vertical NMAE across models.
    *   **CSVs**:
        *   `vertical_profiles_nmae_<var>_summary.csv`: Per-variable summary tables listing mean metric by band, hemisphere, and model.
    *   *Note: Legacy `*_pl_nmae_combined*` files are still supported.*

*   **Deterministic Metrics**
    *   **CSVs**:
        *   `metrics_combined.csv`: Combined deterministic metrics.
        *   `metrics_per_level_combined.csv`: Combined per-level deterministic metrics.
        *   `metrics_standardized_combined.csv`: Combined standardized metrics.
        *   `metrics_standardized_per_level_combined.csv`: Combined standardized per-level metrics.
        *   `temporal_metrics_combined.csv`: Combined temporal metrics (metrics vs lead time).
    *   **Plots**:
        *   `<metric>_compare.png`: Bar charts for each metric (e.g., RMSE, MAE) by variable and model.
        *   `temporal_<metric>_<variable>_compare.png`: Line plots of metric vs lead time.

*   **ETS**
    *   **CSVs**:
        *   `ets_metrics_combined.csv`: Combined ETS metrics.
    *   **Plots**:
        *   `ets_<var>_ETS_<thresh>pct_compare.png`: ETS vs Lead Time plots.


*   **Probabilistic**
    *   **CSVs**:
        *   `crps_summary_combined.csv`: Combined CRPS summary.
        *   `temporal_metrics_combined.csv`: Combined temporal probabilistic metrics from `crps_line_*_by_lead*.csv` and `ssr_line_*_by_lead*.csv`.
        *   `ssr_combined.csv`: Combined Spread-Skill Ratio.
    *   **Plots**:
        *   `temporal_<metric>_<variable>[_level<lvl>]_compare.png`: Line plots of probabilistic metrics vs lead time.
        *   `crps_spatial_<var>_mean_compare.png`: CRPS spatial map averaged over lead times (one panel per model).
        *   `crps_spatial_<var>_per_lead_compare.png`: CRPS spatial map grid (rows = lead times with `Target (+Xh)` labels, cols = models).
        *   `pit_hist_<var>[_<level>]_compare.png`: PIT histogram comparison (per level for 3D variables).
        *   `spaghetti_<var>[_<level>]_compare.png`: Side-by-side spaghetti timeseries comparison (one panel per model). Each panel shows ensemble members as thin coloured lines (model's assigned colour) and the shared ground-truth target as a thick black line. Requires `output_mode` to include `npz` in the single-model run.

*   **Multivariate**
    *   **Plots**:
        *   `bivariate_<var_x>_<var_y>[_level<lvl>]_compare.png`: Side-by-side 2D histogram density compare for each variable pair (target vs each model panel).
