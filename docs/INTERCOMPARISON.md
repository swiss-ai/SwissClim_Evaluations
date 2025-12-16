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

What gets combined:

*   **Maps**
    *   **Plots**:
        *   `map_<var>[_level<lvl>]_compare.png`: Panel maps with DS in the first column and each model as subsequent columns.

*   **Histograms**
    *   **Plots**:
        *   `hist_*global*_compare.png`: Global histogram comparison (log scale).
        *   `hist_*latbands*_compare.png`: Per-latitude band distributions (DS line + model lines).

*   **WD KDE**
    *   **Plots**:
        *   `wd_kde_*global*_compare.png`: Global normalized KDE comparison.
        *   `wd_kde_*latbands*_compare.png`: Standardized KDE overlays by latitude band (DS + models).
    *   **CSVs**:
        *   `wd_kde_wasserstein_averaged_combined.csv`: Averaged Wasserstein distance metrics.

*   **Energy Spectra**
    *   **Plots**:
        *   `*_compare.png`: Overlays of DS baseline + model spectra per variable (and per level).
        *   `*_compare_ratio.png`: Ratio of Model/Target energy density vs wavenumber.
        *   `energy_spectrum_<var>_lead<hhh>h_compare.png`: Per-lead time energy spectra comparison.
    *   **CSVs**:
        *   `lsd_metrics_averaged_combined.csv`: Global averaged Log Spectral Distance (LSD).
        *   `lsd_metrics_banded_averaged_combined.csv`: Banded averaged LSD metrics.
        *   `lsd_metrics_per_level_combined.csv`: Per-level LSD metrics.
        *   `lsd_metrics_banded_per_level_combined.csv`: Banded per-level LSD metrics.

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

*   **SSIM**
    *   **CSVs**:
        *   `ssim_combined.csv`: Combined SSIM metrics.
    *   **Plots**:
        *   `ssim_comparison.png`: Comparison bar plot of the average SSIM.

*   **Probabilistic**
    *   **CSVs**:
        *   `crps_summary_combined.csv`: Combined CRPS summary.
        *   `crps_summary_per_level_combined.csv`: Combined per-level CRPS summary.
        *   `temporal_metrics_combined.csv`: Combined temporal probabilistic metrics.
        *   `spread_skill_ratio_combined.csv`: Combined Spread-Skill Ratio.
        *   `crps_ensemble_combined.csv`: Combined CRPS ensemble metrics.
    *   **Plots**:
        *   `temporal_<metric>_<variable>_compare.png`: Line plots of probabilistic metrics vs lead time.
        *   `crps_map_<var>_compare.png`: CRPS map panels.
        *   `pit_hist_<var>_compare.png`: PIT histogram comparison.
