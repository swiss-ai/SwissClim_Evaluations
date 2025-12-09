# Intercomparison of Saved Artifacts

This repo includes a lightweight CLI to combine plots and CSVs from multiple model runs that wrote artifacts (NPZ/CSV) to disk. It reuses the saved outputs under each model's output folder and generates combined visualizations for quick model-vs-model comparisons. A separate config is available for intercomparison.

Run the intercomparison:

```bash
python -m swissclim_evaluations.intercompare --config config/intercomparison.yaml
```

Outputs are written under `output/intercomparison/` mirroring the module folders. The tool is read-only on the source folders and will only generate figures/CSVs by loading the existing artifacts.

What gets combined:

*   **Energy Spectra**
    *   Overlays of DS baseline + model spectra per variable (and per level).
    *   Merged CSVs: `lsd_metrics_averaged_combined.csv`, `lsd_metrics_per_level_combined.csv`, and banded variants.

*   **Histograms**
    *   Per-latitude band distributions (DS line + model lines) using saved combined NPZs.

*   **WD KDE**
    *   Standardized KDE overlays by latitude band (DS + models) using saved NPZs.

*   **Maps**
    *   Panel maps with DS in the first column and each model as subsequent columns.

*   **Deterministic Metrics**
    *   Merged CSVs: `metrics_combined.csv`, `metrics_standardized_combined.csv`, `metrics_per_level_combined.csv`, `metrics_standardized_per_level_combined.csv`.
    *   Simple bar charts for MAE/RMSE/FSS when data is present.

*   **ETS**
    *   Merged CSVs: `ets_metrics_combined.csv`, `ets_metrics_per_level_combined.csv`.

*   **Multivariate**
    *   Merged CSVs: `multivariate_ssim_combined.csv`.

*   **Probabilistic**
    *   Merged CSVs: `crps_summary_combined.csv`, `crps_summary_per_level_combined.csv`, `spread_skill_ratio_combined.csv`, `crps_ensemble_combined.csv`.
    *   PIT histogram overlays and CRPS map panels (when NPZ map exports exist).
    *   **WBX Integration**: Merges spatial/temporal aggregates into `spatial_metrics_combined.csv` and `temporal_metrics_combined.csv`.
    *   **Plots**: Simple region-wise bar charts, time-bin line plots, and a single availability panel covering all probabilistic artifacts.

*   **Vertical Profiles (vprof)**
    *   Overlay plots per variable of latitude-band vertical NMAE across models (`vprof_nmae_<variable>_multi_combined_compare.png`).
    *   Per-variable summary tables (`vprof_nmae_<variable>_multi_combined_summary.csv`) listing mean metric by band, hemisphere, and model.
    *   *Note: Legacy `*_pl_nmae_combined*` files are still supported.*
