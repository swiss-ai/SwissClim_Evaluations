# SwissClim Evaluations

Fast, reproducible evaluation of weather/climate model outputs against ERA5 (or other targets). Compute deterministic and probabilistic scores, spectral metrics, and helpful diagnostic plots — all driven by a single YAML config.

## Quickstart

We recommend the container workflow for fastest, reproducible setup.

1.  **Clone the repository** (see [Installation Guide](docs/INSTALLATION.md) for path conventions).
2.  **Build the container** (Podman) or set up your environment (uenv/conda).
3.  **Configure your run** by copying `config/example_config.yaml`.
4.  **Run the evaluation**:

    ```bash
    python -m swissclim_evaluations.cli --config config/my_run.yaml
    ```

For detailed installation instructions, including Podman, uenv, and conda setups, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Configuration

The YAML config is the single source of truth.

See [config/example_config.yaml](config/example_config.yaml) for a fully commented example explaining every key and valid value.

## Dataset Requirements

This verification is based on xarray Datasets with the following structure.
Currently the dataloader expects a zarr archive of the following type:
- the dataset **must** have the same dimensions and coordinates as shown below
- the size of these dimensions and coordinates may vary

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

There are two special cases:
- The `ensemble` dimension must **always** be present in both datasets. If your data (e.g., ERA5) lacks this dimension, it will be automatically added with a single member (index 0).
- The `level` dimension **MUST NOT** be present in datasets containing only 2D variables. In mixed datasets, 2D variables must not have the `level` dimension.

### Chunking policy (xarray/dask)

- The level dimension is required for 3D variables, but must be absent for 2D variables.
- The ensemble dimension is required and should be present even for deterministic datasets (use ensemble dimension of size 1, e.g. coordinate value `[0]`).
- The repository enforces a default Dask chunking policy in code:
  - init_time: 1, lead_time: 1, level: 1, latitude: -1, longitude: -1, ensemble: -1 (-1 = no chunking)
  - If a dataset deviates, it will be rechunked automatically with a warning.

## Outputs

The evaluation generates organized results for each enabled module under `paths.output_root`.

Modules include:
*   **Maps**: Global and per-level maps.
*   **Histograms & KDE**: Distributions by latitude bands.
*   **Energy Spectra**: Zonal energy spectra + LSD table.
*   **Vertical Profiles**: NMAE vertical profiles.
*   **Deterministic Metrics**: MAE, RMSE, FSS, SSIM, etc.
*   **ETS**: Equitable Threat Score.
*   **Probabilistic**: CRPS, PIT, Spread-Skill Ratio.

For a detailed overview of outputs, file naming conventions, and ensemble handling, see [docs/OUTPUTS.md](docs/OUTPUTS.md).

## Intercomparison

You can combine plots and CSVs from multiple model runs using the intercomparison tool:

```bash
python -m swissclim_evaluations.intercompare --config config/intercomparison.yaml
```

See [docs/INTERCOMPARISON.md](docs/INTERCOMPARISON.md) for details.

## Notebooks

You can explore the outputs interactively using the provided notebooks:

- notebooks/deterministic_verification.ipynb (classic deterministic metrics, maps, histograms, spectra, profiles)
- notebooks/probabilistic_verification.ipynb (Probabilistic metrics: CRPS/PIT via Xarray, Spread–Skill Ratio and temporal/spatial metrics via WeatherBenchX)
- notebooks/model_intercomparison.ipynb (Compare multiple model runs, generating comparative plots)

Notebook tips

- Use the YAML config and `prepare_datasets` to ensure alignment, selection, and chunking are consistent with the CLI.

## Intercomparison of Saved Artifacts

This repo includes a lightweight CLI to combine plots and CSVs from multiple model runs that wrote artifacts (NPZ/CSV) to disk. It reuses the saved outputs under each model's output folder and generates combined visualizations for quick model-vs-model comparisons. A separate config is available for intercomparison.

Run the intercomparison:

```bash
python -m swissclim_evaluations.intercompare --config config/intercomparison.yaml
```

Outputs are written under `output/intercomparison/` mirroring the module folders. The tool is read-only on the source folders and will only generate figures/CSVs by loading the existing artifacts.

What gets combined:

- energy_spectra: overlays of DS baseline + model spectra per variable (and per level), plus `lsd_metrics_averaged_combined.csv`, `lsd_metrics_per_level_combined.csv`, and banded variants when available.
- histograms: per-latitude band distributions (DS line + model lines) using saved combined NPZs.
- wd_kde: standardized KDE overlays by latitude band (DS + models) using saved NPZs.
- maps: panel maps with DS in the first column and each model as subsequent columns.
- deterministic: merged CSVs (`metrics_combined.csv`, `metrics_standardized_combined.csv`, `metrics_per_level_combined.csv`, `metrics_standardized_per_level_combined.csv`) and simple bar charts for MAE/RMSE/FSS when data is present.
- ets: merged CSVs (`ets_metrics_combined.csv`, `ets_metrics_per_level_combined.csv`).
- ssim: merged CSVs (`ssim_combined.csv`) and a comparison bar plot of the average SSIM (`ssim_comparison.png`).
- probabilistic: merged CSVs (`crps_summary_combined.csv`, `crps_summary_per_level_combined.csv`, `spread_skill_ratio_combined.csv`), PIT histogram overlays, and CRPS map panels when NPZ map exports exist.
  - Additionally merges WBX spatial and temporal aggregates from `prob_metrics_{spatial,temporal}_*.nc` (or legacy names) into (`spatial_metrics_combined.csv`, `temporal_metrics_combined.csv`), with simple region-wise bar charts and time-bin line plots if the corresponding dimensions are present.
  - A single availability panel covers all probabilistic artifacts (PIT, CRPS maps, spatial/temporal WBX).
- vertical profiles (vprof): overlay plots per variable of latitude-band vertical NMAE across models — saved as `vertical_profiles/vertical_profiles_nmae_<variable>_multi_combined_compare.png` — plus per-variable summary tables (`vertical_profiles_nmae_<variable>_multi_combined_summary.csv`) listing mean metric by band, hemisphere, and model. Legacy `*_pl_nmae_combined*` files are still supported as input.

## Development

*   **Python**: 3.11
*   **Key libs**: xarray, numpy, scipy, pandas, matplotlib, cartopy, scores, weatherbenchX

For debugging, testing, and contribution guidelines, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
