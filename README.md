# SwissClim Evaluations

Fast, reproducible evaluation of weather/climate model outputs against ERA5 (or other references). Compute deterministic and probabilistic scores, spectral metrics, and helpful diagnostic plots — all driven by a single YAML config.

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
- The `ensemble` dimension must **always** be present in both the dataset. If your data (e.g., ERA5 ground truth) lacks this dimension, it will be automatically added with a single member (index 0).
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
*   **Deterministic Metrics**: MAE, RMSE, FSS, etc.
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

Explore outputs interactively:
*   `notebooks/deterministic_verification.ipynb`
*   `notebooks/probabilistic_verification.ipynb`
*   `notebooks/probabilistic_verification_wbx.ipynb`

## Development

*   **Python**: 3.11
*   **Key libs**: xarray, numpy, scipy, pandas, matplotlib, cartopy, scores, weatherbenchX

For debugging, testing, and contribution guidelines, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
