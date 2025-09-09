# SwissClim_Evaluations

A Python package for evaluating weather and climate model forecasts using probabilistic and deterministic metrics.

## TL;DR

- Recommended install (uv):
  - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Create env + deps: `uv sync && source .venv/bin/activate`
- Hello World (Python): see the short snippet below to compute CRPS on synthetic data in seconds.
- CLI runs full "chapters" from a YAML config (examples below).

## Overview

SwissClim_Evaluations provides tools to assess any weather or climate model output by computing verification metrics. The package works with any model data as long as it follows the expected Xarray format conventions. Built-in support is provided for ERA5, IFS ensemble forecasts, and ESMF model outputs.

## Features

- **Probabilistic Metrics**: CRPS, Probability Integral Transform (PIT), ensemble statistics
- **Universal Model Support**: Works with any weather/climate model output in proper Xarray format
- **Built-in Data Loaders**: ERA5, IFS ensemble, and ESMF model data
- **Spatial Aggregation**: Latitude-weighted averaging and histogram computations
- **WeatherBench-X Integration**: Compatible with weatherbenchX metrics framework
- **Efficient Processing**: Chunked processing for large datasets using Xarray and Dask

## Installation

Pick one of the following:

1. uv (recommended, fastest)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

1. pip (vanilla Python)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# WeatherBench-X may be needed; either from PyPI or directly from GitHub
python -m pip install git+https://github.com/google-research/weatherbenchX.git
python -m pip install -e .
```

1. CSCS (Clariden) with modules and uv

```bash
module load prgenv-gnu/24.11:v1
./tools/setup_env_uv.sh
source .venv/bin/activate
```

## Quick Start

### Hello World (Python in < 30s)

Compute CRPS and PIT on tiny synthetic data to see the library in action:

```python
import numpy as np
import xarray as xr
from swissclim_evaluations.metrics.probabilistic import crps_ensemble, probability_integral_transform

# Synthetic grid and time
time = xr.cftime_range("2021-01-01", periods=4)
lat = xr.DataArray(np.linspace(-60, 60, 8), dims=["latitude"]) 
lon = xr.DataArray(np.linspace(0, 357.5, 16), dims=["longitude"]) 
ens = xr.DataArray(np.arange(5), dims=["ensemble"]) 

# Observations and ensemble forecasts for a single variable
rng = np.random.default_rng(0)
obs = xr.Dataset({
  "2m_temperature": ("time", rng.normal(size=time.size))
}).assign_coords(time=time).expand_dims({"latitude": lat, "longitude": lon}).transpose("time","latitude","longitude")

fct = xr.Dataset({
  "2m_temperature": ("time", rng.normal(size=time.size))
}).assign_coords(time=time).expand_dims({"latitude": lat, "longitude": lon, "ensemble": ens}).transpose("time","ensemble","latitude","longitude")

# Compute scores
crps = crps_ensemble(obs["2m_temperature"], fct["2m_temperature"], ensemble_dim="ensemble")
pit = probability_integral_transform(obs["2m_temperature"], fct["2m_temperature"], ensemble_dim="ensemble")

print("CRPS mean:", float(crps.mean()))
print("PIT mean:", float(pit.mean()))
```

That’s it—no data downloads needed.

### Using Your Own Model Data

The package works with any model output that follows Xarray conventions:

```python
import xarray as xr
from swissclim_evaluations.metrics.probabilistic import crps_ensemble

# Load your model data (must have proper dimensions)
forecasts = xr.open_dataset("your_model_output.nc")  # e.g., dims: [time, ensemble, lat, lon]
observations = xr.open_dataset("observations.nc")    # e.g., dims: [time, lat, lon]

# Compute CRPS
crps = crps_ensemble(observations, forecasts, ensemble_dim="ensemble")
```

### Using Built-in Data Loaders

```python
from swissclim_evaluations.data import era5, ifs
from swissclim_evaluations.metrics.probabilistic import crps_ensemble

# Load data using built-in loaders
obs = era5(variables=["2m_temperature"])
forecasts = ifs(variables=["2m_temperature"])

# Compute metrics
crps = crps_ensemble(obs, forecasts, ensemble_dim="ensemble")
```

## Data Format Requirements

Your model data should be Xarray Datasets/DataArrays with standard dimension names:

- **Time dimensions**: `time`, `init_time`, `lead_time`
- **Spatial dimensions**: `latitude`/`lat`, `longitude`/`lon`
- **Ensemble dimension**: `ensemble`, `member`, or specify via `ensemble_dim` parameter

## Available Metrics

### Probabilistic Metrics ([`src/swissclim_evaluations/metrics/probabilistic.py`](src/swissclim_evaluations/metrics/probabilistic.py))

- **CRPS (Continuous Ranked Probability Score)**: [`crps_ensemble`](src/swissclim_evaluations/metrics/probabilistic.py), [`crps_e1`](src/swissclim_evaluations/metrics/probabilistic.py), [`crps_e2`](src/swissclim_evaluations/metrics/probabilistic.py)
- **PIT (Probability Integral Transform)**: [`probability_integral_transform`](src/swissclim_evaluations/metrics/probabilistic.py)
- **Ensemble Statistics**: [`ensemble_mean_se`](src/swissclim_evaluations/metrics/probabilistic.py), [`ensemble_std`](src/swissclim_evaluations/metrics/probabilistic.py)

## Project Structure

```text
SwissClim_Evaluations/
├── src/swissclim_evaluations/
│   ├── __init__.py
│   ├── data.py                    # Data loading functions
│   ├── helpers.py                 # Utility functions
│   ├── aggregations.py           # Spatial aggregation tools
│   └── metrics/
│       ├── probabilistic.py      # Core probabilistic + WBX-compatible metrics
│       └── deterministic.py      # Deterministic (formerly objective) metrics
├── notebooks/                     # Analysis notebooks
├── tools/setup_env.sh            # Environment setup script
└── README.md
```

## CLI runner

This repo also provides a YAML-driven CLI to run verification “chapters” end-to-end. Chapters can be enabled in your YAML config under `chapters` or selected explicitly via the CLI `--chapters` flag.

- plots: `maps`, `histograms`, `wd_kde`, `vertical_profiles`
- metrics: `energy_spectra`, `deterministic`, `ets`, `probabilistic`, `probabilistic_wbx`

Configuration examples can be found in `config/example_config.yaml`. Notable keys:

- `plotting.save_plot_data` (bool): Save data used to generate plots in addition to images.
- `metrics.deterministic.include` and `metrics.deterministic.standardized_include`: Control which deterministic metrics are computed and stored.
- `metrics.ets.thresholds`: Quantile thresholds (in %) for ETS computation.
- `metrics.ets.save_csv` (bool): Save ETS results to `output_root/ets/ets_metrics.csv`.

## Try it

Run chapters via the CLI using your YAML config:

```bash
swissclim-evaluations --config config/example_config.yaml --chapters deterministic probabilistic
```

If you omit `--chapters`, the CLI uses the toggles under `chapters:` in your YAML.

### Minimal YAML example

Create a small `config/minimal.yaml` pointing to your datasets and choose a couple of chapters:

```yaml
paths:
  nwp: "/path/to/era5.zarr"      # or NetCDF; must follow xarray conventions
  ml: "/path/to/model_output.zarr"
  output_root: "./output/verification"

selection:
  variables_2d: ["2m_temperature"]
  temporal_resolution_hours: 6    # optional downsampling

chapters:
  deterministic: true
  probabilistic: true

plotting:
  save_plot_data: false
```

Then run:

```bash
swissclim-evaluations --config config/minimal.yaml
```

## Examples

See the [`notebooks/`](notebooks/) directory for detailed examples:

- [`probabilistic_verification.ipynb`](notebooks/probabilistic_verification.ipynb): Probabilistic forecast verification
- [`probabilistic_verification_wbx.ipynb`](notebooks/probabilistic_verification_wbx.ipynb): Using WeatherBench-X metrics

## Contributing

This package is designed for evaluating climate foundational models. When adding new metrics or data sources, ensure compatibility with the Xarray/Dask ecosystem and follow the existing code patterns.

## Dependencies

- `xarray`: Multi-dimensional data handling
- `numpy`: Numerical computations
- `dask`: Parallel and chunked computing
- `weatherbenchX`: Weather forecast verification framework

## Notes

- The former “objective metrics” have been renamed to “deterministic” throughout the code and config (metrics.deterministic.*). The CLI still accepts `--chapters metrics` for backward compatibility.
- If building in a container and you see `/root/.cargo/bin/uv: not found`, either install uv in your Dockerfile or replace the uv step with pip:
  - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or use pip: `python -m pip install -e .`

### Open the notebooks

After activating the environment, launch Jupyter and open the examples:

```bash
python -m ipykernel install --user --name swissclim-evals --display-name "Python (swissclim-evals)"
python -m jupyter notebook notebooks/
```

Select the installed kernel when prompted.
