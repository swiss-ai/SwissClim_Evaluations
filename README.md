# SwissClim_Evaluations

A Python package for evaluating weather and climate model forecasts using probabilistic and deterministic metrics.

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

On CSCS systems (Clariden):

```bash
module load prgenv-gnu/24.11:v1
cd SwissClim_Evaluations
./tools/setup_env.sh
source .venv/bin/activate
```

Or with `uv`:

```bash
uv sync && source .venv/bin/activate
```

## Quick Start

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

```
SwissClim_Evaluations/
├── src/swissclim_evaluations/
│   ├── __init__.py
│   ├── data.py                    # Data loading functions
│   ├── helpers.py                 # Utility functions
│   ├── aggregations.py           # Spatial aggregation tools
│   └── metrics/
│       ├── probabilistic.py      # Core probabilistic metrics
│       └── wbx.py                # WeatherBench-X compatible metrics
├── notebooks/                     # Analysis notebooks
├── tools/setup_env.sh            # Environment setup script
└── README.md
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
