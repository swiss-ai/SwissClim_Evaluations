#!/bin/bash
#SBATCH --job-name=swissclim-eval-ml-debug
#SBATCH --output=logs/swissclim_ml_debug%j.out
#SBATCH --error=logs/swissclim_ml_debug%j.err
#SBATCH --time=00:20:00
#SBATCH --account=a122
#SBATCH --partition=debug

# Multi-lead evaluation using the container defined by your EDF
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"
CONFIG_PATH_IN_REPO="config/debug_multi_lead.yaml"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "${CONFIG_PATH_IN_REPO}" = /* ]]; then
  CONFIG_FILE="${CONFIG_PATH_IN_REPO}"
else
  CONFIG_FILE="${SUBMIT_DIR}/${CONFIG_PATH_IN_REPO}"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
  exit 2
fi

export SWISSCLIM_COLOR=never
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SUBMIT_DIR}/src:${PYTHONPATH}"

srun \
  --container-writable \
  --environment="${EDF_CONFIG}" \
  python -u -m swissclim_evaluations.cli --config "${CONFIG_FILE}"
