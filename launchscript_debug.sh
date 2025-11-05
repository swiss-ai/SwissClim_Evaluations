#!/bin/bash
#SBATCH --job-name=swissclim-eval-debug   # job name
#SBATCH --output=logs/swissclim_debug%j.out
#SBATCH --error=logs/swissclim_debug%j.err
#SBATCH --time=00:30:00                   # shorter time for debug
#SBATCH --account=a122
#SBATCH --partition=debug

# Debug single-lead evaluation using the container defined by your EDF
# 1) Ensure you have ~/.edf/swissclim-eval.toml (see README Quickstart)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"
# 2) Use the small, single-lead config shipped in this repo
CONFIG_PATH_IN_REPO="config/debug_single_lead.yaml"

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

srun \
  --container-writable \
  --environment="${EDF_CONFIG}" \
  python -u -m swissclim_evaluations.cli --config "${CONFIG_FILE}"
