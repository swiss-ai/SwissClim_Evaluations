#!/bin/bash
#SBATCH --job-name=swissclim-eval
#SBATCH --output=logs/swissclim%j.out
#SBATCH --error=logs/swissclim%j.err
#SBATCH --time=01:30:00
#SBATCH --account=a122
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# -------------------------------------------------------------
# EDIT THESE TWO LINES FOR YOUR SETUP
# 1) Path to your Enroot/EDF TOML file (or EDF name if your site supports it)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"
# 2) Config path relative to this repo OR an absolute path
#    Examples: "config/my_run.yaml"  OR  "/abs/path/to/config.yaml"
CONFIG_PATH_IN_REPO="config/my_run.yaml"
# -------------------------------------------------------------

# Resolve config relative to the job submission directory (SLURM_SUBMIT_DIR)
# This avoids using the Slurm spool path where the script is copied to.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "${CONFIG_PATH_IN_REPO}" = /* ]]; then
  CONFIG_FILE="${CONFIG_PATH_IN_REPO}"
else
  CONFIG_FILE="${SUBMIT_DIR}/${CONFIG_PATH_IN_REPO}"
fi

# Early check with a clear error if config is missing
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
  echo "Hint: set CONFIG_PATH_IN_REPO to a path relative to your submission directory (${SUBMIT_DIR}), or use an absolute path." >&2
  exit 2
fi

# Disable rich/ANSI output so SLURM log files remain clean
export SWISSCLIM_COLOR=never

# Ensure Python can import the mounted source directly (no rebuild needed)
# Uncomment if you want to quickly test local changes without rebuilding the container
# export PYTHONPATH="${SUBMIT_DIR}/src:${PYTHONPATH}"

# Run with SLURM CPU binding inside the container defined by EDF_CONFIG
export PYTHONUNBUFFERED=1
srun \
  --cpu-bind=cores \
  --container-writable \
  --environment="${EDF_CONFIG}" \
  python -u -m swissclim_evaluations.cli --config "${CONFIG_FILE}"
