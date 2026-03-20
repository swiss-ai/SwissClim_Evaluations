#!/bin/bash
#SBATCH --job-name=swissclim-eval-single
#SBATCH --output=logs/swissclim_single_%j.out
#SBATCH --error=logs/swissclim_single_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================
# USER INPUT (edit this section only)
# =============================================================
# Evaluation config to run (absolute path or relative to submit/project directory)
CONFIG_FILE="config/firat/train_config_ESFMs_umv3_1aa8et_esfmi_meanmaeafcrps_sgd_atm_adalndecsb0scposemb_ro.yaml"

# Enroot/EDF TOML (or EDF name if your site supports it)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"

# Dask spill directory
DASK_TEMPORARY_DIRECTORY="/iopsstor/scratch/cscs/$USER/dask-tmp"

# PYTHONPATH behavior for project src directory:
# - prepend  : PROJECT_ROOT/src:$PYTHONPATH
# - overwrite: PROJECT_ROOT/src only
# - keep     : do not modify PYTHONPATH (default)
PYTHONPATH_MODE="prepend"
# =============================================================

# Resolve paths relative to the job submission directory when running under Slurm.
# Using BASH_SOURCE alone is not reliable because Slurm can execute a spool copy.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# -------------------------------------------------------------
# CONFIG TO RUN - EDIT THIS TO POINT TO YOUR DESIRED CONFIG FILE
CONFIG_FILE="/capstor/store/cscs/swissai/a122/sadamov/SwissClim_Evaluations/config/example_config.yaml"

# -------------------------------------------------------------
# EDIT THESE TWO LINES FOR YOUR SETUP
# 1) Path to your Enroot/EDF TOML file (or EDF name if your site supports it)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"
# -------------------------------------------------------------
export PYTHONPATH="${SUBMIT_DIR}/src:${PYTHONPATH}"

# -------------------------------------------------------------
# DASK SCRATCH CONFIGURATION
# Set the directory for Dask spillover to avoid filling /tmp
# -------------------------------------------------------------
export DASK_TEMPORARY_DIRECTORY="/iopsstor/scratch/cscs/$USER/dask-tmp"
mkdir -p "$DASK_TEMPORARY_DIRECTORY"


# Disable rich/ANSI output so SLURM log files remain clean
export SWISSCLIM_COLOR=never
export PYTHONUNBUFFERED=1


# Create logs directory if it doesn't exist
mkdir -p logs


# Generate robust job name from config (parent_dir + filename)
filename=$(basename "${CONFIG_FILE}" .yaml)
parent_dir=$(basename "$(dirname "${CONFIG_FILE}")")
job_name="${parent_dir}_${filename}"

echo "Starting evaluation for config: $CONFIG_FILE"

# Run the evaluation
# We use srun to launch the python process within the container environment defined by EDF_CONFIG
srun --ntasks=1 --container-writable --environment="${EDF_CONFIG}" \
    python -u -m swissclim_evaluations.cli --config "$CONFIG_FILE"
EXIT_CODE=$?

echo "Evaluation finished."

# --- Copy Logs ---
# Extract output_root using python
OUTDIR=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg.get('output_root') or cfg.get('paths', {}).get('output_root', ''))")

if [ -n "$OUTDIR" ] && [ -d "$OUTDIR" ]; then
    echo "Copying logs to $OUTDIR"
    cp "logs/swissclim_single_${SLURM_JOB_ID}.out" "$OUTDIR/${job_name}.out" 2>/dev/null || echo "Could not copy .out log"
    cp "logs/swissclim_single_${SLURM_JOB_ID}.err" "$OUTDIR/${job_name}.err" 2>/dev/null || echo "Could not copy .err log"
    cp "logs/dask_distributed_${SLURM_JOB_ID}_0.log" "$OUTDIR/${job_name}_dask.log" 2>/dev/null || \
    echo "Could not copy dask log"
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo "Evaluation failed with exit code $EXIT_CODE. Aborting notebook rendering."
    exit $EXIT_CODE
fi

# --- Notebook Rendering ---
echo "Rendering notebooks..."

# Create a temporary bash script to handle notebook rendering
cat <<'EOF' > render_single_notebook.sh
#!/bin/bash

# Install missing dependencies for notebook rendering

CONFIG_PATH="$1"

# Extract output_root using python
OUTDIR=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_PATH')); print(cfg.get('output_root') or cfg.get('paths', {}).get('output_root', ''))")

if [ -z "$OUTDIR" ]; then
    echo "No output_root found in $CONFIG_PATH"
    exit 1
fi

if [ ! -d "$OUTDIR" ]; then
    echo "Output directory does not exist: $OUTDIR"
    exit 1
fi

echo "Output directory: $OUTDIR"

# List of notebooks to render
NOTEBOOKS=("deterministic_verification.ipynb" "probabilistic_verification.ipynb")

for nb_name in "${NOTEBOOKS[@]}"; do
    nb="notebooks/${nb_name}"
    if [ ! -f "$nb" ]; then
        echo "Notebook not found: $nb"
        continue
    fi

    out_nb_path="${OUTDIR}/${nb_name}"
    echo "Rendering $nb_name..."

    # Execute notebook with papermill
    # We pass the absolute path of the config
    if ! python -m papermill "$nb" "$out_nb_path" -p config_path_str "$CONFIG_PATH"; then
        echo "ERROR: Failed to render $nb_name"
        continue
    fi

    echo "Converting to HTML..."
    if ! python -m jupyter nbconvert --to html "$out_nb_path"; then
        echo "ERROR: Failed to convert $nb_name to HTML"
    fi
done
EOF

# Run the rendering script
srun --ntasks=1 --container-writable --environment="${EDF_CONFIG}" \
    bash render_single_notebook.sh "$CONFIG_FILE"

# Cleanup
rm render_single_notebook.sh

echo "All tasks completed."
