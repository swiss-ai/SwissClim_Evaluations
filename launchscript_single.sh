#!/bin/bash
#SBATCH --job-name=swissclim-eval-single
#SBATCH --output=logs/swissclim_single_%j.out
#SBATCH --error=logs/swissclim_single_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Resolve paths relative to the job submission directory when running under Slurm.
# Using BASH_SOURCE alone is not reliable because Slurm can execute a spool copy.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SUBMIT_DIR"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
fi
LOG_DIR="${PROJECT_ROOT}/logs"

cd "$PROJECT_ROOT" || {
    echo "ERROR: Failed to change directory to project root: $PROJECT_ROOT"
    exit 1
}

# -------------------------------------------------------------
# CONFIG TO RUN - EDIT THIS TO POINT TO YOUR DESIRED CONFIG FILE
CONFIG_FILE="config/example_config.yaml"

# Resolve CONFIG_FILE robustly (absolute or relative to submit directory)
resolve_config_path() {
    local cfg="$1"
    if [ -z "$cfg" ]; then
        return 1
    fi
    if [ -f "$cfg" ]; then
        python - <<'PY' "$cfg"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
        return 0
    fi
    if [ -f "${SUBMIT_DIR}/${cfg}" ]; then
        python - <<'PY' "${SUBMIT_DIR}/${cfg}"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
        return 0
    fi
    if [ -f "${PROJECT_ROOT}/${cfg}" ]; then
        python - <<'PY' "${PROJECT_ROOT}/${cfg}"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
        return 0
    fi
    return 1
}

if ! CONFIG_FILE_RESOLVED="$(resolve_config_path "$CONFIG_FILE")"; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Checked: '$CONFIG_FILE', '${SUBMIT_DIR}/${CONFIG_FILE}', '${PROJECT_ROOT}/${CONFIG_FILE}'"
    exit 1
fi
CONFIG_FILE="$CONFIG_FILE_RESOLVED"

# -------------------------------------------------------------
# Path to your Enroot/EDF TOML file (or EDF name if your site supports it)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"
# -------------------------------------------------------------
# Override PYTHONPATH to include src directory (latest code changes)
# export PYTHONPATH="${SUBMIT_DIR}/src:${PYTHONPATH}"

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
mkdir -p "$LOG_DIR"


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
    cp "${LOG_DIR}/swissclim_single_${SLURM_JOB_ID}.out" "$OUTDIR/${job_name}.out" 2>/dev/null || echo "Could not copy .out log"
    cp "${LOG_DIR}/swissclim_single_${SLURM_JOB_ID}.err" "$OUTDIR/${job_name}.err" 2>/dev/null || echo "Could not copy .err log"
    cp "${LOG_DIR}/dask_distributed_${SLURM_JOB_ID}_0.log" "$OUTDIR/${job_name}_dask.log" 2>/dev/null || \
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
PROJECT_ROOT="$2"

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

# Select notebooks based on available outputs
NOTEBOOKS_RAW=$(python - <<'PY' "$CONFIG_PATH"
import yaml
from pathlib import Path
import sys

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text()) or {}
outdir_val = cfg.get("output_root") or (cfg.get("paths", {}) or {}).get("output_root", "")
outdir = Path(outdir_val)

def has_files(p: Path) -> bool:
    return p.exists() and any(x.is_file() for x in p.rglob("*"))

notebooks: list[str] = []
if has_files(outdir / "deterministic"):
    notebooks.append("deterministic_verification.ipynb")
if has_files(outdir / "probabilistic"):
    notebooks.append("probabilistic_verification.ipynb")

# Intercomparison artifacts are typically combined/compare outputs.
has_intercompare = (
    has_files(outdir / "intercomparison")
    or any(outdir.rglob("*_combined*.csv"))
    or any(outdir.rglob("*_compare*.png"))
)
if has_intercompare:
    notebooks.append("model_intercomparison.ipynb")

print(" ".join(notebooks))
PY
)

if [ -z "$NOTEBOOKS_RAW" ]; then
    echo "No matching deterministic/probabilistic/intercomparison outputs found; skipping notebook rendering."
    exit 0
fi

read -r -a NOTEBOOKS <<< "$NOTEBOOKS_RAW"
echo "Selected notebooks: ${NOTEBOOKS[*]}"

for nb_name in "${NOTEBOOKS[@]}"; do
    nb="${PROJECT_ROOT}/notebooks/${nb_name}"
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
    bash render_single_notebook.sh "$CONFIG_FILE" "$PROJECT_ROOT"

# Cleanup
rm render_single_notebook.sh

echo "All tasks completed."
