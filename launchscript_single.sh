#!/bin/bash
#SBATCH --job-name=swissclim-eval-single
#SBATCH --output=logs/swissclim_single_%j.out
#SBATCH --error=logs/swissclim_single_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================
# USER INPUT (edit this section only)
# =============================================================
# Evaluation config to run (absolute path or relative to submit/project directory)
CONFIG_FILE="config/example_config.yaml"

# Enroot/EDF TOML (or EDF name if your site supports it)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"

# Dask spill directory
DASK_TEMPORARY_DIRECTORY="/iopsstor/scratch/cscs/$USER/dask-tmp"

# PYTHONPATH behavior for project src directory:
# - prepend  : PROJECT_ROOT/src:$PYTHONPATH
# - overwrite: PROJECT_ROOT/src only
# - keep     : do not modify PYTHONPATH (default)
PYTHONPATH_MODE="keep"
# =============================================================

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

resolve_output_dir() {
    local cfg_path="$1"
    local project_root="$2"
    python - <<'PY' "$cfg_path" "$project_root"
import os
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()
cfg = yaml.safe_load(cfg_path.read_text()) or {}
out = cfg.get("output_root") or (cfg.get("paths", {}) or {}).get("output_root", "")
if not out:
    print("")
    raise SystemExit(0)

out_path = Path(out)
if not out_path.is_absolute():
    cfg_based = (cfg_path.parent / out_path).resolve()
    project_based = (project_root / out_path).resolve()
    out_path = cfg_based if cfg_based.exists() else project_based

print(str(out_path))
PY
}

if ! CONFIG_FILE_RESOLVED="$(resolve_config_path "$CONFIG_FILE")"; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Checked: '$CONFIG_FILE', '${SUBMIT_DIR}/${CONFIG_FILE}', '${PROJECT_ROOT}/${CONFIG_FILE}'"
    exit 1
fi
CONFIG_FILE="$CONFIG_FILE_RESOLVED"

# Configure PYTHONPATH behavior
SRC_PATH="${PROJECT_ROOT}/src"
case "$PYTHONPATH_MODE" in
    prepend)
        export PYTHONPATH="${SRC_PATH}:${PYTHONPATH:-}"
        echo "PYTHONPATH mode: prepend (${SRC_PATH} added first)"
        ;;
    overwrite)
        export PYTHONPATH="${SRC_PATH}"
        echo "PYTHONPATH mode: overwrite (set to ${SRC_PATH})"
        ;;
    keep)
        echo "PYTHONPATH mode: keep (no changes)"
        ;;
    *)
        echo "ERROR: Invalid PYTHONPATH_MODE='$PYTHONPATH_MODE'. Use: prepend|overwrite|keep"
        exit 1
        ;;
esac

export DASK_TEMPORARY_DIRECTORY
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
# Extract and resolve output_root robustly
OUTDIR="$(resolve_output_dir "$CONFIG_FILE" "$PROJECT_ROOT")"

if [ -n "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
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
cat <<'EOF' > "${PROJECT_ROOT}/render_single_notebook.sh"
#!/bin/bash

# Install missing dependencies for notebook rendering

CONFIG_PATH="$1"
PROJECT_ROOT="$2"
OUTDIR="$3"

if [ -z "$OUTDIR" ]; then
    echo "No output_root found in $CONFIG_PATH"
    exit 1
fi

mkdir -p "$OUTDIR"

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
NOTEBOOK_LOG="${LOG_DIR}/${job_name}_notebooks.log"
srun --ntasks=1 --container-writable --environment="${EDF_CONFIG}" \
    bash "${PROJECT_ROOT}/render_single_notebook.sh" "$CONFIG_FILE" "$PROJECT_ROOT" "$OUTDIR" > "$NOTEBOOK_LOG" 2>&1

if [ -n "$OUTDIR" ] && [ -f "$NOTEBOOK_LOG" ]; then
    cp "$NOTEBOOK_LOG" "$OUTDIR/${job_name}_notebooks.log" 2>/dev/null || echo "Could not copy notebook render log"
fi

# Cleanup
rm "${PROJECT_ROOT}/render_single_notebook.sh"

echo "All tasks completed."
