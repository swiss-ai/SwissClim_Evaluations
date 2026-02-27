#!/bin/bash
#SBATCH --job-name=swissclim-eval-multi
#SBATCH --output=logs/swissclim_multi_%j.out
#SBATCH --error=logs/swissclim_multi_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=1
# ntasks-per-node and cpus-per-task are derived from PARALLEL_BATCH_SIZE below.
# Allocation: PARALLEL_BATCH_SIZE tasks × (144 / PARALLEL_BATCH_SIZE) CPUs = 144 CPUs total.
# If you change PARALLEL_BATCH_SIZE you MUST also update these two directives to match:
#   --ntasks-per-node = PARALLEL_BATCH_SIZE
#   --cpus-per-task   = 144 / PARALLEL_BATCH_SIZE  (must be a whole number)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=36

# =============================================================
# USER INPUT (edit this section only)
# =============================================================
# Enroot/EDF TOML (or EDF name if your site supports it)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"

# File listing evaluation configs, one path per line
EVAL_CONFIG_LIST_FILE="eval_configs.txt"

# Dask spill directory
DASK_TEMPORARY_DIRECTORY="/iopsstor/scratch/cscs/$USER/dask-tmp"

# Maximum number of eval configs to run in parallel at one time.
# Configs beyond this limit are processed in sequential batches so that
# memory is always split by at most PARALLEL_BATCH_SIZE, preventing OOMs.
# MUST match --ntasks-per-node above.
PARALLEL_BATCH_SIZE=4

# Approx. usable node RAM in GiB for this job (set to your node memory)
NODE_RAM_GIB=480

# Fraction of node RAM allowed for the parallel evals in one batch (0-1).
# Keep headroom for scheduler/container/OS overhead.
NODE_RAM_USAGE_FRACTION=0.90

# Fraction of per-eval budget passed to Dask worker pool (0-1).
# Remaining memory stays as process/output overhead outside workers.
DASK_EVAL_MEMORY_FRACTION=0.90

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

TEMP_FILES=()
cleanup() {
    for f in "${TEMP_FILES[@]}"; do
        [ -n "$f" ] && rm -f "$f"
    done
}
trap cleanup EXIT

# --- Step 1: Parallel Evaluation Jobs ---
echo "Starting parallel evaluation jobs..."

# List of evaluation configs
# Read from shared file
if [ ! -f "${PROJECT_ROOT}/${EVAL_CONFIG_LIST_FILE}" ]; then
    echo "Error: ${EVAL_CONFIG_LIST_FILE} not found!"
    exit 1
fi
mapfile -t EVAL_CONFIGS < "${PROJECT_ROOT}/${EVAL_CONFIG_LIST_FILE}"

# Filter out empty lines
VALID_CONFIGS=()
for config in "${EVAL_CONFIGS[@]}"; do
    [[ -n "$config" ]] && VALID_CONFIGS+=("$config")
done
EVAL_CONFIGS=("${VALID_CONFIGS[@]}")

# Resolve config paths robustly (absolute or relative to submit directory)
RESOLVED_CONFIGS=()
for config in "${EVAL_CONFIGS[@]}"; do
    [ -z "$config" ] && continue
    resolved=""
    if [ -f "$config" ]; then
        resolved=$(python - <<'PY' "$config"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)
    elif [ -f "${SUBMIT_DIR}/${config}" ]; then
        resolved=$(python - <<'PY' "${SUBMIT_DIR}/${config}"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)
    elif [ -f "${PROJECT_ROOT}/${config}" ]; then
        resolved=$(python - <<'PY' "${PROJECT_ROOT}/${config}"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)
    else
        echo "ERROR: Config file not found: $config"
        echo "Checked: '$config', '${SUBMIT_DIR}/${config}', '${PROJECT_ROOT}/${config}'"
        exit 1
    fi
    RESOLVED_CONFIGS+=("$resolved")
done
EVAL_CONFIGS=("${RESOLVED_CONFIGS[@]}")

TOTAL_PARALLEL_EVALS=${#EVAL_CONFIGS[@]}
if [ "$TOTAL_PARALLEL_EVALS" -lt 1 ]; then
    echo "ERROR: No evaluation configs found after filtering/resolution."
    exit 1
fi

if ! [[ "$NODE_RAM_GIB" =~ ^[0-9]+$ ]] || [ "$NODE_RAM_GIB" -lt 1 ]; then
    echo "ERROR: NODE_RAM_GIB must be a positive integer, got '$NODE_RAM_GIB'"
    exit 1
fi

if ! [[ "$NODE_RAM_USAGE_FRACTION" =~ ^0(\.[0-9]+)?|1(\.0+)?$ ]]; then
    echo "ERROR: NODE_RAM_USAGE_FRACTION must be between 0 and 1, got '$NODE_RAM_USAGE_FRACTION'"
    exit 1
fi

if ! [[ "$DASK_EVAL_MEMORY_FRACTION" =~ ^0(\.[0-9]+)?|1(\.0+)?$ ]]; then
    echo "ERROR: DASK_EVAL_MEMORY_FRACTION must be between 0 and 1, got '$DASK_EVAL_MEMORY_FRACTION'"
    exit 1
fi

if ! [[ "$PARALLEL_BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$PARALLEL_BATCH_SIZE" -lt 1 ]; then
    echo "ERROR: PARALLEL_BATCH_SIZE must be a positive integer, got '$PARALLEL_BATCH_SIZE'"
    exit 1
fi

TOTAL_USABLE_RAM_GIB=$(python - <<'PY' "$NODE_RAM_GIB" "$NODE_RAM_USAGE_FRACTION"
import sys
node_ram = float(sys.argv[1])
fraction = float(sys.argv[2])
print(max(1, int(node_ram * fraction)))
PY
)

# Memory per eval is always based on the fixed batch width, NOT the total number of
# configs.  This guarantees that no matter how many configs are in the list, each
# running eval gets the same generous slice of RAM and never races against N others.
EVAL_MEMORY_BUDGET_GIB=$(python - <<'PY' "$TOTAL_USABLE_RAM_GIB" "$PARALLEL_BATCH_SIZE"
import sys
usable = float(sys.argv[1])
batch  = int(sys.argv[2])
print(max(1, int(usable / batch)))
PY
)

export SWISSCLIM_DASK_MEMORY_BUDGET_GIB="$EVAL_MEMORY_BUDGET_GIB"
export SWISSCLIM_DASK_MEMORY_BUDGET_FRACTION="$DASK_EVAL_MEMORY_FRACTION"

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-36}"

echo "Resource plan: node_ram=${NODE_RAM_GIB}GiB, usable=${TOTAL_USABLE_RAM_GIB}GiB"
echo "  batch_size=${PARALLEL_BATCH_SIZE}, per_eval_budget=${EVAL_MEMORY_BUDGET_GIB}GiB, dask_fraction=${DASK_EVAL_MEMORY_FRACTION}"
echo "  total_configs=${TOTAL_PARALLEL_EVALS}, batches=ceil(${TOTAL_PARALLEL_EVALS}/${PARALLEL_BATCH_SIZE})"
echo "  cpus_per_eval=${CPUS_PER_TASK}"

# ---------------------------------------------------------------------------
# Process evaluation configs in sequential batches of PARALLEL_BATCH_SIZE.
# Each batch is launched as a single blocking srun --multi-prog step so the
# node's memory is always shared by at most PARALLEL_BATCH_SIZE processes.
# ---------------------------------------------------------------------------
MP_CONF="${PROJECT_ROOT}/eval_multiprog.conf"
TEMP_FILES+=("$MP_CONF")

batch_num=0
batch_start=0
while [ "$batch_start" -lt "$TOTAL_PARALLEL_EVALS" ]; do
    batch_end=$(( batch_start + PARALLEL_BATCH_SIZE ))
    if [ "$batch_end" -gt "$TOTAL_PARALLEL_EVALS" ]; then
        batch_end="$TOTAL_PARALLEL_EVALS"
    fi
    BATCH_CONFIGS=("${EVAL_CONFIGS[@]:$batch_start:$(( batch_end - batch_start ))}")
    current_batch_size=${#BATCH_CONFIGS[@]}
    (( batch_num++ ))

    echo ""
    echo "=== Batch ${batch_num}: running configs ${batch_start}–$(( batch_end - 1 )) (${current_batch_size} parallel evals) ==="

    rm -f "$MP_CONF"
    local_idx=0
    for config in "${BATCH_CONFIGS[@]}"; do
        [ -z "$config" ] && continue
        filename=$(basename "${config}" .yaml)
        parent_dir=$(basename "$(dirname "${config}")")
        job_name="${parent_dir}_${filename}"
        out_log="${LOG_DIR}/${job_name}.out"
        err_log="${LOG_DIR}/${job_name}.err"

        # Append exit code to the .out file so we can detect success/failure later.
        echo "${local_idx} bash -c 'python -u -m swissclim_evaluations.cli --config \"${config}\" > \"${out_log}\" 2> \"${err_log}\"; echo \"SWISSCLIM_JOB_EXIT_CODE: \$?\" >> \"${out_log}\"'" >> "$MP_CONF"
        (( local_idx++ ))
    done

    # srun is blocking: this call returns only when all tasks in the batch complete.
    srun --ntasks="${current_batch_size}" --cpus-per-task="${CPUS_PER_TASK}" \
        --container-writable --environment="${EDF_CONFIG}" \
        --multi-prog "$MP_CONF"

    echo "=== Batch ${batch_num} finished ==="
    batch_start="$batch_end"
done

rm -f "$MP_CONF"
echo ""
echo "All evaluation batches finished."

# Check for failures and identify successful jobs
FAILURES=0
SUCCESSFUL_CONFIGS=()

# --- Step 2: Check Job Status and Collect Logs ---
for config in "${EVAL_CONFIGS[@]}"; do
    [ -z "$config" ] && continue
    filename=$(basename "${config}" .yaml)
    parent_dir=$(basename "$(dirname "${config}")")
    job_name="${parent_dir}_${filename}"
    out_log="${LOG_DIR}/${job_name}.out"
    err_log="${LOG_DIR}/${job_name}.err"

    # Check status from log file
    status_code=""
    if [ -f "$out_log" ]; then
        # Extract exit code from the specific line
        status_line=$(grep "SWISSCLIM_JOB_EXIT_CODE:" "$out_log" | tail -n 1)
        if [ -n "$status_line" ]; then
            status_code=$(echo "$status_line" | awk -F': ' '{print $2}')
        fi
    fi

    if [ "$status_code" == "0" ]; then
        echo "SUCCESS: Job $job_name completed successfully."
        SUCCESSFUL_CONFIGS+=("$config")
    else
        FAILURES=1
        echo "ERROR: Job $job_name failed (Exit code: ${status_code:-UNKNOWN})"
        if [ -f "$out_log" ]; then
            echo "=== TAIL OF OUT LOG: $out_log ==="
            tail -n 20 "$out_log"
            echo "========================================="
        else
            echo "Out log not found: $out_log"
        fi
        if [ -f "$err_log" ]; then
            echo "=== TAIL OF ERR LOG: $err_log ==="
            tail -n 20 "$err_log"
            echo "========================================="
        else
            echo "Err log not found: $err_log"
        fi
    fi
done

# Copy logs to output folders
for config in "${EVAL_CONFIGS[@]}"; do
    [ -z "$config" ] && continue
    filename=$(basename "${config}" .yaml)
    parent_dir=$(basename "$(dirname "${config}")")
    job_name="${parent_dir}_${filename}"
    out_log="${LOG_DIR}/${job_name}.out"
    err_log="${LOG_DIR}/${job_name}.err"

    if [ -f "$out_log" ] || [ -f "$err_log" ]; then
        outdir="$(resolve_output_dir "$config" "$PROJECT_ROOT")"

        if [ -n "$outdir" ]; then
            mkdir -p "$outdir"
            if [ -f "$out_log" ]; then
                cp "$out_log" "$outdir/"
                echo "Copied $out_log to $outdir/"
            fi
            if [ -f "$err_log" ]; then
                cp "$err_log" "$outdir/"
                echo "Copied $err_log to $outdir/"
            fi
            # Note: dask log PROCID restarts at 0 each batch, so we cannot map
            # global config index → PROCID reliably.  Dask logs land in LOG_DIR
            # and are available there for manual inspection.
        fi
    fi
done

if [ $FAILURES -ne 0 ]; then
    echo "One or more evaluation jobs failed. Proceeding with notebook rendering for successful jobs only."
fi

# --- Step 3: Render Notebooks ---
echo "Rendering notebooks..."

# Create a bash script to handle notebook rendering for a single config
cat <<'EOF' > "${PROJECT_ROOT}/render_single_notebook.sh"
#!/bin/bash

config=$1
logfile=$2
project_root=$3

# Redirect all output to logfile
exec > "$logfile" 2>&1

if [ ! -f "$config" ]; then
    echo "Config file not found: $config"
    exit 0
fi

# Get absolute path to config to ensure notebooks find it correctly
abs_config=$(python -c "import os; print(os.path.abspath('$config'))")

# Extract output_root using python for robust YAML parsing
outdir=$(python -c "import yaml; cfg=yaml.safe_load(open('$config')); print(cfg.get('output_root') or cfg.get('paths', {}).get('output_root', ''))")

if [ -z "$outdir" ]; then
    echo "No output_root found in $config"
    exit 0
fi

if [[ "$outdir" != /* ]]; then
    cfg_dir=$(python - <<'PY' "$config"
import os
import sys
print(os.path.dirname(os.path.abspath(sys.argv[1])))
PY
)
    if [ -d "${cfg_dir}/${outdir}" ]; then
        outdir="${cfg_dir}/${outdir}"
    else
        outdir="${project_root}/${outdir}"
    fi
fi

mkdir -p "$outdir"

echo "Processing notebooks for config: $config -> $outdir"

# Select notebooks based on config type and available outputs
notebooks_raw=$(python - <<'PY' "$config"
import yaml
from pathlib import Path
import sys

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text()) or {}
paths_block = cfg.get("paths", {}) or {}

is_intercomparison = bool(cfg.get("models")) and not (
    paths_block.get("target")
    or paths_block.get("nwp")
    or paths_block.get("prediction")
    or paths_block.get("ml")
)

outdir_val = cfg.get("output_root") or paths_block.get("output_root", "")
outdir = Path(outdir_val)

def has_files(p: Path) -> bool:
    return p.exists() and any(x.is_file() for x in p.rglob("*"))

notebooks: list[str] = []

if is_intercomparison:
    # Only render the intercomparison notebook; deterministic/probabilistic notebooks
    # require zarr datasets and must not be rendered for intercomparison configs.
    if has_files(outdir):
        notebooks.append("model_intercomparison.ipynb")
else:
    if has_files(outdir / "deterministic"):
        notebooks.append("deterministic_verification.ipynb")
    if has_files(outdir / "probabilistic"):
        notebooks.append("probabilistic_verification.ipynb")

print(" ".join(notebooks))
PY
)

if [ -z "$notebooks_raw" ]; then
    echo "No matching deterministic/probabilistic/intercomparison outputs found; skipping notebook rendering."
    echo "NOTEBOOK_RENDER_STATUS: SUCCESS"
    exit 0
fi

read -r -a notebooks <<< "$notebooks_raw"
echo "Selected notebooks: ${notebooks[*]}"

failed_notebooks=()
for nb_name in "${notebooks[@]}"; do
    nb="${project_root}/notebooks/${nb_name}"
    if [ ! -f "$nb" ]; then
        echo "Notebook not found: $nb"
        continue
    fi

    out_nb_path="${outdir}/${nb_name}"

    echo "  Rendering $nb_name..."
    # Execute notebook with papermill
    if ! python -m papermill "$nb" "$out_nb_path" -p config_path_str "$abs_config"; then
        echo "ERROR: Failed to render $nb_name for config $config"
        failed_notebooks+=("$nb_name")
        continue
    fi

    echo "  Converting to HTML..."
    # Convert to HTML
    if ! python -m jupyter nbconvert --to html "$out_nb_path"; then
        echo "ERROR: Failed to convert $nb_name to HTML"
    fi
done

if [ ${#failed_notebooks[@]} -eq 0 ]; then
    echo "NOTEBOOK_RENDER_STATUS: SUCCESS"
else
    echo "NOTEBOOK_RENDER_STATUS: FAILED (${failed_notebooks[*]})"
fi
exit 0
EOF

# Render notebooks in sequential batches of PARALLEL_BATCH_SIZE.
MP_NB_CONF="${PROJECT_ROOT}/notebook_multiprog.conf"
TEMP_FILES+=("$MP_NB_CONF")

TOTAL_SUCCESSFUL=${#SUCCESSFUL_CONFIGS[@]}
if [ "$TOTAL_SUCCESSFUL" -gt 0 ]; then
    nb_batch_num=0
    nb_batch_start=0
    while [ "$nb_batch_start" -lt "$TOTAL_SUCCESSFUL" ]; do
        nb_batch_end=$(( nb_batch_start + PARALLEL_BATCH_SIZE ))
        if [ "$nb_batch_end" -gt "$TOTAL_SUCCESSFUL" ]; then
            nb_batch_end="$TOTAL_SUCCESSFUL"
        fi
        NB_BATCH=("${SUCCESSFUL_CONFIGS[@]:$nb_batch_start:$(( nb_batch_end - nb_batch_start ))}")
        nb_current_size=${#NB_BATCH[@]}
        (( nb_batch_num++ ))

        echo ""
        echo "=== Notebook batch ${nb_batch_num}: rendering ${nb_current_size} configs ==="

        rm -f "$MP_NB_CONF"
        nb_local_idx=0
        for config in "${NB_BATCH[@]}"; do
            [ -z "$config" ] && continue
            filename=$(basename "${config}" .yaml)
            parent_dir=$(basename "$(dirname "${config}")")
            job_name="${parent_dir}_${filename}"
            echo "${nb_local_idx} bash ${PROJECT_ROOT}/render_single_notebook.sh $config ${LOG_DIR}/notebook_${job_name}.log ${PROJECT_ROOT}" >> "$MP_NB_CONF"
            (( nb_local_idx++ ))
        done

        srun --ntasks="${nb_current_size}" --cpus-per-task="${CPUS_PER_TASK}" \
            --container-writable --environment="${EDF_CONFIG}" \
            --multi-prog "$MP_NB_CONF"

        echo "=== Notebook batch ${nb_batch_num} finished ==="
        nb_batch_start="$nb_batch_end"
    done
else
    echo "No successful jobs to render."
fi

# Check notebook rendering status
echo "Checking notebook rendering results..."
for config in "${SUCCESSFUL_CONFIGS[@]}"; do
    [ -z "$config" ] && continue
    filename=$(basename "${config}" .yaml)
    parent_dir=$(basename "$(dirname "${config}")")
    job_name="${parent_dir}_${filename}"
    log_file="${LOG_DIR}/notebook_${job_name}.log"
    outdir="$(resolve_output_dir "$config" "$PROJECT_ROOT")"

    if [ -f "$log_file" ]; then
        status_line=$(grep "NOTEBOOK_RENDER_STATUS:" "$log_file" | tail -n 1)
        if [[ "$status_line" == *"SUCCESS"* ]]; then
             echo "Notebooks for $job_name: SUCCESS"
        else
             echo "Notebooks for $job_name: FAILED"
             # Print details if available
             if [ -n "$status_line" ]; then
                 echo "  $status_line"
             fi
             echo "  See log: $log_file"
        fi
    else
        echo "Notebooks for $job_name: UNKNOWN (Log file missing: $log_file)"
    fi

    if [ -n "$outdir" ] && [ -f "$log_file" ]; then
        mkdir -p "$outdir"
        cp "$log_file" "$outdir/${job_name}_notebooks.log" 2>/dev/null || \
            echo "Could not copy notebook log $log_file to $outdir"
    fi
done

# Cleanup
rm -f "$MP_NB_CONF" "${PROJECT_ROOT}/render_single_notebook.sh"

echo "All tasks completed."
