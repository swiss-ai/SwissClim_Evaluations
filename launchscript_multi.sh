#!/bin/bash
#SBATCH --job-name=swissclim-eval-multi
#SBATCH --output=logs/swissclim_multi_%j.out
#SBATCH --error=logs/swissclim_multi_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32

# -------------------------------------------------------------
# EDIT THESE TWO LINES FOR YOUR SETUP
# 1) Path to your Enroot/EDF TOML file (or EDF name if your site supports it)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"
# -------------------------------------------------------------

# -------------------------------------------------------------
# DASK SCRATCH CONFIGURATION
# Set the directory for Dask spillover to avoid filling /tmp
# -------------------------------------------------------------
export DASK_TEMPORARY_DIRECTORY="/iopsstor/scratch/cscs/$USER/dask-tmp"
mkdir -p "$DASK_TEMPORARY_DIRECTORY"

# Resolve config relative to the job submission directory (SLURM_SUBMIT_DIR)
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Disable rich/ANSI output so SLURM log files remain clean
export SWISSCLIM_COLOR=never
export PYTHONUNBUFFERED=1

export PYTHONPATH="${SUBMIT_DIR}/src:${PYTHONPATH}"

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Step 1: Parallel Evaluation Jobs ---
echo "Starting parallel evaluation jobs..."

# List of evaluation configs
# Read from shared file
if [ ! -f "eval_configs.txt" ]; then
    echo "Error: eval_configs.txt not found!"
    exit 1
fi
mapfile -t EVAL_CONFIGS < eval_configs.txt

# Filter out empty lines
VALID_CONFIGS=()
for config in "${EVAL_CONFIGS[@]}"; do
    [[ -n "$config" ]] && VALID_CONFIGS+=("$config")
done
EVAL_CONFIGS=("${VALID_CONFIGS[@]}")

# Create multi-prog config file
MP_CONF="eval_multiprog.conf"
rm -f "$MP_CONF"

idx=0
for config in "${EVAL_CONFIGS[@]}"; do
    [ -z "$config" ] && continue
    # Use parent dir + filename to avoid collisions
    filename=$(basename "${config}" .yaml)
    parent_dir=$(basename "$(dirname "${config}")")
    job_name="${parent_dir}_${filename}"

    # Map task ID to command.
    # We append the exit code to the log file to avoid creating separate status files
    echo "$idx bash -c 'python -u -m swissclim_evaluations.cli --config \"${config}\" > logs/${job_name}.log 2>&1; echo \"SWISSCLIM_JOB_EXIT_CODE: \$?\" >> logs/${job_name}.log'" >> "$MP_CONF"
    ((idx++))
done

echo "Evaluation configs: ${EVAL_CONFIGS[*]}"
# Run all tasks in a single step
srun --ntasks=${#EVAL_CONFIGS[@]} --cpus-per-task=32 --multi-prog "$MP_CONF" \
    --container-writable --environment="${EDF_CONFIG}"

# Cleanup
rm "$MP_CONF"
echo "Evaluation jobs finished."

# Check for failures and identify successful jobs
FAILURES=0
SUCCESSFUL_CONFIGS=()

# --- Step 2: Check Job Status and Collect Logs ---
for config in "${EVAL_CONFIGS[@]}"; do
    [ -z "$config" ] && continue
    filename=$(basename "${config}" .yaml)
    parent_dir=$(basename "$(dirname "${config}")")
    job_name="${parent_dir}_${filename}"
    log_file="logs/${job_name}.log"

    # Check status from log file
    status_code=""
    if [ -f "$log_file" ]; then
        # Extract exit code from the specific line
        status_line=$(grep "SWISSCLIM_JOB_EXIT_CODE:" "$log_file" | tail -n 1)
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
        if [ -f "$log_file" ]; then
            echo "=== TAIL OF LOG: $log_file ==="
            tail -n 20 "$log_file"
            echo "========================================="
        else
            echo "Log file not found: $log_file"
        fi
    fi
done

# Copy logs to output folders
idx=0
for config in "${EVAL_CONFIGS[@]}"; do
    # Use parent dir + filename to avoid collisions
    filename=$(basename "${config}" .yaml)
    parent_dir=$(basename "$(dirname "${config}")")
    job_name="${parent_dir}_${filename}"
    log_file="logs/${job_name}.log"

    if [ -f "$log_file" ]; then
        # Extract output_root
        outdir=$(python -c "import yaml; cfg=yaml.safe_load(open('$config')); print(cfg.get('output_root') or cfg.get('paths', {}).get('output_root', ''))")

        if [ -n "$outdir" ] && [ -d "$outdir" ]; then
            cp "$log_file" "$outdir/"
            echo "Copied $log_file to $outdir/"

            # Copy dask log if it exists (using idx as PROCID)
            dask_log="logs/dask_distributed_${SLURM_JOB_ID}_${idx}.log"
            if [ -f "$dask_log" ]; then
                cp "$dask_log" "$outdir/${job_name}_dask.log"
                echo "Copied $dask_log to $outdir/"
            fi
        fi
    fi
    ((idx++))
done

if [ $FAILURES -ne 0 ]; then
    echo "One or more evaluation jobs failed. Proceeding with notebook rendering for successful jobs only."
fi

# --- Step 3: Render Notebooks ---
echo "Rendering notebooks..."

# Create a bash script to handle notebook rendering for a single config
cat <<'EOF' > render_single_notebook.sh
#!/bin/bash

config=$1
logfile=$2
shift 2
notebooks=("$@")

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

if [ ! -d "$outdir" ]; then
    echo "Output directory does not exist: $outdir"
    exit 0
fi

echo "Processing notebooks for config: $config -> $outdir"

failed_notebooks=()
for nb_name in "${notebooks[@]}"; do
    nb="notebooks/${nb_name}"
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

# Create multi-prog config for notebooks
MP_NB_CONF="notebook_multiprog.conf"
rm -f "$MP_NB_CONF"

task_id=0
for config in "${SUCCESSFUL_CONFIGS[@]}"; do
    [ -z "$config" ] && continue
    # Use parent dir + filename to avoid collisions
    filename=$(basename "${config}" .yaml)
    parent_dir=$(basename "$(dirname "${config}")")
    job_name="${parent_dir}_${filename}"

    echo "$task_id bash render_single_notebook.sh $config logs/notebook_${job_name}.log deterministic_verification.ipynb" >> "$MP_NB_CONF"
    ((task_id++))
done

if [ $task_id -gt 0 ]; then
    echo "Rendering notebooks in parallel for $task_id jobs..."
    # Move flags before --multi-prog to avoid them being passed as arguments to the command
    srun --ntasks=$task_id --cpus-per-task=32 \
        --container-writable --environment="${EDF_CONFIG}" \
        --multi-prog "$MP_NB_CONF"
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
    log_file="logs/notebook_${job_name}.log"

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
done

# Cleanup
rm "$MP_NB_CONF" "render_single_notebook.sh"

echo "All tasks completed."
