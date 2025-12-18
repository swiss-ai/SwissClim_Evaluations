#!/bin/bash
#SBATCH --job-name=swissclim-eval-large
#SBATCH --output=logs/swissclim_large_%j.out
#SBATCH --error=logs/swissclim_large_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32

# -------------------------------------------------------------
# EDIT THESE TWO LINES FOR YOUR SETUP
# 1) Path to your Enroot/EDF TOML file (or EDF name if your site supports it)
EDF_CONFIG="/users/$USER/.edf/swissclim-eval.toml"
# -------------------------------------------------------------

# Resolve config relative to the job submission directory (SLURM_SUBMIT_DIR)
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Disable rich/ANSI output so SLURM log files remain clean
export SWISSCLIM_COLOR=never
export PYTHONUNBUFFERED=1

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
    job_name=$(basename "${config}" .yaml)
    # Map task ID to command.
    echo "$idx bash -c 'python -u -m swissclim_evaluations.cli --config \"${config}\" > logs/${job_name}.log 2>&1'" >> "$MP_CONF"
    ((idx++))
done

echo "Evaluation configs: ${EVAL_CONFIGS[*]}"
# Run all tasks in a single step
srun --ntasks=${#EVAL_CONFIGS[@]} --cpus-per-task=32 --multi-prog "$MP_CONF" \
    --container-writable --environment="${EDF_CONFIG}"

# Cleanup
rm "$MP_CONF"
echo "Evaluation jobs finished."

# Copy logs to output folders
for config in "${EVAL_CONFIGS[@]}"; do
    job_name=$(basename "${config}" .yaml)
    log_file="logs/${job_name}.log"

    if [ -f "$log_file" ]; then
        # Extract output_root
        outdir=$(python -c "import yaml; cfg=yaml.safe_load(open('$config')); print(cfg.get('output_root') or cfg.get('paths', {}).get('output_root', ''))")

        if [ -n "$outdir" ] && [ -d "$outdir" ]; then
            cp "$log_file" "$outdir/"
            echo "Copied $log_file to $outdir/"
        fi
    fi
done

# --- Step 3: Render Notebooks ---
echo "Rendering notebooks..."

# Create a bash script to handle notebook rendering using papermill CLI
cat <<'EOF' > render_notebooks.sh
#!/bin/bash
# set -e removed to allow continuing after errors

# Define groups
# Read from shared file
if [ ! -f "eval_configs.txt" ]; then
    echo "Error: eval_configs.txt not found!"
    exit 1
fi
mapfile -t eval_configs < eval_configs.txt

# Filter out empty lines
valid_configs=()
for config in "${eval_configs[@]}"; do
    [[ -n "$config" ]] && valid_configs+=("$config")
done
eval_configs=("${valid_configs[@]}")

# Function to render notebooks
render_notebooks() {
    local config=$1
    shift
    local notebooks=("$@")

    if [ ! -f "$config" ]; then
        echo "Config file not found: $config"
        return
    fi

    # Get absolute path to config to ensure notebooks find it correctly
    abs_config=$(python -c "import os; print(os.path.abspath('$config'))")

    # Extract output_root using python for robust YAML parsing
    outdir=$(python -c "import yaml; cfg=yaml.safe_load(open('$config')); print(cfg.get('output_root') or cfg.get('paths', {}).get('output_root', ''))")

    if [ -z "$outdir" ]; then
        echo "No output_root found in $config"
        return
    fi

    if [ ! -d "$outdir" ]; then
        echo "Output directory does not exist: $outdir"
        return
    fi

    echo "Processing notebooks for config: $config -> $outdir"

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
            continue
        fi

        echo "  Converting to HTML..."
        # Convert to HTML
        if ! python -m jupyter nbconvert --to html "$out_nb_path"; then
            echo "ERROR: Failed to convert $nb_name to HTML"
        fi
    done
}

# Process Eval Configs
for config in "${eval_configs[@]}"; do
    [ -z "$config" ] && continue
    render_notebooks "$config" "deterministic_verification.ipynb"
done
EOF

# Run the rendering script
srun --ntasks=1 --cpus-per-task=32 --exclusive --output="logs/notebook_rendering.log" \
    --container-writable --environment="${EDF_CONFIG}" \
    bash render_notebooks.sh

# Cleanup temporary script
rm render_notebooks.sh

echo "All tasks completed."
