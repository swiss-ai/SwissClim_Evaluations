
#!/bin/bash
set -euo pipefail

# Path to conda.sh
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

# Check if conda is available
if ! command -v conda > /dev/null; then
    echo "⚠️ conda command not found in PATH. Trying to source conda.sh..."
    if [ -f "$CONDA_SH" ]; then
        # shellcheck source=/dev/null
        source "$CONDA_SH"
        if command -v conda > /dev/null; then
            echo "✅ conda successfully sourced from $CONDA_SH."
        else
            echo "❌ Failed to source conda from $CONDA_SH. Please install or initialize conda."
            exit 1
        fi
    else
        echo "❌ conda.sh not found at $CONDA_SH. Please install conda or adjust the path."
        exit 1
    fi
else
    echo "✅ conda is already available."
fi


# Create conda environment if not already created
if ! conda env list | grep -q "swissclim-eval"; then
    echo "🔧 Creating conda environment from environment.yml..."
    conda env create -f tools/environment.yml
else
    echo "✅ Conda environment 'swissclim-eval' already exists."
fi

# Initialize conda for this non-interactive shell and activate env
echo "🔁 Activating 'swissclim-eval'..."
eval "$(conda shell.bash hook)"
conda activate swissclim-eval

# Install the parent project in editable mode via uv
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo "📦 Installing project (editable) with uv..."
uv pip install -e "$SCRIPT_DIR/.."

# Reminder: activation inside a script doesn't persist after it exits
echo "You can now activate the conda environment using:"
echo "   conda activate swissclim-eval"
