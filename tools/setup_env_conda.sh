#!/bin/bash
set -euo pipefail
# Check if conda is available
if ! command -v conda > /dev/null; then
    echo "❌ conda is not available. Please install conda first."
    exit 1
else
    echo "✅ conda is available."
fi

# Create conda environment if not already created
if ! conda env list | grep -q "swissai-eval"; then
    echo "🔧 Creating conda environment from environment.yml..."
    conda env create -f tools/environment.yml
else
    echo "✅ Conda environment 'swissai-eval' already exists."
fi

# Initialize conda for this non-interactive shell and activate env
echo "🔁 Activating 'swissai-eval'..."
eval "$(conda shell.bash hook)"
conda activate swissai-eval

# Install the parent project in editable mode via uv
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo "📦 Installing project (editable) with uv..."
uv pip install -e "$SCRIPT_DIR/.."

# Reminder: activation inside a script doesn't persist after it exits
echo "You can now activate the conda environment using:"
echo "   conda activate swissai-eval"