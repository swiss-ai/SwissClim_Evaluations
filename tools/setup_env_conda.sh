#!/bin/bash
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
    conda env create -f environment.yml
else
    echo "✅ Conda environment 'swissai-eval' already exists."
fi

# Now the user can activate the environment manually
echo "You can now activate the conda environment using:"
echo "   conda activate swissai-eval"