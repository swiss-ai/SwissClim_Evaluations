#!/bin/bash

# Check if already in uenv environment
if uenv status | grep -q "there is no uenv loaded"; then
    echo "Starting uenv with prgenv-gnu..."
    uenv start prgenv-gnu/24.11:v1 --view=modules
else
    echo "✅ Already in uenv environment: $UENV_SESSION"
fi
# load required modules
module load gcc
module load cmake 

# install uv if not already installed
if ! command -v uv > /dev/null; then
    echo "🔧 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "✅ uv already installed."
fi

# create environment if not already created
if [ ! -d ".venv" ]; then
    echo "🔧 Creating environment..."
    uv venv --prompt=swissai-eval --python=3.12.9
else
    echo "✅ Environment already created."
fi

# Clone weatherbenchX from GitHub to parent directory
echo "Cloning weatherbenchX from GitHub..."
cd ..
if [ ! -d "weatherbenchX" ]; then
    git clone https://github.com/google-research/weatherbenchX.git weatherbenchX
    echo "weatherbenchX cloned successfully"
else
    echo "weatherbenchX directory already exists, skipping clone"
fi
cd SwissClim_Evaluations

uv sync 

# activate environment
source .venv/bin/activate