#!/bin/bash

ENV_NAME="prgenv-gnu/24.11:v1"

# Check if uenv image is already pulled
if ! uenv image list | grep -q "$ENV_NAME"; then
    echo "🔄 Pulling uenv image $ENV_NAME to local storage..."
    uenv image pull "$ENV_NAME"
else
    echo "✅ uenv image $ENV_NAME already available locally."
fi

# Check if already in uenv environment
if uenv status | grep -q "there is no uenv loaded"; then
    echo "🚀 Starting uenv with $ENV_NAME..."
    uenv start "$ENV_NAME" --view=modules
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

echo "🔧 Creating environment..."
uv venv --prompt=swissclim-eval --python=3.11.9
uv sync
uv pip install pip

# activate environment
source .venv/bin/activate
