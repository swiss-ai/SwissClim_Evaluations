#!/bin/bash

# NOTE
# this setup script assumes the prgenv-gnu/24.11:v1 uenv on clariden is active

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

uv sync 

# activate environment
source .venv/bin/activate