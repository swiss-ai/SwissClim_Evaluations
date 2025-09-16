# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git bzip2 g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda for ARM architecture
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Accept Terms of Service for required channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy dependency files
COPY tools/environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Copy the rest of the application, including pyproject.toml or setup.py
COPY . .

# Activate the environment and install the project
RUN /bin/bash -c "source activate swissai-eval && uv pip install -e ."

# Set the default command
CMD ["python"]