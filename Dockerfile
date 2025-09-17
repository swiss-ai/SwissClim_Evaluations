FROM debian:bookworm-slim

# Bring in uv (fast Python package manager) from official image
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

# System deps (gcc, gdal, etc.) — adjust as needed for Cartopy/GDAL
RUN apt update && \
    apt install -y --no-install-recommends \
    ca-certificates python3-dev curl git bzip2 g++ \
    libexpat1 libgdal-dev gdal-bin build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the project
WORKDIR /app
ADD . /app

# Create and sync the environment with uv (resolves from pyproject.toml)
RUN uv sync

# Make uv's Python the default
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python"]
