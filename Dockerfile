FROM debian:bookworm-slim

# Bring in uv (fast Python package manager) from official image
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
#  - Core build & scientific stack (GDAL, etc.)
#  - VS Code Server / tunnel runtime deps (libsecret-1-0, libnss3, libxkbfile1, libgbm1, libxshmfence1, libdrm2, openssh-client)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates python3-dev curl git bzip2 g++ build-essential \
    libexpat1 libgdal-dev gdal-bin \
    libsecret-1-0 libnss3 libxkbfile1 libglib2.0-0 libxshmfence1 libdrm2 libgbm1 \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Copy the project
WORKDIR /app
ADD . /app

# Create and sync the environment with uv (resolves from pyproject.toml)
RUN uv sync

# Make uv's Python the default
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python"]
