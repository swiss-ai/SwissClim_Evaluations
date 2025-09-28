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
    openssh-client zsh && \
    rm -rf /var/lib/apt/lists/*

# Copy the project
WORKDIR /app
ADD . /app

## Create environment outside /app (bind-mounted at runtime) so it persists.
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# Preferred interactive shell inside the container
ENV SHELL=/bin/bash
# NOTE:
# 1) Virtual env location: We place the venv at /opt/venv instead of the project tree (/app/.venv).
#    At runtime /app is bind-mounted from the host (EDF/Enroot), which would hide any venv baked
#    into the image under /app/.venv. Locating it outside the mount keeps the environment immutable
#    and available regardless of host overlays.
# 2) Wrapper vs symlink: Plain symlinks (/usr/local/bin/python -> /opt/venv/bin/python) intermittently
#    failed to trigger venv detection (pyvenv.cfg not found) in this container+overlay setup, leading
#    to missing packages. Lightweight wrapper scripts exec the canonical interpreter path so Python
#    always resolves the correct sys.prefix.
RUN uv venv "$VIRTUAL_ENV" \
    && uv sync \
    && uv pip install pip \
    && for bin in python python3 pip pip3; do \
    printf '#!/bin/sh\nexec /opt/venv/bin/%s "$@"\n' "$bin" > /usr/local/bin/$bin; \
    chmod +x /usr/local/bin/$bin; \
    done
