FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Install system dependencies (build-essential, git, libgomp1 for LightGBM, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv using official pre-built binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Cache dependency layer: copy configuration and lockfile first
COPY pyproject.toml uv.lock ./

# Install project dependencies into virtualenv without copying full source yet
RUN uv sync --frozen --no-install-project --dev

# Copy application source code, tests, scripts and configuration
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY sql/ ./sql/
COPY AGENTS.md README.md ./

# Final synchronization to install project package itself
RUN uv sync --frozen --dev

# Default entrypoint runs the main application
CMD ["uv", "run", "python", "-m", "trading_bot.interfaces.main"]
