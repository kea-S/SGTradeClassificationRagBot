# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 1. INSTALL SYSTEM DEPENDENCIES & NODE.JS
# We add 'nodejs' and 'npm' here so we can install promptfoo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g promptfoo \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy the application
COPY . .

# Install the project
RUN uv pip install --system -e .

# 3. UPDATE THE COMMAND
# Now 'promptfoo' exists in the container.
# We don't strictly need 'uv run' for promptfoo itself since it's a global binary,
# but keeping it doesn't hurt.
CMD ["promptfoo", "eval", "-c", "src/sg_trade_ragbot/utils/evals/eval_configs/bare_config.yaml"]
