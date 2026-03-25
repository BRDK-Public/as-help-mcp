# Build stage with uv for dependency management
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml uv.lock ./
COPY src ./src

# Install dependencies using uv
# Use CPU-only PyTorch to avoid downloading ~2.5GB of CUDA packages
RUN uv venv && \
    uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache -r pyproject.toml

# Copy pre-exported model from host.
# Corporate proxies with TLS inspection prevent downloading from HuggingFace
# inside Docker (SSL cert verify fails).  Instead, the model is exported on
# the host via: uv run python prepare_model.py
# and then copied into the image.  This is fast (~22MB) and repeatable.
COPY .model_cache /app/.model_cache

# Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libsqlite3-0 \
    libxml2 \
    libxslt1.1 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy pre-exported model (from host's .model_cache/ via prepare_model.py)
COPY --from=builder /app/.model_cache /app/.model_cache

# Copy source code
COPY --from=builder /app/src /app/src

# Set Python path to use venv
# Disable tqdm/HF progress bars — they write to stdout which conflicts with
# the MCP stdio transport piped by Docker.
# Skip optional transformer backends to speed up import.
# HF_HUB_OFFLINE=1 ensures no network calls to HuggingFace at runtime.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    TQDM_DISABLE=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_XET=1 \
    TRANSFORMERS_OFFLINE=1 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    TOKENIZERS_PARALLELISM=false

# Default environment variables (can be overridden)
# Note: AS_HELP_DB_PATH is NOT set here - server.py auto-detects based on help root hash
# Default stdio transport is used for local development; for HTTP, set MCP_TRANSPORT=streamable-http and configure host/port with MCP_HOST/MCP_PORT
ENV AS_HELP_ROOT=/data/help \
    AS_HELP_FORCE_REBUILD=false\
    MCP_TRANSPORT=stdio \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# Create data directories
RUN mkdir -p /data/help /data/db

# Expose port for SSE mode (optional)
EXPOSE 8000

# Health check (start-period allows for first-run model download + index build)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from server import mcp; print('OK')" || exit 1

# Run the MCP server
CMD ["python", "-u", "src/server.py"]
