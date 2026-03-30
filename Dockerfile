# Build stage with uv for dependency management
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml uv.lock ./
COPY src ./src

# Install dependencies using uv
RUN uv venv && \
    uv pip install --no-cache -r pyproject.toml

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

# Copy source code
COPY --from=builder /app/src /app/src

# Set Python path to use venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

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
