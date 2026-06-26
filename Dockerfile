# ─────────────────────────────────────────────────────────────
#  Multi-Agent Triage Router — Production Dockerfile
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── 1. Install OS-level build dependencies (grpcio wheels may need them) ──
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── 2. Copy and install Python dependencies FIRST (cache-friendly) ────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── 3. Copy source code and project metadata ──────────────────────────────
COPY pyproject.toml .
COPY src/ src/

# Install the project itself in editable-compatible mode
RUN pip install --no-cache-dir .

# ── 4. Create a non-root user for runtime security ────────────────────────
RUN groupadd --system appuser && \
    useradd --system --gid appuser --no-create-home appuser
USER appuser

# ── 5. Expose the future API port ────────────────────────────────────────
EXPOSE 8000

# Default entrypoint — swap for uvicorn / gunicorn when the API layer lands
CMD ["python", "-m", "http.server", "8000"]

