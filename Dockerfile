# ============================================
# Stage 1: Builder - Install Python dependencies
# ============================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create virtual environment
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 support for RTX 5090 (sm_120/Blackwell)
# PyTorch 2.7+ with CUDA 12.8 is required for Blackwell architecture
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: Flash attention causes segfaults with jina-embeddings-v3 on RTX 5090 (Blackwell)
# The xlm-roberta-flash-implementation used by V3 is not compatible with Blackwell yet
# Skipping flash-attn installation - PyTorch's native SDPA will be used instead
RUN echo "Skipping flash-attention - not compatible with RTX 5090 Blackwell for V3 model"

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 runtime only (minimal)
RUN apt-get update && apt-get install -y \
    python3.11 \
    libgomp1 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser config/ ./config/

# Create cache directory for HuggingFace models
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache

# Switch to non-root user
USER appuser

# Environment variables
ENV DEVICE=cuda \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TORCH_HOME=/app/.cache/torch \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    MODELS_TO_LOAD=all \
    MAX_BATCH_SIZE=32 \
    TORCH_DTYPE=float16

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

# Run the application with uvicorn
# Using single worker since models are loaded into GPU memory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
