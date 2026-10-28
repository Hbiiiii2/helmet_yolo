# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    YOLO_CONFIG_DIR=/tmp

WORKDIR /app

# Install system dependencies for OpenCV, PyTorch, and Container Healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (saves ~3GB bandwidth & prevents build timeouts on Railway)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (filtered by .dockerignore)
COPY . .

# Setup runtime storage directories and non-root user for security
RUN mkdir -p uploads snapshots /tmp/Ultralytics && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp/Ultralytics

USER appuser

# Expose container port (Railway binds dynamic $PORT automatically)
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch FastAPI web server
CMD ["python", "run_web.py"]
