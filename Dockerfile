# Multi-stage build for optimized production image
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ========================================
# Production stage
FROM python:3.9-slim

# Set labels for container metadata
LABEL maintainer="Jay Guwalani <jguwalan@umd.edu>"
LABEL version="1.0.0"
LABEL description="Driver Behavior Analytics System"
LABEL build_date=${BUILD_DATE}

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libopenblas-base \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r analytics && useradd -r -g analytics analytics

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY setup.py .
COPY README.md .

# Create necessary directories with correct permissions
RUN mkdir -p models data logs && \
    chown -R analytics:analytics /app

# Switch to non-root user
USER analytics

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose application port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "4"]
