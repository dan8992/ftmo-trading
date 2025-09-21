# DAX LLM Service Dockerfile (Multi-container service)
FROM python:3.11-slim

LABEL maintainer="FTMO Trading Team"
LABEL description="DAX LLM Service with sentiment analysis and financial analyzer"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./

# Create model cache directory
RUN mkdir -p /tmp/transformers_cache

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /tmp/transformers_cache
USER appuser

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV MODEL_NAME=nlptown/bert-base-multilingual-uncased-sentiment
ENV GPU_MEMORY_UTILIZATION=0.9
ENV MAX_MODEL_LEN=4096
ENV TENSOR_PARALLEL_SIZE=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Expose ports
EXPOSE 8000 8080

# Run the application
CMD ["python", "main.py"]