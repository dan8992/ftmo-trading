# FinBERT Trading LLM Dockerfile
FROM python:3.11-slim

LABEL maintainer="FTMO Trading Team"
LABEL description="FinBERT Trading LLM for financial sentiment analysis"

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
RUN mkdir -p /app/model_cache \
    && mkdir -p /tmp/transformers_cache

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /tmp/transformers_cache
USER appuser

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/transformers_cache
ENV MODEL_NAME=ProsusAI/finbert

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health', timeout=5)" || exit 1

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "main.py"]