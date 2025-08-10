# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV PYTORCH_HOME=/app/models
ENV HF_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    gcc \
    g++ \
    make \
    curl \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs logs cache temp models

# Download NeMo models during build (with Docker flag for fallback handling)
RUN python scripts/download_nemo_models.py --docker --models vad_multilingual_marblenet titanet_large ecapa_tdnn --output-dir models

# Verify models were processed
RUN ls -la models/ && echo "Models processed successfully"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with proper Python path
CMD ["python", "-m", "app.main"]
