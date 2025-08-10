# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs logs cache

# Ensure app/models directory exists and copy the real files
RUN mkdir -p app/models && \
    echo "=== Checking source files ===" && \
    ls -la app/models/ && \
    echo "=== Copying real model files ===" && \
    cp app/models/transcription.py app/models/transcription.py 2>/dev/null || echo "transcription.py copy failed" && \
    cp app/models/__init__.py app/models/__init__.py 2>/dev/null || echo "__init__.py copy failed" && \
    echo "=== Model files after copy ===" && \
    ls -la app/models/ && \
    echo "=== transcription.py content preview ===" && \
    head -5 app/models/transcription.py 2>/dev/null || echo "transcription.py not accessible"

# Debug: Check the final structure
RUN echo "=== Final app directory structure ===" && \
    ls -la app/ && \
    echo "=== App/models directory contents ===" && \
    ls -la app/models/ && \
    echo "=== App/models __init__.py ===" && \
    cat app/models/__init__.py

# Verify Python path and package structure
RUN echo 'import sys; print("Python path:", sys.path); import app; print("App package imported successfully"); print("App dir contents:", dir(app)); from app.models.transcription import TranscriptionRequest; print("TranscriptionRequest imported successfully"); print("All imports verified successfully")' > /tmp/verify_imports.py && \
    python /tmp/verify_imports.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
