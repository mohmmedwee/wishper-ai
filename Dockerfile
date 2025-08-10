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

# Debug: Check what was copied
RUN echo "=== After COPY . . ===" && \
    echo "=== Root directory ===" && ls -la && \
    echo "=== App directory ===" && ls -la app/ && \
    echo "=== App/models directory ===" && ls -la app/models/

# Verify Python path and package structure (AFTER installing dependencies)
RUN echo 'import sys; print("Python path:", sys.path); import app; print("App package imported successfully"); print("App dir contents:", dir(app)); from app.models.transcription import TranscriptionRequest; print("TranscriptionRequest imported successfully"); print("All imports verified successfully")' > /tmp/verify_imports.py && \
    python /tmp/verify_imports.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
