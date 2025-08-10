# Use Python 3.11 for ML package compatibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in stages to avoid disk space issues
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.20.0 \
    python-multipart>=0.0.6 \
    pydantic>=2.0.0 \
    pydantic-settings>=0.0.6 \
    click>=8.0.0 \
    python-dotenv>=1.0.0 \
    structlog>=23.0.0 \
    aiofiles>=23.0.0

# Install audio processing dependencies
RUN pip install --no-cache-dir \
    librosa>=0.10.0 \
    soundfile>=0.12.1 \
    pydub>=0.25.1

# Install ML dependencies (these are the heavy ones)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchaudio>=2.0.0

# Install remaining ML dependencies
RUN pip install --no-cache-dir \
    transformers>=4.30.0 \
    faster-whisper>=0.9.0

# Install NeMo toolkit (very heavy - install last)
RUN pip install --no-cache-dir \
    nemo-toolkit[asr]>=1.20.0 \
    nemo-toolkit[nlp]>=1.20.0

# Install remaining dependencies
RUN pip install --no-cache-dir \
    demucs>=4.0.0 \
    ctc-forced-aligner>=0.1.0 \
    deepmultilingualpunctuation>=0.1.0 \
    nltk>=3.8.0 \
    pyannote.audio>=3.0.0 \
    pyannote.core>=5.0.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    tqdm>=4.65.0 \
    redis>=4.6.0 \
    sqlalchemy>=2.0.0 \
    alembic>=1.11.0 \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    httpx>=0.24.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs logs cache models

# Ensure app/models directory exists and has the right structure
RUN ls -la app/ && echo "--- Creating app/models if missing ---" && \
    mkdir -p app/models && \
    cp -r models/* app/models/ 2>/dev/null || echo "Models already in place"

# Keep original structure - no need to move files

# Debug: Check what we actually have in the app directory
RUN echo "=== After fixing models directory ===" && \
    echo "=== App directory ===" && ls -la app/ && \
    echo "=== App/models directory ===" && ls -la app/models/ && \
    echo "=== App/models __init__.py ===" && cat app/models/__init__.py

# Verify Python path and package structure
RUN python -c "import sys; print('Python path:', sys.path)" && \
    python -c "import app; print('App package imported successfully')" && \
    python -c "print('App dir contents:', dir(app))" && \
    python -c "from app.models.transcription import TranscriptionRequest; print('TranscriptionRequest imported successfully')" && \
    python -c "print('All imports verified successfully')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
