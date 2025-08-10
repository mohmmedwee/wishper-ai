#!/bin/bash

# Enhanced Whisper Diarization Service Startup Script
# This script starts the service with proper environment setup

echo "🚀 Starting Enhanced Whisper Diarization Service..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "🔧 Checking dependencies..."
pip install -r requirements.txt --quiet

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads outputs models

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PORT=${PORT:-8000}

# Start the service
echo "🌟 Starting service on port $PORT..."
echo "📖 API Documentation: http://localhost:$PORT/docs"
echo "🔍 Health Check: http://localhost:$PORT/health"
echo "⚡ Features: http://localhost:$PORT/api/v1/features"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload
