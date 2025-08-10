#!/bin/bash

# Whisper Diarization Service Startup Script

echo "Starting Whisper Diarization Service..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads outputs models

# Set environment variables
export PYTORCH_HOME="./models"
export HF_HOME="./models"

# Start the service
echo "Starting service..."
python -m app.main
