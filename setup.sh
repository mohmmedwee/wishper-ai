#!/bin/bash

# Whisper Diarization Service Setup Script

echo "🚀 Setting up Whisper Diarization Service for your company..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.10+ first"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if FFmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  Warning: FFmpeg is not installed"
    echo "FFmpeg is required for audio processing"
    echo "Install with: brew install ffmpeg (macOS) or sudo apt install ffmpeg (Ubuntu)"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads outputs models

# Set environment variables
export PYTORCH_HOME="./models"
export HF_HOME="./models"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Copy config.env to .env and customize settings:"
echo "   cp config.env .env"
echo ""
echo "2. Start the service with Docker (recommended):"
echo "   docker-compose up -d"
echo ""
echo "3. Or start manually:"
echo "   source venv/bin/activate"
echo "   python -m app.main"
echo ""
echo "4. Test the API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "5. Use the CLI tool:"
echo "   source venv/bin/activate"
echo "   python cli.py --help"
echo ""
echo "📖 For more information, see README.md"
echo "🔧 Configuration file: config.env"
echo "🐳 Docker setup: docker-compose.yml"
