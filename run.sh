#!/bin/bash

# Enhanced Whisper Diarization Service - Unified Setup & Run Script
# This script handles both initial setup and service startup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup the environment
setup_environment() {
    print_status "🚀 Setting up Enhanced Whisper Diarization Service..."
    
    # Check Python version
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        print_status "Please install Python 3.10+ first"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python 3 found: $PYTHON_VERSION"
    
    # Check FFmpeg
    if ! command_exists ffmpeg; then
        print_warning "FFmpeg is not installed"
        print_status "FFmpeg is required for audio processing"
        print_status "Install with: brew install ffmpeg (macOS) or sudo apt install ffmpeg (Ubuntu)"
    else
        print_success "FFmpeg found: $(ffmpeg -version | head -n1)"
    fi
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_status "📦 Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_status "🔧 Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "⬆️  Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install dependencies
    print_status "📚 Installing Python dependencies..."
    pip install -r requirements.txt --quiet
    
    # Create necessary directories
    print_status "📁 Creating necessary directories..."
    mkdir -p uploads outputs models
    
    # Set environment variables
    export PYTORCH_HOME="./models"
    export HF_HOME="./models"
    
    print_success "🎉 Setup completed successfully!"
}

# Function to start the service
start_service() {
    print_status "🌟 Starting Enhanced Whisper Diarization Service..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Running setup first..."
        setup_environment
    fi
    
    # Activate virtual environment
    print_status "📦 Activating virtual environment..."
    source venv/bin/activate
    
    # Check dependencies
    print_status "🔧 Checking dependencies..."
    pip install -r requirements.txt --quiet
    
    # Create necessary directories
    print_status "📁 Creating directories..."
    mkdir -p uploads outputs models
    
    # Set environment variables
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export PORT=${PORT:-80}
    
    # Check if .env exists, if not copy from config.env
    if [ ! -f ".env" ] && [ -f "config.env" ]; then
        print_status "📋 Creating .env file from config.env..."
        cp config.env .env
    fi
    
    print_success "Service starting on port $PORT"
    print_status "📖 API Documentation: http://localhost:$PORT/docs"
    print_status "🔍 Health Check: http://localhost:$PORT/health"
    print_status "⚡ Features: http://localhost:$PORT/api/v1/features"
    echo ""
    print_status "Press Ctrl+C to stop the service"
    echo ""
    
    python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload
}

# Function to show help
show_help() {
    echo "Enhanced Whisper Diarization Service - Unified Setup & Run Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup     - Setup the environment (install dependencies, create venv)"
    echo "  start     - Start the service (runs setup if needed)"
    echo "  docker    - Start with Docker Compose"
    echo "  stop      - Stop Docker services"
    echo "  logs      - Show Docker logs"
    echo "  clean     - Clean up Docker containers and volumes"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Setup environment only"
    echo "  $0 start     # Setup and start service"
    echo "  $0 docker    # Start with Docker"
    echo "  $0           # Default: setup and start"
    echo ""
    echo "Environment Variables:"
    echo "  PORT        - Service port (default: 80)"
    echo "  DEBUG       - Enable debug mode"
}

# Function to handle Docker operations
docker_operations() {
    case "$1" in
        "start")
            print_status "🐳 Starting services with Docker Compose..."
            docker-compose up -d
            print_success "Services started! Check status with: $0 logs"
            ;;
        "stop")
            print_status "🛑 Stopping Docker services..."
            docker-compose down
            print_success "Services stopped"
            ;;
        "logs")
            print_status "📋 Showing Docker logs..."
            docker-compose logs -f whisper-diarization
            ;;
        "clean")
            print_status "🧹 Cleaning up Docker containers and volumes..."
            docker-compose down -v --remove-orphans
            docker system prune -f
            print_success "Cleanup completed"
            ;;
        *)
            print_error "Unknown Docker operation: $1"
            show_help
            exit 1
            ;;
    esac
}

# Main script logic
case "${1:-start}" in
    "setup")
        setup_environment
        ;;
    "start")
        start_service
        ;;
    "docker")
        docker_operations "${2:-start}"
        ;;
    "stop")
        docker_operations "stop"
        ;;
    "logs")
        docker_operations "logs"
        ;;
    "clean")
        docker_operations "clean"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_status "No option specified, running setup and start..."
        setup_environment
        start_service
        ;;
esac
