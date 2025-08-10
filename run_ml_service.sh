#!/bin/bash

# Whisper Diarization ML Service Runner
# This script runs the full ML-enabled service using Docker

set -e

echo "🚀 Starting Whisper Diarization ML Service..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if we have the required files
if [ ! -f "Dockerfile" ] || [ ! -f "docker-compose.yml" ]; then
    echo "❌ Required files not found. Please ensure Dockerfile and docker-compose.yml exist."
    exit 1
fi

# Build and start the service
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Service is running and healthy!"
    echo ""
    echo "🌐 Service URL: http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo "📁 Uploads directory: ./uploads"
    echo "📁 Outputs directory: ./outputs"
    echo ""
    echo "📋 Available endpoints:"
    echo "   POST /api/transcribe - Upload and transcribe audio"
    echo "   GET  /api/health     - Service health check"
    echo "   GET  /api/status     - Service status and capabilities"
    echo ""
    echo "🛑 To stop the service: docker-compose down"
    echo "📊 To view logs: docker-compose logs -f"
else
    echo "❌ Service health check failed. Check logs with: docker-compose logs"
    exit 1
fi
