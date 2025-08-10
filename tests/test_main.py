"""
Basic tests for the main application
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Whisper Diarization Service" in data["message"]

def test_health_endpoint():
    """Test the detailed health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "whisper-diarization"
    assert data["version"] == "1.0.0"

def test_api_docs():
    """Test that API documentation is available"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_schema():
    """Test that OpenAPI schema is available"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == "Whisper Diarization Service"
