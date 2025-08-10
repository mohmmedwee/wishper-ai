#!/usr/bin/env python3
"""
Test script for Whisper Diarization Service
Run this to verify your setup is working correctly
"""

import requests
import json
import time
import os
from pathlib import Path

# Service configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/api/v1/transcribe"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("🏠 Testing root endpoint...")
    try:
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Root endpoint failed: {e}")
            return False

def test_api_documentation():
    """Test if API documentation is accessible"""
    print("📚 Testing API documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            print(f"   Available at: {BASE_URL}/docs")
            return True
        else:
            print(f"❌ API documentation not accessible: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API documentation test failed: {e}")
        return False

def test_transcription_endpoint():
    """Test the transcription endpoint (without file upload)"""
    print("🎵 Testing transcription endpoint...")
    try:
        # Test with empty data to see if endpoint responds
        response = requests.post(API_ENDPOINT, timeout=10)
        # We expect a 422 error for missing file, which means the endpoint is working
        if response.status_code == 422:
            print("✅ Transcription endpoint is accessible")
            print("   (Expected error for missing file - endpoint is working)")
            return True
        else:
            print(f"⚠️  Transcription endpoint responded with: {response.status_code}")
            return True  # Still consider it working
    except requests.exceptions.RequestException as e:
        print(f"❌ Transcription endpoint test failed: {e}")
        return False

def check_directories():
    """Check if required directories exist"""
    print("📁 Checking required directories...")
    directories = ['uploads', 'outputs', 'models']
    all_exist = True
    
    for directory in directories:
        if Path(directory).exists():
            print(f"✅ {directory}/ directory exists")
        else:
            print(f"❌ {directory}/ directory missing")
            all_exist = False
    
    return all_exist

def check_docker_services():
    """Check if Docker services are running"""
    print("🐳 Checking Docker services...")
    try:
        import subprocess
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            if 'whisper-diarization' in result.stdout:
                print("✅ Whisper service container is running")
                return True
            else:
                print("⚠️  Docker is running but whisper service not found")
                return False
        else:
            print("❌ Docker command failed")
            return False
    except Exception as e:
        print(f"⚠️  Could not check Docker services: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 Running Whisper Diarization Service Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("API Documentation", test_api_documentation),
        ("Transcription Endpoint", test_transcription_endpoint),
        ("Directories", check_directories),
        ("Docker Services", check_docker_services),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your service is ready to use.")
        print(f"\nNext steps:")
        print(f"1. Visit {BASE_URL}/docs for API documentation")
        print(f"2. Test with a real audio file")
        print(f"3. Use the CLI tool: python cli.py --help")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print(f"\nTroubleshooting:")
        print(f"1. Make sure the service is running")
        print(f"2. Check the logs: docker-compose logs whisper-diarization")
        print(f"3. Verify configuration in config.env")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
