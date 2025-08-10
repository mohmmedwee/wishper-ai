#!/usr/bin/env python3
"""
Test script for the Whisper Diarization Service
Demonstrates the service capabilities and API usage
"""

import asyncio
import json
from pathlib import Path
from app.services.diarization_service import DiarizationService
from app.models.transcription import TranscriptionRequest

async def test_service():
    """Test the diarization service"""
    print("🚀 Testing Whisper Diarization Service")
    print("=" * 50)
    
    # Create service instance
    service = DiarizationService()
    
    try:
        # Initialize service
        print("📋 Initializing service...")
        await service.initialize()
        print("✅ Service initialized successfully")
        
        # Check capabilities
        print("\n🔍 Service Capabilities:")
        capabilities = service.get_supported_features()
        for feature, available in capabilities.items():
            status = "✅" if available else "❌"
            print(f"  {status} {feature}")
        
        # Check upgrade path
        print("\n📈 Upgrade Information:")
        upgrade_info = service.get_upgrade_path()
        for key, value in upgrade_info.items():
            print(f"  {key}: {value}")
        
        # Get available models and languages
        print("\n🎯 Available Models:")
        models = service.get_whisper_models()
        for model in models:
            print(f"  - {model}")
        
        print("\n🌍 Supported Languages:")
        languages = service.get_supported_languages()[:10]  # Show first 10
        for lang in languages:
            print(f"  - {lang}")
        print(f"  ... and {len(service.get_supported_languages()) - 10} more")
        
        # Test with mock audio file
        print("\n🎵 Testing Audio Processing:")
        mock_audio = Path("test_audio.wav")
        
        # Create a simple test audio file if it doesn't exist
        if not mock_audio.exists():
            print("  Creating test audio file...")
            import numpy as np
            import soundfile as sf
            
            # Generate 5 seconds of test audio
            sample_rate = 16000
            duration = 5.0
            samples = int(sample_rate * duration)
            audio_data = np.random.randn(samples) * 0.1  # Low volume noise
            
            sf.write(str(mock_audio), audio_data, sample_rate)
            print(f"  ✅ Created test audio: {mock_audio}")
        
        # Create transcription request
        request = TranscriptionRequest(
            language="en",
            whisper_model="base",
            output_format="json",
            enable_diarization=True,
            source_separation=False,
            enhanced_alignment=False
        )
        
        # Process audio
        print(f"  Processing {mock_audio}...")
        result = await service.process_audio(mock_audio, request)
        
        print("  ✅ Audio processing completed!")
        print(f"  📊 Results:")
        print(f"    - Duration: {result.audio_duration:.2f}s")
        print(f"    - Language: {result.language}")
        print(f"    - Speakers: {result.num_speakers}")
        print(f"    - Segments: {len(result.segments)}")
        print(f"    - Processing time: {result.processing_time:.2f}s")
        
        # Show first segment
        if result.segments:
            first_segment = result.segments[0]
            print(f"    - First segment: '{first_segment.text}' (Speaker: {first_segment.speaker})")
        
        # Cleanup
        await service.cleanup()
        print("\n🧹 Service cleanup completed")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_service())
