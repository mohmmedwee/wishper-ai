#!/usr/bin/env python3
"""
Model management script for the Whisper Diarization Service
This script helps download, verify, and manage NeMo models
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import json

def check_model_status(model_dir: str = "models"):
    """Check the status of downloaded models"""
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        print(f"❌ Models directory {model_dir} does not exist")
        return False
    
    print(f"📁 Checking models in: {model_dir}")
    print("-" * 50)
    
    models = {
        "vad_multilingual_marblenet": "Voice Activity Detection",
        "titanet_large": "Speaker Embedding (Large)",
        "ecapa_tdnn": "Speaker Embedding (ECAPA-TDNN)"
    }
    
    status = {}
    
    for model_name, description in models.items():
        model_path = model_dir / model_name
        nemo_file = model_dir / f"{model_name}.nemo"
        
        if model_path.exists() and model_path.is_dir():
            # Check if it's a proper Hugging Face model
            config_file = model_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    model_type = config.get('model_type', 'Unknown')
                    print(f"✅ {model_name}: {description} ({model_type})")
                    status[model_name] = "downloaded"
                except:
                    print(f"⚠️  {model_name}: {description} (corrupted)")
                    status[model_name] = "corrupted"
            else:
                print(f"⚠️  {model_name}: {description} (incomplete)")
                status[model_name] = "incomplete"
        elif nemo_file.exists():
            # Check if it's a placeholder file
            try:
                with open(nemo_file, 'r') as f:
                    content = f.read()
                if content.startswith("# Placeholder"):
                    print(f"📝 {model_name}: {description} (placeholder)")
                    status[model_name] = "placeholder"
                else:
                    print(f"✅ {model_name}: {description} (.nemo file)")
                    status[model_name] = "nemo_file"
            except:
                print(f"❓ {model_name}: {description} (unknown)")
                status[model_name] = "unknown"
        else:
            print(f"❌ {model_name}: {description} (missing)")
            status[model_name] = "missing"
    
    return status

def download_missing_models(status: dict, model_dir: str = "models"):
    """Download missing models"""
    print("\n🔄 Downloading missing models...")
    
    missing_models = [name for name, status in status.items() if status in ["missing", "placeholder"]]
    
    if not missing_models:
        print("✅ All models are available!")
        return True
    
    print(f"📥 Models to download: {', '.join(missing_models)}")
    
    # Use the existing download script
    cmd = [
        sys.executable, "scripts/download_nemo_models.py",
        "--models", *missing_models,
        "--output-dir", model_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Models downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download models: {e}")
        print(f"Error output: {e.stderr}")
        return False

def verify_models(model_dir: str = "models"):
    """Verify that models are working correctly"""
    print("\n🔍 Verifying models...")
    
    # This would typically involve loading the models and running a quick test
    # For now, we'll just check file sizes and basic structure
    
    model_dir = Path(model_dir)
    
    for model_name in ["vad_multilingual_marblenet", "titanet_large", "ecapa_tdnn"]:
        model_path = model_dir / model_name
        nemo_file = model_dir / f"{model_name}.nemo"
        
        if model_path.exists() and model_path.is_dir():
            size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            print(f"📊 {model_name}: {size / (1024*1024):.1f} MB")
        elif nemo_file.exists():
            size = nemo_file.stat().st_size
            if size > 1000:  # More than 1KB (not a placeholder)
                print(f"📊 {model_name}: {size / (1024*1024):.1f} MB (.nemo)")
            else:
                print(f"📊 {model_name}: Placeholder file")
        else:
            print(f"📊 {model_name}: Not available")

def main():
    parser = argparse.ArgumentParser(description="Manage NeMo models for the Whisper Diarization Service")
    parser.add_argument("--check", action="store_true", help="Check model status")
    parser.add_argument("--download", action="store_true", help="Download missing models")
    parser.add_argument("--verify", action="store_true", help="Verify model integrity")
    parser.add_argument("--model-dir", default="models", help="Models directory")
    parser.add_argument("--all", action="store_true", help="Run all checks and downloads")
    
    args = parser.parse_args()
    
    if not any([args.check, args.download, args.verify, args.all]):
        args.all = True  # Default to running all
    
    print("🤖 NeMo Model Manager for Whisper Diarization Service")
    print("=" * 60)
    
    # Check status
    if args.check or args.all:
        status = check_model_status(args.model_dir)
    
    # Download missing models
    if args.download or args.all:
        if 'status' not in locals():
            status = check_model_status(args.model_dir)
        download_missing_models(status, args.model_dir)
    
    # Verify models
    if args.verify or args.all:
        verify_models(args.model_dir)
    
    print("\n🎯 Model management completed!")
    print("\nNext steps:")
    print("1. Start the service: ./run.sh docker start")
    print("2. Test transcription: curl -X POST http://localhost:80/api/v1/transcribe")
    print("3. Check service status: curl http://localhost:80/health")

if __name__ == "__main__":
    main()
