#!/usr/bin/env python3
"""
Download NeMo models for speaker diarization
This script downloads the required models that are referenced in the diarization config
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import logging

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from app.core.config import settings

def download_nemo_model(model_name: str, output_dir: str = None):
    """Download a NeMo model using NGC CLI or direct download"""
    
    if output_dir is None:
        output_dir = Path("models")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {model_name} to {output_dir}")
    
    try:
        # Try using NGC CLI first
        cmd = [
            "ngc", "model", "download", 
            "--source", "nvidia", 
            "--model", model_name,
            "--dest", str(output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully downloaded {model_name} using NGC CLI")
            return True
        else:
            print(f"NGC CLI failed: {result.stderr}")
            print("Trying alternative download method...")
            
            # Alternative: Direct download from Hugging Face or other sources
            return download_model_alternative(model_name, output_dir)
            
    except FileNotFoundError:
        print("NGC CLI not found. Trying alternative download method...")
        return download_model_alternative(model_name, output_dir)
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def download_model_alternative(model_name: str, output_dir: str):
    """Alternative download method for models"""
    
    # Map of model names to their download URLs or alternative sources
    model_sources = {
        "vad_multilingual_marblenet": {
            "url": "https://huggingface.co/nvidia/vad_multilingual_marblenet",
            "type": "huggingface"
        },
        "titanet_large": {
            "url": "https://huggingface.co/nvidia/titanet_large",
            "type": "huggingface"
        },
        "ecapa_tdnn": {
            "url": "https://huggingface.co/nvidia/ecapa_tdnn",
            "type": "huggingface"
        }
    }
    
    if model_name not in model_sources:
        print(f"Unknown model: {model_name}")
        return False
    
    source = model_sources[model_name]
    
    if source["type"] == "huggingface":
        try:
            # Use git lfs to download from Hugging Face
            model_dir = output_dir / model_name
            if model_dir.exists():
                print(f"Model {model_name} already exists at {model_dir}")
                return True
            
            cmd = [
                "git", "clone", 
                source["url"], 
                str(model_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully downloaded {model_name} from Hugging Face")
                return True
            else:
                print(f"Failed to download from Hugging Face: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
            return False
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Download NeMo models for diarization")
    parser.add_argument("--models", nargs="+", default=[
        "vad_multilingual_marblenet",
        "titanet_large"
    ], help="Models to download")
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    
    args = parser.parse_args()
    
    print("Downloading NeMo models for speaker diarization...")
    print(f"Output directory: {args.output_dir}")
    
    success_count = 0
    total_count = len(args.models)
    
    for model in args.models:
        print(f"\n--- Downloading {model} ---")
        if download_nemo_model(model, args.output_dir):
            success_count += 1
        else:
            print(f"Failed to download {model}")
    
    print(f"\n--- Download Summary ---")
    print(f"Successfully downloaded: {success_count}/{total_count} models")
    
    if success_count == total_count:
        print("All models downloaded successfully!")
        print("\nNext steps:")
        print("1. Update the config/diarization_inference.yaml file with correct model paths")
        print("2. Test the diarization service")
    else:
        print("Some models failed to download. Check the errors above.")
        print("You may need to manually download the models or check your internet connection.")

if __name__ == "__main__":
    main()
