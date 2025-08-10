#!/usr/bin/env python3
"""
Enhanced CLI for Whisper Diarization Service
Provides access to advanced features like source separation, parallel processing, and language detection
"""

import click
import requests
import json
import time
from pathlib import Path
from typing import Optional, List
import asyncio
import aiohttp
import aiofiles

# Service configuration
BASE_URL = "http://localhost:80"
ENHANCED_API_BASE = f"{BASE_URL}/api/v2"

@click.group()
def cli():
    """Enhanced Whisper Diarization Service CLI"""
    pass

@cli.command()
@click.option('--file', '-f', required=True, help='Audio file path')
@click.option('--language', '-l', help='Language code (auto-detect if not specified)')
@click.option('--source-separation/--no-source-separation', default=True, 
              help='Enable source separation for better quality')
@click.option('--parallel-processing/--no-parallel-processing', default=True,
              help='Enable parallel processing for faster results')
@click.option('--enhanced-alignment/--no-enhanced-alignment', default=True,
              help='Enable enhanced alignment and punctuation')
@click.option('--whisper-model', help='Whisper model size')
@click.option('--suppress-numerals/--no-suppress-numerals', default=True,
              help='Suppress numerals for better alignment')
@click.option('--output-format', '-o', default='json', 
              type=click.Choice(['json', 'srt', 'vtt', 'txt']),
              help='Output format')
@click.option('--output-dir', '-d', default='./outputs',
              help='Output directory')
def transcribe(file, language, source_separation, parallel_processing, 
               enhanced_alignment, whisper_model, suppress_numerals, 
               output_format, output_dir):
    """Enhanced transcription with advanced features"""
    
    click.echo("🎵 Starting enhanced transcription...")
    
    # Check if file exists
    file_path = Path(file)
    if not file_path.exists():
        click.echo(f"❌ Error: File {file} not found")
        return
    
    # Prepare API parameters
    params = {
        'source_separation': source_separation,
        'parallel_processing': parallel_processing,
        'enhanced_alignment': enhanced_alignment,
        'suppress_numerals': suppress_numerals
    }
    
    if language:
        params['language'] = language
    if whisper_model:
        params['whisper_model'] = whisper_model
    
    try:
        # Upload file and start transcription
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'audio/wav')}
            
            click.echo("📤 Uploading file...")
            response = requests.post(
                f"{ENHANCED_API_BASE}/transcribe",
                files=files,
                params=params,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                click.echo("✅ Transcription completed successfully!")
                click.echo(f"📊 Results:")
                click.echo(f"   - Transcription ID: {result['transcription_id']}")
                click.echo(f"   - Language: {result['language']}")
                click.echo(f"   - Duration: {result['total_duration']:.2f}s")
                click.echo(f"   - Speakers: {result['speaker_count']}")
                click.echo(f"   - Segments: {result['segments_count']}")
                click.echo(f"   - Features used: {', '.join([k for k, v in result['features_used'].items() if v])}")
                
                # Save result
                output_path = Path(output_dir) / f"{file_path.stem}_enhanced.{output_format}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                click.echo(f"💾 Result saved to: {output_path}")
                
            else:
                click.echo(f"❌ Transcription failed: {response.status_code}")
                click.echo(f"Error: {response.text}")
                
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@cli.command()
@click.option('--files', '-f', required=True, multiple=True, help='Audio files (can specify multiple)')
@click.option('--language', '-l', help='Language code for all files')
@click.option('--source-separation/--no-source-separation', default=True,
              help='Enable source separation')
@click.option('--parallel-processing/--no-parallel-processing', default=True,
              help='Enable parallel processing')
@click.option('--enhanced-alignment/--no-enhanced-alignment', default=True,
              help='Enable enhanced alignment')
@click.option('--output-dir', '-d', default='./outputs',
              help='Output directory')
def batch_transcribe(files, language, source_separation, parallel_processing, 
                     enhanced_alignment, output_dir):
    """Enhanced batch transcription with advanced features"""
    
    click.echo(f"🎵 Starting enhanced batch transcription for {len(files)} files...")
    
    # Check if files exist
    file_paths = [Path(f) for f in files]
    for file_path in file_paths:
        if not file_path.exists():
            click.echo(f"❌ Error: File {file_path} not found")
            return
    
    # Prepare API parameters
    params = {
        'source_separation': source_separation,
        'parallel_processing': parallel_processing,
        'enhanced_alignment': enhanced_alignment
    }
    
    if language:
        params['language'] = language
    
    try:
        # Prepare files for upload
        file_data = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                file_data.append(('files', (file_path.name, f.read(), 'audio/wav')))
        
        # Start batch transcription
        click.echo("📤 Starting batch transcription...")
        response = requests.post(
            f"{ENHANCED_API_BASE}/transcribe/batch",
            files=file_data,
            params=params,
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            click.echo("✅ Batch transcription completed!")
            click.echo(f"📊 Batch Results:")
            click.echo(f"   - Batch ID: {result['batch_id']}")
            click.echo(f"   - Total files: {result['total_files']}")
            click.echo(f"   - Successful: {result['successful']}")
            click.echo(f"   - Failed: {result['failed']}")
            
            # Show individual results
            for file_result in result['results']:
                status_icon = "✅" if file_result['status'] == 'completed' else "❌"
                click.echo(f"   {status_icon} {file_result['filename']}")
                if file_result['status'] == 'completed':
                    click.echo(f"      - Language: {file_result['language']}")
                    click.echo(f"      - Duration: {file_result['duration']:.2f}s")
                    click.echo(f"      - Speakers: {file_result['speakers']}")
                else:
                    click.echo(f"      - Error: {file_result['error']}")
            
            # Save batch result
            output_path = Path(output_dir) / f"batch_{int(time.time())}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            click.echo(f"💾 Batch result saved to: {output_path}")
            
        else:
            click.echo(f"❌ Batch transcription failed: {response.status_code}")
            click.echo(f"Error: {response.text}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@cli.command()
def features():
    """Show available enhanced features"""
    
    try:
        click.echo("🔍 Checking enhanced features...")
        response = requests.get(f"{ENHANCED_API_BASE}/features", timeout=10)
        
        if response.status_code == 200:
            features = response.json()
            click.echo("✅ Enhanced Features Available:")
            click.echo(f"   - Source Separation: {'✅' if features['enhanced_features']['source_separation'] else '❌'}")
            click.echo(f"   - Parallel Processing: {'✅' if features['enhanced_features']['parallel_processing'] else '❌'}")
            click.echo(f"   - Enhanced Alignment: {'✅' if features['enhanced_features']['enhanced_alignment'] else '❌'}")
            click.echo(f"   - Language Detection: {'✅' if features['enhanced_features']['language_detection'] else '❌'}")
            click.echo(f"   - Multilingual Punctuation: {'✅' if features['enhanced_features']['multilingual_punctuation'] else '❌'}")
            click.echo(f"   - GPU Support: {'✅' if features['enhanced_features']['gpu_support'] else '❌'}")
            
            click.echo(f"\n🌍 Supported Languages: {len(features['supported_languages'])}")
            click.echo(f"   {', '.join(features['supported_languages'][:10])}{'...' if len(features['supported_languages']) > 10 else ''}")
            
            click.echo(f"\n📝 Punctuation Languages: {len(features['punctuation_languages'])}")
            click.echo(f"   {', '.join(features['punctuation_languages'])}")
            
            click.echo(f"\n⚙️  Configuration:")
            click.echo(f"   - Whisper Model: {features['configuration']['whisper_model']}")
            click.echo(f"   - Device: {features['configuration']['device']}")
            click.echo(f"   - GPU Available: {'✅' if features['configuration']['gpu_available'] else '❌'}")
            
        else:
            click.echo(f"❌ Failed to get features: {response.status_code}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@cli.command()
def health():
    """Check enhanced service health"""
    
    try:
        click.echo("🏥 Checking enhanced service health...")
        response = requests.get(f"{ENHANCED_API_BASE}/health", timeout=10)
        
        if response.status_code == 200:
            health_info = response.json()
            click.echo("✅ Enhanced Service is Healthy!")
            click.echo(f"   - Status: {health_info['status']}")
            click.echo(f"   - Version: {health_info['version']}")
            click.echo(f"   - GPU Available: {'✅' if health_info['gpu_available'] else '❌'}")
            
            click.echo(f"\n🔧 Models Loaded:")
            for model, loaded in health_info['models_loaded'].items():
                status_icon = "✅" if loaded else "❌"
                click.echo(f"   - {model.title()}: {status_icon}")
                
        elif response.status_code == 503:
            click.echo("❌ Enhanced Service is Unhealthy!")
            click.echo("   - Service not available or not initialized")
        else:
            click.echo(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@cli.command()
def info():
    """Show service information"""
    
    click.echo("🚀 Enhanced Whisper Diarization Service")
    click.echo("=" * 50)
    click.echo("Advanced features for production use:")
    click.echo("   • Source separation using Demucs")
    click.echo("   • Parallel processing for faster results")
    click.echo("   • Enhanced alignment with CTC forced aligner")
    click.echo("   • Multilingual punctuation and language detection")
    click.echo("   • GPU acceleration support")
    click.echo("")
    click.echo("API Endpoints:")
    click.echo(f"   • Enhanced API: {ENHANCED_API_BASE}")
    click.echo(f"   • Documentation: {BASE_URL}/docs")
    click.echo("")
    click.echo("Usage Examples:")
    click.echo("   • Enhanced transcription: enhanced-cli transcribe -f audio.wav")
    click.echo("   • Batch processing: enhanced-cli batch-transcribe -f file1.wav -f file2.wav")
    click.echo("   • Check features: enhanced-cli features")
    click.echo("   • Health check: enhanced-cli health")

if __name__ == '__main__':
    cli()
