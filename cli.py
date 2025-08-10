#!/usr/bin/env python3
"""
Command-line interface for the Whisper Diarization Service
"""

import click
import asyncio
import json
from pathlib import Path
from app.services.diarization_service import DiarizationService
from app.models.transcription import TranscriptionRequest

@click.group()
def cli():
    """Whisper Diarization Service CLI"""
    pass

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--language', '-l', help='Language code (e.g., en, es)')
@click.option('--model', '-m', default='medium.en', help='Whisper model to use')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', default='json', type=click.Choice(['json', 'txt', 'srt']), help='Output format')
@click.option('--no-diarization', is_flag=True, help='Disable speaker diarization')
def transcribe(audio_file, language, model, output, format, no_diarization):
    """Transcribe audio file with speaker diarization"""
    
    async def run_transcription():
        try:
            # Initialize service
            service = DiarizationService()
            await service.initialize()
            
            # Create request
            request = TranscriptionRequest(
                language=language,
                whisper_model=model,
                enable_diarization=not no_diarization,
                output_format=format
            )
            
            # Process audio
            result = await service.process_audio(Path(audio_file), request)
            
            # Output result
            if output:
                output_path = Path(output)
                if format == 'json':
                    with open(output_path, 'w') as f:
                        json.dump(result.dict(), f, indent=2, default=str)
                elif format == 'txt':
                    with open(output_path, 'w') as f:
                        f.write(result.text)
                elif format == 'srt':
                    with open(output_path, 'w') as f:
                        for i, segment in enumerate(result.segments):
                            f.write(f"{i+1}\n")
                            f.write(f"{segment.start:02.0f}:{segment.start%1*60:05.2f} --> {segment.end:02.0f}:{segment.end%1*60:05.2f}\n")
                            f.write(f"{segment.text}\n\n")
                
                click.echo(f"Transcription saved to {output_path}")
            else:
                click.echo(json.dumps(result.dict(), indent=2, default=str))
            
            # Cleanup
            await service.cleanup()
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            return 1
        
        return 0
    
    # Run async function
    exit_code = asyncio.run(run_transcription())
    exit(exit_code)

@cli.command()
def models():
    """List available Whisper models"""
    models = [
        "tiny", "tiny.en",
        "base", "base.en", 
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v2", "large-v3"
    ]
    
    click.echo("Available Whisper models:")
    for model in models:
        click.echo(f"  - {model}")

@cli.command()
def info():
    """Show service information"""
    click.echo("Whisper Diarization Service")
    click.echo("Version: 1.0.0")
    click.echo("Description: Production-ready service for ASR with speaker diarization")
    click.echo("\nFeatures:")
    click.echo("  - OpenAI Whisper transcription")
    click.echo("  - Speaker diarization")
    click.echo("  - RESTful API")
    click.echo("  - Docker support")
    click.echo("  - Production ready")

if __name__ == '__main__':
    cli()
