"""
Audio processing utilities for the diarization service
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import structlog

import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize

logger = structlog.get_logger(__name__)

class AudioProcessor:
    """Handles audio file preprocessing and conversion"""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    
    async def preprocess_audio(
        self, 
        audio_file: Path, 
        transcription_id: str
    ) -> Path:
        """Preprocess audio file for optimal transcription"""
        
        try:
            logger.info("Preprocessing audio file", file=str(audio_file))
            
            # Check if preprocessing is needed
            if not self._needs_preprocessing(audio_file):
                logger.info("No preprocessing needed", file=str(audio_file))
                return audio_file
            
            # Load audio
            audio = await self._load_audio(audio_file)
            
            # Apply preprocessing steps
            processed_audio = await self._apply_preprocessing(audio)
            
            # Save processed audio
            output_path = await self._save_processed_audio(
                processed_audio, 
                audio_file, 
                transcription_id
            )
            
            logger.info("Audio preprocessing completed", 
                       input_file=str(audio_file),
                       output_file=str(output_path))
            
            return output_path
            
        except Exception as e:
            logger.error("Failed to preprocess audio", error=str(e), file=str(audio_file))
            raise
    
    def _needs_preprocessing(self, audio_file: Path) -> bool:
        """Check if audio file needs preprocessing"""
        
        # Check file format
        if audio_file.suffix.lower() not in self.supported_formats:
            return True
        
        # Check if it's already WAV format
        if audio_file.suffix.lower() == '.wav':
            return False
        
        return True
    
    async def _load_audio(self, audio_file: Path) -> AudioSegment:
        """Load audio file asynchronously"""
        
        try:
            # Use pydub for loading
            audio = AudioSegment.from_file(str(audio_file))
            return audio
            
        except Exception as e:
            logger.error("Failed to load audio file", error=str(e), file=str(audio_file))
            raise
    
    async def _apply_preprocessing(self, audio: AudioSegment) -> AudioSegment:
        """Apply preprocessing steps to audio"""
        
        try:
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                logger.debug("Converted audio to mono")
            
            # Normalize audio levels
            audio = normalize(audio)
            logger.debug("Normalized audio levels")
            
            # Ensure sample rate is 16kHz (optimal for Whisper)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                logger.debug("Set sample rate to 16kHz")
            
            return audio
            
        except Exception as e:
            logger.error("Failed to apply preprocessing", error=str(e))
            raise
    
    async def _save_processed_audio(
        self, 
        audio: AudioSegment, 
        original_file: Path, 
        transcription_id: str
    ) -> Path:
        """Save processed audio to temporary file"""
        
        try:
            # Create temporary file
            temp_dir = Path(tempfile.gettempdir())
            output_filename = f"{transcription_id}_processed.wav"
            output_path = temp_dir / output_filename
            
            # Export as WAV
            audio.export(
                str(output_path),
                format="wav",
                parameters=["-ar", "16000", "-ac", "1"]
            )
            
            logger.debug("Saved processed audio", output_path=str(output_path))
            return output_path
            
        except Exception as e:
            logger.error("Failed to save processed audio", error=str(e))
            raise
    
    async def get_audio_info(self, audio_file: Path) -> dict:
        """Get audio file information"""
        
        try:
            # Load audio with librosa for metadata
            y, sr = librosa.load(str(audio_file), sr=None)
            
            # Get duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get basic info
            info = {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else 2,
                "file_size": audio_file.stat().st_size,
                "format": audio_file.suffix.lower()
            }
            
            return info
            
        except Exception as e:
            logger.error("Failed to get audio info", error=str(e), file=str(audio_file))
            raise
    
    async def extract_audio_segment(
        self, 
        audio_file: Path, 
        start_time: float, 
        end_time: float
    ) -> AudioSegment:
        """Extract a segment from audio file"""
        
        try:
            # Load audio
            audio = await self._load_audio(audio_file)
            
            # Convert times to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Extract segment
            segment = audio[start_ms:end_ms]
            
            return segment
            
        except Exception as e:
            logger.error("Failed to extract audio segment", 
                        error=str(e), 
                        start_time=start_time, 
                        end_time=end_time)
            raise
