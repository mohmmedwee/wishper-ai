"""
Whisper transcription utilities
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path
import structlog

from faster_whisper import WhisperModel
from app.models.transcription import TranscriptionRequest

logger = structlog.get_logger(__name__)

class WhisperProcessor:
    """Handles Whisper transcription processing"""
    
    def __init__(self):
        self.default_options = {
            "beam_size": 5,
            "best_of": 5,
            "patience": 1,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "no_speech_threshold": 0.6,
            "log_prob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": True,
            "initial_prompt": None,
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": [-1],
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "word_timestamps": True,
            "prepend_punctuations": "\"'"¿([{-",
            "append_punctuations": "\"'.。,，!！?？:：")]}、",
        }
    
    async def transcribe(
        self,
        model: WhisperModel,
        audio_file: Path,
        request: TranscriptionRequest
    ) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        
        try:
            logger.info("Starting Whisper transcription", file=str(audio_file))
            start_time = time.time()
            
            # Prepare transcription options
            options = self._prepare_options(request)
            
            # Run transcription
            segments, info = model.transcribe(
                str(audio_file),
                **options
            )
            
            # Process results
            result = await self._process_transcription_result(
                segments, info, request, start_time
            )
            
            logger.info("Whisper transcription completed", 
                       file=str(audio_file),
                       language=result.get("language"),
                       processing_time=result.get("processing_time"))
            
            return result
            
        except Exception as e:
            logger.error("Failed to transcribe with Whisper", 
                        error=str(e), file=str(audio_file))
            raise
    
    def _prepare_options(self, request: TranscriptionRequest) -> Dict[str, Any]:
        """Prepare transcription options from request"""
        
        options = self.default_options.copy()
        
        # Override with request parameters
        if request.language:
            options["language"] = request.language
        
        if request.task == "translate":
            options["task"] = "translate"
        
        if request.suppress_numerals:
            options["suppress_tokens"].extend([i for i in range(0, 10)])
        
        if request.word_timestamps:
            options["word_timestamps"] = True
        else:
            options["word_timestamps"] = False
        
        # Batch size
        if request.batch_size:
            options["batch_size"] = request.batch_size
        
        return options
    
    async def _process_transcription_result(
        self,
        segments,
        info,
        request: TranscriptionRequest,
        start_time: float
    ) -> Dict[str, Any]:
        """Process Whisper transcription results"""
        
        try:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract text and segments
            full_text = ""
            processed_segments = []
            
            for segment in segments:
                # Extract segment data
                segment_data = {
                    "id": len(processed_segments),
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "confidence": getattr(segment, 'avg_logprob', None)
                }
                
                # Add word timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    segment_data["words"] = []
                    for word in segment.words:
                        word_data = {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "confidence": getattr(word, 'probability', None)
                        }
                        segment_data["words"].append(word_data)
                
                processed_segments.append(segment_data)
                full_text += segment.text + " "
            
            # Clean up text
            full_text = full_text.strip()
            
            # Create result
            result = {
                "text": full_text,
                "segments": processed_segments,
                "language": info.language,
                "processing_time": processing_time,
                "model_used": request.whisper_model or "default"
            }
            
            return result
            
        except Exception as e:
            logger.error("Failed to process transcription result", error=str(e))
            raise
    
    async def validate_audio_file(self, audio_file: Path) -> bool:
        """Validate audio file for transcription"""
        
        try:
            # Check if file exists
            if not audio_file.exists():
                logger.error("Audio file does not exist", file=str(audio_file))
                return False
            
            # Check file size
            file_size = audio_file.stat().st_size
            if file_size == 0:
                logger.error("Audio file is empty", file=str(audio_file))
                return False
            
            # Check file extension
            supported_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
            if audio_file.suffix.lower() not in supported_extensions:
                logger.error("Unsupported audio format", 
                           file=str(audio_file), 
                           extension=audio_file.suffix)
                return False
            
            logger.debug("Audio file validation passed", file=str(audio_file))
            return True
            
        except Exception as e:
            logger.error("Failed to validate audio file", 
                        error=str(e), file=str(audio_file))
            return False
    
    async def get_available_models(self) -> list:
        """Get list of available Whisper models"""
        
        models = [
            "tiny", "tiny.en",
            "base", "base.en", 
            "small", "small.en",
            "medium", "medium.en",
            "large", "large-v2", "large-v3"
        ]
        
        return models
