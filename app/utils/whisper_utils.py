"""
Whisper Utilities for Audio Transcription
Simplified version for Python 3.13 compatibility
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class WhisperUtils:
    """Utility class for Whisper transcription operations"""
    
    def __init__(self):
        self.default_options = {
            "language": None,
            "task": "transcribe",
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "no_speech_threshold": 0.6,
            "log_prob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": True,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "initial_prompt": None,
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": [-1],
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "word_timestamps": True,
            "prepend_punctuations": "\"'([{-",
            "append_punctuations": "\"'.!?):]}",
        }
    
    async def validate_audio_file(self, audio_file: Path) -> bool:
        """Validate audio file for transcription"""
        try:
            if not audio_file.exists():
                logger.error(f"Audio file does not exist: {audio_file}")
                return False
            
            # Check file size
            file_size = audio_file.stat().st_size
            if file_size == 0:
                logger.error(f"Audio file is empty: {audio_file}")
                return False
            
            # Check file extension
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
            if audio_file.suffix.lower() not in supported_formats:
                logger.error(f"Unsupported audio format: {audio_file.suffix}")
                return False
            
            logger.info(f"Audio file validation passed: {audio_file}")
            return True
            
        except Exception as e:
            logger.error(f"Audio file validation failed: {e}")
            return False
    
    async def get_available_models(self) -> list:
        """Get list of available Whisper models"""
        # Note: This is a placeholder until ML packages are available
        return [
            "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        model_info = {
            "tiny": {"params": "39M", "multilingual": True, "english_only": False},
            "base": {"params": "74M", "multilingual": True, "english_only": False},
            "small": {"params": "244M", "multilingual": True, "english_only": False},
            "medium": {"params": "769M", "multilingual": True, "english_only": False},
            "large": {"params": "1550M", "multilingual": True, "english_only": False},
            "large-v2": {"params": "1550M", "multilingual": True, "english_only": False},
            "large-v3": {"params": "1550M", "multilingual": True, "english_only": False}
        }
        
        return model_info.get(model_name, {"params": "Unknown", "multilingual": False, "english_only": False})
    
    async def estimate_transcription_time(self, audio_duration: float, model_name: str = "base") -> float:
        """Estimate transcription time based on audio duration and model"""
        # Rough estimates based on model complexity
        model_multipliers = {
            "tiny": 0.1,
            "base": 0.2,
            "small": 0.5,
            "medium": 1.0,
            "large": 2.0,
            "large-v2": 2.0,
            "large-v3": 2.5
        }
        
        multiplier = model_multipliers.get(model_name, 0.5)
        estimated_time = audio_duration * multiplier
        
        # Add some overhead
        estimated_time += 5.0
        
        return estimated_time
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    
    def get_language_name(self, language_code: str) -> str:
        """Get full language name from language code"""
        language_names = {
            "en": "English",
            "zh": "Chinese",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "ko": "Korean",
            "fr": "French",
            "ja": "Japanese",
            "pt": "Portuguese",
            "tr": "Turkish",
            "pl": "Polish",
            "ca": "Catalan",
            "nl": "Dutch",
            "ar": "Arabic",
            "sv": "Swedish",
            "it": "Italian",
            "id": "Indonesian",
            "hi": "Hindi",
            "fi": "Finnish",
            "vi": "Vietnamese"
        }
        
        return language_names.get(language_code, language_code.upper())
