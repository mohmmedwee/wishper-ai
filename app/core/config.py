"""
Configuration settings for the Whisper Diarization Service
"""

import os
from typing import List
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings"""
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=80, env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"],
        env="ALLOWED_ORIGINS"
    )
    
    # Whisper settings
    WHISPER_MODEL: str = Field(default="medium.en", env="WHISPER_MODEL")
    WHISPER_DEVICE: str = Field(default="auto", env="WHISPER_DEVICE")
    WHISPER_BATCH_SIZE: int = Field(default=16, env="WHISPER_BATCH_SIZE")
    WHISPER_SUPPRESS_NUMERALS: bool = Field(default=True, env="WHISPER_SUPPRESS_NUMERALS")
    
    # NeMo settings
    NEMO_DEVICE: str = Field(default="auto", env="NEMO_DEVICE")
    NEMO_BATCH_SIZE: int = Field(default=32, env="NEMO_BATCH_SIZE")
    
    # Audio processing settings
    MAX_AUDIO_DURATION: int = Field(default=3600, env="MAX_AUDIO_DURATION")  # 1 hour
    SUPPORTED_FORMATS: List[str] = Field(
        default=["wav", "mp3", "m4a", "flac", "ogg"],
        env="SUPPORTED_FORMATS"
    )
    
    # Storage settings
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    OUTPUT_DIR: str = Field(default="./outputs", env="OUTPUT_DIR")
    MAX_FILE_SIZE: int = Field(default=500 * 1024 * 1024, env="MAX_FILE_SIZE")  # 500MB
    
    # Database settings
    DATABASE_URL: str = Field(default="sqlite:///./whisper_diarization.db", env="DATABASE_URL")
    
    # Redis settings
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # API settings
    API_KEY_HEADER: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    API_KEYS: List[str] = Field(default=[], env="API_KEYS")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    # Monitoring settings
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
