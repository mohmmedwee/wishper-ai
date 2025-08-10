"""
Core package for the Whisper Diarization Service
"""

from .config import settings
from .logging import setup_logging

__all__ = ["settings", "setup_logging"]
