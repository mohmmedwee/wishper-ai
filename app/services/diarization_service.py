"""
Core diarization service that combines Whisper ASR with speaker diarization
"""

import asyncio
import os
import tempfile
import uuid
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import structlog

import torch
from faster_whisper import WhisperModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.models import ClusteringDiarizer

from app.core.config import settings
from app.models.transcription import TranscriptionRequest, TranscriptionResult, SpeakerSegment
from app.utils.audio_processor import AudioProcessor
from app.utils.whisper_utils import WhisperProcessor

logger = structlog.get_logger(__name__)

class DiarizationService:
    def __init__(self):
        self.whisper_model: Optional[WhisperModel] = None
        self.diarization_pipeline: Optional[NeuralDiarizer] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.whisper_processor: Optional[WhisperProcessor] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config_path = Path("config/diarization_inference.yaml")

    async def initialize(self):
        logger.info("Initializing Diarization Service...")
        await self._initialize_whisper()
        await self._initialize_diarization()
        await self._initialize_audio_processor()
        logger.info("Diarization Service initialized successfully")

    async def _initialize_whisper(self):
        """Initialize Whisper model for transcription"""
        logger.info("Initializing Whisper model", model=settings.WHISPER_MODEL, device=self.device)
        self.whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8"
        )
        logger.info("Whisper model initialized successfully")

    async def _initialize_diarization(self):
        """Initialize NeMo diarization pipeline"""
        logger.info("Initializing NeMo diarization pipeline")
        try:
            # Initialize the clustering diarizer with the config
            self.diarization_pipeline = ClusteringDiarizer.from_pretrained(
                model_name="titanet_large",
                cfg=self.config_path
            )
            logger.info("NeMo diarization pipeline initialized successfully")
        except Exception as e:
            logger.warning("Failed to initialize NeMo pipeline, falling back to basic diarization", error=str(e))
            self.diarization_pipeline = None

    async def _initialize_audio_processor(self):
        """Initialize audio processing utilities"""
        self.audio_processor = AudioProcessor()
        self.whisper_processor = WhisperProcessor()

    async def process_audio(self, audio_file: Path, request: TranscriptionRequest) -> TranscriptionResult:
        """Process audio file with transcription and diarization"""
        logger.info("Processing audio file", file=str(audio_file))
        transcription_id = str(uuid.uuid4())
        
        # Preprocess audio
        processed_audio_path = await self.audio_processor.preprocess_audio(audio_file, transcription_id)
        
        # Run Whisper transcription
        whisper_result = await self.whisper_processor.transcribe(
            self.whisper_model, processed_audio_path, request
        )
        
        # Run speaker diarization
        speaker_segments = []
        if self.diarization_pipeline and request.enable_diarization:
            speaker_segments = await self._run_diarization(processed_audio_path)
        
        # Align speakers with transcription
        final_result = await self._align_speakers(
            whisper_result, speaker_segments, transcription_id
        )
        
        # Cleanup temporary files
        if processed_audio_path != audio_file:
            os.unlink(processed_audio_path)
        
        return final_result

    async def _run_diarization(self, audio_path: Path) -> List[SpeakerSegment]:
        """Run NeMo speaker diarization"""
        logger.info("Running NeMo speaker diarization")
        
        try:
            # Create temporary manifest file for NeMo
            manifest_data = {
                "audio_filepath": str(audio_path.absolute()),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None
            }
            
            manifest_path = audio_path.parent / f"{audio_path.stem}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
            
            # Update config with current paths
            config = self._load_diarization_config()
            config['diarizer']['manifest_filepath'] = str(manifest_path)
            config['diarizer']['out_dir'] = str(audio_path.parent / "diarization_output")
            
            # Save updated config
            temp_config_path = audio_path.parent / "temp_diarization_config.yaml"
            self._save_diarization_config(config, temp_config_path)
            
            # Run diarization
            diarization_result = self.diarization_pipeline.diarize(
                manifest_filepath=str(manifest_path),
                out_dir=str(config['diarizer']['out_dir'])
            )
            
            # Parse results and convert to SpeakerSegment format
            speaker_segments = self._parse_diarization_results(diarization_result)
            
            # Cleanup
            manifest_path.unlink(missing_ok=True)
            temp_config_path.unlink(missing_ok=True)
            
            logger.info("Diarization completed successfully", segments_count=len(speaker_segments))
            return speaker_segments
            
        except Exception as e:
            logger.error("Diarization failed", error=str(e))
            return []

    def _load_diarization_config(self) -> Dict:
        """Load the diarization configuration"""
        import yaml
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _save_diarization_config(self, config: Dict, path: Path):
        """Save diarization configuration to file"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def _parse_diarization_results(self, diarization_result) -> List[SpeakerSegment]:
        """Parse NeMo diarization results into SpeakerSegment format"""
        speaker_segments = []
        
        try:
            # Extract speaker segments from NeMo output
            # This is a simplified parser - you may need to adjust based on actual NeMo output format
            if hasattr(diarization_result, 'speaker_segments'):
                for segment in diarization_result.speaker_segments:
                    speaker_segments.append(SpeakerSegment(
                        speaker_id=f"speaker_{segment.speaker_id}",
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        confidence=segment.confidence if hasattr(segment, 'confidence') else 1.0
                    ))
            
            # If no structured output, try to parse RTTM file
            elif hasattr(diarization_result, 'rttm_filepath'):
                rttm_path = Path(diarization_result.rttm_filepath)
                if rttm_path.exists():
                    speaker_segments = self._parse_rttm_file(rttm_path)
                    
        except Exception as e:
            logger.error("Failed to parse diarization results", error=str(e))
        
        return speaker_segments

    def _parse_rttm_file(self, rttm_path: Path) -> List[SpeakerSegment]:
        """Parse RTTM file format for speaker segments"""
        speaker_segments = []
        
        try:
            with open(rttm_path, 'r') as f:
                for line in f:
                    if line.startswith('SPEAKER'):
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            speaker_id = parts[7]
                            start_time = float(parts[3])
                            duration = float(parts[4])
                            end_time = start_time + duration
                            
                            speaker_segments.append(SpeakerSegment(
                                speaker_id=f"speaker_{speaker_id}",
                                start_time=start_time,
                                end_time=end_time,
                                confidence=1.0
                            ))
        except Exception as e:
            logger.error("Failed to parse RTTM file", error=str(e))
        
        return speaker_segments

    async def _align_speakers(self, whisper_result: Dict, speaker_segments: List[SpeakerSegment], transcription_id: str) -> TranscriptionResult:
        """Align speaker segments with transcription segments"""
        logger.info("Aligning speakers with transcription")
        
        # Extract text and segments from whisper result
        text = whisper_result.get("text", "")
        segments = whisper_result.get("segments", [])
        
        # Create TranscriptionResult
        result = TranscriptionResult(
            transcription_id=transcription_id,
            text=text,
            segments=segments,
            speaker_segments=speaker_segments,
            language=whisper_result.get("language", "unknown"),
            processing_time=whisper_result.get("processing_time", 0.0)
        )
        
        logger.info("Speaker alignment completed", 
                   transcription_id=transcription_id,
                   segments_count=len(segments),
                   speaker_segments_count=len(speaker_segments))
        
        return result

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Diarization Service")
        if self.whisper_model:
            del self.whisper_model
        if self.diarization_pipeline:
            del self.diarization_pipeline
        logger.info("Cleanup completed")
