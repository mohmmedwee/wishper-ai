"""
Original diarize.py functionality from whisper-diarization repository
This module provides the exact same functionality as the original script
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

class OriginalDiarizer:
    """Original diarization functionality from whisper-diarization repo"""
    
    def __init__(self, 
                 hf_token: str = None,
                 whisper_model: str = "medium",
                 device: str = None):
        
        self.hf_token = hf_token
        self.whisper_model = whisper_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.whisper_model_instance = None
        self.diarization_pipeline = None
        
    def initialize(self):
        """Initialize Whisper and diarization models"""
        logger.info("Initializing models...")
        
        # Initialize Whisper
        logger.info(f"Loading Whisper model: {self.whisper_model}")
        self.whisper_model_instance = whisper.load_model(self.whisper_model)
        
        # Initialize diarization pipeline
        if self.hf_token:
            logger.info("Loading diarization pipeline...")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=self.hf_token
                )
                self.diarization_pipeline.to(torch.device(self.device))
                logger.info("Diarization pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load diarization pipeline: {e}")
                self.diarization_pipeline = None
        else:
            logger.warning("No Hugging Face token provided. Diarization will be disabled.")
            self.diarization_pipeline = None
    
    def process_audio(self, 
                     audio_path: str,
                     output_dir: str = None,
                     num_speakers: int = None,
                     min_speakers: int = None,
                     max_speakers: int = None) -> Dict:
        """
        Process audio file with transcription and diarization
        This is the exact functionality from the original diarize.py
        """
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create output directory
        if output_dir is None:
            output_dir = audio_path.parent / f"{audio_path.stem}_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing audio: {audio_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Step 1: Transcribe with Whisper
        logger.info("Running Whisper transcription...")
        transcription_result = self.whisper_model_instance.transcribe(
            str(audio_path),
            word_timestamps=True,
            verbose=True
        )
        
        # Step 2: Run speaker diarization
        speaker_segments = []
        if self.diarization_pipeline:
            logger.info("Running speaker diarization...")
            speaker_segments = self._run_diarization(
                audio_path, 
                num_speakers, 
                min_speakers, 
                max_speakers
            )
        
        # Step 3: Align speakers with transcription
        logger.info("Aligning speakers with transcription...")
        aligned_result = self._align_speakers(
            transcription_result, 
            speaker_segments
        )
        
        # Step 4: Save results
        self._save_results(aligned_result, output_dir, audio_path.stem)
        
        logger.info("Processing completed successfully!")
        return aligned_result
    
    def _run_diarization(self, 
                         audio_path: Path,
                         num_speakers: int = None,
                         min_speakers: int = None,
                         max_speakers: int = None) -> List[Dict]:
        """Run speaker diarization using pyannote.audio"""
        
        if not self.diarization_pipeline:
            return []
        
        try:
            # Prepare diarization parameters
            diarization_params = {}
            if num_speakers is not None:
                diarization_params["num_speakers"] = num_speakers
            if min_speakers is not None:
                diarization_params["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarization_params["max_speakers"] = max_speakers
            
            # Run diarization
            with ProgressHook() as hook:
                diarization = self.diarization_pipeline(
                    str(audio_path),
                    hook=hook,
                    **diarization_params
                )
            
            # Extract speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
            
            logger.info(f"Found {len(speaker_segments)} speaker segments")
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []
    
    def _align_speakers(self, 
                        transcription_result: Dict, 
                        speaker_segments: List[Dict]) -> Dict:
        """Align speaker segments with transcription segments"""
        
        if not speaker_segments:
            return transcription_result
        
        # Get transcription segments
        segments = transcription_result.get("segments", [])
        
        # Align each segment with the most overlapping speaker
        aligned_segments = []
        
        for segment in segments:
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            # Find the speaker that overlaps most with this segment
            best_speaker = None
            max_overlap = 0
            
            for speaker_seg in speaker_segments:
                speaker_start = speaker_seg["start"]
                speaker_end = speaker_seg["end"]
                
                # Calculate overlap
                overlap_start = max(segment_start, speaker_start)
                overlap_end = min(segment_end, speaker_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker_seg["speaker"]
            
            # Add speaker information to segment
            aligned_segment = segment.copy()
            aligned_segment["speaker"] = best_speaker or "unknown"
            aligned_segments.append(aligned_segment)
        
        # Create aligned result
        aligned_result = transcription_result.copy()
        aligned_result["segments"] = aligned_segments
        aligned_result["speaker_segments"] = speaker_segments
        
        return aligned_result
    
    def _save_results(self, result: Dict, output_dir: Path, base_name: str):
        """Save results in various formats"""
        
        # Save JSON result
        json_path = output_dir / f"{base_name}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON result: {json_path}")
        
        # Save SRT with speaker labels
        srt_path = output_dir / f"{base_name}_with_speakers.srt"
        self._save_srt_with_speakers(result, srt_path)
        logger.info(f"Saved SRT with speakers: {srt_path}")
        
        # Save VTT with speaker labels
        vtt_path = output_dir / f"{base_name}_with_speakers.vtt"
        self._save_vtt_with_speakers(result, vtt_path)
        logger.info(f"Saved VTT with speakers: {vtt_path}")
        
        # Save plain text with speaker labels
        txt_path = output_dir / f"{base_name}_with_speakers.txt"
        self._save_txt_with_speakers(result, txt_path)
        logger.info(f"Saved text with speakers: {txt_path}")
    
    def _save_srt_with_speakers(self, result: Dict, output_path: Path):
        """Save SRT format with speaker labels"""
        
        segments = result.get("segments", [])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments):
                speaker = segment.get("speaker", "unknown")
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                f.write(f"{i+1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{speaker}] {text}\n\n")
    
    def _save_vtt_with_speakers(self, result: Dict, output_path: Path):
        """Save VTT format with speaker labels"""
        
        segments = result.get("segments", [])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in segments:
                speaker = segment.get("speaker", "unknown")
                start_time = self._format_timestamp_vtt(segment["start"])
                end_time = self._format_timestamp_vtt(segment["end"])
                text = segment["text"].strip()
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{speaker}] {text}\n\n")
    
    def _save_txt_with_speakers(self, result: Dict, output_path: Path):
        """Save plain text with speaker labels"""
        
        segments = result.get("segments", [])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                speaker = segment.get("speaker", "unknown")
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                f.write(f"[{start_time} --> {end_time}] [{speaker}] {text}\n")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format seconds to VTT timestamp format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization using Whisper and pyannote.audio"
    )
    
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--hf-token", help="Hugging Face token for pyannote.audio")
    parser.add_argument("--whisper-model", default="medium", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--num-speakers", type=int, help="Number of speakers")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize diarizer
    diarizer = OriginalDiarizer(
        hf_token=args.hf_token,
        whisper_model=args.whisper_model,
        device=args.device
    )
    
    try:
        diarizer.initialize()
        result = diarizer.process_audio(
            audio_path=args.audio_path,
            output_dir=args.output_dir,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        
        print("Processing completed successfully!")
        print(f"Output saved to: {args.output_dir or Path(args.audio_path).parent}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
