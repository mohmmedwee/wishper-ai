"""
Output format converters for transcription results
Supports SRT, VTT, TXT, and JSON formats with speaker labels
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)

class OutputFormatConverter:
    """Convert transcription results to various output formats"""
    
    @staticmethod
    def to_srt(result: Dict[str, Any], include_speakers: bool = True) -> str:
        """Convert to SRT format"""
        segments = result.get("segments", [])
        srt_content = []
        
        for i, segment in enumerate(segments):
            # Format timestamps
            start_time = OutputFormatConverter._format_timestamp_srt(segment["start"])
            end_time = OutputFormatConverter._format_timestamp_srt(segment["end"])
            
            # Format text
            text = segment["text"].strip()
            if include_speakers and "speaker" in segment:
                speaker = segment["speaker"]
                text = f"[{speaker}] {text}"
            
            # Build SRT entry
            srt_content.append(f"{i+1}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    @staticmethod
    def to_vtt(result: Dict[str, Any], include_speakers: bool = True) -> str:
        """Convert to VTT format"""
        segments = result.get("segments", [])
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            # Format timestamps
            start_time = OutputFormatConverter._format_timestamp_vtt(segment["start"])
            end_time = OutputFormatConverter._format_timestamp_vtt(segment["end"])
            
            # Format text
            text = segment["text"].strip()
            if include_speakers and "speaker" in segment:
                speaker = segment["speaker"]
                text = f"[{speaker}] {text}"
            
            # Build VTT entry
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
        
        return "\n".join(vtt_content)
    
    @staticmethod
    def to_txt(result: Dict[str, Any], include_speakers: bool = True, include_timestamps: bool = True) -> str:
        """Convert to plain text format"""
        segments = result.get("segments", [])
        txt_content = []
        
        for segment in segments:
            # Format text
            text = segment["text"].strip()
            
            # Add speaker label
            if include_speakers and "speaker" in segment:
                speaker = segment["speaker"]
                text = f"[{speaker}] {text}"
            
            # Add timestamps
            if include_timestamps:
                start_time = OutputFormatConverter._format_timestamp_readable(segment["start"])
                end_time = OutputFormatConverter._format_timestamp_readable(segment["end"])
                text = f"[{start_time} - {end_time}] {text}"
            
            txt_content.append(text)
        
        return "\n".join(txt_content)
    
    @staticmethod
    def to_json(result: Dict[str, Any], pretty: bool = True) -> str:
        """Convert to JSON format"""
        if pretty:
            return json.dumps(result, indent=2, ensure_ascii=False)
        else:
            return json.dumps(result, ensure_ascii=False)
    
    @staticmethod
    def to_rttm(result: Dict[str, Any]) -> str:
        """Convert to RTTM format (Rich Transcription Time Marked)"""
        segments = result.get("segments", [])
        rttm_content = []
        
        for segment in segments:
            if "speaker" in segment:
                speaker = segment["speaker"]
                start_time = segment["start"]
                duration = segment["end"] - segment["start"]
                
                # RTTM format: SPEAKER <file-id> <channel-id> <start-time> <duration> <ortho> <speaker-type> <speaker-name> <conf>
                rttm_line = f"SPEAKER audio 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} <NA>"
                rttm_content.append(rttm_line)
        
        return "\n".join(rttm_content)
    
    @staticmethod
    def to_csv(result: Dict[str, Any]) -> str:
        """Convert to CSV format"""
        segments = result.get("segments", [])
        csv_content = ["start_time,end_time,speaker,text"]
        
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment.get("speaker", "unknown")
            text = segment["text"].strip().replace('"', '""')  # Escape quotes
            
            csv_line = f"{start_time},{end_time},{speaker},\"{text}\""
            csv_content.append(csv_line)
        
        return "\n".join(csv_content)
    
    @staticmethod
    def save_to_file(result: Dict[str, Any], 
                     output_path: Path, 
                     format_type: str = "auto",
                     **kwargs) -> bool:
        """Save result to file in specified format"""
        
        try:
            # Determine format from file extension if auto
            if format_type == "auto":
                format_type = output_path.suffix.lower().lstrip(".")
            
            # Generate content based on format
            if format_type in ["srt"]:
                content = OutputFormatConverter.to_srt(result, **kwargs)
            elif format_type in ["vtt"]:
                content = OutputFormatConverter.to_vtt(result, **kwargs)
            elif format_type in ["txt"]:
                content = OutputFormatConverter.to_txt(result, **kwargs)
            elif format_type in ["json"]:
                content = OutputFormatConverter.to_json(result, **kwargs)
            elif format_type in ["rttm"]:
                content = OutputFormatConverter.to_rttm(result)
            elif format_type in ["csv"]:
                content = OutputFormatConverter.to_csv(result)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Saved {format_type.upper()} output to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {format_type.upper()} output", error=str(e))
            return False
    
    @staticmethod
    def save_all_formats(result: Dict[str, Any], 
                        output_dir: Path, 
                        base_name: str,
                        include_speakers: bool = True) -> Dict[str, Path]:
        """Save result in all available formats"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Define formats to save
        formats = [
            ("json", "json"),
            ("srt", "srt"),
            ("vtt", "vtt"),
            ("txt", "txt"),
            ("rttm", "rttm"),
            ("csv", "csv")
        ]
        
        for format_name, extension in formats:
            output_path = output_dir / f"{base_name}.{extension}"
            
            if OutputFormatConverter.save_to_file(
                result, 
                output_path, 
                format_name,
                include_speakers=include_speakers
            ):
                saved_files[format_name] = output_path
        
        logger.info(f"Saved {len(saved_files)} output formats to {output_dir}")
        return saved_files
    
    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Format seconds to VTT timestamp (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    @staticmethod
    def _format_timestamp_readable(seconds: float) -> str:
        """Format seconds to human-readable timestamp (HH:MM:SS)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
