"""
API routes for the Whisper Diarization Service
"""

import uuid
import time
from typing import List, Optional
from pathlib import Path
import aiofiles
import structlog

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.requests import Request

from app.models.transcription import (
    TranscriptionRequest, 
    TranscriptionResult, 
    TranscriptionStatus,
    BatchTranscriptionRequest,
    BatchTranscriptionResult
)
from app.core.config import settings
from app.core.logging import log_request, log_response, log_error

logger = structlog.get_logger(__name__)

# Create router
transcription_router = APIRouter()

# Import parallel routes
from app.api.parallel_routes import parallel_router

# In-memory storage for demo purposes (use Redis/DB in production)
transcription_jobs = {}

async def get_diarization_service(request: Request):
    """Dependency to get the diarization service"""
    if not hasattr(request.app.state, 'diarization_service'):
        raise HTTPException(status_code=503, detail="Diarization service not available")
    return request.app.state.diarization_service

@transcription_router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language code"),
    task: str = Query("transcribe", description="Task type"),
    whisper_model: Optional[str] = Query(None, description="Whisper model"),
    enable_diarization: bool = Query(True, description="Enable speaker diarization"),
    source_separation: bool = Query(True, description="Enable source separation for better quality"),
    parallel_processing: bool = Query(True, description="Enable parallel processing for faster results"),
    enhanced_alignment: bool = Query(True, description="Enable enhanced alignment and punctuation"),
    output_format: str = Query("json", description="Output format")
):
    """Transcribe audio file with optional speaker diarization"""
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Log request
        log_request(request_id, "/transcribe", "POST", 
                   file_size=file.size, language=language, task=task)
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Check file format
        file_ext = Path(file.filename).suffix.lower().lstrip('.')
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {', '.join(settings.SUPPORTED_FORMATS)}"
            )
        
        # Save uploaded file
        upload_path = Path(settings.UPLOAD_DIR) / f"{request_id}_{file.filename}"
        async with aiofiles.open(upload_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Create transcription request
        transcription_request = TranscriptionRequest(
            language=language,
            task=task,
            whisper_model=whisper_model or settings.WHISPER_MODEL,
            enable_diarization=enable_diarization,
            source_separation=source_separation,
            parallel_processing=parallel_processing,
            enhanced_alignment=enhanced_alignment,
            output_format=output_format
        )
        
        # Get diarization service
        diarization_service = await get_diarization_service(request)
        
        # Process audio
        result = await diarization_service.process_audio(upload_path, transcription_request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log response
        log_response(request_id, 200, response_time, 
                    transcription_id=result.transcription_id)
        
        # Cleanup uploaded file
        background_tasks.add_task(lambda: upload_path.unlink(missing_ok=True))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to transcribe audio: {str(e)}"
        log_error(request_id, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@transcription_router.post("/transcribe/batch", response_model=BatchTranscriptionResult)
async def batch_transcribe(
    request: Request,
    batch_request: BatchTranscriptionRequest
):
    """Batch transcription endpoint"""
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        log_request(request_id, "/transcribe/batch", "POST", 
                   num_files=len(batch_request.files))
        
        # This is a placeholder for batch processing
        # In production, you'd implement a proper job queue system
        
        raise HTTPException(
            status_code=501, 
            detail="Batch transcription not yet implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to process batch transcription: {str(e)}"
        log_error(request_id, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@transcription_router.get("/transcribe/{transcription_id}", response_model=TranscriptionStatus)
async def get_transcription_status(
    transcription_id: str,
    request: Request
):
    """Get transcription job status"""
    
    request_id = str(uuid.uuid4())
    
    try:
        log_request(request_id, f"/transcribe/{transcription_id}", "GET")
        
        # Check if job exists
        if transcription_id not in transcription_jobs:
            raise HTTPException(status_code=404, detail="Transcription job not found")
        
        status = transcription_jobs[transcription_id]
        log_response(request_id, 200, 0.0, transcription_id=transcription_id)
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get transcription status: {str(e)}"
        log_error(request_id, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@transcription_router.get("/transcribe/{transcription_id}/download")
async def download_transcription(
    transcription_id: str,
    request: Request,
    format: str = Query("json", description="Download format")
):
    """Download transcription in specified format"""
    
    request_id = str(uuid.uuid4())
    
    try:
        log_request(request_id, f"/transcribe/{transcription_id}/download", "GET", format=format)
        
        # Check if job exists and is completed
        if transcription_id not in transcription_jobs:
            raise HTTPException(status_code=404, detail="Transcription job not found")
        
        status = transcription_jobs[transcription_id]
        if status.status != "completed":
            raise HTTPException(status_code=400, detail="Transcription not yet completed")
        
        # Generate output file based on format
        output_path = Path(settings.OUTPUT_DIR) / f"{transcription_id}.{format}"
        
        # This is a placeholder - implement format conversion logic
        raise HTTPException(
            status_code=501, 
            detail=f"Download in {format} format not yet implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to download transcription: {str(e)}"
        log_error(request_id, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@transcription_router.delete("/transcribe/{transcription_id}")
async def delete_transcription(
    transcription_id: str,
    request: Request
):
    """Delete transcription job and associated files"""
    
    request_id = str(uuid.uuid4())
    
    try:
        log_request(request_id, f"/transcribe/{transcription_id}", "DELETE")
        
        # Check if job exists
        if transcription_id not in transcription_jobs:
            raise HTTPException(status_code=404, detail="Transcription job not found")
        
        # Remove job from storage
        del transcription_jobs[transcription_id]
        
        # Cleanup associated files
        # This would be implemented in production
        
        log_response(request_id, 200, 0.0, transcription_id=transcription_id)
        
        return {"message": "Transcription deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to delete transcription: {str(e)}"
        log_error(request_id, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@transcription_router.get("/features")
async def get_features(request: Request):
    """Get information about supported features"""
    try:
        diarization_service = await get_diarization_service(request)
        return diarization_service.get_supported_features()
    except Exception as e:
        logger.error("Failed to get features", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get features")
