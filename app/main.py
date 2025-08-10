"""
Whisper Diarization Service
A production-ready service for automatic speech recognition with speaker diarization
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

from app.core.config import settings
from app.api.routes import transcription_router
from app.api.parallel_routes import parallel_router
from app.services.diarization_service import DiarizationService
from app.services.parallel_diarization_service import ParallelDiarizationService
from app.core.logging import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Whisper Diarization Service...")
    
    # Initialize regular diarization service
    app.state.diarization_service = DiarizationService()
    await app.state.diarization_service.initialize()
    logger.info("Regular diarization service initialized successfully")
    
    # Initialize parallel diarization service
    app.state.parallel_diarization_service = ParallelDiarizationService(
        max_workers=getattr(settings, 'MAX_WORKERS', 2),
        use_process_pool=getattr(settings, 'USE_PROCESS_POOL', False)
    )
    await app.state.parallel_diarization_service.initialize()
    logger.info("Parallel diarization service initialized successfully")
    
    logger.info("All services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Whisper Diarization Service...")
    
    # Cleanup parallel service
    if hasattr(app.state, 'parallel_diarization_service'):
        await app.state.parallel_diarization_service.cleanup()
        logger.info("Parallel diarization service cleaned up")
    
    # Cleanup regular service
    if hasattr(app.state, 'diarization_service'):
        await app.state.diarization_service.cleanup()
        logger.info("Regular diarization service cleaned up")
    
    logger.info("All services cleaned up successfully")

# Create FastAPI app
app = FastAPI(
    title="Whisper Diarization Service",
    description="Production-ready service for ASR with speaker diarization and parallel processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(transcription_router, prefix="/api/v1", tags=["transcription"])
app.include_router(parallel_router, tags=["parallel-processing"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Whisper Diarization Service is running", 
        "status": "healthy",
        "features": {
            "regular_processing": True,
            "parallel_processing": True,
            "speaker_diarization": True,
            "batch_processing": True
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "whisper-diarization",
        "version": "1.0.0",
        "services": {
            "regular_diarization": "running",
            "parallel_diarization": "running"
        },
        "endpoints": {
            "regular": "/api/v1/transcribe",
            "parallel": "/api/v1/parallel/transcribe",
            "parallel_async": "/api/v1/parallel/transcribe/async",
            "parallel_batch": "/api/v1/parallel/batch",
            "parallel_stats": "/api/v1/parallel/stats",
            "parallel_features": "/api/v1/parallel/features"
        }
    }

@app.get("/api")
async def api_info():
    """API information and available endpoints"""
    return {
        "service": "Whisper Diarization Service",
        "version": "1.0.0",
        "description": "Production-ready ASR service with advanced speaker diarization",
        "endpoints": {
            "regular_processing": {
                "transcribe": "/api/v1/transcribe",
                "batch_transcribe": "/api/v1/transcribe/batch",
                "get_status": "/api/v1/transcribe/{id}",
                "download": "/api/v1/transcribe/{id}/download",
                "delete": "/api/v1/transcribe/{id}",
                "features": "/api/v1/features"
            },
            "parallel_processing": {
                "transcribe": "/api/v1/parallel/transcribe",
                "transcribe_async": "/api/v1/parallel/transcribe/async",
                "get_status": "/api/v1/parallel/status/{task_id}",
                "get_result": "/api/v1/parallel/result/{task_id}",
                "batch_transcribe": "/api/v1/parallel/batch",
                "stats": "/api/v1/parallel/stats",
                "features": "/api/v1/parallel/features"
            }
        },
        "features": {
            "parallel_processing": "True parallel Whisper + NeMo processing for 2-3x speedup",
            "speaker_diarization": "Advanced speaker identification and segmentation",
            "batch_processing": "Process multiple files concurrently",
            "async_processing": "Non-blocking audio processing with status tracking",
            "multiple_formats": "Support for various audio formats",
            "gpu_acceleration": "CUDA support for optimal performance"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
