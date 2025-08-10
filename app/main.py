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
from app.services.diarization_service import DiarizationService
from app.core.logging import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Whisper Diarization Service...")
    app.state.diarization_service = DiarizationService()
    await app.state.diarization_service.initialize()
    logger.info("Service initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Whisper Diarization Service...")
    await app.state.diarization_service.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Whisper Diarization Service",
    description="Production-ready service for ASR with speaker diarization",
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Whisper Diarization Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "whisper-diarization",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
