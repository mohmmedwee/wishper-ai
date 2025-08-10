"""
Batch processing service for multiple audio files
Handles concurrent processing with progress tracking and result aggregation
"""

import asyncio
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.config import settings
from app.models.transcription import BatchTranscriptionRequest, BatchTranscriptionResult, TranscriptionStatus
from app.services.diarization_service import DiarizationService
from app.utils.output_formats import OutputFormatConverter

logger = structlog.get_logger(__name__)

class BatchProcessor:
    """Handles batch processing of multiple audio files"""
    
    def __init__(self, diarization_service: DiarizationService):
        self.diarization_service = diarization_service
        self.max_workers = settings.MAX_CONCURRENT_JOBS
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        self.failed_jobs: Dict[str, Dict[str, Any]] = {}
    
    async def process_batch(self, request: BatchTranscriptionRequest) -> str:
        """Start batch processing and return batch ID"""
        
        batch_id = str(uuid.uuid4())
        logger.info("Starting batch processing", batch_id=batch_id, file_count=len(request.files))
        
        # Initialize batch job
        self.active_jobs[batch_id] = {
            "id": batch_id,
            "request": request,
            "status": "processing",
            "started_at": datetime.utcnow(),
            "total_files": len(request.files),
            "completed_files": 0,
            "failed_files": 0,
            "results": {},
            "progress": 0.0
        }
        
        # Start processing in background
        asyncio.create_task(self._process_batch_async(batch_id, request))
        
        return batch_id
    
    async def _process_batch_async(self, batch_id: str, request: BatchTranscriptionRequest):
        """Process batch asynchronously"""
        
        try:
            # Create output directory for batch
            batch_output_dir = Path(settings.OUTPUT_DIR) / f"batch_{batch_id}"
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process files with limited concurrency
            semaphore = asyncio.Semaphore(self.max_workers)
            
            # Create tasks for each file
            tasks = []
            for file_info in request.files:
                task = asyncio.create_task(
                    self._process_single_file(batch_id, file_info, batch_output_dir, semaphore)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Finalize batch
            await self._finalize_batch(batch_id, batch_output_dir)
            
        except Exception as e:
            logger.error("Batch processing failed", batch_id=batch_id, error=str(e))
            self.active_jobs[batch_id]["status"] = "failed"
            self.active_jobs[batch_id]["error"] = str(e)
    
    async def _process_single_file(self, 
                                  batch_id: str, 
                                  file_info: Dict[str, Any], 
                                  batch_output_dir: Path,
                                  semaphore: asyncio.Semaphore) -> None:
        """Process a single file within the batch"""
        
        async with semaphore:
            file_id = str(uuid.uuid4())
            file_path = Path(file_info["file_path"])
            
            try:
                logger.info("Processing file in batch", 
                           batch_id=batch_id, 
                           file_id=file_id, 
                           file=str(file_path))
                
                # Update progress
                self.active_jobs[batch_id]["results"][file_id] = {
                    "file_path": str(file_path),
                    "status": "processing",
                    "started_at": datetime.utcnow()
                }
                
                # Process the file
                result = await self.diarization_service.process_audio(
                    file_path, 
                    file_info.get("transcription_request", {})
                )
                
                # Save results in multiple formats
                file_output_dir = batch_output_dir / file_path.stem
                saved_files = OutputFormatConverter.save_all_formats(
                    result, 
                    file_output_dir, 
                    file_path.stem,
                    include_speakers=True
                )
                
                # Update file result
                self.active_jobs[batch_id]["results"][file_id].update({
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "result": result,
                    "output_files": {k: str(v) for k, v in saved_files.items()}
                })
                
                # Update batch progress
                self.active_jobs[batch_id]["completed_files"] += 1
                self.active_jobs[batch_id]["progress"] = (
                    self.active_jobs[batch_id]["completed_files"] / 
                    self.active_jobs[batch_id]["total_files"]
                )
                
                logger.info("File completed in batch", 
                           batch_id=batch_id, 
                           file_id=file_id, 
                           file=str(file_path))
                
            except Exception as e:
                logger.error("File processing failed in batch", 
                           batch_id=batch_id, 
                           file_id=file_id, 
                           file=str(file_path), 
                           error=str(e))
                
                # Update file result with error
                self.active_jobs[batch_id]["results"][file_id].update({
                    "status": "failed",
                    "completed_at": datetime.utcnow(),
                    "error": str(e)
                })
                
                # Update batch progress
                self.active_jobs[batch_id]["failed_files"] += 1
    
    async def _finalize_batch(self, batch_id: str, batch_output_dir: Path):
        """Finalize batch processing and generate summary"""
        
        try:
            batch_info = self.active_jobs[batch_id]
            
            # Create batch summary
            summary = {
                "batch_id": batch_id,
                "total_files": batch_info["total_files"],
                "completed_files": batch_info["completed_files"],
                "failed_files": batch_info["failed_files"],
                "success_rate": batch_info["completed_files"] / batch_info["total_files"],
                "started_at": batch_info["started_at"],
                "completed_at": datetime.utcnow(),
                "output_directory": str(batch_output_dir)
            }
            
            # Save batch summary
            summary_path = batch_output_dir / "batch_summary.json"
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Update batch status
            if batch_info["failed_files"] == 0:
                batch_info["status"] = "completed"
            else:
                batch_info["status"] = "completed_with_errors"
            
            batch_info["summary"] = summary
            
            # Move to completed jobs
            self.completed_jobs[batch_id] = batch_info
            del self.active_jobs[batch_id]
            
            logger.info("Batch processing completed", 
                       batch_id=batch_id, 
                       summary=summary)
            
        except Exception as e:
            logger.error("Failed to finalize batch", batch_id=batch_id, error=str(e))
            self.active_jobs[batch_id]["status"] = "failed"
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch job"""
        
        # Check active jobs
        if batch_id in self.active_jobs:
            return self.active_jobs[batch_id]
        
        # Check completed jobs
        if batch_id in self.completed_jobs:
            return self.completed_jobs[batch_id]
        
        # Check failed jobs
        if batch_id in self.failed_jobs:
            return self.failed_jobs[batch_id]
        
        return None
    
    def get_all_batches(self) -> Dict[str, Dict[str, Any]]:
        """Get all batch jobs"""
        all_batches = {}
        all_batches.update(self.active_jobs)
        all_batches.update(self.completed_jobs)
        all_batches.update(self.failed_jobs)
        return all_batches
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch job"""
        
        if batch_id not in self.active_jobs:
            return False
        
        batch_info = self.active_jobs[batch_id]
        batch_info["status"] = "cancelled"
        batch_info["cancelled_at"] = datetime.utcnow()
        
        # Move to failed jobs
        self.failed_jobs[batch_id] = batch_info
        del self.active_jobs[batch_id]
        
        logger.info("Batch cancelled", batch_id=batch_id)
        return True
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed/failed jobs"""
        
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        # Clean up completed jobs
        jobs_to_remove = []
        for batch_id, job_info in self.completed_jobs.items():
            if job_info["completed_at"].timestamp() < cutoff_time:
                jobs_to_remove.append(batch_id)
        
        for batch_id in jobs_to_remove:
            del self.completed_jobs[batch_id]
        
        # Clean up failed jobs
        jobs_to_remove = []
        for batch_id, job_info in self.failed_jobs.items():
            if job_info.get("completed_at", job_info["started_at"]).timestamp() < cutoff_time:
                jobs_to_remove.append(batch_id)
        
        for batch_id in jobs_to_remove:
            del self.failed_jobs[batch_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        
        total_active = len(self.active_jobs)
        total_completed = len(self.completed_jobs)
        total_failed = len(self.failed_jobs)
        
        # Calculate success rate
        total_processed = total_completed + total_failed
        success_rate = total_completed / total_processed if total_processed > 0 else 0
        
        # Calculate average processing time
        processing_times = []
        for job_info in self.completed_jobs.values():
            if "started_at" in job_info and "completed_at" in job_info:
                duration = (job_info["completed_at"] - job_info["started_at"]).total_seconds()
                processing_times.append(duration)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "active_jobs": total_active,
            "completed_jobs": total_completed,
            "failed_jobs": total_failed,
            "total_processed": total_processed,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "max_concurrent_jobs": self.max_workers
        }
