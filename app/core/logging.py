"""
Logging configuration for the Whisper Diarization Service
"""

import logging
import sys
import structlog
from typing import Any, Dict

def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured logging configuration"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)

def log_request(request_id: str, endpoint: str, method: str, **kwargs: Any) -> None:
    """Log API request details"""
    logger = get_logger("api.request")
    logger.info(
        "API request",
        request_id=request_id,
        endpoint=endpoint,
        method=method,
        **kwargs
    )

def log_response(request_id: str, status_code: int, response_time: float, **kwargs: Any) -> None:
    """Log API response details"""
    logger = get_logger("api.response")
    logger.info(
        "API response",
        request_id=request_id,
        status_code=status_code,
        response_time=response_time,
        **kwargs
    )

def log_error(request_id: str, error: str, **kwargs: Any) -> None:
    """Log error details"""
    logger = get_logger("api.error")
    logger.error(
        "API error",
        request_id=request_id,
        error=error,
        **kwargs
    )
