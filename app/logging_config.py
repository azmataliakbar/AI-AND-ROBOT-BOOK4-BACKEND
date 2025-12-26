import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel


class LogRecord(BaseModel):
    timestamp: str
    level: str
    service: str
    event: str
    user_id: Optional[str] = None  # ✅ FIXED: Made optional
    chapter_id: Optional[str] = None  # ✅ FIXED: Made optional
    query: Optional[str] = None  # ✅ FIXED: Made optional
    response: Optional[str] = None  # ✅ FIXED: Made optional
    confidence_score: Optional[float] = None  # ✅ FIXED: Made optional
    execution_time: Optional[float] = None  # ✅ FIXED: Made optional
    source_type: Optional[str] = None  # ✅ FIXED: Made optional
    endpoint: Optional[str] = None  # ✅ ADDED: For API logging
    method: Optional[str] = None  # ✅ ADDED: For API logging
    status_code: Optional[int] = None  # ✅ ADDED: For API logging
    error_type: Optional[str] = None  # ✅ ADDED: For error logging
    error_message: Optional[str] = None  # ✅ ADDED: For error logging
    results_count: Optional[int] = None  # ✅ ADDED: For search logging
    threshold: Optional[float] = None  # ✅ ADDED: For search logging
    sections_processed: Optional[int] = None  # ✅ ADDED: For embedding logging
    metadata: Optional[Dict[str, Any]] = None  # ✅ FIXED: Made optional


class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create handler if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log(self, level: str, event: str, **kwargs):
        log_record = LogRecord(
            timestamp=datetime.utcnow().isoformat(),
            level=level,
            service="ai-robotics-book",
            event=event,
            **kwargs
        )

        self.logger.info(log_record.model_dump_json())

    def info(self, event: str, **kwargs):
        self._log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs):
        self._log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs):
        self._log("ERROR", event, **kwargs)

    def debug(self, event: str, **kwargs):
        self._log("DEBUG", event, **kwargs)

    def log_chat_interaction(self, user_id: Optional[str], query: str, response: str,
                           confidence_score: float, execution_time: float,
                           source_type: Optional[str] = None, chapter_id: Optional[str] = None):
        """Log chat interaction with relevant details"""
        self.info(
            "chat_interaction",
            user_id=user_id,
            query=query,
            response=response[:200] + "..." if len(response) > 200 else response,  # Truncate long responses
            confidence_score=confidence_score,
            execution_time=execution_time,
            source_type=source_type,
            chapter_id=chapter_id
        )

    def log_search_query(self, user_id: Optional[str], query: str, results_count: int,
                        execution_time: float, threshold: Optional[float] = None):
        """Log search query with results"""
        self.info(
            "search_query",
            user_id=user_id,
            query=query,
            results_count=results_count,
            execution_time=execution_time,
            threshold=threshold
        )

    def log_api_request(self, endpoint: str, method: str, user_id: Optional[str] = None,
                       execution_time: Optional[float] = None, status_code: int = 200):
        """Log API request details"""
        self.info(
            "api_request",
            endpoint=endpoint,
            method=method,
            user_id=user_id,
            execution_time=execution_time,
            status_code=status_code
        )

    def log_embedding_creation(self, chapter_id: str, sections_processed: int,
                             execution_time: float, status: str):
        """Log embedding creation process"""
        self.info(
            "embedding_creation",
            chapter_id=chapter_id,
            sections_processed=sections_processed,
            execution_time=execution_time,
            status=status
        )

    def log_error(self, error_type: str, error_message: str, endpoint: Optional[str] = None,
                  user_id: Optional[str] = None, traceback: Optional[str] = None,
                  status_code: Optional[int] = None):
        """Log error details"""
        self.error(
            "system_error",
            error_type=error_type,
            error_message=error_message,
            endpoint=endpoint,
            user_id=user_id,
            status_code=status_code,
            metadata={"traceback": traceback} if traceback else None
        )


# Global logger instance
logger = StructuredLogger(__name__)