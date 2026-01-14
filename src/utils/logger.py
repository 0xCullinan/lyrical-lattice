"""
File: src/utils/logger.py
Purpose: Structured logging configuration per REQ-DEPLOY-006 and REQ-DEPLOY-007
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any
from src.core.config import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production.
    
    Outputs log records as JSON objects for easy parsing by log aggregators.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON-formatted log string.
        """
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            }:
                log_obj[key] = value
        
        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for development console output.
    
    Uses ANSI color codes for improved readability.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.
        
        Args:
            record: Log record to format.
            
        Returns:
            Colored log string.
        """
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> None:
    """Configure logging based on environment.
    
    In production, uses JSON formatting for log aggregation.
    In development, uses colored console output for readability.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, settings.log_level))
    
    if settings.is_production:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ColoredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    
    root_logger.addHandler(handler)
    
    # Reduce noise from third-party libraries
    for noisy_logger in ["uvicorn.access", "httpx", "httpcore", "asyncpg"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Name for the logger (usually __name__).
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds contextual information to log messages.
    
    Useful for adding request IDs, user IDs, etc. to all log messages.
    """
    
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Process log message to add context.
        
        Args:
            msg: Log message.
            kwargs: Additional arguments.
            
        Returns:
            Tuple of processed message and kwargs.
        """
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        return msg, kwargs


def get_request_logger(request_id: str) -> LoggerAdapter:
    """Get a logger adapter with request ID context.
    
    Args:
        request_id: Unique request identifier.
        
    Returns:
        Logger adapter with request ID in context.
    """
    logger = get_logger("oronym.request")
    return LoggerAdapter(logger, {"request_id": request_id})


# Initialize logging on module import
setup_logging()
