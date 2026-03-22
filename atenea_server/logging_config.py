"""
Centralized logging configuration for Atenea Server.

Import this module early in the application startup to configure logging once.
"""

import logging
import os
import sys

# Default log level - can be overridden via environment variable
DEFAULT_LOG_LEVEL = "INFO"


def setup_logging(level: str | None = None) -> None:
    """
    Configure logging for the Atenea server.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses ATENEA_LOG_LEVEL env var or defaults to INFO.
    """
    log_level = level or os.environ.get("ATENEA_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    
    # Set specific loggers that might be too verbose
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)

