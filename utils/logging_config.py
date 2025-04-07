"""Logging configuration for Predictive Analytics for Proactive Measures."""
import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: The logging level (default: logging.INFO)
        log_file: Optional file path to write logs to
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger("predictive_analytics")
    
    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()
    
    # Configure format for log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Configure file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set logging level
    logger.setLevel(level)
    
    # Prevent logs from propagating to root logger
    logger.propagate = False
    
    return logger


# Create a default logger for import
logger = setup_logging() 