"""Logger utility for the ETL Architect Agent V2."""

import logging
import sys
from typing import Optional
from pathlib import Path

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_str: Log format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 