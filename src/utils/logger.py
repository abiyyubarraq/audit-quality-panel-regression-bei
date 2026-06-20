"""
Logging Configuration Module

Provides consistent logging across the project.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logger with console and optional file handlers

    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
        console_output: Whether to output to console (default: True)

    Returns:
        Configured logger instance

    Usage:
        logger = setup_logger('data_processing', 'logs/data.log')
        logger.info('Processing started')
        logger.warning('Warning message')
        logger.error('Error message')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


__all__ = ['setup_logger']
