"""
Logging utilities for DEIM training and inference
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """
    Logger for DEIM operations

    Provides consistent logging across training and inference
    """

    def __init__(self,
                 name: str = 'DEIM',
                 log_file: Optional[str] = None,
                 level: int = logging.INFO):
        """
        Initialize logger

        Args:
            name: Logger name
            log_file: Optional file to save logs
            level: Logging level
        """

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear any existing handlers
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)