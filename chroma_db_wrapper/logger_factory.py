"""
**********************************************************************************************************************************************
*****************************************************       LOGGER FACTORY       *************************************************************
**********************************************************************************************************************************************

This module provides functions to set up a logger for the ChromaDB wrapper. It includes options for logging to both
console and files, with support for both human-readable and JSON formats. The logger is designed to handle large log files
by using a rotating file handler, which creates new log files when the current one reaches a specified size. The logger
also includes a decorator for logging the start and end of operations in the ChromaDB wrapper.

The module uses the following libraries:
- logging: The standard Python library for logging.
- json: The standard Python library for JSON serialization.
- os: The standard Python library for interacting with the operating system.
- logging.handlers: The standard Python library for handling log files, including rotating file handlers.
"""

import logging
import json
import os
from logging.handlers import RotatingFileHandler



class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for logging.
    This formatter converts log records into JSON format, including the timestamp,
    log level, message, module name, function name, and line number.
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }
        return json.dumps(log_record)
    

def setup_logger(name="vectordb", log_dir="./logs", level=logging.INFO, console_output=True):
    """
    Set up a logger for the ChromaDB wrapper.
    This function creates a logger that logs to both console and files, with options for
    human-readable and JSON formats. The logger uses a rotating file handler to manage
    log file sizes, creating new log files when the current one reaches a specified size.
    Args:
        name (str): The name of the logger. Default is "vectordb".
        log_dir (str): The directory where log files will be stored. Default is "./logs".
        level (int): The logging level. Default is logging.INFO.
        console_output (bool): Whether to log to console. Default is True.
    Returns:
        logger (logging.Logger): The configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs

    # Human-readable log
    readable_handler = RotatingFileHandler(os.path.join(log_dir, "vectordb.log"), maxBytes=5_000_000, backupCount=2)
    readable_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(readable_handler)

    # JSON structured log
    json_handler = RotatingFileHandler(os.path.join(log_dir, "vectordb.json.log"), maxBytes=5_000_000, backupCount=2)
    json_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_handler)

    # Optional: print to console
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(console_handler)

    return logger


def log_operation(func):
    """
    Decorator to log the start and end of a function.
    This decorator logs the function name and a separator line before and after the function execution.
    Args:
        func (function): The function to be decorated.
    Returns:
        wrapper (function): The wrapped function with logging.
    """
    def wrapper(self, *args, **kwargs):
        self.logger.info(f"{'*' * 20} Starting: {func.__name__}")
        result = func(self, *args, **kwargs)
        self.logger.info(f"{'*' * 20} Finished: {func.__name__}")
        return result
    return wrapper
