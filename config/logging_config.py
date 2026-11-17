"""
Centralized Logging Configuration
==================================
This module provides a unified logging setup for the AI Invoice Auditor system.
Logs are written to both console and file based on rules.yaml configuration.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_DIR = "./logs"
DEFAULT_LOG_FILE = "invoice_auditor.log"
DEFAULT_LOG_LEVEL = "INFO"


def load_logging_config(config_path: str = "./config/rules.yaml") -> dict:
    """
    Load logging configuration from rules.yaml.

    Args:
        config_path: Path to rules.yaml

    Returns:
        Logging configuration dictionary
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get("logging", {})
    except Exception:
        return {}


def setup_logging(
    name: str = "InvoiceAuditor",
    config_path: str = "./config/rules.yaml",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging with both file and console handlers.

    Args:
        name: Logger name
        config_path: Path to rules.yaml for configuration
        log_dir: Override log directory
        log_file: Override log filename
        log_level: Override log level

    Returns:
        Configured logger instance
    """
    config = load_logging_config(config_path)

    enable_file_log = config.get("enable_audit_log", True)

    if log_dir is None:
        log_path = config.get("log_file", f"{DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE}")
        log_dir = os.path.dirname(log_path) or DEFAULT_LOG_DIR
        log_file = os.path.basename(log_path) if log_file is None else log_file

    if log_file is None:
        log_file = DEFAULT_LOG_FILE

    if log_level is None:
        log_level = config.get("log_level", DEFAULT_LOG_LEVEL)

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    if not any(getattr(handler, "_invoice_auditor_console", False) for handler in root_logger.handlers):
        console_handler._invoice_auditor_console = True
        root_logger.addHandler(console_handler)

    if enable_file_log:
        log_file_path = log_dir_path / (os.getenv("LOG_FILE") or log_file)
        if not any(getattr(handler, "baseFilename", None) == str(log_file_path.resolve()) for handler in root_logger.handlers):
            file_handler = RotatingFileHandler(log_file_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the standard configuration.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance
    """
    root_logger = logging.getLogger("InvoiceAuditor")
    if not logging.getLogger().handlers:
        setup_logging("InvoiceAuditor")

    logger = logging.getLogger(f"InvoiceAuditor.{name}")
    return logger


def log_workflow_event(
    event_type: str,
    invoice_id: str,
    details: dict,
    logger: Optional[logging.Logger] = None,
):
    """
    Log a workflow event with structured data.

    Args:
        event_type: Type of event (e.g., 'extraction', 'validation', 'approval')
        invoice_id: Invoice identifier
        details: Additional event details
        logger: Logger instance (uses default if not provided)
    """
    if logger is None:
        logger = get_logger("Workflow")

    event_data = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "invoice_id": invoice_id,
        **details,
    }

    logger.info("WORKFLOW_EVENT %s", event_data)


_root_logger = None


def init_logging():
    """Initialize the logging system."""
    global _root_logger
    if _root_logger is None:
        _root_logger = setup_logging("InvoiceAuditor")
        _root_logger.info("=" * 60)
        _root_logger.info("AI Invoice Auditor - Logging Initialized")
        _root_logger.info(f"Log directory: {Path(DEFAULT_LOG_DIR).absolute()}")
        _root_logger.info("=" * 60)
    return _root_logger
