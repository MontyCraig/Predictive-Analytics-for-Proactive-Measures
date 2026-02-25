"""Logging configuration for Predictive Analytics for Proactive Measures.

Provides a pre-configured :class:`logging.Logger` instance and a
:func:`setup_logging` factory that can be called to customise the
logging level or add a file handler.

Typical usage::

    from predictive_analytics.config.logging import logger, setup_logging

    # Use the default module-level logger directly:
    logger.info("Application started")

    # Or reconfigure with a file handler:
    setup_logging(level=logging.DEBUG, log_file="/var/log/pa.log")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

__all__: list[str] = ["setup_logging", "logger"]


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    force_reset: bool = False,
) -> logging.Logger:
    """Set up and return the application logger.

    When the logger already has handlers and *force_reset* is ``False``
    the existing configuration is returned as-is (idempotent).

    Args:
        level: The :mod:`logging` level (e.g. ``logging.DEBUG``).
            Defaults to ``logging.INFO``.
        log_file: Optional filesystem path.  When provided a
            :class:`~logging.FileHandler` is attached in addition to
            the console handler.
        force_reset: When ``True`` any existing handlers are removed
            and the logger is reconfigured from scratch.

    Returns:
        The configured :class:`~logging.Logger` for the
        ``predictive_analytics`` namespace.
    """
    logger_instance: logging.Logger = logging.getLogger("predictive_analytics")

    # Return existing logger if already configured (idempotent).
    if logger_instance.handlers and not force_reset:
        return logger_instance

    # Clear existing handlers to avoid duplicate logs.
    if logger_instance.handlers:
        logger_instance.handlers.clear()

    # Configure format for log messages.
    formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout).
    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)

    # File handler (optional).
    if log_file is not None:
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)

    # Set logging level.
    logger_instance.setLevel(level)

    # Prevent logs from propagating to root logger.
    logger_instance.propagate = False

    return logger_instance


# Create a default logger for convenient import.
logger: logging.Logger = setup_logging()
