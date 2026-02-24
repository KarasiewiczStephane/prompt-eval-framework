"""Structured logging configuration for the framework.

Call :func:`setup_logging` once at application startup to configure the
root logger with a consistent format and level.
"""

import logging
import sys


def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    """Configure the root logger.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Optional format string.  Falls back to a sensible default.
    """
    if fmt is None:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if not root.handlers:
        root.addHandler(handler)
