"""Tests for structured logging setup."""

import logging

from src.utils.logger import setup_logging


class TestSetupLogging:
    """setup_logging should configure the root logger."""

    def test_sets_level(self) -> None:
        setup_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_default_level_is_info(self) -> None:
        setup_logging()
        assert logging.getLogger().level == logging.INFO

    def test_handler_added(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging()
        assert len(root.handlers) >= 1

    def test_no_duplicate_handlers(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging()
        count = len(root.handlers)
        setup_logging()
        assert len(root.handlers) == count
