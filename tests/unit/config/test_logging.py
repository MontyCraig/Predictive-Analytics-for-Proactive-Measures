"""Tests for predictive_analytics.config.logging module."""

from __future__ import annotations

import logging
from pathlib import Path

from predictive_analytics.config.logging import setup_logging


class TestSetupLogging:
    def test_returns_logger(self) -> None:
        result = setup_logging()
        assert isinstance(result, logging.Logger)
        assert result.name == "predictive_analytics"

    def test_default_level_is_info(self) -> None:
        result = setup_logging()
        assert result.level == logging.INFO

    def test_custom_level(self) -> None:
        result = setup_logging(level=logging.DEBUG)
        assert result.level == logging.DEBUG

    def test_has_console_handler(self) -> None:
        result = setup_logging()
        assert any(isinstance(h, logging.StreamHandler) for h in result.handlers)

    def test_no_file_handler_by_default(self) -> None:
        result = setup_logging()
        assert not any(isinstance(h, logging.FileHandler) for h in result.handlers)

    def test_file_handler_when_log_file_provided(self, tmp_path: Path) -> None:
        log_file = str(tmp_path / "test.log")
        result = setup_logging(log_file=log_file)
        file_handlers = [h for h in result.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == str(tmp_path / "test.log")

    def test_file_handler_writes_log(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        result = setup_logging(log_file=str(log_file))
        result.info("hello from test")
        # Flush handlers so content is written
        for h in result.handlers:
            h.flush()
        content = log_file.read_text()
        assert "hello from test" in content

    def test_clears_existing_handlers(self) -> None:
        first = setup_logging()
        handler_count_first = len(first.handlers)
        second = setup_logging()
        assert len(second.handlers) == handler_count_first

    def test_propagate_is_false(self) -> None:
        result = setup_logging()
        assert result.propagate is False
