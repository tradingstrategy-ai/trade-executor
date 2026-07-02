"""Tests for command line logging setup."""

import logging
import os

import pytest
from eth_defi.coloured_logging import EthDefiRichHandler

from tradeexecutor.cli.log import setup_logging


@pytest.fixture(autouse=True)
def restore_root_logging() -> None:
    """Restore root logger handlers after setup_logging() mutates global logging.

    1. Save the current root logger handlers and level.
    2. Let the test exercise the real setup_logging() implementation.
    3. Close test-created handlers and restore the original logger state.
    """

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    # 1. Save the current root logger handlers and level.
    try:
        # 2. Let the test exercise the real setup_logging() implementation.
        yield
    finally:
        # 3. Close test-created handlers and restore the original logger state.
        for handler in root.handlers:
            if handler not in original_handlers:
                handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def test_setup_logging_uses_eth_defi_rich_handler_when_colour_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check CLI logging delegates coloured terminal output to eth_defi.

    1. Enable ANSI colour output through the standard environment variable.
    2. Set up trade-executor command line logging.
    3. Verify the root logger uses eth_defi's Rich handler.
    """

    # 1. Enable ANSI colour output through the standard environment variable.
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    # 2. Set up trade-executor command line logging.
    logger = setup_logging("info")

    # 3. Verify the root logger uses eth_defi's Rich handler.
    assert logger is logging.getLogger()
    assert any(isinstance(handler, EthDefiRichHandler) for handler in logger.handlers)


def test_setup_logging_prefers_explicit_log_level_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check explicit CLI log level still wins over LOG_LEVEL.

    1. Set LOG_LEVEL to a less verbose environment default.
    2. Set up trade-executor command line logging with an explicit level.
    3. Verify the root logger and console handler use the explicit level.
    """

    # 1. Set LOG_LEVEL to a less verbose environment default.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("LOG_LEVEL", "info")

    # 2. Set up trade-executor command line logging with an explicit level.
    logger = setup_logging("debug")

    # 3. Verify the root logger and console handler use the explicit level.
    assert logger.level == logging.DEBUG
    assert any(handler.level == logging.DEBUG for handler in logger.handlers)
    assert os.environ["LOG_LEVEL"] == "info"


def test_setup_logging_ignores_disabled_environment_when_level_is_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check LOG_LEVEL=disabled does not leak into eth_defi logging setup.

    1. Set LOG_LEVEL to trade-executor's disabled test sentinel.
    2. Set up trade-executor command line logging with a resolved level.
    3. Verify logging uses the resolved level and leaves the environment intact.
    """

    # 1. Set LOG_LEVEL to trade-executor's disabled test sentinel.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("LOG_LEVEL", "disabled")

    # 2. Set up trade-executor command line logging with a resolved level.
    logger = setup_logging(logging.WARNING)

    # 3. Verify logging uses the resolved level and leaves the environment intact.
    assert logger.level == logging.WARNING
    assert any(handler.level == logging.WARNING for handler in logger.handlers)
    assert os.environ["LOG_LEVEL"] == "disabled"


def test_setup_logging_uses_plain_handler_when_colour_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check CLI logging honours NO_COLOR through eth_defi.

    1. Disable ANSI colour output through the standard environment variable.
    2. Set up trade-executor command line logging.
    3. Verify the root logger falls back to the standard stream handler.
    """

    # 1. Disable ANSI colour output through the standard environment variable.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("CLICOLOR_FORCE", raising=False)

    # 2. Set up trade-executor command line logging.
    logger = setup_logging("info")

    # 3. Verify the root logger falls back to the standard stream handler.
    assert logger is logging.getLogger()
    assert any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, EthDefiRichHandler)
        for handler in logger.handlers
    )
