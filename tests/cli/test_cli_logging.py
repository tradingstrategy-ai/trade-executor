"""Tests for command line logging setup."""

import json
import logging
import os
import socket
import sys
from collections.abc import Iterator

import logstash
import pytest
from eth_defi.coloured_logging import EthDefiRichHandler

from tradeexecutor.cli.log import (
    MAX_LOGSTASH_UDP_PAYLOAD_SIZE,
    create_logstash_handler,
    setup_logging,
    setup_logstash_logging,
)


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


@pytest.fixture()
def logstash_udp_socket() -> Iterator[socket.socket]:
    """Bind a local UDP socket standing in for a Logstash server.

    A real socket is used instead of a mock, because the bug under test is
    raised by the operating system - a datagram larger than the socket limit
    fails with ``OSError: [Errno 90] Message too long`` - and a mocked
    ``sendto()`` would happily accept any payload size.

    1. Bind a datagram socket on an ephemeral loopback port.
    2. Hand the socket to the test.
    3. Close the socket afterwards.
    """

    # 1. Bind a datagram socket on an ephemeral loopback port.
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(5)

    try:
        # 2. Hand the socket to the test.
        yield server
    finally:
        # 3. Close the socket afterwards.
        server.close()


def receive_messages(server: socket.socket, count: int) -> dict[str, bytes]:
    """Read datagrams and index them by their log message.

    UDP does not guarantee ordering, so tests must not assume the datagrams
    arrive in the order they were logged.
    """

    datagrams = [server.recvfrom(65535)[0] for _ in range(count)]
    return {json.loads(datagram)["message"]: datagram for datagram in datagrams}


def test_logstash_logging_truncates_oversized_messages(
    logstash_udp_socket: socket.socket,
    capsys: pytest.CaptureFixture,
) -> None:
    """Check a huge diagnostic table still reaches Logstash as a truncated datagram.

    Diagnostic tables like the stale vault listing are tens of kilobytes and
    used to fail with ``OSError: [Errno 90] Message too long``, printing
    ``--- Logging error ---`` and dropping the record entirely.

    The table is built from CJK and emoji vault names as seen in production,
    because ``json.dumps()`` escapes those into ``\\uXXXX`` sequences of up to 12
    bytes per character. A budget counted in characters instead of bytes would
    therefore still produce an oversized datagram.

    1. Point Logstash logging at a local UDP socket.
    2. Log a small message and an oversized diagnostic table.
    3. Verify the small message is delivered intact.
    4. Verify the oversized message is delivered truncated, with its beginning preserved.
    5. Verify logging did not report an error.
    """

    # 1. Point Logstash logging at a local UDP socket.
    port = logstash_udp_socket.getsockname()[1]
    setup_logstash_logging(
        "127.0.0.1",
        application_name="test-executor",
        port=port,
    )
    logger = logging.getLogger(__name__)

    # 2. Log a small message and an oversized diagnostic table.
    table = "\n".join(
        f"● MONO BLACK TOKYO「MORI 森」💩 {i:<20} 0x{'a' * 40} 2026-08-29 00:00:00 2 days 12:51:14"
        for i in range(345)
    )
    assert len(table) > MAX_LOGSTASH_UDP_PAYLOAD_SIZE
    logger.warning("Vault candle data is fresh")
    logger.warning("Vault candle data is stale for %d vault(s):\n%s", 345, table)
    messages = receive_messages(logstash_udp_socket, 2)

    # 3. Verify the small message is delivered intact.
    assert len(messages["Vault candle data is fresh"]) <= MAX_LOGSTASH_UDP_PAYLOAD_SIZE

    # 4. Verify the oversized message is delivered truncated, with its beginning preserved.
    truncated_message = next(
        message for message in messages if message.startswith("Vault candle data is stale")
    )
    assert len(messages[truncated_message]) <= MAX_LOGSTASH_UDP_PAYLOAD_SIZE
    assert truncated_message.startswith(
        "Vault candle data is stale for 345 vault(s):\n● MONO BLACK TOKYO「MORI 森」💩 0 "
    )
    assert "truncated" in truncated_message

    # 5. Verify logging did not report an error.
    assert "--- Logging error ---" not in capsys.readouterr().err


def test_logstash_handler_leaves_fitting_records_untouched() -> None:
    """Check the truncating handler is transparent for ordinary log records.

    Truncation must not become a tax on every log line: a record that fits has
    to serialise exactly as the stock handler serialises it, and an oversized
    record must not corrupt the shared record that console, file and Sentry
    handlers read afterwards.

    1. Create a truncating handler and a stock handler with the same settings.
    2. Serialise a small record with both.
    3. Verify the two payloads are byte for byte identical.
    4. Serialise an oversized record with the truncating handler.
    5. Verify the oversized record itself was not modified.
    """

    # 1. Create a truncating handler and a stock handler with the same settings.
    truncating_handler = create_logstash_handler(
        "127.0.0.1",
        5959,
        tags=["python"],
        extra_fields={"application": "test-executor"},
    )
    reference_handler = logstash.UDPLogstashHandler(
        "127.0.0.1",
        5959,
        version=1,
        tags=["python"],
        extra_fields={"application": "test-executor"},
    )

    # 2. Serialise a small record with both.
    small_record = logging.LogRecord(
        "test", logging.WARNING, "/tmp/test.py", 10, "Vault %s is fresh", ("Storm",), None
    )

    # 3. Verify the two payloads are byte for byte identical.
    assert truncating_handler.makePickle(small_record) == reference_handler.makePickle(small_record)

    # 4. Serialise an oversized record with the truncating handler.
    table = "森" * MAX_LOGSTASH_UDP_PAYLOAD_SIZE
    try:
        raise RuntimeError("Vault price feed is stale")
    except RuntimeError:
        large_record = logging.LogRecord(
            "test", logging.WARNING, "/tmp/test.py", 10, "Stale:\n%s", (table,), sys.exc_info()
        )
    payload = truncating_handler.makePickle(large_record)
    assert len(payload) <= MAX_LOGSTASH_UDP_PAYLOAD_SIZE

    # 5. Verify the oversized record itself was not modified.
    assert large_record.getMessage() == f"Stale:\n{table}"
    assert large_record.args == (table,)
    assert large_record.exc_info is not None


def test_logstash_logging_replaces_records_that_cannot_be_shortened(
    logstash_udp_socket: socket.socket,
    capsys: pytest.CaptureFixture,
) -> None:
    """Check a record bloated outside its message is replaced by a placeholder.

    The Logstash formatter turns every custom record attribute into an extra
    field, so a short message can still produce an oversized datagram. Cutting
    the message cannot help there, and the record must not be lost silently.

    1. Point Logstash logging at a local UDP socket.
    2. Log a short message carrying a megabyte-sized extra field.
    3. Verify a placeholder datagram is delivered instead.
    4. Verify logging did not report an error.
    """

    # 1. Point Logstash logging at a local UDP socket.
    port = logstash_udp_socket.getsockname()[1]
    setup_logstash_logging(
        "127.0.0.1",
        application_name="test-executor",
        port=port,
    )
    logger = logging.getLogger(__name__)

    # 2. Log a short message carrying a megabyte-sized extra field.
    logger.warning(
        "Position dump",
        extra={"position_dump": "x" * 1_000_000},
    )

    # 3. Verify a placeholder datagram is delivered instead.
    messages = receive_messages(logstash_udp_socket, 1)
    message, datagram = next(iter(messages.items()))
    assert len(datagram) <= MAX_LOGSTASH_UDP_PAYLOAD_SIZE
    assert "dropped" in message

    # 4. Verify logging did not report an error.
    assert "--- Logging error ---" not in capsys.readouterr().err
