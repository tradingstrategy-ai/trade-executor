"""Webhook testing helpers."""

import errno
from queue import Queue

from eth_defi.utils import find_free_port

from tradeexecutor.state.metadata import Metadata
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.webhook.server import WebhookServer, create_webhook_server


def create_webhook_server_with_retries(
    host: str,
    username: str,
    password: str,
    queue: Queue,
    store,
    metadata: Metadata,
    execution_state: RunState,
    *,
    port_start: int = 20_000,
    port_end: int = 40_000,
    port_search_tries: int = 20,
    bind_tries: int = 5,
) -> WebhookServer:
    """Create a webhook server while retrying transient port binding races.

    The test suite runs webhook tests in parallel, so a port returned by
    `find_free_port()` may be claimed by another worker before Waitress binds it.
    Retry only `EADDRINUSE` errors so real server startup failures still surface.
    """

    last_exception: OSError | None = None

    for _attempt in range(bind_tries):
        port = find_free_port(port_start, port_end, port_search_tries)

        try:
            return create_webhook_server(
                host,
                port,
                username,
                password,
                queue,
                store,
                metadata,
                execution_state,
            )
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise

            last_exception = exc

    assert last_exception is not None
    raise last_exception


def get_webhook_test_url(server: WebhookServer, username: str, password: str) -> str:
    """Build an authenticated URL for a started webhook test server."""
    return f"http://{username}:{password}@127.0.0.1:{server.effective_port}"
