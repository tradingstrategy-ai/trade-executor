"""Check that the webhook authenticates correctly"""
from queue import Queue

import requests
from eth_defi.utils import find_free_port

from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.store import NoneStore
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.webhook.server import create_webhook_server


def test_auth_ok(logger):
    """Username and password allow to access the webhook"""
    queue = Queue()
    port = find_free_port(20_000, 40_000, 20)
    server = create_webhook_server("127.0.0.1", port, "test", "test", queue, NoneStore(), Metadata.create_dummy(), RunState())
    server_url = f"http://test:test@127.0.0.1:{port}"
    # Test home view
    try:
        resp = requests.get(server_url)
        assert resp.status_code == 200
    finally:
        server.shutdown()


def test_auth_failed(logger):
    """Wrong password denies the access to the webhook"""
    queue = Queue()
    port = find_free_port(20_000, 40_000, 20)
    server = create_webhook_server("127.0.0.1", port, "test", "test", queue, NoneStore(), Metadata.create_dummy(), RunState())
    server_url = f"http://test:wrong@127.0.0.1:{port}"
    # Test home view
    try:
        resp = requests.get(server_url)
        assert resp.status_code == 403
    finally:
        server.shutdown()
