"""Check that the webhook authenticates correctly"""
from queue import Queue

import requests
import flaky

from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.store import NoneStore
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.testing.webhook import create_webhook_server_with_retries, get_webhook_test_url


#  OSError: [Errno 98] Address already in use
@flaky.flaky()
def test_auth_ok(logger):
    """Username and password allow to access the webhook"""
    queue = Queue()
    server = create_webhook_server_with_retries("127.0.0.1", "test", "test", queue, NoneStore(), Metadata.create_dummy(), RunState())
    server_url = get_webhook_test_url(server, "test", "test")
    # Test home view
    try:
        resp = requests.get(server_url)
        assert resp.status_code == 200
    finally:
        server.shutdown()


def test_auth_failed(logger):
    """Wrong password denies the access to the webhook"""
    queue = Queue()
    server = create_webhook_server_with_retries("127.0.0.1", "test", "test", queue, NoneStore(), Metadata.create_dummy(), RunState())
    server_url = get_webhook_test_url(server, "test", "wrong")
    # Test home view
    try:
        resp = requests.get(server_url)
        assert resp.status_code == 403
    finally:
        server.shutdown()
