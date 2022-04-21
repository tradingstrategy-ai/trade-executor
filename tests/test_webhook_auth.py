"""Check that the webhook authenticates correctly"""
from queue import Queue
from threading import Thread

import requests

from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.webhook.server import create_webhook_server


def test_auth_ok():
    """Username and password allow to access the webhook"""
    queue = Queue()
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue, JSONFileStore(None), Metadata.create_dummy())
    server_url = "http://test:test@127.0.0.1:5000"
    # Test home view
    try:
        resp = requests.get(server_url)
        assert resp.status_code == 200
    finally:
        server.shutdown()


def test_auth_failed():
    """Wrong password denies the access to the webhook"""
    queue = Queue()
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue, JSONFileStore(None), Metadata.create_dummy())
    server_url = "http://test:wrong@127.0.0.1:5000"
    # Test home view
    try:
        resp = requests.get(server_url)
        assert resp.status_code == 403
    finally:
        server.shutdown()
