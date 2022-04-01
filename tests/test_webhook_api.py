"""Check API endpoints."""
from queue import Queue
from threading import Thread

import pytest
import requests

from tradeexecutor.webhook.server import create_webhook_server


@pytest.fixture()
def server():
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue)
    server_url = "http://test:test@127.0.0.1:5000"
    yield server_url
    server


def test_auth_ok():
    """Username and password allow to access the webhook"""
    queue = Queue()
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue)
    server_url = "http://test:test@127.0.0.1:5000"
    webhook_thread = Thread(target=server.run)
    webhook_thread.start()
    # Test home view
    try:
        resp = requests.get(server_url)
        assert resp.status_code == 200
    finally:
        server.close()


def test_auth_failed():
    """Wrong password denies the access to the webhook"""
    queue = Queue()
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue)
    server_url = "http://test:wrong@127.0.0.1:5000"
    webhook_thread = Thread(target=server.run)
    webhook_thread.start()
    # Test home view
    try:
        resp = requests.get(server_url)
        assert resp.status_code == 403
    finally:
        server.close()