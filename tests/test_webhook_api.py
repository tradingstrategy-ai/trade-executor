"""Check API endpoints."""
from queue import Queue
from threading import Thread

import pytest
import requests

from tradeexecutor.webhook.server import create_webhook_server


@pytest.fixture()
def server_url():
    queue = Queue()
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue)
    server_url = "http://test:test@127.0.0.1:5000"
    yield server_url
    server.shutdown()


def test_home(server_url):
    """Username and password allow to access the webhook"""
    resp = requests.get(server_url)
    assert resp.status_code == 200
    # Chuck the Trade Executor server, version 0.1.0, our URL is http://127.0.0.1:5000
    assert resp.headers["content-type"] == "text/plain; charset=UTF-8"
    assert "Chuck the Trade Executor server" in resp.text

