"""Check that the webhook authenticates correctly"""
from queue import Queue
from threading import Thread

import requests

from tradeexecutor.webhook.server import create_webhook_server


def test_auth_ok():
    """Username and password allow to access the webhook"""
    queue = Queue()
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue)
    server_url = "http://test:test@127.0.0.1:5000"
    webhook_thread = Thread(target=server.run)
    webhook_thread.start()
    # Test home view
    resp = requests.get(server_url)
    assert resp.status_code == 200
    server.close()


def test_auth_failed():
    """Wrong password denies the access to the webhook"""
    queue = Queue()
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue)
    server_url = "http://test:wrong@127.0.0.1:5000"
    webhook_thread = Thread(target=server.run)
    webhook_thread.start()
    # Test home view
    resp = requests.get(server_url)
    assert resp.status_code == 200
    server.close()