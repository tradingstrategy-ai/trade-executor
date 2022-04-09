"""Check API endpoints."""

from queue import Queue

import pytest
import requests

from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.webhook.server import create_webhook_server


@pytest.fixture()
def store() -> JSONFileStore:
    """Dummy state and store for the tests."""
    portfolio = Portfolio()
    state = State(portfolio=portfolio)
    store = JSONFileStore("/tmp/webhook-test.json")
    store.sync(state)
    return store


@pytest.fixture()
def server_url(store):
    queue = Queue()
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue, store)
    server_url = "http://test:test@127.0.0.1:5000"
    yield server_url
    server.shutdown()


def test_home(server_url):
    """Homepage renders plain text"""
    resp = requests.get(server_url)
    assert resp.status_code == 200
    # Chuck the Trade Executor server, version 0.1.0, our URL is http://127.0.0.1:5000
    assert resp.headers["content-type"] == "text/plain; charset=UTF-8"
    assert "Chuck the Trade Executor server" in resp.text


def test_ping(server_url):
    """Get pong for ping"""
    resp = requests.get(f"{server_url}/ping")
    assert resp.status_code == 200
    assert resp.json() == {"ping": "pong"}


def test_cors(server_url):
    """Cors headers are in place."""
    resp = requests.get(f"{server_url}/ping")
    assert "Access-Control-Allow-Origin" in resp.headers


def test_state(server_url):
    """Download an empty state."""

    # Create a state.
    resp = requests.get(f"{server_url}/state")
    assert resp.status_code == 200

    # Test deserialisation
    state_dict = resp.json()
    state = State.from_dict(state_dict)

    assert state.portfolio.next_trade_id == 1
    assert state.portfolio.next_position_id == 1
