import os

import pytest
from tradingstrategy.client import Client


@pytest.fixture()
def strategy_folder():
    """Where unit test strategies are located."""
    return os.path.join(os.path.dirname(__file__), "../strategies/test_only")


@pytest.fixture()
def persistent_test_cache_path() -> str:
    return "/tmp/trading-strategy-tests"


@pytest.fixture(scope="session")
def persistent_test_client(persistent_test_cache_path) -> Client:
    """Create a client that never redownloads data in a local dev env.

    Read API key from TRADING_STRATEGY_API_KEY env variable.
    """
    c = Client.create_test_client(persistent_test_cache_path)
    return c
