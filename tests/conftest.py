import os
from logging import Logger

import pytest
from tradingstrategy.client import Client

from tradeexecutor.cli.log import setup_pytest_logging


@pytest.fixture(scope="session")
def strategy_folder():
    """Where unit test strategies are located."""
    return os.path.join(os.path.dirname(__file__), "../strategies/test_only")


@pytest.fixture(scope="session")
def persistent_test_cache_path() -> str:
    return "/tmp/trading-strategy-tests"


@pytest.fixture(scope="session")
def persistent_test_client(persistent_test_cache_path) -> Client:
    """Create a client that never redownloads data in a local dev env.

    Read API key from TRADING_STRATEGY_API_KEY env variable.
    """
    c = Client.create_test_client(persistent_test_cache_path)
    return c


@pytest.fixture()
def logger() -> Logger:
    """Get rid of pyramid_openapi warnings in test output.

    .. code-block::

        WARNING  pyramid_openapi3 settings not found. Did you forget to call config.pyramid_openapi3_spec?

    Only seem to affect multitest runs.
    """
    return setup_pytest_logging()
