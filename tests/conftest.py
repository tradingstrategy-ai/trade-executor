import os
from logging import Logger

import pytest

from tradeexecutor.testing.pytest_helpers import phase_report_key
from tradingstrategy.client import Client

from tradeexecutor.cli.log import setup_pytest_logging


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """See tradeexecutor.testing.pytest_helpers."""
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # store test results for each phase of a call, which can
    # be "setup", "call", "teardown"
    item.stash.setdefault(phase_report_key, {})[rep.when] = rep


@pytest.fixture(scope="session")
def strategy_folder():
    """Where unit test strategies are located."""
    return os.path.join(os.path.dirname(__file__), "../strategies/test_only")


@pytest.fixture(scope="session")
def persistent_test_cache_path() -> str:
    """The path where tests store and cache the downloaded datsets.

    - Matches one used in tradingstrategy.tests.conftest
    """
    path = os.path.expanduser("~/.cache/trading-strategy-tests")
    return path


@pytest.fixture(scope="session")
def persistent_test_client(persistent_test_cache_path) -> Client:
    """Create a client that never redownloads data in a local dev env.

    Read API key from TRADING_STRATEGY_API_KEY env variable.
    """
    c = Client.create_test_client(persistent_test_cache_path)
    yield c
    c.close()


@pytest.fixture()
def logger() -> Logger:
    """Get rid of pyramid_openapi warnings in test output.

    .. code-block::

        WARNING  pyramid_openapi3 settings not found. Did you forget to call config.pyramid_openapi3_spec?

    Only seem to affect multitest runs.
    """
    return setup_pytest_logging()


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    # Make sure dataclasses-json is monkey patched
    from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
    patch_dataclasses_json()


# Use this to track RAM usage (RSS) over the execution
# to debug PyArrow memory leaks
#
# @pytest.fixture(autouse=True)
# def cleanup(request):
#     """Try to release pyarrow memory and avoid leaking."""
#     import gc
#     import psutil
#     p = psutil.Process()
#     rss = p.memory_info().rss
#     print(f"RSS is {rss:,}")
#     #gc.collect()
#     #import pyarrow
#     #pool = pyarrow.default_memory_pool()
#     #pool.release_unused()
#

