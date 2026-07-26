import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from logging import Logger
from pathlib import Path

import pytest

from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.token_cache import (
    install_token_cache,
    is_token_cache_rebuild_requested,
    is_token_cache_seeding_disabled,
    merge_into_token_cache_seed,
)
from eth_defi.testing.rpc_cache import seed_default_foundry_rpc_cache, seed_foundry_rpc_cache

from tradeexecutor.testing.pytest_helpers import phase_report_key
from tradingstrategy.client import Client

from tradeexecutor.cli.log import setup_pytest_logging


@pytest.fixture(scope="session", autouse=True)
def _seed_token_cache(worker_id: str) -> Iterator[None]:
    """Install eth-defi's private per-worker token cache."""
    if is_token_cache_seeding_disabled():
        yield
        return

    cache = install_token_cache(worker_id)
    try:
        yield
    finally:
        if is_token_cache_rebuild_requested():
            merge_into_token_cache_seed(cache)


@pytest.fixture(scope="session", autouse=True)
def _seed_foundry_rpc_cache() -> None:
    """Populate Foundry's mutable fork cache from eth-defi and project seeds."""
    seed_default_foundry_rpc_cache()
    seed_foundry_rpc_cache(Path(__file__).parent / "rpc_cache_seed")


@pytest.fixture(scope="session")
def anvil_fork_pool() -> Iterator[AnvilForkPool]:
    """Provide one worker-local pool of reusable fixed-block Anvil forks."""
    pool = AnvilForkPool()
    try:
        yield pool
    finally:
        pool.close_all()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Require every warm fork user to be scheduled with compatible fork users."""
    for marker_name in ("warm_rpc_test_group", "warm_rpc_high_value_group"):
        warm_items = [item for item in items if item.get_closest_marker(marker_name)]
        module_markers: dict[object, set[str | None]] = defaultdict(set)
        groups: set[str] = set()

        for item in warm_items:
            marker = item.get_closest_marker("xdist_group")
            group = marker.args[0] if marker and marker.args else None
            module_markers[item.module].add(group)
            if group:
                groups.add(group)

        missing_groups = [module.__name__ for module, markers in module_markers.items() if None in markers]
        if missing_groups:
            raise pytest.UsageError(f"{marker_name} modules need xdist_group markers: {', '.join(missing_groups)}")

        partial_groups = [module.__name__ for module, markers in module_markers.items() if len(markers) > 1]
        if partial_groups:
            raise pytest.UsageError(f"Warm RPC modules must use one xdist_group marker: {', '.join(partial_groups)}")

        if len(groups) > 3:
            raise pytest.UsageError(f"{marker_name} invocation has {len(groups)} fork groups; maximum is 3")


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


@pytest.fixture(autouse=True)
def _suppress_info_logging():
    """Reset root logger to WARNING after each test.

    CLI tests call setup_logging() which sets root logger to INFO.
    This persists across tests in the same xdist
    worker process, clogging CI output with thousands of INFO lines.
    """
    logging.getLogger().setLevel(logging.WARNING)
    yield
    logging.getLogger().setLevel(logging.WARNING)


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
