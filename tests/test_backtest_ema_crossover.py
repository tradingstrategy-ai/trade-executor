"""Test EMA cross-over strategy.

To run:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="secret-token:tradingstrategy-6ce98...."
    export BNB_CHAIN_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest --log-cli-level=info -s -k test_bnb_chain_16h_momentum

"""

import logging
import os
import pickle
from pathlib import Path
from unittest import mock

import pytest
from eth_defi.hotwallet import HotWallet
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "strategies", "ema-crossover-long-only-no-stop-loss.py"))


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="This test is too flaky on Github CI. Manual runs only.")
def test_ema_crossover(
        logger: logging.Logger,
        strategy_path: Path,
        ganache_bnb_chain_fork,
        hot_wallet: HotWallet,
        persistent_test_cache_path,
    ):
    """Run the strategy test

    - Use decision data from the past

    - Trade against live exchanges
    """

    run_backtest(
        strategy_path,
        start_at="2021-01-01",
        end_at="2022-01-01",
    )
