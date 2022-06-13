"""Test EMA cross-over strategy.

To run:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="secret-token:tradingstrategy-6ce98...."
    export BNB_CHAIN_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest --log-cli-level=info -s -k test_bnb_chain_16h_momentum

"""
import datetime
import logging
import os

from pathlib import Path


import pytest
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest
from tradeexecutor.cli.log import setup_pytest_logging


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "strategies", "ema-crossover-long-only-no-stop-loss.py"))


def test_ema_crossover_synthetic_data(
    strategy_path,
    logger: logging.Logger,
    persistent_test_client,
    ):
    """Check that EMA crossover strategy does not have syntax bugs or similar."""

    client = persistent_test_client

    setup = setup_backtest(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2022, 1, 1),
        initial_deposit=10_000,
    )

    state, debug_dump = run_backtest(setup, client)

    assert len(debug_dump) == 214

    portfolio = state.portfolio
    assert len(list(portfolio.get_all_trades())) == 214
    buys = [t for t in portfolio.get_all_trades() if t.is_buy()]
    sells = [t for t in portfolio.get_all_trades() if t.is_sell()]

    assert len(buys) == 107
    assert len(sells) == 107

    # The actual result might vary, but we should slowly leak
    # portfolio valuation because losses on trading fees
    assert portfolio.get_current_cash() > 9000
    assert portfolio.get_current_cash() < 10_500
