"""Run quick smoke test on eth-btc-usdc-credit strategy."""
import os
from pathlib import Path
from unittest import mock

import pytest

from tradeexecutor.cli.commands.app import app


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "eth-btc-usdc-credit.py"))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    strategy_file: Path,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",

        # "LOG_LEVEL": "info",
        # "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],

        # Force to use DEX + lending data in the backtest
        "USE_BINANCE": "false",
    }
    return environment


def test_eth_btc_momentum_credit(
    environment: dict,
):
    """Backtest the strategy with a short period.

    - Ensure credit handling code does not crash outright
    """

    # Accounting is detect to be incorrect
    with mock.patch.dict('os.environ', environment, clear=True):
        with pytest.raises(SystemExit) as sys_exit:
            app(["backtest"], standalone_mode=False)
        assert sys_exit.value.code == 0

