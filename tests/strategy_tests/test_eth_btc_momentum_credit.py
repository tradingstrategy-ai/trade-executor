"""Run quick smoke test on eth-btc-usdc-credit strategy."""
import os
from pathlib import Path
from unittest import mock

import pytest

from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "eth-btc-usdc-credit.py"))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """The strategy module where the broken accounting happened."""
    return Path(tmp_path / "state.json")


@pytest.fixture()
def environment(
    strategy_file: Path,
    state_file: Path,
) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "USE_BINANCE": "false",  # Force to use DEX + lending data in the backtest
    }
    return environment


def test_eth_btc_momentum_credit(
    environment: dict,
    state_file: Path,
):
    """Backtest the strategy with a short period.

    - Ensure credit handling code does not crash outright
    """

    # Accounting is detect to be incorrect
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["backtest"], standalone_mode=False)

    state = State.read_json_file(state_file)
    credit_positions = [p for p in state.portfolio.get_all_positions() if p.is_credit_supply()]
    assert len(credit_positions) == 15

