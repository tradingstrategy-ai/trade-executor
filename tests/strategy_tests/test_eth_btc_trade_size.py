"""Check trade sizes are capped on eth-btc-usdc-size-risk strategy."""
import os
from pathlib import Path
from unittest import mock

import pytest

from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State


@pytest.fixture()
def strategy_file() -> Path:
    p = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "eth-btc-usdc-size-risk.py"))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def state_file(tmp_path) -> Path:
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
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "USE_BINANCE": "false",  # Force to use DEX data for TVL and lending
        "GENERATE_REPORT": "true"
    }
    return environment


@pytest.mark.slow_test_group
def test_eth_btc_trade_size(
    environment: dict,
    state_file: Path,
):
    """Backtest the strategy with a short period.

    - Ensure credit handling code does not crash outright and we open backtest supply positions

    - The strategy module sets $10M as the backtest Parameters.initial_cash so it should be trade size limited a lot
    """

    with mock.patch.dict('os.environ', environment, clear=True):
        app(["backtest"], standalone_mode=False)

    state = State.read_json_file(state_file)

    all_positions = list(state.portfolio.get_all_positions())
    assert len(all_positions) == 10
    for p in state.portfolio.get_all_positions():
        print(p)

    # credit_positions = [p for p in state.portfolio.get_all_positions() if p.is_credit_supply()]
    #assert len(credit_positions) >= 15  # TODO: Why this number varies - calendar?

