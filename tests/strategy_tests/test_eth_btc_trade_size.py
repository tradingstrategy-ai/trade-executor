"""Check trade sizes are capped on eth-btc-usdc-size-risk strategy."""
import os
from pathlib import Path
from unittest import mock

import pytest

from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.state import State


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


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
        "LOG_LEVEL": "info",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "USE_BINANCE": "false",  # Force to use DEX data for TVL and lending
        "GENERATE_REPORT": "false"
                           ""
                           ""
    }
    return environment


@pytest.mark.slow_test_group
def test_eth_btc_trade_size(
    logger,
    environment: dict,
    state_file: Path,
):
    """Backtest the strategy with a short period.

    - Do sample trading with a dummy strategy

    - Ensure credit handling code does not crash outright and we open backtest supply positions

    - The strategy module sets $10M as the backtest Parameters.initial_cash so it should be trade size limited a lot
    """

    with mock.patch.dict('os.environ', environment, clear=True):
        app(["backtest"], standalone_mode=False)

    state = State.read_json_file(state_file)

    # Check we stay within some tolerance
    for t in state.portfolio.get_all_trades():
        position_size_risk = t.position_size_risk

        assert t.price_impact_tolerance == 0.03

        if not t.pair.is_spot():
            # Ignore credit supply/withdraw
            continue

        if not t.is_sell():
            # Currently we store the size risk details only for trades increasing position,
            # it might be missing for close trades
            assert position_size_risk is not None, f"Trade lacks position size risk information: {t}"

            # Check only entries / increases and they stay good compared to 100M cash
            assert position_size_risk.accepted_size < 1_000_000

    # Check we also get credit positions
    credit_positions = [p for p in state.portfolio.get_all_positions() if p.is_credit_supply()]
    assert len(credit_positions) == 11

