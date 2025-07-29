"""Test dummy execution of a trading strategy cycle.

To run:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="secret-token:tradingstrategy-6ce98...."
    export POLYGON_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest --log-cli-level=info -s -k test_strategy_cycle_trigger

"""

import os

from pathlib import Path
from unittest import mock
import flaky

import pytest

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON"), reason="Set POLYGON_JSON_RPC environment variable to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "strategies", "test_only", "quickswap_dummy.py"))


@flaky.flaky
@pytest.mark.slow_test_group
def test_run_one_live_cycle(
        logger,
        strategy_path: Path,
    ):
    """Test dummy execution of a trading strategy cycle.
z
    - Takes ~3 minutes to complete

    - Run the trading cycle once

    - Downloads MATIC-USDC data for Quickswap

    - The strategy is a dummy placeholder, it will never give any trades to execute

    - Can be run with free or private Polygon RPC node

    - Uses live oracle for the data

    - Does not do any trades or need keys - uses DummyExecution model

    """

    state_file = "/tmp/test_run_one_live_cycle.json"

    # Set up the configuration for the backtesting,
    # run the loop 6 cycles using Ganache + live BNB Chain fork
    environment = {
        "STRATEGY_FILE": strategy_path.as_posix(),
        "JSON_RPC_POLYGON": os.environ["JSON_RPC_POLYGON"],
        "STATE_FILE": state_file,
        "RESET_STATE": "true",
        "ASSET_MANAGEMENT_MODE": "dummy",
        "STRATEGY_CYCLE_TRIGGER": "cycle_offset",
        "CACHE_PATH": "/tmp/test_run_one_live_cycle",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "CYCLE_DURATION": "1m",
        "UNIT_TESTING": "true",
        "MAX_CYCLES": "1",
        "LOG_LEVEL": "disabled",
    }

    # Don't use CliRunner.invoke() here,
    # as it patches stdout/stdin and causes our pdb to stop working
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["start"], standalone_mode=False)

    # Load state and check we got uptime
    with open(state_file, "rt") as inp:
        text = inp.read()
        try:
            state = State.from_json(text)
        except Exception as e:
            raise RuntimeError(f"Could not deserialise: {text}") from e

    # Check we compelted 1 cycle
    assert len(state.uptime.cycles_completed_at) == 1
    assert 1 in state.uptime.cycles_completed_at
