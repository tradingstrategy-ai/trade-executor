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

import pytest
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON"), reason="Set POLYGON_JSON_RPC environment variable to run this test")


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "strategies", "test_only", "quickswap_dummy.py"))


def test_run_one_live_cycle(
        strategy_path: Path,
    ):
    """Test dummy execution of a trading strategy cycle.

    - Takes ~3 minutes to complete

    - Run the trading cycle once

    - The strateg is a dummy placeholder, it will never give any trades to execute

    - Can be run with free or private Polygon RPC node

    - Uses live oracle for the data

    - Does not do any trades or need keys - uses DummyExecution model

    """

    debug_dump_file = "/tmp/trading_data_availability_based_strategy_cycle_trigger.debug.json"

    state_file = "/tmp/trading_data_availability_based_strategy_cycle_trigger.json"

    # Set up the configuration for the backtesting,
    # run the loop 6 cycles using Ganache + live BNB Chain fork
    environment = {
        "STRATEGY_FILE": strategy_path.as_posix(),
        "JSON_RPC_POLYGON": os.environ["JSON_RPC_POLYGON"],
        "STATE_FILE": state_file,
        "RESET_STATE": "true",
        "EXECUTION_TYPE": "dummy",
        "STRATEGY_CYCLE_TRIGGER": "cycle_offset",
        "CACHE_PATH": "/tmp/trading_data_availability_based_strategy_cycle_trigger",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "DEBUG_DUMP_FILE": debug_dump_file,
        "CYCLE_DURATION": "1m",
        "CONFIRMATION_BLOCK_COUNT": "8",
        "MAX_POSITIONS": "2",
        "UNIT_TESTING": "true",
        "MAX_CYCLES": "1",
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
