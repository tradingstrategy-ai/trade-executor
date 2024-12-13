"""Test trading data availability based strategy cycle strategy.

To run:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="secret-token:tradingstrategy-6ce98...."
    export POLYGON_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest --log-cli-level=info -s -k test_strategy_cycle_trigger

"""

import os
import pickle
import datetime
from pathlib import Path
from unittest import mock

import pytest

from tradeexecutor.cli.main import app
from tradeexecutor.cli.log import setup_pytest_logging

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON"), reason="Set POLYGON_JSON_RPC environment variable to run this test")


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "strategies", "test_only", "quickswap_dummy.py"))


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


temp_disable = datetime.datetime(2024, 12, 15)

@pytest.mark.skipif(datetime.datetime.utcnow() < temp_disable, reason="Temporary disabled, oracle server having an issue")
@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TEST"), reason="Slow tests skipping enabled")
@pytest.mark.slow_test_group
def test_trading_data_availability_based_strategy_cycle_trigger(
    logger,
    strategy_path: Path,
    tmp_path: Path,
    ):
    """Test live decision making triggers using trading data availability endpoint

    - This test will take > 5 minutes to complete

    - Uses live oracle for the data

    - Does not do any trades or need private keys - uses DummyExecution model

    - Web3 connection is still needed as it is used for the tested WMATIC-USDC
      asset live pricing
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
        "ASSET_MANAGEMENT_MODE": "dummy",
        "STRATEGY_CYCLE_TRIGGER": "trading_pair_data_availability",
        "CACHE_PATH": tmp_path.as_posix(),
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "DEBUG_DUMP_FILE": debug_dump_file,
        "CYCLE_DURATION": "1m",
        "CONFIRMATION_BLOCK_COUNT": "8",
        "MAX_POSITIONS": "2",
        "UNIT_TESTING": "true",
        "MAX_CYCLES": "1",
        "LOG_LEVEL": "info",
    }

    # Don't use CliRunner.invoke() here,
    # as it patches stdout/stdin and causes our pdb to stop working
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["start"], standalone_mode=False)

    # We did one cycle
    with open(debug_dump_file, "rb") as inp:
        debug_dump = pickle.load(inp)
        assert len(debug_dump) == 1
        cycle_1 = debug_dump[1]
        print(cycle_1)
        assert cycle_1["universe_update_poll_cycles"] > 0
