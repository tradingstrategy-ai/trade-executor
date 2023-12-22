"""Test for a bug case from a productoin state."""

import os
import shutil
import tempfile
import logging

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.main import get_command

from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.hotwallet import HotWallet

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON_ARCHIVE"), reason="Set JSON_RPC_POLYGON_ARCHIVE environment variable to run this test")


@pytest.fixture
def anvil_polygon_chain_fork() -> str:
    """Create a testable fork of live Polygon.

    :return: JSON-RPC URL for Web3
    """
    mainnet_rpc = os.environ["JSON_RPC_POLYGON_ARCHIVE"]
    launch = fork_network_anvil(
        mainnet_rpc,
        fork_block_number=51_156_615
    )
    try:
        yield launch
    finally:
        # Wind down Anvil process after the test is complete
        # launch.close(log_level=logging.ERROR)
        pass


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only" / "reconfirm.py"


@pytest.fixture()
def state_file() -> Path:
    """Return mutable test state copy."""
    path = Path(tempfile.mkdtemp()) / "test-correct-account.json"
    source = os.path.join(os.path.dirname(__file__), "correct-accounts-token-price-missing.json")
    shutil.copy(source, path)
    return path


@pytest.fixture()
def environment(
    anvil_polygon_chain_fork: AnvilLaunch,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    ) -> dict:
    """Set up environment vars for all CLI commands."""

    environment = {
        "EXECUTOR_ID": "test_cli_correct_account_price_missing",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),  # Irrelevant
        "JSON_RPC_ANVIL": anvil_polygon_chain_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        # Set parameters from Enzyme vault deployment
        "VAULT_ADDRESS": "0xDD06559A12d99a5301602213FBcB3c40Dcc71F4E",
        "VAULT_ADAPTER_ADDRESS": "0x58FDa1d623e54B0d2f27f1D7fB38c3aB5eCcbd3b",
        "VAULT_PAYMENT_FORWARDER_ADDRESS": "0x6D1A63C9679afa68Dc61AB88F16542D6F1bFA3A3",
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": "43828921",
    }
    return environment


def test_cli_correct_account_price_missing(
    logger: logging.Logger,
    environment: dict,
    state_file: Path,
):
    """Perform account corrections.

    - The state has a missing price information:

    - Perform reconfirm for transactions broken in this state,
      taken from at a specific Polygon mainnet fork

    .. code-block:: text

          File "/Users/moo/code/ts/trade-executor/tradeexecutor/strategy/account_correction.py", line 329, in calculate_account_corrections
            usd_value = position.calculate_quantity_usd_value(diff) if position else None
          File "/Users/moo/code/ts/trade-executor/tradeexecutor/state/position.py", line 1510, in calculate_quantity_usd_value
            assert self.last_token_price, f"Asset price not available when calculating price for quantity: {quantity}"
        AssertionError: Asset price not available when calculating price for quantity: 0.425383816875787613

    TODO: We cannot simulate a full test, because we do not have HotWallet account unlock yet.
    """

    state = State.read_json_file(state_file)

    position = state.portfolio.closed_positions[1]
    value = position.calculate_quantity_usd_value(1)
    assert value is not None

