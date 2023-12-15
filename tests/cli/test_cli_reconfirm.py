"""Test transaction reconfirmation."""

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
    path = Path(tempfile.mkdtemp()) / "test-reconfirm.json"
    source = os.path.join(os.path.dirname(__file__), "reconfirm-test-state.json")
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
        "EXECUTOR_ID": "test_cli_reconfirm",
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


@pytest.mark.skip(reason="We do not have currently a good state trace to finish this test. Placeholder left so we can later poke this.")
def test_cli_reconfirm(
    logger: logging.Logger,
    environment: dict,
    state_file: Path,
):
    """Perform reconfirm command

    - Load a broken state snapshot from a test straetgy

    - Perform reconfirm for transactions broken in this state,
      taken from at a specific Polygon mainnet fork
    """

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 0

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["repair"])
        assert e.value.code == 0

    # After reconfirm, ETH-USDC spot position should be open
    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 1
