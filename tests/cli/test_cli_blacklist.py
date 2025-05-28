"""Test blacklist CLI command."""
import os
import tempfile

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.main import get_command

from eth_defi.provider.anvil import AnvilLaunch
from eth_typing import HexAddress

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from tradeexecutor.state.identifier import AssetIdentifier

from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only" / "enzyme_end_to_end_multipair.py"


@pytest.fixture()
def state_file() -> Path:
    return Path(tempfile.mkdtemp()) / "test-cli-blacklist.json"


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    deployer: HexAddress,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    multipair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    ) -> dict:
    """Passed to init and start commands as multipair_environment variables"""
    # Set up the configuration for the live trader
    multipair_environment = {
        "EXECUTOR_ID": "test_cli_blacklists",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "TEST_EVM_UNISWAP_V2_ROUTER": uniswap_v2.router.address,
        "TEST_EVM_UNISWAP_V2_FACTORY": uniswap_v2.factory.address,
        "TEST_EVM_UNISWAP_V2_INIT_CODE_HASH": uniswap_v2.init_code_hash,
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
    }
    return multipair_environment


def test_cli_add_blacklist(
    environment: dict,
    state_file: Path,
    weth_asset: AssetIdentifier,
):
    """Perform close-all command

    - End-to-end high level test for the command

    - Create test EVM trading environment

    - Initialise strategy command

    - Perform buy only test trade command

    - Perform close all command
    """

    cli = get_command(app)

    # trade-executor init
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["blacklist"])
        assert e.value.code == 0

    add_env = environment.copy()
    add_env["ADD_TOKEN"] = weth_asset.address
    with patch.dict(os.environ, add_env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["blacklist"])
        assert e.value.code == 0

    remove_env = environment.copy()
    remove_env["REMOVE_TOKEN"] = weth_asset.address
    with patch.dict(os.environ, remove_env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["blacklist"])
        assert e.value.code == 0
