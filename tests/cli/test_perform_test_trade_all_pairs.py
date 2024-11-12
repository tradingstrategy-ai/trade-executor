"""Perform a hot-wallet based test trade on all pairs in the trading universe."""
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

from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only" / "enzyme_end_to_end_multipair.py"


@pytest.fixture()
def state_file() -> Path:
    return Path(tempfile.mkdtemp()) / "test-close-all-state.json"


@pytest.fixture()
def multipair_environment_all_pairs(
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
        "EXECUTOR_ID": "test_perform_test_trade_all_pairs",
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
        "MAX_CYCLES": "5",  # Run decide_trades() 5 times
        "ALL_PAIRS": 'True',
    }
    return multipair_environment


def test_perform_test_trade_all_pairs(
    multipair_environment_all_pairs: dict,
    state_file: Path,
):
    """Perform close-all command

    - End-to-end high level test for the command

    - Create test EVM trading environment

    - Initialise strategy command

    - Perform buy only test trade command

    - Perform close all command
    """

    # trade-executor init
    cli = get_command(app)
    with patch.dict(os.environ, multipair_environment_all_pairs, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    # trade-executor perform-test-trade --buy-only
    cli = get_command(app)
    with patch.dict(os.environ, multipair_environment_all_pairs, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade"])
        assert e.value.code == 0

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 0

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 0