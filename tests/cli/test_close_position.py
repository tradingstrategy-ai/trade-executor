"""close-all CLI command end-to-end test."""
import json
import os
import secrets
import tempfile
import logging

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


logger = logging.getLogger(__name__)


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only" / "enzyme_end_to_end_multipair.py"


@pytest.fixture()
def state_file() -> Path:
    return Path(tempfile.mkdtemp()) / "test-close-all-state.json"


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    deployer: HexAddress,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    ) -> dict:
    """Set up environment vars for all CLI commands."""

    environment = {
        "EXECUTOR_ID": "test_close_all",
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
        "PATH": os.environ["PATH"],
    }
    return environment


def test_close_all(
    environment: dict,
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
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    # trade-executor perform-test-trade --buy-only
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade", "--buy-only"])
        assert e.value.code == 0

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 1

    # trade-executor close-all
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["close-all"])
        assert e.value.code == 0

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 0


def test_close_position_single(
    environment: dict,
    state_file: Path,
):
    """Perform close-position command

    - End-to-end high level test for the command

    - Create test EVM trading environment

    - Initialise strategy command

    - Perform buy only test trade command

    - Perform close singgle command
    """

    # trade-executor init
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    # trade-executor perform-test-trade --buy-only
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade", "--buy-only"])
        assert e.value.code == 0

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 1

    position = next(iter(state.portfolio.open_positions.values()))

    environment["POSITION_ID"] = str(position.position_id)

    # trade-executor close-all
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["close-position"])
        assert e.value.code == 0

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 0



def test_close_all_simulate(
    environment: dict,
    state_file: Path,
):
    """Perform close-all --simulate command"""

    environment = environment.copy()

    # trade-executor init
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    # trade-executor perform-test-trade --buy-only
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade", "--buy-only"])
        assert e.value.code == 0

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 1

    environment["SIMULATE"] = "true"

    # trade-executor close-all
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["close-all"])
        assert e.value.code == 0

    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 1
