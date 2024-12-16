"""Mainnet fork + state dump test case for a zero quantity bug."""
import os
import secrets
import shutil
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest

from eth_defi.provider.anvil import launch_anvil, AnvilLaunch
from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do Ethereum mainnet fork from the damaged situation."""

    mainnet_rpc = os.environ["JSON_RPC_ETHEREUM"]

    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=21135827,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        #anvil.close(log_level=logging.INFO)
        anvil.close()


@pytest.mark.skip(reaso="Needs fixing of Trading Universe - too many paris")
#@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") == "true", reason="This test seems to block Github CI for some reason")
def test_repair_zero_quantity_position(
    logger,
    persistent_test_cache_path: str,
    mocker,
    anvil: AnvilLaunch,
    tmp_path,
):
    """Check we repair positions that did not manage to open due to a failed trade."""

    strategy_path = os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "ethereum-memecoin-vol-basket.py")

    state_source = os.path.join(os.path.dirname(__file__), "zero-quantity-positions.json")
    assert Path(state_source).exists()
    state_path = tmp_path / "state.json"
    shutil.copy(state_source, state_path)

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "STATE_FILE": state_path.as_posix(),
        "CACHE_PATH": persistent_test_cache_path,
        "JSON_RPC_ETHEREUM": anvil.json_rpc_url,
        "PRIVATE_KEY": "0x" + secrets.token_hex(32),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "RAISE_ON_UNCLEAN": "true",  # check-accounts should complete
        "AUTO_APPROVE": "true",
    }

    mocker.patch.dict("os.environ", environment, clear=True)

    app(["repair"], standalone_mode=False)

    # Check we are good
    state = State.read_json_file(state_path)
    for p in state.portfolio.get_open_positions():
        assert p.get_quantity() > 0

    app(["correct-accounts"], standalone_mode=False)
    app(["check-accounts"], standalone_mode=False)
