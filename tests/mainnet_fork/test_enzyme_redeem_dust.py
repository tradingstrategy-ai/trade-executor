"""Test Enzyme redemption where the redemption request has a closed position with dust."""
import os
import secrets
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradeexecutor.cli.main import app


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_POLYGON and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture(scope="module")
def end_block() -> int:
    """The chain point of time when we simulate the events."""
    block_time = 2
    days = 6
    return 62514843 - days*24*3600//block_time


@pytest.fixture()
def anvil(request: FixtureRequest, end_block) -> AnvilLaunch:
    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]
    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=end_block,  # The timestamp on when the broken position was created
    )
    try:
        yield anvil
    finally:
        #anvil.close(log_level=logging.INFO)
        anvil.close()


@pytest.fixture()
def state_file() -> Path:
    p = Path(os.path.join(os.path.dirname(__file__), "redeem-dust.json"))
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "enzyme-polygon-eth-btc-rsi.py"))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    state_file: Path,
    strategy_file: Path,
    end_block: int,
    ) -> dict:
    """Used by CLI commands, for setting up this test environment"""
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": "0x" + secrets.token_bytes(32).hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        # "LOG_LEVEL": "disabled",
        "LOG_LEVEL": "info",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_ADDRESS": "0xbba6B781e0BAC1798e4E715ef9b1113Bf2387544",
        "VAULT_ADAPTER_ADDRESS": "0x519f26bE61889656e83262ab56D75f00DFDAAEc1",
        "VAULT_PAYMENT_FORWARDER_ADDRESS": "0x638241c16aB5002298B24ED2F0074B4662042258",
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": "60554042",
        "SKIP_SAVE": "true",
        "PROCESS_REDEMPTION": "true",
        "PROCESS_REDEMPTION_END_BLOCK_HINT": str(end_block),
    }
    return environment


@pytest.mark.slow_test_group
def test_enzyme_redeem_dust(
    environment: dict,
    mocker,
):
    """Test Enzyme redemption where the redemption request has a closed position with dust.

    - The state file contains a closed aPolUSDC position

    - This position has small dust value left

    - Enzyme redemption request tries to redeem part of this dust (1 unit of aPolUSDC)

    - Make sure we can handle the dust redemption request on a closed position
    """

    mocker.patch.dict("os.environ", environment, clear=True)

    with pytest.raises(SystemExit) as sys_exit:
        app(["repair"], standalone_mode=False)

    with pytest.raises(SystemExit) as sys_exit:
        app(["correct-accounts"], standalone_mode=False)
    assert sys_exit.value.code == 0


