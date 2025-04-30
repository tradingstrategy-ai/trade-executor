"""Check entering the positions using Velvet Enso API."""

import os
import secrets
from pathlib import Path

import pytest
from typer.main import get_command
from web3 import Web3

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State

JSON_RPC_BINANCE = os.environ.get("JSON_RPC_BINANCE")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

CI = os.environ.get("CI") == "true"


pytestmark = pytest.mark.skipif(
    (not JSON_RPC_BINANCE or not TRADING_STRATEGY_API_KEY),
     reason="Set JSON_RPC_BINANCE and TRADING_STRATEGY_API_KEY needed to run this test"
)


@pytest.fixture()
def anvil() -> AnvilLaunch:
    """Launch mainnet fork."""

    anvil = launch_anvil(
        fork_url=JSON_RPC_BINANCE,
    )
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil) -> Web3:
    web3 = create_multi_provider_web3(anvil.json_rpc_url)
    assert web3.eth.chain_id == 56
    return web3


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "velvet-position-enter-check.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    path = tmp_path / "test_velvet_enter_check_position.json"
    return path


@pytest.fixture()
def vault_address():
    """Which Velvet vault to use for this test.

    - Private key is set as the vault owner

    - Private key is prefunded with ETH

    - There is $100 capital deposited, no open positions
    """
    # https://dapp.velvet.capital/ManagerVaultDetails/0x2213a945a93c2aa10bf4b6f0cfb1db12dadc61ba
    # https://basescan.org/address/0xc4db1ce83f6913cf667da4247aa0971dd0147349
    return "0x806b760f99ce80fa01bf9b3a8de6dd3590d4d1a9"


@pytest.fixture()
def environment(
    strategy_file,
    anvil,
    state_file,
    vault_address,
    persistent_test_client,

):
    cache_path = persistent_test_client.transport.cache_path
    environment = {
        "EXECUTOR_ID": "test_velvet_enter_check_position",
        "NAME": "test_velvet_enter_check_position",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": "0x" + secrets.token_bytes(32).hex(),
        "CACHE_PATH": cache_path,
        "JSON_RPC_BINANCE": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "velvet",
        "VAULT_ADDRESS": vault_address,
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "RUN_SINGLE_CYCLE": "true",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),  # 10 years or "disabled""
        "MIN_GAS_BALANCE": "0.0",
    }
    return environment


def test_velvet_check_enter_position(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run a single cycle test strategy and see we get correct replies in PositionAvailabilityResponse

    """

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)

    state = State.read_json_file(state_file)
    assert state.cycle == 1

    cli.main(args=["start"], standalone_mode=False)

    state = State.read_json_file(state_file)
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_value() > 5.0  # Should have 100 USDC starting balance

    messages = state.visualisation.get_messages_tail(5)
    assert messages[0] == "Ok"

