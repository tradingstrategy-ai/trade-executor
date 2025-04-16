"""Test CLI command and strateggy running with Velvet Capital vault."""

import os
from pathlib import Path

import pytest
from typer.main import get_command
from web3 import Web3

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State

JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")
VELVET_VAULT_OWNER_PRIVATE_KEY = os.environ.get("VELVET_VAULT_OWNER_PRIVATE_KEY")
CI = os.environ.get("CI") == "true"


pytestmark = pytest.mark.skipif(
    (not JSON_RPC_BASE or not TRADING_STRATEGY_API_KEY or not VELVET_VAULT_OWNER_PRIVATE_KEY) or CI,
     reason="Set JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY and VELVET_VAULT_OWNER_PRIVATE_KEYneeded to run this test"
)


@pytest.fixture()
def anvil() -> AnvilLaunch:
    """Launch mainnet fork."""

    anvil = launch_anvil(
        fork_url=JSON_RPC_BASE,
    )
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil) -> Web3:
    web3 = create_multi_provider_web3(anvil.json_rpc_url)
    assert web3.eth.chain_id == 8453
    return web3


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "base-memecoin-index.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    path = tmp_path / "base-memecoin-index.json"
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
    return "0x2213a945a93c2aa10bf4b6f0cfb1db12dadc61ba"


@pytest.fixture()
def environment(
    strategy_file,
    anvil,
    state_file,
    vault_address,
):
    environment = {
        "EXECUTOR_ID": "test_base_memecoin_inddex_velvet",
        "NAME": "test_base_memecoin_inddex_velvet",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": VELVET_VAULT_OWNER_PRIVATE_KEY,
        "JSON_RPC_BASE": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "velvet",
        "VAULT_ADDRESS": vault_address,
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "RUN_SINGLE_CYCLE": "true",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),  # 10 years or "disabled""
        "MIN_GAS_BALANCE": "0.005",
        "RAISE_ON_UNCLEAN": "true",  # For correct-accounts
        "GENERATE_REPORT": "false",
    }
    return environment


def test_velvet_check_wallet(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run check-walet Velvet vault."""

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["check-wallet"], standalone_mode=False)


def test_velvet_check_universe(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run check-walet Velvet vault."""

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["check-universe"], standalone_mode=False)


def test_velvet_perform_test_trade(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run a single test trade using Velvet vault."""

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["perform-test-trade", "--pair", "(base, uniswap-v2, KEYCAT, WETH, 0.0030)"], standalone_mode=False)

    state = State.read_json_file(state_file)
    keycat_trades = [t for t in state.portfolio.get_all_trades() if t.pair.base.token_symbol == "KEYCAT"]
    assert len(keycat_trades) == 2

    for t in keycat_trades:
        assert t.is_success()


@pytest.mark.skipif(CI, reason="Too slow to run on Github, > 600 seconds")
def test_velvet_backtest(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run backtest using a Velvet vault strat."""

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["backtest"], standalone_mode=False)


def test_base_memecoin_index_velvet_single_cycle(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run a single cycle of Memecoin index strategy to see everything works.

    - Should attempt to open multiple positions using Enso
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
    assert len(state.visualisation.get_messages_tail(5)) == 1
    assert len(state.portfolio.frozen_positions) == 0


def test_velvet_correct_accounts(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Check correct-accounts works with Velvet sync model.

    - This test checks code runs, but does not attempt to repair any errors
    """

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["correct-accounts"], standalone_mode=False)
