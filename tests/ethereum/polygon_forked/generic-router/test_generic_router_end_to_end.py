"""Generic routing end-to-end test.

All tests use forked polygon mainnet.
"""
import os
import secrets
import logging

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import Result
from typer.main import get_command
from typer.testing import CliRunner

from web3 import Web3
from eth_account import Account
from hexbytes import HexBytes


from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State


logger = logging.getLogger(__name__)


@pytest.fixture
def hot_wallet(
    web3,
    large_usdc_holder,
    usdc: TokenDetails,
) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": large_usdc_holder, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.contract.functions.transfer(wallet.address, 500 * 10**6).transact({"from": large_usdc_holder})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    strat = Path(os.path.dirname(__file__)) / ".." / ".." / ".." / ".." / "strategies" / "test_only" / "generic_routing_end_to_end.py"
    strat = strat.resolve()
    assert strat.exists(), f"Does not exist: {strat}"
    return strat


@pytest.fixture()
def state_file() -> Path:
    """The state file for the tests.

    Always start with an empty file.
    """
    path = Path("/tmp/test_generic_routing_end_to_end.json")
    if path.exists():
        os.remove(path)
    return path


@pytest.fixture()
def environment(
    anvil_polygon_chain_fork: AnvilLaunch,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "EXECUTOR_ID": "test_generic_router_end_to_end",
        "NAME": "test_generic_router_end_to_end",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil_polygon_chain_fork,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "MAX_CYCLES": "10",  # Run decide_trades() N times
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
    }
    return environment


def run_init(environment: dict) -> Result:
    """Run vault init command"""

    # https://typer.tiangolo.com/tutorial/testing/
    runner = CliRunner()

    # Need to use patch here, or parent shell env vars will leak in and cause random test failres
    with patch.dict(os.environ, environment, clear=True):
        result = runner.invoke(app, "init", env=environment)

    if result.exception:
        raise result.exception

    return result


def test_generic_routing_live_trading_init(
    environment: dict,
    state_file: Path,
):
    """Initialize execution for generic routing strategy."""

    result = run_init(environment)
    assert result.exit_code == 0

    # Check the initial state sync set some of the variables
    with state_file.open("rt") as inp:
        state = State.from_json(inp.read())
        assert state.sync.deployment.initialised_at is not None
        assert state.sync.deployment.block_number > 1


def test_generic_routing_live_trading_start(
    environment: dict,
    state_file: Path,
):
    """Run generic routing based executor live.

    - Use a forked polygon
    """

    # Need to be initialised first
    result = run_init(environment)
    assert result.exit_code == 0

    # Run strategy for few cycles.
    # Manually call the main() function so that Typer's CliRunner.invoke() does not steal
    # stdin and we can still set breakpoints
    cli = get_command(app)

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["start"])

        assert e.value.code == 0

    # Check that trades completed
    with state_file.open("rt") as inp:
        state = State.from_json(inp.read())
        assert len(state.portfolio.closed_positions) == 1
        assert len(state.portfolio.open_pos**18 * 0.03112978758721282)


def test_generic_routing_test_trade(
    environment: dict,
    web3: Web3,
    state_file: Path,
):
    """Perform a test trade on Enzymy vault via CLI.

    - Use a vault deployed by the test fixtures

    - Initialise the strategy to use this vault

    - Perform a test trade on this fault
    """

    cli = get_command(app)

    # Deposit some USDC to start
    deposit_amount = 500 * 10**6
    tx_hash = usdc.functions.approve(vault.comptroller.address, deposit_amount).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = vault.comptroller.functions.buyShares(deposit_amount, 1).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.functions.balanceOf(vault.address).call() == deposit_amount
    logger.info("Deposited %d %s at block %d", deposit_amount, usdc.address, web3.eth.block_number)

    # Check we have a deposit event
    logs = vault.comptroller.events.SharesBought.get_logs()
    logger.info("Got logs %s", logs)
    assert len(logs) == 1

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade"])
        assert e.value.code == 0

    assert usdc.functions.balanceOf(vault.address).call() < deposit_amount, "No deposits where spent; trades likely did not happen"

    # Check the resulting state and see we made some trade for trading fee losses
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        assert len(list(state.portfolio.get_all_trades())) == 2
        reserve_value = state.portfolio.get_default_reserve_position().get_value()
        assert reserve_value == pytest.approx(499.994009)


