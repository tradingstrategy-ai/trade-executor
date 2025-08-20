"""Generic routing end-to-end test.

All tests use forked polygon mainnet.
"""
import os
import secrets
import logging

from pathlib import Path
from unittest.mock import patch

import flaky
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


os = os.environ.get("CI") == "true"

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
def spot_strategy_file() -> Path:
    """Where do we load our strategy file."""
    strat = Path(os.path.dirname(__file__)) / ".." / ".." / ".." / ".." / "strategies" / "test_only" / "generic_routing_end_to_end_spot.py"
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
        "MAX_DATA_DELAY_MINUTES": "1440",  # Don't crash on not doing candle refresh properly
        "GAS_BALANCE_WARNING_LEVEL": "0",  # Avoid unnecessary gas warnings
        "UNIT_TEST_FORCE_ANVIL": "true",
        "PATH": os.environ["PATH"],
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


# Flaky due to Anvil randomly reverting tx and causing a frozen position
@pytest.mark.skipif(CI, "Anvil too unstable on Github CI")
@flaky.flaky
def test_generic_routing_live_trading_start_spot_and_short(
    environment: dict,
    state_file: Path,
):
    """Run generic routing based executor live.

    - Use a forked polygon

    - Pretty slow test due to its nature ~1 minute
    """

    # Need to be initialised first
    result = run_init(environment)
    assert result.exit_code == 0

    cli = get_command(app)

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["start"])

        assert e.value.code == 0

    # Check that trades completed
    with state_file.open("rt") as inp:
        state = State.from_json(inp.read())
        assert len(state.portfolio.closed_positions) >= 7  # Sometimes 7, sometimes 8


# FAILED tests/ethereum/polygon_forked/generic-router/test_generic_router_end_to_end.py::test_generic_routing_test_trade_spot_and_short - assert 999.990918 == 499.990849 ± 5.0e-04
@flaky.flaky
def test_generic_routing_test_trade_spot_and_short(
    environment: dict,
    web3: Web3,
    state_file: Path,
):
    """Perform a test trade on short and spot strategy

    - Perform-test-trade should try both spot and short position
    """

    cli = get_command(app)

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade", "--all-pairs", "--no-test-credit-supply"])
        assert e.value.code == 0

    # Check the resulting state and see we made some trade for trading fee losses
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        assert len(list(state.portfolio.get_all_trades())) == 6
        reserve_value = state.portfolio.get_default_reserve_position().get_value()
        assert reserve_value == pytest.approx(499.990849)


@pytest.mark.slow_test_group
def test_generic_routing_live_trading_start_spot_only(
    environment: dict,
    state_file: Path,
    spot_strategy_file,
):
    """Run generic routing based executor live.

    - Use a forked polygon

    - Use a spot only stragegy (so we avoid issues with short brokeness)
    """

    environment["STRATEGY_FILE"] = spot_strategy_file.as_posix()

    # Need to be initialised first
    result = run_init(environment)
    assert result.exit_code == 0

    cli = get_command(app)

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["start"])

        assert e.value.code == 0

    # Check that trades completed
    with state_file.open("rt") as inp:
        state = State.from_json(inp.read())
        assert len(state.portfolio.closed_positions) == 8


@flaky.flaky
def test_generic_routing_test_trade_spot_only(
    environment: dict,
    web3: Web3,
    state_file: Path,
):
    """Perform a test trade using a spot only generic routing strategy

    - Forked Polygon mainnet

    """

    cli = get_command(app)

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade", "--all-pairs", "--no-test-short"])
        assert e.value.code == 0

    # Check the resulting state and see we made some trade for trading fee losses
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        assert len(list(state.portfolio.get_all_trades())) == 4
        reserve_value = state.portfolio.get_default_reserve_position().get_value()
        assert reserve_value == pytest.approx(499.993009)


def test_generic_routing_check_wallet(
    environment: dict,
    web3: Web3,
    state_file: Path,
):
    """Perform check wallet command for generic rounting.

    - Forked Polygon mainnet
    """

    cli = get_command(app)

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["check-wallet"])
        assert e.value.code == 0
