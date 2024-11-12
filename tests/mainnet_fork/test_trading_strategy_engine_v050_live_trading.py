"""Test live trading with trading strategy engine 0.5 and create_indicators based module.

To run:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="secret-token:tradingstrategy-6ce98...."
    export JSON_RPC_POLYGON=
    pytest --log-cli-level=info -s -k test_trading_strategy_engine_v050_live_trading

"""
import logging
import os
import pickle
from pathlib import Path
from unittest import mock

import flaky
import pytest
from eth_account import Account

from eth_defi.abi import get_deployed_contract
from eth_defi.provider.anvil import fork_network_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.gas import node_default_gas_price_strategy
from eth_typing import HexAddress, HexStr

from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

from tradeexecutor.cli.log import setup_pytest_logging


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("JSON_RPC_POLYGON") is None, reason="Set JSON_RPC_POLYGON environment variable to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture
def usdc_contract(web3) -> Contract:
    """BUSD with $4B supply."""
    # https://bscscan.com/address/0xe9e7cea3dedca5984780bafc599bd69add087d56
    token = get_deployed_contract(web3, "ERC20MockDecimals.json", "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56")
    return token


@pytest.fixture()
def large_polygon_usdc_holder() -> HexAddress:
    """A random account picked from Polygon mainnet that holds USDC.

    - We use bridged USDC.e here
    """
    return HexAddress(HexStr("0xe7804c37c13166fF0b37F5aE0BB07A3aEbb6e245"))  # Binance Hot Wallet 2


@pytest.fixture()
def large_matic_holder() -> HexAddress:
    """A random account picked from Polygon mainnet that holds MATIC.
    """
    return HexAddress(HexStr("0x0000000000000000000000000000000000001010"))


@pytest.fixture()
def anvil_polygon_chain_fork_rpc(logger, large_polygon_usdc_holder, large_matic_holder) -> str:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]

    launch = fork_network_anvil(
        mainnet_rpc,
        unlocked_addresses=[large_polygon_usdc_holder, large_matic_holder])
    try:
        yield launch.json_rpc_url
    finally:
        # launch.close(log_level=logging.INFO)
        launch.close()


@pytest.fixture
def web3(anvil_polygon_chain_fork_rpc: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    web3 = Web3(HTTPProvider(anvil_polygon_chain_fork_rpc, request_kwargs={"timeout": 5}))
    web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)
    install_chain_middleware(web3)
    return web3


@pytest.fixture
def chain_id(web3):
    return web3.eth.chain_id


@pytest.fixture()
def hot_wallet(
    web3: Web3,
    usdc_contract: Contract,
    large_polygon_usdc_holder: HexAddress,
    large_matic_holder: HexAddress,
) -> HotWallet:
    """Our trade-executor hot wallet account.

    Start with 10,000 USDC cash and 2 MATIC.
    """
    account = Account.create()
    tx_hash = web3.eth.send_transaction({"from": large_matic_holder, "to": account.address, "value": 2*10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc_contract.functions.transfer(account.address, 10_000 * 10**18).transact({"from": large_polygon_usdc_holder})
    assert_transaction_success_with_explanation(web3, tx_hash)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


def test_trading_strategy_engine_v050_live_trading(
    logger: logging.Logger,
    anvil_polygon_chain_fork_rpc,
    hot_wallet: HotWallet,
    persistent_test_cache_path,
    tmp_path,
    ):
    """Run the live strategy test using engine v0.5 for few seconds.

    - Fork Polygon mainnet

    - Check that the create_indicator and indicator result data looks correct

    - Check that the live strategy re-creates indicator data on every cycle
    """

    strategy_path = Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "test_trading_strategy_engine_v050.py"))
    assert strategy_path.exists()

    debug_dump_file = f"{tmp_path}/test_trading_strategy_engine_v050_live_trading.debug.json"
    state_file = f"{tmp_path}/test_trading_strategy_engine_v050_live_trading.json"

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path.as_posix(),
        "CACHE_PATH": persistent_test_cache_path,
        "JSON_RPC_ANVIL": anvil_polygon_chain_fork_rpc,
        "PRIVATE_KEY": hot_wallet.private_key.hex(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "CYCLE_DURATION": "1s",
        "MAX_CYCLES": "4",  # Run for 4 seconds, 4 cycles
        "MAX_DATA_DELAY_MINUTES": f"{10*60*24*365}",  # 10 years or "disabled"
        "DEBUG_DUMP_FILE": debug_dump_file,
        "STATE_FILE": state_file,
    }

    # Don't use CliRunner.invoke() here,
    # as it patches stdout/stdin and causes our pdb to stop working
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["start"], standalone_mode=False)

    # We should have three cycles worth of debug data
    with open(debug_dump_file, "rb") as inp:
        debug_dump = pickle.load(inp)
        # {3: {'cycle': 3, 'unrounded_timestamp': datetime.datetime(2024, 3, 18, 14, 5, 38, 2508), 'timestamp': datetime.datetime(2024, 3, 18, 14, 5, 38), 'strategy_cycle_trigger': 'cycle_offset', 'reserve_update_events': [], 'total_equity_at_start': 0, 'total_cash_at_start': 0, 'rsi_WETH': None, 'custom_test_indicator': [1, 2, 3, 4], 'rsi_WMATIC': None, 'rebalance_trades': []}}
        assert len(debug_dump) == 3
        cycle_3 = debug_dump[3]
        assert cycle_3["custom_test_indicator"] == [1, 2, 3, 4]

    # See we can load the state after all this testing.
    # Mainly stresses on serialization/deserialization issues.
    json_text = open(state_file, "rt").read()
    state = State.from_json(json_text)
    state.perform_integrity_check()
    logger.info("All ok")


def test_trading_strategy_engine_v050_backtest(
    logger: logging.Logger,
    persistent_test_cache_path,
    tmp_path,
    ):
    """Run the backtest command for v0.5 strategy module.

    - Check that we get backtest results
    """

    strategy_path = Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "test_trading_strategy_engine_v050.py"))
    assert strategy_path.exists()

    environment = {
        "ID": "test_trading_strategy_engine_v050_backtest",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path.as_posix(),
        "CACHE_PATH": persistent_test_cache_path,
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
    }

    # Don't use CliRunner.invoke() here,
    # as it patches stdout/stdin and causes our pdb to stop working
    curr_path = os.getcwd()
    try:
        os.chdir(tmp_path)
        os.makedirs("state", exist_ok=True)  # TODO: fix in backtest command
        with mock.patch.dict('os.environ', environment, clear=True):
            app(["backtest"], standalone_mode=False)
    finally:
        os.chdir(curr_path)

    # Generated in backtest.py
    state_file = Path(f"{tmp_path}/state/test_trading_strategy_engine_v050-backtest.json")
    assert state_file.exists(), f"backtest command generated different path for the state file, {state_file.resolve()} did not exist"

    # See we can load the state after all this testing.
    # Mainly stresses on serialization/deserialization issues.
    json_text = open(state_file, "rt").read()
    state = State.from_json(json_text)
    state.perform_integrity_check()

    logger.info("All ok")
