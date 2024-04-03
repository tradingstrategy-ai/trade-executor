"""Test EMA cross-over strategy.

To run:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="secret-token:tradingstrategy-6ce98...."
    export BNB_CHAIN_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest --log-cli-level=info -s -k test_bnb_chain_16h_momentum

"""
import datetime
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
from eth_defi.confirmation import wait_transactions_to_complete
from eth_defi.gas import node_default_gas_price_strategy
from eth_typing import HexAddress, HexStr

from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

from tradeexecutor.cli.log import setup_pytest_logging


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture
def busd_token(web3) -> Contract:
    """BUSD with $4B supply."""
    # https://bscscan.com/address/0xe9e7cea3dedca5984780bafc599bd69add087d56
    token = get_deployed_contract(web3, "ERC20MockDecimals.json", "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56")
    return token


@pytest.fixture()
def large_busd_holder() -> HexAddress:
    """A random account picked from BNB Smart chain that holds a lot of BUSD.

    This account is unlocked on Ganache, so you have access to good BUSD stash.

    `To find large holder accounts, use bscscan <https://bscscan.com/token/0xe9e7cea3dedca5984780bafc599bd69add087d56#balances>`_.
    """
    # Binance Hot Wallet 6
    return HexAddress(HexStr("0x8894E0a0c962CB723c1976a4421c95949bE2D4E3"))


@pytest.fixture()
def anvil_bnb_chain_fork(logger, large_busd_holder) -> str:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["BNB_CHAIN_JSON_RPC"]

    launch = fork_network_anvil(
        mainnet_rpc,
        unlocked_addresses=[large_busd_holder])
    try:
        yield launch.json_rpc_url
    finally:
        launch.close(log_level=logging.INFO)


@pytest.fixture
def web3(anvil_bnb_chain_fork: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    web3 = Web3(HTTPProvider(anvil_bnb_chain_fork, request_kwargs={"timeout": 5}))
    web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)
    install_chain_middleware(web3)
    return web3


@pytest.fixture
def chain_id(web3):
    return web3.eth.chain_id


@pytest.fixture()
def hot_wallet(web3: Web3, busd_token: Contract, large_busd_holder: HexAddress) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 BNB.
    """
    account = Account.create()
    web3.eth.send_transaction({"from": large_busd_holder, "to": account.address, "value": 2*10**18})
    tx_hash = busd_token.functions.transfer(account.address, 10_000 * 10**18).transact({"from": large_busd_holder})
    wait_transactions_to_complete(web3, [tx_hash])
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "ema-crossover-long-only-no-stop-loss.py"))


# web3.exceptions.BlockNotFound: Block with id: 'latest' not found. with third party JSON-RPC
# caused by gas pricing middleware
@flaky.flaky()
def test_ema_crossover(
        logger: logging.Logger,
        strategy_path: Path,
        anvil_bnb_chain_fork,
        hot_wallet: HotWallet,
        persistent_test_cache_path,
    ):
    """Run the strategy test

    - Use decision data from the past

    - Trade against live exchanges
    """

    debug_dump_file = "/tmp/test_bnb_chain_16h_momentum.debug.json"

    state_file = "/tmp/test_bnb_chain_16h_momentum.json"

    # Set up the configuration for the backtesting,
    # run the loop 6 cycles using Ganache + live BNB Chain fork
    environment = {
        "EXECUTOR_ID": "test_ema",
        "NAME": "test_ema",
        "STRATEGY_FILE": strategy_path.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "HTTP_ENABLED": "false",
        "JSON_RPC": anvil_bnb_chain_fork,
        "GAS_PRICE_METHOD": "legacy",
        "STATE_FILE": state_file,
        "RESET_STATE": "true",
        "ASSET_MANAGEMENT_MODE": "backtest",
        "APPROVAL_TYPE": "unchecked",
        "CACHE_PATH": persistent_test_cache_path,
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "DEBUG_DUMP_FILE": debug_dump_file,
        "BACKTEST_START": "2021-12-07",
        "BACKTEST_END": "2021-12-09",
        "BACKTEST_CANDLE_TIME_FRAME_OVERRIDE": "1d",  # Speed up testing / reduce download data
        "TICK_OFFSET_MINUTES": "10",
        "CONFIRMATION_BLOCK_COUNT": "8",
        "UNIT_TESTING": "true",
        "DISCORD_WEBHOOK_URL": "",  # Always disable,
        "LOG_LEVEL": "disabled",
    }

    # Don't use CliRunner.invoke() here,
    # as it patches stdout/stdin and causes our pdb to stop working
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["start"], standalone_mode=False)

    # We should have three cycles worth of debug data
    with open(debug_dump_file, "rb") as inp:
        debug_dump = pickle.load(inp)
        assert len(debug_dump) == 2

    # See we can load the state after all this testing.
    # Mainly stresses on serialization/deserialization issues.
    json_text = open(state_file, "rt").read()
    state = State.from_json(json_text)
    state.perform_integrity_check()

    # Check the stats of the first position when it was opened
    assert len(state.visualisation.plots) > 0

    logger.info("All ok")
