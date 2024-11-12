"""Test QuickSwap momentum strategy.

To run:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="secret-token:tradingstrategy-6ce98...."
    export POLYGON_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest --log-cli-level=info -s -k test_quickswap_momentum

.. note ::

    There seems to be one frozen position among the trades.

"""
import datetime
import logging
import os
import pickle
from pathlib import Path
from unittest import mock

import pytest
from eth_account import Account

from eth_defi.abi import get_deployed_contract
from eth_defi.confirmation import wait_transactions_to_complete
from eth_typing import HexAddress, HexStr

from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.provider.ganache import fork_network
from eth_defi.hotwallet import HotWallet
from tradeexecutor.cli.main import app

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.qstrader import HAS_QSTRADER

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("POLYGON_JSON_RPC") is None or not HAS_QSTRADER, reason="Set POLYGON_JSON_RPC environment variable to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture
def usdc_token(web3) -> Contract:
    """USDC on Polygon.

    Has 6 decimals.
    """
    # https://tradingstrategy.ai/trading-view/polygon/tokens/0x2791bca1f2de4661ed88a30c99a7a9449aa84174
    token = get_deployed_contract(web3, "ERC20MockDecimals.json", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
    return token


@pytest.fixture()
def large_usdc_holder() -> HexAddress:
    # https://polygonscan.com/address/0x06959153b974d0d5fdfd87d561db6d8d4fa0bb0b
    return HexAddress(HexStr("0x06959153B974D0D5fDfd87D561db6d8d4FA0bb0B"))


@pytest.fixture()
def ganache_fork(logger, large_usdc_holder) -> str:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["POLYGON_JSON_RPC"]

    launch = fork_network(
        mainnet_rpc,
        block_time=1,  # Insta mining cannot be done in this test
        evm_version="istanbul",  # For Polygon
        unlocked_addresses=[large_usdc_holder],  # Unlock WBNB stealing
        quiet=True,  # Otherwise the Ganache output is millions lines of long
    )
    yield launch.json_rpc_url
    # Wind down Ganache process after the test is complete
    launch.close(verbose=True)


@pytest.fixture
def web3(ganache_fork: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return Web3(HTTPProvider(ganache_fork))


@pytest.fixture
def chain_id(web3):
    return web3.eth.chain_id


@pytest.fixture()
def hot_wallet(web3: Web3, usdc_token: Contract, large_usdc_holder: HexAddress) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 MATIC.
    """
    account = Account.create()
    web3.eth.send_transaction({"from": large_usdc_holder, "to": account.address, "value": 2*10**18})
    tx_hash = usdc_token.functions.transfer(account.address, 10_000 * 10**6).transact({"from": large_usdc_holder})
    wait_transactions_to_complete(web3, [tx_hash])
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "quickswap-momentum.py"))


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="This test is too flaky on Github CI. Manual runs only.")
def test_quickswap_momentum(
        logger: logging.Logger,
        strategy_path: Path,
        ganache_fork,
        hot_wallet: HotWallet,
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
        "EXECUTOR_ID": "test_bnb_chain_16h_momentum",
        "NAME": "test_bnb_chain_16h_momentum",
        "STRATEGY_FILE": strategy_path.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "HTTP_ENABLED": "false",
        "JSON_RPC": ganache_fork,
        "GAS_PRICE_METHOD": "legacy",
        "STATE_FILE": state_file,
        "RESET_STATE": "true",
        "EXECUTION_TYPE": "uniswap_v2_hot_wallet",
        "APPROVAL_TYPE": "unchecked",
        "CACHE_PATH": "/tmp/main_loop_tests",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "DEBUG_DUMP_FILE": debug_dump_file,
        "BACKTEST_START": "2021-12-07",
        "BACKTEST_END": "2021-12-09",
        "TICK_OFFSET_MINUTES": "10",
        "CYCLE_DURATION": "16h",
        "CONFIRMATION_BLOCK_COUNT": "8",
        "MAX_POSITIONS": "2",
        "UNIT_TESTING": "true",
    }

    # Don't use CliRunner.invoke() here,
    # as it patches stdout/stdin and causes our pdb to stop working
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["start"], standalone_mode=False)

    with open(debug_dump_file, "rb") as inp:
        debug_dump = pickle.load(inp)

        assert len(debug_dump) == 3

        cycle_1 = debug_dump[1]
        cycle_2 = debug_dump[2]
        cycle_3 = debug_dump[3]

        # Check that we made trades based on 2 max position count
        assert cycle_1["cycle"] == 1
        assert cycle_1["timestamp"] == datetime.datetime(2021, 12, 7, 0, 0)
        assert len(cycle_1["approved_trades"]) == 2

        assert cycle_2["cycle"] == 2
        assert cycle_2["timestamp"].replace(minute=0) == datetime.datetime(2021, 12, 7, 16, 0)
        assert len(cycle_2["approved_trades"]) == 3

        assert cycle_3["cycle"] == 3
        assert len(cycle_3["approved_trades"]) == 4
        assert cycle_3["timestamp"].replace(minute=0) == datetime.datetime(2021, 12, 8, 8, 0)

