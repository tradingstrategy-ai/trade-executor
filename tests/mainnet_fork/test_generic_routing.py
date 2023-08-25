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
from eth_defi.anvil import fork_network_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.confirmation import wait_transactions_to_complete
from eth_defi.gas import node_default_gas_price_strategy
from eth_typing import HexAddress, HexStr

from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.ganache import fork_network
from eth_defi.hotwallet import HotWallet
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.testing.pytest_helpers import is_failed_test
from tradeexecutor.cli.log import setup_pytest_logging


pytestmark = pytest.mark.skipif(
    os.environ.get("JSON_RPC_POLYGON") is None,
    reason="Set JSON_RPC_POLYGON environment variable to Polygon node to run this test",
)


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture()
def large_usdc_holder() -> HexAddress:
    """A random account picked from Polygon chain that holds a lot of usdc.

    This account is unlocked on Ganache, so you have access to good usdc stash.

    `To find large holder accounts, use polygonscan <https://polygonscan.com/token/0x2791bca1f2de4661ed88a30c99a7a9449aa84174#balances>`_.
    """
    # Binance Hot Wallet 2
    return HexAddress(HexStr("0xe7804c37c13166fF0b37F5aE0BB07A3aEbb6e245"))


@pytest.fixture()
def anvil_polygon_chain_fork(request, logger, large_usdc_holder) -> str:
    """Create a testable fork of live polygon chain.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]

    launch = fork_network_anvil(mainnet_rpc, unlocked_addresses=[large_usdc_holder])
    try:
        yield launch.json_rpc_url
    finally:
        verbose_anvil_exit = is_failed_test(request)
        stdout, stderr = launch.close()

        if verbose_anvil_exit:
            print(f"Anvil stdout:\n{stdout.decode('utf-8')}")
            print(f"Anvil stderr:\n{stderr.decode('utf-8')}")


@pytest.fixture
def web3(anvil_polygon_chain_fork: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    web3 = Web3(HTTPProvider(anvil_polygon_chain_fork, request_kwargs={"timeout": 5}))
    install_chain_middleware(web3)
    return web3


@pytest.fixture
def chain_id(web3) -> int:
    """The test chain id (67)."""
    return web3.eth.chain_id


@pytest.fixture
def usdc_token(web3) -> Contract:
    """usdc with $4B supply."""
    # https://polygonscan.com/address/0x2791bca1f2de4661ed88a30c99a7a9449aa84174
    token = get_deployed_contract(
        web3, "ERC20MockDecimals.json", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    )
    return token


@pytest.fixture()
def hot_wallet(
    web3: Web3, usdc_token: Contract, large_usdc_holder: HexAddress
) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 polygon.
    """
    account = Account.create()
    web3.eth.send_transaction(
        {"from": large_usdc_holder, "to": account.address, "value": 2 * 10**18}
    )
    tx_hash = usdc_token.functions.transfer(account.address, 10_000 * 10**6).transact(
        {"from": large_usdc_holder}
    )
    wait_transactions_to_complete(web3, [tx_hash])
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "polygon-momentum-multipair-generic-routing.py"))


def test_generic_routing_backtest(
        logger: logging.Logger,
        strategy_path: Path,
        anvil_polygon_chain_fork,
        hot_wallet: HotWallet,
        persistent_test_cache_path,
    ):
    """Run the strategy test

    - Use decision data from the past

    - Trade against live exchanges
    """

    debug_dump_file = "/tmp/test_generic_routing.debug.json"

    state_file = "/tmp/test_generic_routing.json"

    # Set up the configuration for the backtesting,
    # run the loop 6 cycles using Ganache + live BNB Chain fork
    environment = {
        "EXECUTOR_ID": "test_generic_routing",
        "NAME": "test_generic_routing",
        "STRATEGY_FILE": strategy_path.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "HTTP_ENABLED": "false",
        "JSON_RPC": anvil_polygon_chain_fork,
        "GAS_PRICE_METHOD": "legacy",
        "STATE_FILE": state_file,
        "RESET_STATE": "true",
        "ASSET_MANAGEMENT_MODE": "backtest",
        "APPROVAL_TYPE": "unchecked",
        "CACHE_PATH": persistent_test_cache_path,
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "DEBUG_DUMP_FILE": debug_dump_file,
        "BACKTEST_START": "2023-1-1",
        "BACKTEST_END": "2023-08-20",
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
        assert len(debug_dump) == 231

    # See we can load the state after all this testing.
    # Mainly stresses on serialization/deserialization issues.
    json_text = open(state_file, "rt").read()
    state = State.from_json(json_text)
    state.perform_integrity_check()

    print(len(list(state.portfolio.get_all_trades())))

    # Check the stats of the first position when it was opened
    # assert len(state.visualisation.plots) > 0

    logger.info("All ok")