"""Sets up the main loop and a strategy in a Ethereum Tester environment.

We test with ganache-cli mainnet forking.
"""
import json
import logging
import os
import pickle
import secrets
from pathlib import Path
from typing import List

import flaky
import pytest
from eth_account import Account
from eth_defi.provider.anvil import fork_network_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.gas import node_default_gas_price_strategy
from eth_typing import HexAddress, HexStr
from hexbytes import HexBytes
from typer.testing import CliRunner
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.provider.ganache import fork_network
from eth_defi.hotwallet import HotWallet
from eth_defi.confirmation import wait_transactions_to_complete
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from eth_defi.utils import is_localhost_port_listening
from tradeexecutor.cli.main import app
from tradeexecutor.state.identifier import AssetIdentifier

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.qstrader import HAS_QSTRADER

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None or not HAS_QSTRADER, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


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
def hot_wallet_private_key(web3) -> HexBytes:
    """Generate a private key"""
    return HexBytes(secrets.token_bytes(32))


@pytest.fixture
def busd_token(web3) -> Contract:
    """BUSD with $4B supply."""
    # https://bscscan.com/address/0xe9e7cea3dedca5984780bafc599bd69add087d56
    token = get_deployed_contract(web3, "ERC20MockDecimals.json", "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56")
    return token


def cake_token(web3) -> Contract:
    """CAKE token."""
    token = get_deployed_contract(web3, "ERC20MockDecimals.json", "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82")
    return token


@pytest.fixture()
def pancakeswap_v2(web3) -> UniswapV2Deployment:
    """Fetch live PancakeSwap v2 deployment.

    See https://docs.pancakeswap.finance/code/smart-contracts for more information
    """
    deployment = fetch_deployment(
        web3,
        "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
        "0x10ED43C718714eb63d5aA57B78B54704E256024E",
        # Taken from https://bscscan.com/address/0xca143ce32fe78f1f7019d7d551a6402fc5350c73#readContract
        init_code_hash="0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5",
        )
    return deployment


@pytest.fixture
def wbnb_token(pancakeswap_v2: UniswapV2Deployment) -> Contract:
    return pancakeswap_v2.weth


@pytest.fixture
def asset_busd(busd_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, busd_token.address, busd_token.functions.symbol().call(), busd_token.functions.decimals().call())


@pytest.fixture
def asset_wbnb(wbnb_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, wbnb_token.address, wbnb_token.functions.symbol().call(), wbnb_token.functions.decimals().call())


@pytest.fixture
def asset_cake(cake_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, cake_token.address, cake_token.functions.symbol().call(), cake_token.functions.decimals().call())


@pytest.fixture
def cake_busd_uniswap_trading_pair() -> HexAddress:
    return HexAddress(HexStr("0x804678fa97d91b974ec2af3c843270886528a9e6"))


@pytest.fixture
def wbnb_busd_uniswap_trading_pair() -> HexAddress:
    return HexAddress(HexStr("0x58f876857a02d6762e0101bb5c46a8c1ed44dc16"))


@pytest.fixture
def supported_reserves(busd) -> List[AssetIdentifier]:
    """What reserve currencies we support for the strategy."""
    return [busd]


@pytest.fixture()
def hot_wallet(web3: Web3, busd_token: Contract, hot_wallet_private_key: HexBytes, large_busd_holder: HexAddress) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 BNB.
    """
    account = Account.from_key(hot_wallet_private_key)
    web3.eth.send_transaction({"from": large_busd_holder, "to": account.address, "value": 2*10**18})
    tx_hash = busd_token.functions.transfer(account.address, 10_000 * 10**18).transact({"from": large_busd_holder})
    wait_transactions_to_complete(web3, [tx_hash])
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../../strategies/test_only", "pancakeswap_v2_main_loop.py"))


# Confirmation timeout issues
@pytest.mark.skip(reason="Confirmation timeout issues. Check how to change Anvil to zero block time.")
def test_main_loop_success(
        logger: logging.Logger,
        strategy_path: Path,
        anvil_bnb_chain_fork,
        hot_wallet: HotWallet,
        pancakeswap_v2: UniswapV2Deployment,
    ):
    """Run the main loop one time in a backtested date.

    A smoke test for setting up the whole trade executor live trading application in local Ethereum Tester environment
    and then executed one rebalance.
    """

    debug_dump_file = "/tmp/test_main_loop.debug.json"

    # Set up the configuration for the live trader
    environment = {
        "EXECUTOR_ID": "test_main_loop.py",
        "NAME": "test_main_loop.py",
        "STRATEGY_FILE": strategy_path.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "HTTP_ENABLED": "false",
        "JSON_RPC_BINANCE": anvil_bnb_chain_fork,
        "UNISWAP_V2_FACTORY_ADDRESS": pancakeswap_v2.factory.address,
        "UNISWAP_V2_ROUTER_ADDRESS": pancakeswap_v2.router.address,
        "UNISWAP_V2_INIT_CODE_HASH": pancakeswap_v2.init_code_hash,
        "CONFIRMATION_TIMEOUT": "90",
        "STATE_FILE": "/tmp/test_main_loop.json",
        "RESET_STATE": "true",
        "EXECUTION_TYPE": "uniswap_v2_hot_wallet",
        "APPROVAL_TYPE": "unchecked",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "DEBUG_DUMP_FILE": debug_dump_file,
        "BACKTEST_START": "2021-12-07",
        "BACKTEST_END": "2022-01-07",
        "MAX_CYCLES": "1",
        "DISCORD_WEBHOOK_URL": "",
        "CYCLE_DURATION": "1d",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
    }

    # https://typer.tiangolo.com/tutorial/testing/
    runner = CliRunner()

    try:
        result = runner.invoke(app, "start", env=environment)

        if result.exception:
            raise result.exception

        if result.exit_code != 0:
            logger.error("runner failed")
            for line in result.stdout.split('\n'):
                logger.error(line)
            raise AssertionError("runner launch failed")

        assert result.exit_code == 0
    except ValueError:
        # ValueError: I/O operation on closed file.
        # bug in Typer,
        # but the app should still have completed
        pass

    with open(debug_dump_file, "rb") as inp:
        debug_dump = pickle.load(inp)

        # We should have data only for one cycle
        assert len(debug_dump) == 1

        cycle_1 = debug_dump[1]
        assert len(cycle_1["approved_trades"]) == 2

