"""Uniswap v2 routing model tests."""

import datetime
import logging
import os
import pickle
import secrets
from pathlib import Path
from typing import List

import pytest
from eth_account import Account

from eth_defi.gas import estimate_gas_fees, node_default_gas_price_strategy
from eth_defi.txmonitor import wait_transactions_to_complete
from eth_typing import HexAddress, HexStr
from hexbytes import HexBytes
from typer.testing import CliRunner
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.ganache import fork_network
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from eth_defi.utils import is_localhost_port_listening
from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2RoutingState, UniswapV2SimpleRoutingModel
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier

from tradeexecutor.cli.log import setup_pytest_logging


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")


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
def ganache_bnb_chain_fork(logger, large_busd_holder) -> str:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["BNB_CHAIN_JSON_RPC"]

    if not is_localhost_port_listening(19999):
        # Start Ganache
        launch = fork_network(
            mainnet_rpc,
            block_time=1,  # Insta mining cannot be done in this test
            evm_version="berlin",  # BSC is not yet London compatible?
            unlocked_addresses=[large_busd_holder],  # Unlock WBNB stealing
            quiet=True,  # Otherwise the Ganache output is millions lines of long
        )
        yield launch.json_rpc_url
        # Wind down Ganache process after the test is complete
        launch.close(verbose=True)
    else:
        # raise AssertionError("ganache zombie detected")

        # Uncomment to test against manually started Ganache
        yield "http://127.0.0.1:19999"


@pytest.fixture
def web3(ganache_bnb_chain_fork: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    web3 = Web3(HTTPProvider(ganache_bnb_chain_fork))
    web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)
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


@pytest.fixture
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
def busd_asset(busd_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(
        chain_id,
        busd_token.address,
        busd_token.functions.symbol().call(),
        busd_token.functions.decimals().call())


@pytest.fixture
def asset_wbnb(wbnb_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, wbnb_token.address, wbnb_token.functions.symbol().call(), wbnb_token.functions.decimals().call())


@pytest.fixture
def cake_asset(cake_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, cake_token.address, cake_token.functions.symbol().call(), cake_token.functions.decimals().call())


@pytest.fixture
def cake_busd_uniswap_trading_pair() -> HexAddress:
    return HexAddress(HexStr("0x804678fa97d91b974ec2af3c843270886528a9e6"))


@pytest.fixture
def cake_bnb_pair_address() -> HexAddress:
    """See https://tradingstrategy.ai/trading-view/binance/pancakeswap-v2/cake-bnb."""
    return HexAddress(HexStr("0x0ed7e52944161450477ee417de9cd3a859b14fd0"))


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


@pytest.fixture
def cake_busd_trading_pair(cake_asset, busd_asset, pancakeswap_v2) -> TradingPairIdentifier:
    """Cake-BUSD pair representation in the trade executor domain."""
    return TradingPairIdentifier(
        cake_asset,
        busd_asset,
        "0x804678fa97d91B974ec2af3c843270886528a9E6",  #  https://tradingstrategy.ai/trading-view/binance/pancakeswap-v2/cake-busd
        internal_id=1000,  # random number
        exchange_address=pancakeswap_v2.factory.address,
    )


@pytest.fixture(scope="module")
def routing_model():

    # Allowed exchanges as factory -> router pairs
    factory_router_map = {
        # Pancake
        "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73": ("0x10ED43C718714eb63d5aA57B78B54704E256024E", "0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5"),
        # Biswap
        #"0x858e3312ed3a876947ea49d572a7c42de08af7ee": ("0x3a6d8cA21D1CF76F653A67577FA0D27453350dD8", )
        # FSTSwap
        #"0x9A272d734c5a0d7d84E0a892e891a553e8066dce": ("0x1B6C9c20693afDE803B27F8782156c0f892ABC2d", ),
    }

    return UniswapV2SimpleRoutingModel(factory_router_map, set())


def test_simple_routing_one_leg(
        web3,
        hot_wallet,
        busd_asset,
        routing_model,
        cake_busd_trading_pair,
):
    """Make 1x two way trade BUSD -> Cake.

    - Buy Cake with BUSD
    """

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    # Create
    routing_state = UniswapV2RoutingState(tx_builder)

    txs = routing_model.trade(
        routing_state,
        cake_busd_trading_pair,
        busd_asset,
        100 * 10**18,  # Buy Cake worth of 100 BUSD,
        max_slippage=0.01,
    )





