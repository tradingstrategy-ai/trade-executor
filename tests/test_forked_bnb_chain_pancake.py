"""Runs a test strategy with a forked Pancakeswap V2.

To run tests:

.. code-block:: shell

    export BNB_CHAIN_JSON_RPC=https://bsc-dataseed1.defibit.io/
    pytest -s -k test_forked_pancake

"""
import datetime
import logging
import os
import secrets
from decimal import Decimal
from pathlib import Path
from typing import List

import pytest
from eth_account import Account
from eth_typing import HexAddress, HexStr
from hexbytes import HexBytes

from eth_defi.utils import is_localhost_port_listening
from tradingstrategy.client import Client
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.ganache import fork_network
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer
from tradeexecutor.ethereum.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.ethereum.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2_revaluation import UniswapV2PoolRevaluator
from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.bootstrap import import_strategy_file
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.cli.log import setup_pytest_logging


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
from tradeexecutor.utils.timer import timed_task

pytestmark = pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


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
            unlocked_addresses=[large_busd_holder])
        yield launch.json_rpc_url
        # Wind down Ganache process after the test is complete
        launch.close(verbose=True)
    else:
        logger.warning("Detected existing Ganache running - terminate with: kill -9 $(lsof -ti:19999)")
        # Assume ganache-cli manually launched by the dev
        yield "http://localhost:19999"


@pytest.fixture
def web3(ganache_bnb_chain_fork: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return Web3(HTTPProvider(ganache_bnb_chain_fork, request_kwargs={"timeout": 2}))


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


@pytest.fixture()
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
def supported_reserves(web3: Web3, busd) -> List[AssetIdentifier]:
    """What reserve currencies we support for the strategy."""
    return [busd]


@pytest.fixture()
def hot_wallet(web3: Web3, busd_token: Contract, hot_wallet_private_key: HexBytes, large_busd_holder: HexAddress) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 BNB.
    """
    account = Account.from_key(hot_wallet_private_key)
    web3.eth.send_transaction({"from": large_busd_holder, "to": account.address, "value": 2*10**18})

    balance = web3.eth.get_balance(large_busd_holder)
    assert balance  > web3.toWei("1", "ether"), f"Account is empty {large_busd_holder}"

    busd_token.functions.transfer(account.address, 10_000 * 10**18).transact({"from": large_busd_holder})
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "strategies", "pancakeswap_v2_main_loop.py"))


@pytest.fixture()
def portfolio() -> Portfolio:
    """A portfolio loaded with the initial cash.

    We start with 10,000 USDC.
    """
    portfolio = Portfolio()
    return portfolio


@pytest.fixture()
def state(portfolio) -> State:
    return State(portfolio=portfolio)


def test_forked_pancake(
        logger: logging.Logger,
        web3: Web3,
        strategy_path: Path,
        ganache_bnb_chain_fork,
        hot_wallet: HotWallet,
        pancakeswap_v2: UniswapV2Deployment,
        state: State,
        persistent_test_client: Client,
        cake_token: Contract,
    ):
    """Run a strategy tick against PancakeSwap v2 on forked BSC.

    This checks we can trade "live" assets.
    """

    strategy_factory = import_strategy_file(strategy_path)
    approval_model = UncheckedApprovalModel()
    execution_model = UniswapV2ExecutionModelVersion0(pancakeswap_v2, hot_wallet, confirmation_block_count=0, confirmation_timeout=datetime.timedelta(minutes=1))
    sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
    revaluation_method = UniswapV2PoolRevaluator(pancakeswap_v2)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=execution_model,
        timed_task_context_manager=timed_task,
        sync_method=sync_method,
        revaluation_method=revaluation_method,
        pricing_model_factory=uniswap_v2_live_pricing_factory,
        approval_model=approval_model,
        client=persistent_test_client,
    )

    # Deconstruct strategy input
    runner: StrategyRunner = run_description.runner
    universe_constructor = run_description.universe_model

    # Set up internal tracing store
    debug_details = {"cycle": 1}

    # Use a fixed data in the past for the test
    ts = datetime.datetime(2021, 12, 7)

    # Refresh the trading universe for this cycle
    universe = universe_constructor.construct_universe(ts, live=False)

    # Run cycle checks
    runner.pretick_check(ts, universe)

    # Execute the strategy tick and trades
    runner.tick(ts, universe, state, debug_details)

    # The strategy is always going to do some trades
    assert len(debug_details["approved_trades"]) > 0

    # We evaluated trading pair daily candles for momentum
    assert debug_details["timepoint_candles_count"] == 1079

    # The algo executes 4 buys,
    # from the most weighted to least weighted
    trades: List[TradeExecution] = debug_details["rebalance_trades"]
    assert len(trades) == 4
    assert trades[0].pair.base.token_symbol == "Cake"
    assert trades[0].executed_quantity > Decimal(100)  # TODO: Depends on daily Cake price - fix when we have a historical trade simulator
    assert trades[1].pair.base.token_symbol == "BTT"
    assert trades[2].pair.base.token_symbol == "CHR"
    assert trades[3].pair.base.token_symbol == "CUB"

    # Check on-chain Cake balance matches what we traded from Pancake
    assert cake_token.functions.balanceOf(hot_wallet.address).call() > 0
