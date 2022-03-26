"""Buy a ERC-20 token with token tax which the strategy cannot figure out how to sell.

To run tests:

.. code-block:: shell

    export BNB_CHAIN_JSON_RPC=https://bsc-dataseed1.defibit.io/
    pytest -s -k test_frozen_position_and_blacklist


This test executes against LIVE bit-busd and bnb-busd order books on PancakeSwap v2,
so prices may vary.
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
from eth_hentai.txmonitor import wait_transactions_to_complete
from eth_typing import HexAddress, HexStr
from hexbytes import HexBytes

from eth_hentai.utils import is_localhost_port_listening
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import StaticUniverseModel
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_hentai.abi import get_deployed_contract
from eth_hentai.ganache import fork_network
from eth_hentai.hotwallet import HotWallet
from eth_hentai.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer
from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2_revaluation import UniswapV2PoolRevaluator
from tradeexecutor.state.state import AssetIdentifier, Portfolio, State, TradeExecution, TradingPairIdentifier, \
    TradingPosition
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

    Unlike other tests, we use 1 second block time,
    because we need to test failed transaction scenarios
    and otherwise this cannot be emulated.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["BNB_CHAIN_JSON_RPC"]

    assert not is_localhost_port_listening(19999), "Ganache alread running"
    # Start Ganache
    launch = fork_network(
        mainnet_rpc,
        block_time=1,
        unlocked_addresses=[large_busd_holder])
    yield launch.json_rpc_url
    # Wind down Ganache process after the test is complete
    launch.close(verbose=True)


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


@pytest.fixture
def wbnb_token(web3) -> Contract:
    token = get_deployed_contract(web3, "ERC20MockDecimals.json", "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c")
    return token


@pytest.fixture()
def bit_token(web3) -> Contract:
    """Biconomy token is a token with 3% tax.

    Any naive sell with approve() + swap() will fail.

    https://tradingstrategy.ai/trading-view/binance/pancakeswap-v2/bit-busd

    https://bscscan.com/address/0xc864019047b864b6ab609a968ae2725dfaee808a
    """
    token = get_deployed_contract(web3, "ERC20MockDecimals.json", "0xc864019047B864B6ab609a968ae2725DFaee808A")
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
def asset_bit(bit_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, bit_token.address, bit_token.functions.symbol().call(), bit_token.functions.decimals().call())


@pytest.fixture
def asset_wbnb(wbnb_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, wbnb_token.address, wbnb_token.functions.symbol().call(), wbnb_token.functions.decimals().call())


@pytest.fixture
def bit_busd_pair_address() -> HexAddress:
    return HexAddress(HexStr("0xf1b66d220479a5620677518c139d0a33f609793b"))


@pytest.fixture
def wbnb_busd_pair_address() -> HexAddress:
    return HexAddress(HexStr("0x58f876857a02d6762e0101bb5c46a8c1ed44dc16"))


@pytest.fixture
def wbnb_busd_pair(wbnb_busd_pair_address, asset_wbnb, asset_busd) -> TradingPairIdentifier:
    return TradingPairIdentifier(asset_wbnb, asset_busd, wbnb_busd_pair_address, internal_id=int(wbnb_busd_pair_address, 16))


@pytest.fixture
def bit_busd_pair(bit_busd_pair_address, asset_bit, asset_busd) -> TradingPairIdentifier:
    return TradingPairIdentifier(asset_bit, asset_busd, bit_busd_pair_address, internal_id=int(bit_busd_pair_address, 16))


@pytest.fixture
def supported_reserves(web3: Web3, asset_busd) -> List[AssetIdentifier]:
    """What reserve currencies we support for the strategy."""
    return [asset_busd]


@pytest.fixture()
def hot_wallet(web3: Web3, busd_token: Contract, hot_wallet_private_key: HexBytes, large_busd_holder: HexAddress) -> HotWallet:
    """Our trading Ethereum account.

    Start with 100 USDC cash and 2 BNB.
    """
    account = Account.from_key(hot_wallet_private_key)
    web3.eth.send_transaction({"from": large_busd_holder, "to": account.address, "value": 2*10**18})

    balance = web3.eth.getBalance(large_busd_holder)
    assert balance  > web3.toWei("1", "ether"), f"Account is empty {large_busd_holder}"

    txid = busd_token.functions.transfer(account.address, 100 * 10**18).transact({"from": large_busd_holder})
    wait_transactions_to_complete(web3, [txid])

    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "strategies", "frozen_asset.py"))


@pytest.fixture()
def portfolio() -> Portfolio:
    """An empty portfolio."""
    portfolio = Portfolio()
    return portfolio


@pytest.fixture()
def state(portfolio) -> State:
    return State(portfolio=portfolio)


@pytest.fixture()
def exchange_universe(web3, pancakeswap_v2: UniswapV2Deployment) -> ExchangeUniverse:
    """We trade on one Uniswap v2 deployment on tester."""
    return create_exchange_universe(web3, [pancakeswap_v2])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, wbnb_busd_pair, bit_busd_pair) -> PandasPairUniverse:
    """We trade on two trading pairs."""
    exchange = next(iter(exchange_universe.exchanges.values()))
    return create_pair_universe(web3, exchange, [wbnb_busd_pair, bit_busd_pair])


@pytest.fixture()
def universe(web3, exchange_universe: ExchangeUniverse, pair_universe: PandasPairUniverse) -> Universe:
    """Get our trading universe."""
    return Universe(
        time_frame=TimeBucket.d1,
        chains=[ChainId(web3.eth.chain_id)],
        exchanges=list(exchange_universe.exchanges.values()),
        pairs=pair_universe,
        candles=GroupedCandleUniverse.create_empty_qstrader(),
        liquidity=GroupedLiquidityUniverse.create_empty(),
    )


@pytest.fixture()
def universe_model(universe, supported_reserves) -> StaticUniverseModel:
    """Model the trading universe for the trade executor."""
    return StaticUniverseModel(TradingStrategyUniverse(
        universe=universe,
        reserve_assets=supported_reserves
    ))


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "strategies", "frozen_asset.py"))


@pytest.fixture()
def runner(
        pancakeswap_v2,
        strategy_path,
        web3,
        hot_wallet,
        universe_model,
) -> StrategyRunner:
    """Construct the strategy runner."""

    strategy_factory = import_strategy_file(strategy_path)
    approval_model = UncheckedApprovalModel()
    execution_model = UniswapV2ExecutionModel(pancakeswap_v2, hot_wallet)
    sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=execution_model,
        timed_task_context_manager=timed_task,
        sync_method=sync_method,
        revaluation_method=UniswapV2PoolRevaluator(pancakeswap_v2),
        pricing_model_factory=uniswap_v2_live_pricing_factory,
        approval_model=approval_model,
        client=None,
        universe_model=universe_model,
        cash_buffer=0.50,
    )

    # Deconstruct strategy description
    runner: StrategyRunner = run_description.runner

    return runner


def test_buy_and_sell_blacklisted_asset(
        logger: logging.Logger,
        strategy_path: Path,
        web3: Web3,
        hot_wallet: HotWallet,
        pancakeswap_v2: UniswapV2Deployment,
        universe_model: StaticUniverseModel,
        state: State,
        runner,
        wbnb_busd_pair,
        bit_busd_pair
    ):
    """Try to buy/sell BIT token.

    Day 1

    - Bit token buy should be ok

    Day 2

    - Bit token cannot be sold naively, as it has 3% transfer tax
    - See Bit token gets moved to frozen positions
    - See Bit token gets blacklisted

    Day 3

    - See there is no further Bit token buy attempts

    The actual timestamp datetime.datetime(2020, 1, 1) does not matter
    in test, as the real live today prices are used from Ganache RPC.
    """

    assert len(state.asset_blacklist) == 0

    executor_universe: TradingStrategyUniverse = universe_model.universe
    universe = executor_universe.universe

    # Check assets
    exchange = universe.exchanges[0]
    wbnb_busd = universe.pairs.get_one_pair_from_pandas_universe(exchange.exchange_id, "WBNB", "BUSD")
    bit_busd = universe.pairs.get_one_pair_from_pandas_universe(exchange.exchange_id, "BIT", "BUSD")

    assert wbnb_busd
    assert bit_busd

    # see strategy/simulated_uniswap.py for different days we can have 0, 1, 2

    #
    # 1st day
    #

    # We start with day_kind 1 that is all ETH day.
    debug_details = runner.tick(datetime.datetime(2020, 1, 1), executor_universe, state, {"cycle": 1})
    weights = debug_details["alpha_model_weights"]
    assert weights[wbnb_busd.pair_id] == 0.5
    assert weights[bit_busd.pair_id] == 0.5

    #
    # 2nd day - cannot sell BIT
    #
    ts = datetime.datetime(2020, 1, 2)
    state.revalue_positions(ts, UniswapV2PoolRevaluator(pancakeswap_v2))
    debug_details = runner.tick(ts, executor_universe, state, {"cycle": 2})
    weights = debug_details["alpha_model_weights"]
    assert len(weights) == 0
    assert len(debug_details["succeeded_trades"]) == 1
    assert len(debug_details["failed_trades"]) == 1

    # Position is now frozen
    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.frozen_positions) == 1
    assert len(portfolio.closed_positions) == 1

    failed_position: TradingPosition = next(iter(portfolio.frozen_positions.values()))
    assert failed_position.position_id == 2
    assert failed_position.frozen_at is not None
    assert failed_position.is_frozen()
    failed_trade = failed_position.get_last_trade()
    assert failed_trade.trade_id == 4
    assert failed_trade.failed_at is not None
    assert failed_trade.is_failed()
    assert failed_trade.is_sell()
    assert failed_trade.tx_info.revert_reason == "VM Exception while processing transaction: revert TransferHelper: TRANSFER_FROM_FAILED"
    assert failed_position.get_freeze_reason() == "VM Exception while processing transaction: revert TransferHelper: TRANSFER_FROM_FAILED"
    assert portfolio.get_frozen_position_equity() > 0

    # The asset is now blacklisted for the future trades
    assert state.asset_blacklist == {bit_busd_pair.base.address.lower()}

    #
    # 3nd day - we no longer try to buy BIT,
    # the alpha model ignores it as a blacklisted asset
    #
    ts = datetime.datetime(2020, 1, 3)
    state.revalue_positions(ts, UniswapV2PoolRevaluator(pancakeswap_v2))
    debug_details = runner.tick(ts, executor_universe, state, {"cycle": 3})
    weights = debug_details["alpha_model_weights"]
    assert weights[wbnb_busd.pair_id] == 1.0
