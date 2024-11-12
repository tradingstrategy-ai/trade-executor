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
from pathlib import Path
from typing import List

import pytest
from eth_account import Account
from eth_defi.provider.anvil import fork_network_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.confirmation import wait_transactions_to_complete
from eth_typing import HexAddress, HexStr
from hexbytes import HexBytes

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation_v0 import UniswapV2PoolValuationMethodV0
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.qstrader import HAS_QSTRADER
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import StaticUniverseModel
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from tradeexecutor.ethereum.hot_wallet_sync_model import EthereumHotWalletReserveSyncer, HotWalletSyncModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import uniswap_v2_sell_valuation_factory
from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.bootstrap import import_strategy_file
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.valuation import revalue_state

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
from tradeexecutor.utils.timer import timed_task


pytestmark = pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None or not HAS_QSTRADER, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")


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
def anvil_bnb_chain_fork(logger, large_busd_holder) -> str:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["BNB_CHAIN_JSON_RPC"]

    # Start Ganache
    launch = fork_network_anvil(
        mainnet_rpc,
        unlocked_addresses=[large_busd_holder])
    try:
        yield launch.json_rpc_url
        # Wind down Ganache process after the test is complete
    finally:
        launch.close(log_level=logging.INFO)


@pytest.fixture
def web3(anvil_bnb_chain_fork: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    web3 = Web3(HTTPProvider(anvil_bnb_chain_fork, request_kwargs={"timeout": 5}))
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
    return AssetIdentifier(chain_id, busd_token.address.lower(), busd_token.functions.symbol().call(), busd_token.functions.decimals().call())


@pytest.fixture
def asset_bit(bit_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, bit_token.address.lower(), bit_token.functions.symbol().call(), bit_token.functions.decimals().call())


@pytest.fixture
def asset_wbnb(wbnb_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(chain_id, wbnb_token.address.lower(), wbnb_token.functions.symbol().call(), wbnb_token.functions.decimals().call())


@pytest.fixture
def bit_busd_pair_address() -> HexAddress:
    return HexAddress(HexStr("0xf1b66d220479a5620677518c139d0a33f609793b"))


@pytest.fixture
def wbnb_busd_pair_address() -> HexAddress:
    return HexAddress(HexStr("0x58f876857a02d6762e0101bb5c46a8c1ed44dc16"))


@pytest.fixture
def wbnb_busd_pair(pancakeswap_v2, wbnb_busd_pair_address, asset_wbnb, asset_busd) -> TradingPairIdentifier:
    return TradingPairIdentifier(asset_wbnb, asset_busd, wbnb_busd_pair_address, pancakeswap_v2.factory.address, internal_id=int(wbnb_busd_pair_address, 16), fee=0.0025)


@pytest.fixture
def bit_busd_pair(pancakeswap_v2, bit_busd_pair_address, asset_bit, asset_busd) -> TradingPairIdentifier:
    return TradingPairIdentifier(asset_bit, asset_busd, bit_busd_pair_address, pancakeswap_v2.factory.address, internal_id=int(bit_busd_pair_address, 16), fee=0.0025)


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

    balance = web3.eth.get_balance(large_busd_holder)
    assert balance  > web3.to_wei("1", "ether"), f"Account is empty {large_busd_holder}"

    txid = busd_token.functions.transfer(account.address, 100 * 10**18).transact({"from": large_busd_holder})
    wait_transactions_to_complete(web3, [txid])

    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../../strategies/test_only", "frozen_asset.py"))


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
        time_bucket=TimeBucket.d1,
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
        data_universe=universe,
        reserve_assets=supported_reserves
    ))


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../../strategies/test_only", "frozen_asset.py"))


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
    execution_model = UniswapV2ExecutionModelVersion0(pancakeswap_v2, hot_wallet, confirmation_timeout=datetime.timedelta(minutes=1), stop_on_execution_failure=False)
    # sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
    sync_model = HotWalletSyncModel(web3, hot_wallet)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=execution_model,
        timed_task_context_manager=timed_task,
        sync_model=sync_model,
        valuation_model_factory=uniswap_v2_sell_valuation_factory,
        pricing_model_factory=uniswap_v2_live_pricing_factory,
        approval_model=approval_model,
        client=None,
        universe_model=universe_model,
        cash_buffer=0.50,
        execution_context=unit_test_execution_context,
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
    universe = executor_universe.data_universe

    assert universe.pairs

    # Check assets
    exchange = universe.exchange_universe.get_single()
    wbnb_busd = universe.pairs.get_one_pair_from_pandas_universe(exchange.exchange_id, "WBNB", "BUSD")
    bit_busd = universe.pairs. get_one_pair_from_pandas_universe(exchange.exchange_id, "BIT", "BUSD")

    assert wbnb_busd
    assert wbnb_busd.fee
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
    revalue_state(state, ts, UniswapV2PoolValuationMethodV0(pancakeswap_v2))
    debug_details = runner.tick(ts, executor_universe, state, {"cycle": 2, "check_balances": True})
    weights = debug_details["alpha_model_weights"]

    succeeded_trades = [t for t in debug_details["rebalance_trades"] if t.is_success()]
    failed_trades = [t for t in debug_details["rebalance_trades"] if not t.is_success()]

    assert len(succeeded_trades) == 1
    assert len(failed_trades) == 1

    # Position is now frozen
    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 1  # BUSD open
    assert len(portfolio.frozen_positions) == 1  # BIT frozen
    assert len(portfolio.closed_positions) == 0

    failed_position: TradingPosition = next(iter(portfolio.frozen_positions.values()))
    assert failed_position.position_id == 2
    assert failed_position.frozen_at is not None
    assert failed_position.is_frozen()
    failed_trade = failed_position.get_last_trade()
    assert failed_trade.trade_id == 4
    assert failed_trade.failed_at is not None
    assert failed_trade.is_failed()
    assert failed_trade.is_sell()
    tx_info = failed_trade.blockchain_transactions[-1]
    #assert tx_info.revert_reason == "VM Exception while processing transaction: revert TransferHelper: TRANSFER_FROM_FAILED"
    #assert failed_position.get_freeze_reason() == "VM Exception while processing transaction: revert TransferHelper: TRANSFER_FROM_FAILED"
    assert portfolio.get_frozen_position_equity() > 0

    # The asset is now blacklisted for the future trades
    assert state.asset_blacklist == {bit_busd_pair.base.get_identifier()}

    #
    # 3nd day - we no longer try to buy BIT,
    # the alpha model ignores it as a blacklisted asset
    #
    ts = datetime.datetime(2020, 1, 3)
    revalue_state(state, ts, UniswapV2PoolValuationMethodV0(pancakeswap_v2))
    debug_details = runner.tick(ts, executor_universe, state, {"cycle": 3})
    weights = debug_details["alpha_model_weights"]
    assert weights[wbnb_busd.pair_id] == 1.0
