"""Sets up a virtual Uniswap v2 world that is compatible with our Trading Strategy client."""
import logging
import os
import datetime
import secrets
from decimal import Decimal
from pathlib import Path
from typing import List

import pytest
from eth_account import Account
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract

from eth_hentai.hotwallet import HotWallet
from eth_hentai.balances import fetch_erc20_balances_decimal
from eth_hentai.token import create_token
from eth_hentai.uniswap_v2 import UniswapV2Deployment, deploy_trading_pair, deploy_uniswap_v2_like
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer
from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2_revaluation import UniswapV2PoolRevaluator
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.ethereum.wallet import sync_portfolio, sync_reserves
from tradeexecutor.state.state import State, AssetIdentifier, TradingPairIdentifier, Portfolio
from tradeexecutor.strategy.approval import UncheckedApprovalModel

from tradeexecutor.strategy.bootstrap import bootstrap_strategy, import_strategy_file
from tradeexecutor.strategy.runner import Dataset, StrategyRunner
from tradeexecutor.utils.log import setup_pytest_logging
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import PairUniverse, PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture
def tester_provider():
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return EthereumTesterProvider()


@pytest.fixture
def eth_tester(tester_provider):
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return tester_provider.ethereum_tester


@pytest.fixture
def web3(tester_provider):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return Web3(tester_provider)


@pytest.fixture
def chain_id(web3) -> int:
    """The test chain id (67)."""
    return web3.eth.chain_id


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deploy account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def hot_wallet_private_key(web3) -> HexBytes:
    """Generate a private key"""
    return HexBytes(secrets.token_bytes(32))


@pytest.fixture
def usdc_token(web3, deployer: HexAddress) -> Contract:
    """Create USDC with 10M supply."""
    token = create_token(web3, deployer, "Fake USDC coin", "USDC", 10_000_000 * 10**6, 6)
    return token


@pytest.fixture
def aave_token(web3, deployer: HexAddress) -> Contract:
    """Create AAVE with 10M supply."""
    token = create_token(web3, deployer, "Fake Aave coin", "AAVE", 10_000_000 * 10**18, 18)
    return token


@pytest.fixture()
def uniswap_v2(web3, deployer) -> UniswapV2Deployment:
    """Uniswap v2 deployment."""
    deployment = deploy_uniswap_v2_like(web3, deployer)
    return deployment


@pytest.fixture
def weth_token(uniswap_v2: UniswapV2Deployment) -> Contract:
    """Mock some assets"""
    return uniswap_v2.weth


@pytest.fixture
def asset_usdc(usdc_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, usdc_token.address, usdc_token.functions.symbol().call(), usdc_token.functions.decimals().call())


@pytest.fixture
def asset_weth(weth_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, weth_token.address, weth_token.functions.symbol().call(), weth_token.functions.decimals().call())


@pytest.fixture
def asset_aave(aave_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, aave_token.address, aave_token.functions.symbol().call(), aave_token.functions.decimals().call())


@pytest.fixture
def aave_usdc_uniswap_trading_pair(web3, deployer, uniswap_v2, aave_token, usdc_token) -> HexAddress:
    """AAVE-USDC pool with 200k liquidity."""
    pair_address = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        aave_token,
        usdc_token,
        1000 * 10**18,  # 1000 AAVE liquidity
        200_000 * 10**6,  # 200k USDC liquidity
    )
    return pair_address


@pytest.fixture
def weth_usdc_uniswap_trading_pair(web3, deployer, uniswap_v2, weth_token, usdc_token) -> HexAddress:
    """AAVE-USDC pool with 1.7M liquidity."""
    pair_address = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        weth_token,
        usdc_token,
        1000 * 10**18,  # 1000 ETH liquidity
        1_700_000 * 10**6,  # 1.7M USDC liquidity
    )
    return pair_address


@pytest.fixture
def weth_usdc_pair(weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(asset_weth, asset_usdc, weth_usdc_uniswap_trading_pair)


@pytest.fixture
def aave_usdc_pair(aave_usdc_uniswap_trading_pair, asset_usdc, asset_aave) -> TradingPairIdentifier:
    return TradingPairIdentifier(asset_aave, asset_usdc, aave_usdc_uniswap_trading_pair)


@pytest.fixture
def start_ts() -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture
def supported_reserves(usdc) -> List[AssetIdentifier]:
    """Timestamp of action started"""
    return [usdc]


@pytest.fixture()
def hot_wallet(web3: Web3, usdc_token: Contract, hot_wallet_private_key: HexBytes, deployer: HexAddress) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 ETH.
    """
    account = Account.from_key(hot_wallet_private_key)
    web3.eth.send_transaction({"from": deployer, "to": account.address, "value": 2*10**18})
    usdc_token.functions.transfer(account.address, 10_000 * 10**6).transact({"from": deployer})
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture
def supported_reserves(asset_usdc) -> List[AssetIdentifier]:
    """The reserve currencies we support."""
    return [asset_usdc]


@pytest.fixture()
def portfolio() -> Portfolio:
    """A portfolio loaded with the initial cash.

    We start with 10,000 USDC.
    """
    portfolio = Portfolio({}, {}, {})
    return portfolio


@pytest.fixture()
def state(portfolio) -> State:
    return State(portfolio=portfolio)


@pytest.fixture()
def exchange_universe(web3, uniswap_v2: UniswapV2Deployment) -> ExchangeUniverse:
    """We trade on one Uniswap v2 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v2])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_pair, aave_usdc_pair) -> PandasPairUniverse:
    """We trade on two trading pairs."""
    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    return create_pair_universe(web3, exchange, [weth_usdc_pair, aave_usdc_pair])


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
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "strategies", "simulated_uniswap.py"))


def test_simulated_uniswap_qstrader_strategy_single_trade(
        logger: logging.Logger,
        strategy_path: Path,
        web3: Web3,
        hot_wallet: HotWallet,
        uniswap_v2: UniswapV2Deployment,
        universe: Universe,
        state: State,
        supported_reserves,
        weth_usdc_pair,
        weth_token,
        usdc_token,
    ):
    """Tests a strategy that runs against a simulated Uniswap environment.

    Do a single trade and analyse data structures look correct after the trade.
    """

    factory = import_strategy_file(strategy_path)
    approval_model = UncheckedApprovalModel()
    execution_model = UniswapV2ExecutionModel(state, uniswap_v2, hot_wallet)
    sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
    revaluation_method = UniswapV2PoolRevaluator(uniswap_v2)
    pricing_method = UniswapV2LivePricing(uniswap_v2, universe.pairs)

    runner: StrategyRunner = factory(
        timed_task_context_manager=timed_task,
        execution_model=execution_model,
        approval_model=approval_model,
        revaluation_method=revaluation_method,
        sync_method=sync_method,
        pricing_method=pricing_method,
        reserve_assets=supported_reserves)

    now_ = datetime.datetime.utcnow()
    runner.preflight_check(None, universe, now_)

    # Run the trading over 14 days
    # The test strategy will buy in/buy out position and flips every second day
    ts = datetime.datetime(2020, 1, 1)
    day_count = 14

    # Asset identifies used in testing
    exchange = universe.exchanges[0]
    weth_usdc = universe.pairs.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    aave_usdc = universe.pairs.get_one_pair_from_pandas_universe(exchange.exchange_id, "AAVE", "USDC")

    assert weth_usdc
    assert aave_usdc

    # see strategy/simulated_uniswap.py for different days we can have 0, 1, 2

    #
    # 1st day
    #

    # We start with day_kind 1 that is all ETH day.
    debug_details = runner.tick(ts, universe, state)
    assert debug_details["day_kind"] == 1

    # We first check we got our 10,000 USDC deposit from hot_wallet fixture above
    # See StrategyRunner.sync_portfolio()
    assert len(debug_details["reserve_update_events"]) == 1
    assert debug_details["total_equity_at_start"] == 10_000
    assert debug_details["total_cash_at_start"] == 10_000

    # Check that the strategy thinking is 100% ETH
    # This comes from qstrader/portfolio_construction_model
    assert debug_details["alpha_model_weights"] == {weth_usdc.pair_id: 1}
    assert debug_details["target_prices"] == {weth_usdc.pair_id: pytest.approx(1705.12)}
    assert debug_details["target_portfolio"] == {weth_usdc.pair_id: {"quantity": pytest.approx(Decimal('5.571455381439429643'))}}

    # The strategy should use all of our available USDC to buy ETH.
    assert len(debug_details["rebalance_trades"]) == 1

    # Check the executed portfolio balances
    assert state.portfolio.get_total_equity() == pytest.approx(9947.390072492823)
    assert state.portfolio.get_current_cash() == pytest.approx(500)

    # Check the open position
    assert len(state.portfolio.open_positions) == 1
    position = state.portfolio.get_open_position_for_pair(weth_usdc_pair)
    assert position is not None
    assert position.get_quantity() == Decimal('5.54060129052079779')
    assert position.get_value() == pytest.approx(9447.390072492823)
    assert len(position.trades) == 1

    # Check the recorded trade history
    trades = list(state.portfolio.get_all_trades())
    assert len(trades) == 1
    t = trades[0]

    assert t.is_success()
    assert t.tx_info.chain_id == 61   # Ethereum Tester
    assert t.tx_info.tx_hash.startswith("0x")
    assert t.tx_info.nonce == 1

    # Check the raw on-chain token balances
    balances = fetch_erc20_balances_decimal(web3, hot_wallet.address)
    assert balances[weth_token.address].value == pytest.approx(Decimal('5.54060129052079779'))
    assert balances[usdc_token.address].value == pytest.approx(Decimal('500'))
