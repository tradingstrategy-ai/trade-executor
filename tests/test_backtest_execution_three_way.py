"""Backtesting router tests for three way trades.
"""
import datetime
import logging
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.strategy_module import ReserveCurrency
from tradeexecutor.testing.backtest_trader import BacktestTrader
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.backtest_execution import BacktestExecutionModel
from tradeexecutor.backtest.backtest_pricing import BacktestSimplePricingModel
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_sync import BacktestSyncer
from tradeexecutor.backtest.backtest_valuation import BacktestValuationModel
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.ethereum.default_routes import get_pancake_default_routing_parameters
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_model import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, TradingStrategyUniverse, \
    translate_trading_pair, translate_token
from tradeexecutor.utils.timer import timed_task


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)



@pytest.fixture(scope="module")
def execution_context(request) -> ExecutionContext:
    """Setup backtest execution context."""
    return ExecutionContext(mode=ExecutionMode.backtesting, timed_task_context_manager=timed_task)



@pytest.fixture(scope="module")
def universe(request, persistent_test_client, execution_context) -> TradingStrategyUniverse:
    """Backtesting data universe.

    This contains only data for WBNB-BUSD pair on PancakeSwap v2 since 2021-01-01.
    """

    client = persistent_test_client

    # Time bucket for our candles
    candle_time_bucket = TimeBucket.d1

    # Which chain we are trading
    chain_id = ChainId.bsc

    # Which exchange we are trading on.
    exchange_slug = "pancakeswap-v2"

    # Which trading pair we are trading
    trading_pairs = [
        ("WBNB", "BUSD"),
        ("Cake", "WBNB"),
    ]

    # Load all datas we can get for our candle time bucket
    dataset = load_all_data(client, candle_time_bucket, execution_context)

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_limited_pair_universe(
        dataset,
        chain_id,
        exchange_slug,
        trading_pairs,
    )

    return universe


@pytest.fixture(scope="module")
def wbnb(request, universe) -> AssetIdentifier:
    """WBNB asset."""
    token = translate_token(universe.universe.pairs.get_token("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"))
    return token


@pytest.fixture(scope="module")
def busd(request, universe) -> AssetIdentifier:
    """bUSD asset."""
    token = translate_token(universe.universe.pairs.get_token("0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56"))
    return token


@pytest.fixture(scope="module")
def cake(request, universe) -> AssetIdentifier:
    """Cake asset."""
    token = translate_token(universe.universe.pairs.get_token("0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82"))
    return token


@pytest.fixture(scope="module")
def routing_model() -> BacktestRoutingModel:
    routing_parameters = get_pancake_default_routing_parameters(ReserveCurrency.busd)
    routing_model = BacktestRoutingModel(**routing_parameters)
    return routing_model


@pytest.fixture(scope="module")
def pricing_model(routing_model, universe) -> BacktestSimplePricingModel:
    return BacktestSimplePricingModel(universe, routing_model)


@pytest.fixture(scope="module")
def valuation_model(pricing_model) -> BacktestValuationModel:
    return BacktestValuationModel(pricing_model)


@pytest.fixture()
def wallet(universe) -> SimulatedWallet:
    return SimulatedWallet()


@pytest.fixture()
def deposit_syncer(wallet) -> BacktestSyncer:
    """Start with 10,000 USD."""
    return BacktestSyncer(wallet, Decimal(10_000))


@pytest.fixture()
def state(universe: TradingStrategyUniverse, deposit_syncer: BacktestSyncer) -> State:
    """Start with 10,000 USD cash in the portfolio."""
    state = State()
    assert len(universe.reserve_assets) == 1
    assert universe.reserve_assets[0].token_symbol == "BUSD"
    events = deposit_syncer(state.portfolio, datetime.datetime(1970, 1, 1), universe.reserve_assets)
    assert len(events) == 1
    token, usd_exchange_rate = state.portfolio.get_default_reserve_currency()
    assert token.token_symbol == "BUSD"
    assert usd_exchange_rate == 1
    assert state.portfolio.get_current_cash() == 10_000
    return state


def test_create_and_execute_backtest_three_way_trade(
        logger: logging.Logger,
        state: State,
        wallet: SimulatedWallet,
        universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
        pricing_model: BacktestSimplePricingModel,
        wbnb: AssetIdentifier,
        busd: AssetIdentifier,
        cake: AssetIdentifier,
    ):
    """Manually walk through creation and execution of a single backtest trade."""

    assert wallet.get_balance(busd.address) == 10_000

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecutionModel(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, universe, execution_model, routing_model, pricing_model)
    cake_wbnb = translate_trading_pair(universe.universe.pairs.get_by_symbols("Cake", "WBNB"))

    assert cake_wbnb.base.token_symbol == "Cake"
    assert cake_wbnb.quote.token_symbol == "WBNB"

    # Create trade for buying WBNB for 1000 USD
    position, trade = trader.buy(cake_wbnb, reserve=Decimal(1000))

    assert trade.is_buy()
    assert trade.get_status() == TradeStatus.success
    assert trade.executed_price == pytest.approx(17.872633403139105)

    # We bought around 3 BNB
    assert position.get_quantity() == pytest.approx(Decimal('55.95146375152318001945304859'))

    # Check our wallet was credited
    assert wallet.get_balance(busd.address) == 9_000
    assert wallet.get_balance(cake.address) == pytest.approx(Decimal('55.95146375152318001945304859'))


def test_buy_sell_three_way_backtest(
        logger: logging.Logger,
        state: State,
        wallet: SimulatedWallet,
        universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
        pricing_model: BacktestSimplePricingModel,
        wbnb: AssetIdentifier,
        busd: AssetIdentifier,
        cake: AssetIdentifier,
    ):
    """Buying and sell using backtest execution, three way."""

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecutionModel(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, universe, execution_model, routing_model, pricing_model)
    cake_wbnb = translate_trading_pair(universe.universe.pairs.get_by_symbols("Cake", "WBNB"))

    # Create trade for buying Cake for 1000 USD thru WBNB
    position, trade = trader.buy(cake_wbnb, reserve=Decimal(1000))

    assert trade.is_buy()
    assert trade.get_status() == TradeStatus.success
    buy_price = trade.executed_price

    position, trade = trader.sell(cake_wbnb, position.get_quantity())

    assert trade.is_sell()
    assert trade.is_success()
    sell_price = trade.executed_price
    assert position.is_closed()

    # Do simulated markets make any sense?
    assert sell_price < buy_price

    # Check our wallet was credited
    assert wallet.get_balance(busd.address) == pytest.approx(Decimal('9999.990999999999889294777233'))
    assert wallet.get_balance(cake.address) == 0

