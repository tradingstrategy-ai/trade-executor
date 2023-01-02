"""Backtesting router tests.
"""
import os
import datetime
import logging
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.universe_model import UniverseOptions
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
from tradeexecutor.ethereum.routing_data import get_pancake_default_routing_parameters
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, TradingStrategyUniverse, \
    translate_trading_pair
from tradeexecutor.utils.timer import timed_task


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


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
    trading_pair = ("WBNB", "BUSD")

    # Load all datas we can get for our candle time bucket
    dataset = load_all_data(client, candle_time_bucket, execution_context, UniverseOptions())

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        chain_id,
        exchange_slug,
        trading_pair[0],
        trading_pair[1],
    )

    return universe


@pytest.fixture(scope="module")
def wbnb(request, universe) -> AssetIdentifier:
    """WBNB asset."""
    pair = translate_trading_pair(universe.universe.pairs.get_single())
    return pair.base


@pytest.fixture(scope="module")
def busd(request, universe) -> AssetIdentifier:
    """BUSD asset."""
    pair = translate_trading_pair(universe.universe.pairs.get_single())
    return pair.quote


@pytest.fixture(scope="module")
def routing_model() -> BacktestRoutingModel:
    params = get_pancake_default_routing_parameters(ReserveCurrency.busd)
    routing_model = BacktestRoutingModel(
        params["factory_router_map"],
        params["allowed_intermediary_pairs"],
        params["reserve_token_address"],
    )
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
    events = deposit_syncer(state.portfolio, datetime.datetime(1970, 1, 1), universe.reserve_assets)
    assert len(events) == 1
    token, usd_exchange_rate = state.portfolio.get_default_reserve_currency()
    assert token.token_symbol == "BUSD"
    assert usd_exchange_rate == 1
    assert state.portfolio.get_current_cash() == 10_000
    return state


def test_get_historical_price(
        logger: logging.Logger,
        state: State,
        wallet: SimulatedWallet,
        universe: TradingStrategyUniverse,
        pricing_model: BacktestSimplePricingModel,
        routing_model: BacktestRoutingModel,
    ):
    """Retrieve historical buy and sell price."""

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecutionModel(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, universe, execution_model, routing_model, pricing_model)
    wbnb_busd = translate_trading_pair(universe.universe.pairs.get_single())

    # Check the candle price range that we have data to get the price
    price_range = universe.universe.candles.get_candles_by_pair(wbnb_busd.internal_id)
    assert price_range.iloc[0]["timestamp"] < ts

    # Check the candle price range that we have data to get the price
    liquidity_range = universe.universe.liquidity.get_samples_by_pair(wbnb_busd.internal_id)
    assert liquidity_range.iloc[0]["timestamp"] < ts

    # Get the price for buying WBNB for 1000 USD at 2021-1-1
    buy_price = trader.get_buy_price(wbnb_busd, Decimal(1_000))
    assert buy_price == pytest.approx(354.3096008300781)

    # Get the price for sellinb 1 WBNB
    sell_price = trader.get_sell_price(wbnb_busd, Decimal(1))
    assert sell_price == pytest.approx(354.3096008300781)


def test_create_and_execute_backtest_trade(
    logger: logging.Logger,
    state: State,
    wallet: SimulatedWallet,
    universe: TradingStrategyUniverse,
    routing_model: BacktestRoutingModel,
    pricing_model: BacktestSimplePricingModel,
    wbnb: AssetIdentifier,
    busd: AssetIdentifier,
    ):
    """Manually walk through creation and execution of a single backtest trade."""

    assert wallet.get_balance(busd.address) == 10_000

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecutionModel(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, universe, execution_model, routing_model, pricing_model)
    wbnb_busd = translate_trading_pair(universe.universe.pairs.get_single())

    # Create trade for buying WBNB for 1000 USD
    position, trade = trader.buy(wbnb_busd, reserve=Decimal(1000))

    assert trade.is_buy()
    assert trade.get_status() == TradeStatus.success
    assert trade.executed_price == pytest.approx(355.375728014120)

    # We bought around 3 BNB
    assert position.get_quantity() == pytest.approx(Decimal('2.813923183747276159258725343'))

    # Check our wallet was credited
    assert wallet.get_balance(busd.address) == 9_000
    assert wallet.get_balance(wbnb.address) == Decimal('2.813923183747276159258725343')


def test_buy_sell_backtest(
    logger: logging.Logger,
    state: State,
    wallet: SimulatedWallet,
    universe: TradingStrategyUniverse,
    routing_model: BacktestRoutingModel,
    pricing_model: BacktestSimplePricingModel,
    wbnb: AssetIdentifier,
    busd: AssetIdentifier,
    ):
    """Buying and sell using backtest execution."""

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecutionModel(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, universe, execution_model, routing_model, pricing_model)
    wbnb_busd = translate_trading_pair(universe.universe.pairs.get_single())

    # Create trade for buying WBNB for 1000 USD
    position, trade = trader.buy(wbnb_busd, reserve=Decimal(1000))

    assert trade.is_buy()
    assert trade.get_status() == TradeStatus.success
    buy_price = trade.executed_price

    position, trade = trader.sell(wbnb_busd, position.get_quantity())

    assert trade.is_sell()
    assert trade.is_success()
    sell_price = trade.executed_price
    assert position.is_closed()

    # Do simulated markets make any sense?
    assert sell_price < buy_price

    # Check our wallet was credited
    assert wallet.get_balance(busd.address) == pytest.approx(Decimal('9999.990999999999889294777233'))
    assert wallet.get_balance(wbnb.address) == 0


def test_buy_with_fee(
    logger: logging.Logger,
    state: State,
    wallet: SimulatedWallet,
    universe: TradingStrategyUniverse,
    routing_model: BacktestRoutingModel,
    pricing_model: BacktestSimplePricingModel,
    wbnb: AssetIdentifier,
    busd: AssetIdentifier,
    ):
    """Check that trading fee is accounted correctly in the backtest execution.

    Compare to test_create_and_execute_backtest_trade where we do the
    same trade without fees.
    """

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecutionModel(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, universe, execution_model, routing_model, pricing_model)
    wbnb_busd = translate_trading_pair(universe.universe.pairs.get_single())

    # Set 0.5% trading fee
    wbnb_busd.fee = 0.0050

    # Create trade for buying WBNB for 1000 USD
    position, trade = trader.buy(wbnb_busd, reserve=Decimal(1000))

    # Check we calculated LP fees correctly
    assert trade.is_buy()
    assert trade.get_status() == TradeStatus.success
    assert trade.lp_fee == 0.0050
    assert trade.executed_price == pytest.approx(357.15260665419106)
    assert trade.lp_fees_estimated == 5.0
    assert trade.lp_fees_paid == 5.0

    # We bought around 3 BNB
    assert position.get_quantity() == pytest.approx(Decimal('2.799923565917687953079317912'))


def test_buy_sell_backtest_with_fee(
    logger: logging.Logger,
    state: State,
    wallet: SimulatedWallet,
    universe: TradingStrategyUniverse,
    routing_model: BacktestRoutingModel,
    pricing_model: BacktestSimplePricingModel,
    wbnb: AssetIdentifier,
    busd: AssetIdentifier,
    ):
    """Buying and sell using backtest execution."""

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecutionModel(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, universe, execution_model, routing_model, pricing_model)
    wbnb_busd = translate_trading_pair(universe.universe.pairs.get_single())

    # Set 0.5% trading fee
    wbnb_busd.fee = 0.0050

    # Create trade for buying WBNB for 1000 USD
    position, trade = trader.buy(wbnb_busd, reserve=Decimal(1000))

    assert trade.is_buy()
    assert trade.get_status() == TradeStatus.success
    buy_price = trade.executed_price

    position, trade = trader.sell(wbnb_busd, position.get_quantity())

    assert trade.is_sell()
    assert trade.is_success()
    sell_price = trade.executed_price

    assert position.is_closed()
    assert trade.lp_fee == 0.005
    assert trade.lp_fees_estimated == pytest.approx(4.960199004975125)
    assert trade.lp_fees_paid == pytest.approx(4.960199004975125)

    # Do simulated markets make any sense?
    assert sell_price < buy_price

    # Check our wallet was credited
    assert wallet.get_balance(busd.address) == pytest.approx(Decimal('9990.040840796019796453251579'))
    assert wallet.get_balance(wbnb.address) == 0
