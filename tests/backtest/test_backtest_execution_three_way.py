"""Backtesting router tests for three way trades.
"""
import os
import datetime
import logging
import math
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.testing.backtest_trader import BacktestTrader
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.legacy_backtest_sync import BacktestSyncer
from tradeexecutor.backtest.backtest_valuation import BacktestValuationModel
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.ethereum.routing_data import get_pancake_default_routing_parameters
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, TradingStrategyUniverse, \
    translate_trading_pair, translate_token, load_pair_data_for_single_exchange
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
    trading_pairs = {
        ("WBNB", "BUSD"),
        ("Cake", "WBNB"),
    }

    # Load all datas we can get for our candle time bucket
    dataset = load_pair_data_for_single_exchange(
        client,
        execution_context,
        candle_time_bucket,
        chain_id,
        exchange_slug,
        trading_pairs,
        UniverseOptions()
    )

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_limited_pair_universe(
        dataset,
        chain_id,
        exchange_slug,
        trading_pairs,
        reserve_asset_pair_ticker=("WBNB", "BUSD")
    )

    assert universe.reserve_assets[0].token_symbol == "BUSD"

    return universe


@pytest.fixture(scope="module")
def strategy_universe(universe):
    """Legacy alias. Use strategy_universe."""
    return universe


@pytest.fixture(scope="module")
def stop_loss_universe(request, persistent_test_client, execution_context) -> TradingStrategyUniverse:
    """Backtesting data universe w/stop loss data.

    This contains only data for WBNB-BUSD pair on PancakeSwap v2 since 2021-01-01.
    """

    client = persistent_test_client

    # Time bucket for our candles
    candle_time_bucket = TimeBucket.d1

    stop_loss_time_bucket = TimeBucket.h4

    # Which chain we are trading
    chain_id = ChainId.bsc

    # Which exchange we are trading on.
    exchange_slug = "pancakeswap-v2"

    # Which trading pair we are trading
    trading_pairs = {
        ("WBNB", "BUSD"),
        ("Cake", "WBNB"),
    }

    # Load all datas we can get for our candle time bucket
    dataset = load_pair_data_for_single_exchange(
        client,
        execution_context,
        candle_time_bucket,
        chain_id,
        exchange_slug,
        trading_pairs,
        UniverseOptions(),
        stop_loss_time_bucket=stop_loss_time_bucket,
    )

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_limited_pair_universe(
        dataset,
        chain_id,
        exchange_slug,
        trading_pairs,
        reserve_asset_pair_ticker=("WBNB", "BUSD")
    )

    assert universe.has_stop_loss_data()
    return universe


@pytest.fixture(scope="module")
def wbnb(request, strategy_universe) -> AssetIdentifier:
    """WBNB asset."""
    token = translate_token(strategy_universe.data_universe.pairs.get_token("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"))
    return token


@pytest.fixture(scope="module")
def busd(request, strategy_universe) -> AssetIdentifier:
    """bUSD asset."""
    token = translate_token(strategy_universe.data_universe.pairs.get_token("0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56"))
    return token


@pytest.fixture(scope="module")
def cake(request, strategy_universe) -> AssetIdentifier:
    """Cake asset."""
    token = translate_token(strategy_universe.data_universe.pairs.get_token("0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82"))
    return token


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
def pricing_model(routing_model, strategy_universe) -> BacktestPricing:
    return BacktestPricing(strategy_universe, routing_model)


@pytest.fixture(scope="module")
def valuation_model(pricing_model) -> BacktestValuationModel:
    return BacktestValuationModel(pricing_model)


@pytest.fixture()
def wallet(strategy_universe) -> SimulatedWallet:
    return SimulatedWallet()


@pytest.fixture()
def deposit_syncer(wallet) -> BacktestSyncer:
    """Start with 10,000 USD."""
    return BacktestSyncer(wallet, Decimal(10_000))


@pytest.fixture()
def state(strategy_universe, deposit_syncer: BacktestSyncer) -> State:
    """Start with 10,000 USD cash in the portfolio."""
    state = State()
    assert len(strategy_universe.reserve_assets) == 1
    assert strategy_universe.reserve_assets[0].token_symbol == "BUSD"
    events = deposit_syncer(state, datetime.datetime(1970, 1, 1), strategy_universe.reserve_assets)
    assert len(events) == 1
    token, usd_exchange_rate = state.portfolio.get_default_reserve_asset()
    assert token.token_symbol == "BUSD"
    assert usd_exchange_rate == 1
    assert state.portfolio.get_cash() == 10_000
    return state


def test_create_and_execute_backtest_three_way_trade(
        logger: logging.Logger,
        state: State,
        wallet: SimulatedWallet,
        strategy_universe,
        routing_model: BacktestRoutingModel,
        pricing_model: BacktestPricing,
        wbnb: AssetIdentifier,
        busd: AssetIdentifier,
        cake: AssetIdentifier,
    ):
    """Manually walk through creation and execution of a single backtest trade."""

    assert wallet.get_balance(busd.address) == 10_000

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecution(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, strategy_universe, execution_model, routing_model, pricing_model)
    cake_wbnb = translate_trading_pair(strategy_universe.data_universe.pairs.get_by_symbols("Cake", "WBNB"))

    cake_wbnb.fee = 0.0025

    assert cake_wbnb.base.token_symbol == "Cake"
    assert cake_wbnb.quote.token_symbol == "WBNB"

    # Create trade for buying WBNB for 1000 USD
    position, trade = trader.buy(cake_wbnb, reserve=Decimal(1000))

    assert trade.is_buy()
    assert trade.get_status() == TradeStatus.success
    assert trade.executed_price == pytest.approx(18.057216310197482)
    assert trade.planned_mid_price == pytest.approx(18.012185845583524)

    # We bought around 3 BNB
    assert position.get_quantity() == pytest.approx(Decimal('55.37952156198451856881142523'))

    # Check our wallet was credited
    assert wallet.get_balance(busd.address) == 9_000
    assert wallet.get_balance(cake.address) == pytest.approx(Decimal('55.37952156198451856881142523'))


def test_buy_sell_three_way_backtest(
        logger: logging.Logger,
        state: State,
        wallet: SimulatedWallet,
        strategy_universe,
        routing_model: BacktestRoutingModel,
        pricing_model: BacktestPricing,
        wbnb: AssetIdentifier,
        busd: AssetIdentifier,
        cake: AssetIdentifier,
    ):
    """Buying and sell using backtest execution, three way."""

    ts = datetime.datetime(2021, 6, 1)
    execution_model = BacktestExecution(wallet, max_slippage=0.01)
    trader = BacktestTrader(ts, state, strategy_universe, execution_model, routing_model, pricing_model)
    cake_wbnb = translate_trading_pair(strategy_universe.data_universe.pairs.get_by_symbols("Cake", "WBNB"))
    cake_wbnb.fee = 0.0025

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
    assert wallet.get_balance(busd.address) == pytest.approx(Decimal('9995.012468827930204002132236'))
    assert wallet.get_balance(cake.address) == 0


def test_load_backtesting_data_with_stop_loss(stop_loss_universe: TradingStrategyUniverse):
    """We load backtesting data with stop loss."""
    assert stop_loss_universe.has_stop_loss_data()
    assert stop_loss_universe.backtest_stop_loss_time_bucket == TimeBucket.h4

    # there should be ~6 times 4h candles than 1d candles in the same period
    backtest_candle_count = len(stop_loss_universe.data_universe.candles.df)
    stop_loss_candle_count = len(stop_loss_universe.backtest_stop_loss_candles.df)
    assert math.ceil(stop_loss_candle_count / backtest_candle_count) == 6
