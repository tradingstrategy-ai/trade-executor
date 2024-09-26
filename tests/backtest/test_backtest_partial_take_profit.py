"""Run a dummy strategy focus on partial take profit orders.

- Run daily decide_trade cycle that prepares orders for hourly cycle
- Strategy decision cycle is 24h
- The backtest trigger check is 1h signal

"""
import datetime
import random
from decimal import Decimal
from typing import List, Dict

import pytest
import pandas as pd

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradingstrategy.universe import Universe
from tradingstrategy.utils.groupeduniverse import resample_candles
from tradeexecutor.state.trigger import TriggerType


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def strategy_universe() -> TradingStrategyUniverse:
    """Create ETH-USDC universe with only increasing data.

    - Close price increase 1% every hour
    """

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="my-dex",
    )
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    time_bucket = TimeBucket.h1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    # Create 1h underlying trade signal
    stop_loss_candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        pair_id=weth_usdc.internal_id,
        daily_drift=(0.99, 1.02),
        high_drift=1.05,
        low_drift=0.90,
        random_seed=1,
    )
    stop_loss_candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(stop_loss_candles)

    # Create upsampled daily candles
    daily_candles = resample_candles(stop_loss_candles, "D")
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(daily_candles)

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )
    universe.pairs.exchange_universe = universe.exchange_universe

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc],
        backtest_stop_loss_candles=stop_loss_candle_universe,
        backtest_stop_loss_time_bucket=time_bucket,
    )


@pytest.fixture()
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


@pytest.fixture()
def pricing_model(synthetic_universe, routing_model) -> BacktestPricing:
    pricing_model = BacktestPricing(
        synthetic_universe.data_universe.candles,
        routing_model,
        allow_missing_fees=True,
    )
    return pricing_model


def create_indicators(timestamp: datetime.datetime, parameters: StrategyParameters, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
    # No indicators needed
    return IndicatorSet()


def decide_trades_1(input: StrategyInput) -> List[TradeExecution]:
    """Example decide_trades function using partial take profits."""
    position_manager = input.get_position_manager()
    pair = input.get_default_pair()
    cash = input.state.portfolio.get_cash()
    indicators = input.indicators
    portfolio = input.state.portfolio
    parameters = input.parameters

    price = indicators.get_price()
    if price is None:
        # Skip cycle 1
        # We do not have the previous day price available at the first cycle
        return []
    
    for position in portfolio.open_positions.values():
        # Check if the position already has take profit triggers in pending trades
        has_take_profit_triggers = any(
            any(trigger.type == TriggerType.take_profit_partial for trigger in trade.triggers)
            for trade in position.pending_trades.values()
        )

        if not has_take_profit_triggers:
            opening_price = position.get_opening_price()

            total_quantity = position.get_quantity()

            tp1_price = opening_price * 1.1
            tp2_price = opening_price * 1.15
            tp3_price = opening_price * 1.2

            tp1_quantity = -total_quantity * Decimal(0.4)
            tp2_quantity = -total_quantity * Decimal(0.3)

            # NOTE: intentionally set tp3_quantity to be larger than it should
            # but the test should still pass as this amount should be ignored for closing trade
            tp3_quantity = -total_quantity * Decimal(0.9)

            position_manager.prepare_take_profit_trades(
                position,
                [
                    (float(tp1_price), tp1_quantity, False), 
                    (float(tp2_price), tp2_quantity, False),  
                    (float(tp3_price), tp3_quantity, True),   
                ]
            )
            
    # Only set a trigger open if we do not have any position open/pending yet
    if not position_manager.get_current_position_for_pair(pair, pending=True):
        trades = position_manager.open_spot(
            pair=pair,
            value=cash * 0.99,
        )

        return trades

    # We return zero trades here, as all of trades we have constructed
    # are pending for a trigger, and do not need to be executed
    # on this decision cycle
    return []


def decide_trades_2(input: StrategyInput) -> List[TradeExecution]:
    """Example decide_trades function using partial take profits and trailing stoploss."""
    position_manager = input.get_position_manager()
    pair = input.get_default_pair()
    cash = input.state.portfolio.get_cash()
    indicators = input.indicators
    portfolio = input.state.portfolio
    parameters = input.parameters

    price = indicators.get_price()
    if price is None:
        # Skip cycle 1
        # We do not have the previous day price available at the first cycle
        return []
    
    for position in portfolio.open_positions.values():
        # Check if the position already has take profit triggers in pending trades
        has_take_profit_triggers = any(
            any(trigger.type == TriggerType.take_profit_partial for trigger in trade.triggers)
            for trade in position.pending_trades.values()
        )
        
        if not has_take_profit_triggers:
            opening_price = position.get_opening_price()

            total_quantity = position.get_quantity()

            tp1_price = opening_price * 1.1
            tp2_price = opening_price * 1.15
            tp3_price = opening_price * 2

            tp1_quantity = -total_quantity * Decimal(0.4)
            tp2_quantity = -total_quantity * Decimal(0.3)

            # NOTE: intentionally set tp3_quantity to be larger than it should
            # but the test should still pass as this amount should be ignored for closing trade
            tp3_quantity = -total_quantity * Decimal(0.9)

            position_manager.prepare_take_profit_trades(
                position,
                [
                    (float(tp1_price), tp1_quantity, False), 
                    (float(tp2_price), tp2_quantity, False),  
                    (float(tp3_price), tp3_quantity, True),   
                ]
            )
            
    # Only set a trigger open if we do not have any position open/pending yet
    if not position_manager.get_current_position_for_pair(pair, pending=True):
        trades = position_manager.open_spot(
            pair=pair,
            value=cash * 0.99,
            trailing_stop_loss_pct=0.98,
        )

        return trades

    # We return zero trades here, as all of trades we have constructed
    # are pending for a trigger, and do not need to be executed
    # on this decision cycle
    return []


def test_partial_take_profit(strategy_universe):
    """Test partial take profit works and last trade fully close the position 
    with correct quantity
    """

    class Parameters:
        backtest_start = strategy_universe.data_universe.candles.get_timestamp_range()[0].to_pydatetime()
        backtest_end = strategy_universe.data_universe.candles.get_timestamp_range()[0].to_pydatetime()  + datetime.timedelta(days=3)
        initial_cash = 100_000
        cycle_duration = CycleDuration.cycle_1d

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades_1,
        create_indicators=create_indicators,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(Parameters),
        mode=ExecutionMode.unit_testing,
    )

    state = result.state
    assert len(result.diagnostics_data) == 5  # Entry for each day + few extras
    assert len(state.portfolio.closed_positions) == 1
    assert len(state.portfolio.pending_positions) == 0
    assert len(state.portfolio.open_positions) == 0

    closed_position = state.portfolio.closed_positions[1]
    assert len(closed_position.trades) == 4

    last_trade = closed_position.trades[4]
    # check the planned price is the trigger_price
    assert last_trade.planned_quantity == pytest.approx(Decimal(-15.09447875566598427355031245))
    assert last_trade.planned_price == pytest.approx(2361.1282361520357)
    assert last_trade.executed_price == pytest.approx(2361.1282361520357)

    # Check these do not crash on market limit positions
    calculate_summary_statistics(
        state,
        ExecutionMode.unit_testing_trading,
        now_=datetime.datetime.utcnow(),
    )


def test_partial_tp_and_trailing_sl(strategy_universe):
    """Test partial take profit works and should work well with (trailing) stop loss"""

    class Parameters:
        backtest_start = strategy_universe.data_universe.candles.get_timestamp_range()[0].to_pydatetime()
        backtest_end = strategy_universe.data_universe.candles.get_timestamp_range()[0].to_pydatetime()  + datetime.timedelta(days=3)
        initial_cash = 100_000
        cycle_duration = CycleDuration.cycle_1d

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades_2,
        create_indicators=create_indicators,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(Parameters),
        mode=ExecutionMode.unit_testing,
    )

    state = result.state
    # assert len(result.diagnostics_data) == 6  # Entry for each day + few extras
    assert len(state.portfolio.closed_positions) == 1
    assert len(state.portfolio.pending_positions) == 0
    
    closed_position = state.portfolio.closed_positions[1]
    assert len(closed_position.trades) == 5
    assert closed_position.is_stop_loss()

    trades = closed_position.trades

    assert trades[1].is_buy()
    assert trades[1].get_status() == TradeStatus.success

    # 1 partial tp hit
    assert trades[2].is_sell() and trades[2].is_partial_take_profit()
    assert trades[2].get_status() == TradeStatus.success

    # 2 partial tp expired
    assert trades[3].is_sell() and trades[2].is_partial_take_profit()
    assert trades[3].get_status() == TradeStatus.expired
    assert trades[4].is_sell() and trades[2].is_partial_take_profit()
    assert trades[4].get_status() == TradeStatus.expired

    # trailing sl hit
    assert trades[5].is_stop_loss()
    assert trades[5].get_status() == TradeStatus.success

    # Check these do not crash on market limit positions
    calculate_summary_statistics(
        state,
        ExecutionMode.unit_testing_trading,
        now_=datetime.datetime.utcnow(),
    )
