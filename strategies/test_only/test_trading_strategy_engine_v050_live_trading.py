"""A test strategy for trading_strategy_engine_version=0.5

- See test_trading_strategy_engine_v050_live_trading

- This file serves as a skeleton for v0.5 test module
"""
import datetime

import pandas as pd
import pandas_ta

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSource, IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_trading_and_lending_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

TRADING_STRATEGY_ENGINE_VERSION = "0.5"

TAGS = {StrategyTag.unit_testing}

NAME = "test_trading_strategy_engine_v050_live_trading"

SHORT_DESCRIPTION = "Unit testing strategy"

LONG_DESCRIPTION = """
- Test strategy module loading and live execution with create_indicators
- See test_trading_strategy_engine_v050_live_trading.py
"""

class ParameterConfiguration:
    # Will be transformed to StrategyParameters dict
    cycle_duration = CycleDuration.cycle_1s
    chain_id = ChainId.polygon
    routing = TradeRouting.default
    backtest_start = datetime.datetime(2023, 1, 1)
    backtest_end = datetime.datetime(2024, 1, 1)
    rsi_bars = 10
    custom_parameter = 1
    initial_cash = 10_000


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:

    # Resolve our pair metadata for our two pair strategy
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    position_manager.log("decide_trades() start")

    assert input.execution_context.mode in (ExecutionMode.unit_testing_trading, ExecutionMode.backtesting)

    # We should do 1s cycles near real time
    assert datetime.datetime.utcnow() - timestamp < datetime.timedelta(minutes=1)

    # We never execute any trades, only test the live execution main loop
    return []


def custom_test_indicator(
    strategy_universe: TradingStrategyUniverse,
    custom_parameter: int
) -> pd.Series:
    assert custom_parameter == 1
    return pd.Series([1, 2, 3, 4])


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    # Test create_indicators() in live trade execution cycle
    assert execution_context.mode in (ExecutionMode.unit_testing_trading, ExecutionMode.backtesting)
    indicators.add("rsi", pandas_ta.rsi, {"length": parameters.rsi_bars})
    indicators.add("custom_test_indicator", custom_test_indicator, {"custom_parameter": parameters.custom_parameter}, source=IndicatorSource.strategy_universe)


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    # Try to load live data, do some sanity checks
    assert datetime.datetime.utcnow() - timestamp < datetime.timedelta(minutes=1)
    dataset = load_trading_and_lending_data(
        client,
        execution_context=execution_context,
        universe_options=universe_options,
        chain_id=ChainId.polygon,
        exchange_slugs={"uniswap-v3"},
        reserve_assets={"USDC"},
        asset_ids={"WMATIC", "WETH"},
        trading_fee=0.0005,
        time_bucket=TimeBucket.d1,
        stop_loss_time_bucket=TimeBucket.h1,
    )
    # Filter down the dataset to the pairs we specified
    universe = TradingStrategyUniverse.create_from_dataset(dataset)
    return universe