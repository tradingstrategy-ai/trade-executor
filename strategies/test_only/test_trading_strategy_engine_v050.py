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
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput, IndicatorNotFound, InvalidForMultipairStrategy
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_trading_and_lending_data, load_partial_data
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

PAIR_IDS = [
    (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
]


class Parameters:
    # We will override this value with CycleDuration.cycle_1s for live trading unit test.
    # Test cycles by live unit testing, takes 4s.
    cycle_duration = CycleDuration.cycle_1d
    chain_id = ChainId.anvil
    routing = TradeRouting.default
    backtest_start = datetime.datetime(2023, 1, 1)
    backtest_end = datetime.datetime(2024, 1, 1)
    custom_parameter = 1
    initial_cash = 10_000
    time_bucket = TimeBucket.d1
    rsi_length = 21


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
    strategy_universe = input.strategy_universe
    live = input.execution_context.mode.is_live_trading()

    assert input.execution_context.mode in (ExecutionMode.unit_testing_trading, ExecutionMode.backtesting)
    if live:
        assert datetime.datetime.utcnow() - timestamp < datetime.timedelta(minutes=1)      # We should do 1s cycles near real time
    assert parameters.custom_parameter == 1

    # Do various live data assets and
    # pass data to the main unit test function
    for pair_id in PAIR_IDS:
        pair = strategy_universe.get_pair_by_human_description(pair_id)

        # Test individual value
        rsi = indicators.get_indicator_value("rsi", pair=pair, data_delay_tolerance=pd.Timedelta(days=7))
        if live:
            assert rsi is not None, f"get_indicator_value() returned None, timestamp is {indicators.timestamp}"

        # Test whole series
        rsi_series = indicators.get_indicator_series("rsi", pair=pair)
        # assert len(rsi_series) == parameters.rsi_length, f"RSI for {pair} is length {len(rsi_series)}, values:\n{rsi_series}"

        # The RSI is calculated for all loaded data (we load 60 days)
        # Each RSI series cell is21 days backwards for RSI from that point
        # The initial cells have NaN as value
        if live:
            assert len(rsi_series) == 60, f"RSI for {pair} is length {len(rsi_series)}, values:\n{rsi_series}"

        # Test unknown indicator
        try:
            indicators.get_indicator_value("foobaa")
            raise RuntimeError(f"Should not happen")
        except IndicatorNotFound:
            pass

        # Test pair arugment missing
        try:
            indicators.get_indicator_series("rsi")
            raise RuntimeError(f"Should not happen")
        except InvalidForMultipairStrategy:
            pass

        # Test price
        price = indicators.get_price(pair)
        assert price is not None, f"Got None price for {pair} at {timestamp}"
        assert 0 < price < 10_000, f"Got {pair} price {price}"

        input.other_data[f"rsi_{pair.base.token_symbol}"] = rsi
        input.other_data[f"custom_test_indicator"] = indicators.get_indicator_series("custom_test_indicator", unlimited=True).to_list()

    # We never execute any trades, only test the live execution main loop
    return []


def custom_test_indicator(
    strategy_universe: TradingStrategyUniverse,
    custom_parameter: int
) -> pd.Series:
    assert custom_parameter == 1
    return pd.Series([1, 2, 3, 4])


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
) -> IndicatorSet:

    # Test create_indicators() in live trade execution cycle
    # Do some unit test checks
    assert execution_context.mode in (ExecutionMode.unit_testing_trading, ExecutionMode.backtesting)
    if execution_context.mode.is_live_trading():
        # Live execution
        assert timestamp is not None
        assert isinstance(timestamp, datetime.datetime)
        assert timestamp.tzinfo is None
    else:
        # For backtesting we do not get a timestamp
        assert timestamp is None

    indicators = IndicatorSet()
    indicators.add("rsi", pandas_ta.rsi, {"length": parameters.rsi_length})
    indicators.add("custom_test_indicator", custom_test_indicator, {"custom_parameter": parameters.custom_parameter}, source=IndicatorSource.strategy_universe)
    return indicators


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    # Try to load live data, do some sanity checks
    assert datetime.datetime.utcnow() - timestamp < datetime.timedelta(minutes=1)

    # Load data for our trading pair whitelist
    if execution_context.mode.is_backtesting():
        # For backtesting, we use a specific time range from the strategy parameters
        start_at = universe_options.start_at
        end_at = universe_options.end_at
        required_history_period = None
    else:
        # For live trading, we look back 30 days for the data
        assert execution_context.mode.is_live_trading()
        assert isinstance(timestamp, datetime.datetime)
        start_at = None
        end_at = None
        required_history_period = datetime.timedelta(days=60)  # We need 21 days run up for RSI indicator

    dataset = load_partial_data(
        client=client,
        time_bucket=Parameters.time_bucket,
        pairs=PAIR_IDS,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=False,
        stop_loss_time_bucket=None,
        start_at=start_at,
        end_at=end_at,
        required_history_period=required_history_period,
    )

    # Filter down the dataset to the pairs we specified
    universe = TradingStrategyUniverse.create_from_dataset(dataset)
    return universe
