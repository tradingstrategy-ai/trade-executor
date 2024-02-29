"""Strategy decision input.

- Input arguments for `decide_trade` functions
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

import cachetools
import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarPrice
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorResultMap, IndicatorSet, IndicatorKey
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.utils.time import get_prior_timestamp

logger = logging.getLogger(__name__)


SERIES_CACHE_SIZE = 1024


class InvalidForMultipairStrategy(Exception):
    """Try to use single trading pair functions in a multipair strategy."""



@dataclass(slots=True)
class StrategyInputIndicators:
    """Indicator results for the strategy decision.

    Wraps the indicator results to a format that has good developer experience
    when accessed from `decide_trades()`. The default timestamp

    - Indicators are prepared in `create_indicators` function
    - The framework takes care of recalculating indicators when needed,
      for backtest and live access
    - For backtests, this class is instiated only once
    - We assume all indicator data is forward-filled and no gaps

    For simple strategies calling :py:meth:`get_indicator_value` should be only required here.
    """

    #: Trading universe
    #:
    #: - Perform additional pair lookups if needed
    #:
    strategy_universe: TradingStrategyUniverse

    #: Available indicators as defined in create_indicators()
    #:
    available_indicators: IndicatorSet

    #: Raw cached indicator results or ones calculated in the memory
    #:
    indicator_results: IndicatorResultMap

    #: The current decision_cycle() timestamp.
    #:
    #: Stored here, so we do not need to pass it explicitly in API.
    #:
    timestamp: pd.Timestamp | None = None

    def __post_init__(self):
        assert type(self.indicator_results) == dict
        assert isinstance(self.available_indicators, IndicatorSet)
        assert isinstance(self.strategy_universe, TradingStrategyUniverse)

    def get_price(
        self,
        pair: TradingPairIdentifier | None = None,
        data_lag_tolerance=None,
    ) -> USDollarPrice | None:
        """Read the available close price of a trading pair.

        - Returns the latest available close price

        - **Does not** return the current price in the decision_cycle,
          because any decision must be made based on the previous price

        :return:
            The latest available price.

            ``None`` if no price information is yet available at this point of time for the strategy.
        """
        assert self.timestamp, f"prepare_decision_cycle() not called - framework missing something somewhere"

        if pair is None:
            pair = self.strategy_universe.get_single_pair()
        assert isinstance(pair, TradingPairIdentifier)
        assert pair.internal_id, "pair.internal_id missing - bad unit test data?"

        series = self.strategy_universe.data_universe.candles.get_samples_by_pair(pair.internal_id)["close"]

        ts = get_prior_timestamp(series, self.timestamp)
        if not ts:
            return None

        return series[ts]

    def get_indicator_value(
        self,
        name: str,
        column: str | None = None,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
        index: int = -1,
    ) -> float | None:
        """Read the available value of an indicator.

        - Returns the latest available indicator value.

        - **Does not** return the current timestamp value in the decision_cycle,
          because any decision must be made based on the previous price.

        - Normalises missing inputs, NaNs and other data issues to Python ``None``.

         Single pair example with a single series indicator (RSI):

        .. code-block:: python

            def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
                indicators.add("rsi", pandas_ta.rsi, {"length": parameters.rsi_length})

            #
            # Then in decide_traces()
            #

            # Read the RSI value of our only trading pair
            indicator_value = input.indicators.get_indicator_value("rsi")

        Single pair example with a multi-series indicator (Bollinger band):

        .. code-block:: python

            def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
                indicators.add("bb", pandas_ta.bbands, {"length": parameters.bb_length})

            #
            # Then in decide_traces()
            #

            # Read bollinger band value for the current trading pair.
            # Bollinger band look up length was 20 and standard deviation 2.0.
            bb_value = input.indicators.get_indicator_value("bb", "BBL_20_2.0")

        Example accessing latest and previous values for cross over test:

        .. code-block:: python

            current_rsi_values[pair] = indicators.get_indicator_value("rsi", pair=pair)
            previous_rsi_values[pair] = indicators.get_indicator_value("rsi", index=-2, pair=pair)

            # Check for RSI crossing our threshold values in this cycle, compared to the previous cycle
            if current_rsi_values[pair] and previous_rsi_values[pair]:
                rsi_cross_above = current_rsi_values[pair] >= parameters.rsi_high and previous_rsi_values[btc_pair] < parameters.rsi_high
                rsi_cross_below = current_rsi_values[pair] < parameters.rsi_low and previous_rsi_values[pair] > parameters.rsi_low

        :param name:
            Indicator name as defined in `create_indicators`.

        :param column:
            The name of the sub-column to read.

            For multicolumn indicators like Bollinger Bands,
            which produce multiple series of data from one column of price data.

        :param pair:
            Trading pair.

            Must be given if the working with a multipair strategy.

        :param index:
            Access a specific previous timeframe item.

            If not given, always return the previous available value.

            Uses Python list access notation.
            - `-1` is the last item (previous time frame value, yesterday).
            - `-2` is the item before previous time frame (the day before yesterday).
            - `0` is looking to the future (the value at the end of the current day that has not yet passed)

        :return:
            The latest available indicator value.

            Any NaN, NA or not a number value in the indicator data is translated to Python ``None``.

            Return ``None`` if value not yet available when asked at the current decision moment.
        """

        series = self.resolve_indicator_data(name, column, pair)

        ts = self.timestamp
        time_frame = _calculate_and_cache_candle_width(series.index)
        shifted_ts = ts + time_frame * index

        try:
            value = series[shifted_ts]
        except KeyError:
            return None

        if pd.isna(value):
            return None

        return value


    def get_indicator_series(
        self,
        name: str,
        column: str | None = None,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
        unlimited=False,
    ) -> pd.Series | None:
        """Get the whole indicator data series.

        By default, return data that is only available before the current timestamp.

        :param unlimited:
            Get all calculated data, even future one, in backtesting.

        :return:
            Indicator data.

            Data may contain NaN values.

            Return ``None`` if any data is not yet available before this stamp.
        """

        series = self.resolve_indicator_data(name, column, pair)

        if unlimited:
            return series

        ts = get_prior_timestamp(series, self.timestamp)
        if ts is None:
            return None

        return series.loc[:ts]

    def resolve_indicator_data(
        self,
        name: str,
        column: str | None = None,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None
    ) -> pd.Series | pd.DataFrame:
        """Get access to indicator data series/frame.

        Throw friendly error messages for pitfalls.

        :param pair:
            Needed when universe contains multiple trading pairs.

            Can be omitted from non-pair indicators.
        """
        assert type(name) == str
        if column is not None:
            assert type(column) == str

        assert self.timestamp, f"prepare_decision_cycle() not called"

        indicator = self.available_indicators.get_indicator(name)
        assert indicator is not None, f"Indicator with name {name} not defined in create_indicators. Available indicators are: {self.available_indicators.get_label()}"

        if indicator.source.is_per_pair():

            if pair is None:
                assert self.strategy_universe.get_pair_count() == 1, f"The strategy universe contains multiple pairs. You need to pass pair argument to the function to determine which trading pair you are manipulating."
                pair = self.strategy_universe.get_single_pair()

            if type(pair) == tuple:
                # Resolve human description
                pair = self.strategy_universe.get_pair_by_human_description(pair)

            assert isinstance(pair, TradingPairIdentifier)
            assert pair.internal_id, "pair.internal_id missing - bad unit test data?"

            key = IndicatorKey(pair, indicator)
        else:
            # Whole universe/custom indicators
            key = IndicatorKey(None, indicator)

        indicator_result = self.indicator_results.get(key)

        if indicator_result is None:
            all_keys = set(self.indicator_results.keys())
            all_indicators = set(self.available_indicators.indicators.keys())
            raise AssertionError(
                f"Indicator results did not contain key {key} for indicator {name}.\n"
                f"Available indicators: {all_indicators}\n"
                f"Available data series: {all_keys}\n"
            )

        data = indicator_result.data
        assert data is not None, f"Indicator pre-calculated values missing for {name} - lookup key {key}"

        if isinstance(data, pd.DataFrame):
            assert column is not None, f"Indicator {name} has multiple available columns to choose from: {data.columns}"
            assert column in data.columns, f"Indicator {name} subcolumn {column} not in the available columns: {data.columns}"
            series = data[column]
        elif isinstance(data, pd.Series):
            series = data
        else:
            raise NotImplementedError(f"Unknown indicator data type {type(data)}")

        return series

    def prepare_decision_cycle(self, cycle: int, timestamp: pd.Timestamp):
        """Called for each decision cycle by the framework..

        - Instead of making a copy of this data structure each time,
          we just bump the timestamp
        """
        logger.info("Strategy indicators moved to the cycle: %d: %s", cycle, timestamp)
        self.timestamp = timestamp



@dataclass
class StrategyInput:
    """Inputs for a trading decision.

    The data structure used to make trade decisions. Captures
    all values that need to go to a single trade, under different live and backtesting
    circumstances.

    - Inputs for `decide_trades` function

    - Enabled when `trading_strategy_engine_version = "0.5"` or higher
    """

    #: Strategy cycle number
    #:
    #: - Deterministic for a backtest
    #: - May be reset for live execution
    #:
    cycle: int

    #: Timestamp of this strategy cycle
    #:
    #: - Timestamp can/should only access earlier data and cannot peek into the future
    #: - Always in UTC, no timezone
    #:
    timestamp: pd.Timestamp

    #: The current state of a strategy
    #:
    #: - You can peek for open/closed positions
    #: - Use :py:meth:`get_position_manager` to access
    #:
    state: State

    #: The source trading universe for this strategy run
    strategy_universe: TradingStrategyUniverse

    #: Parameters used for this backtest or live run
    parameters: StrategyParameters

    #: All indicators that are precalculated with create_indicators()
    #:
    #: - Indicators calculated in `create_indicators` function
    #: - Cached in backtesting for fast reader
    #: - In livee trading recalculated for every cycle
    #:
    indicators: StrategyInputIndicators

    #: Asset pricing model.
    #:
    #: - Used to determine the position size and value of trades
    #: - Backtesting uses historical pricing whereas live trading will read any data directly on-chain
    #: - Access using :py:meth:`get_position_manager`
    #:
    pricing_model: PricingModel

    #: Information about whether this is live or backtest run.
    #:
    execution_context: ExecutionContext

    #: Diagnostics and debug data
    #:
    #: - Undefined format
    #: - Mostly used in internal testing and logging
    #: - Is mutated in-place, but don't rely on this to work for live strategies
    #:
    other_data: dict

    def get_position_manager(self) -> PositionManager:
        """Create a position manager instance to open/close trading positions in this decision cycle."""
        return PositionManager(
            self.timestamp,
            self.strategy_universe,
            self.state,
            self.pricing_model
        )

    def get_default_pair(self) -> TradingPairIdentifier:
        """Get the default trading pair for this stragegy.

        - Works only for single pair strateiges

        :raise InvalidForMultipairStrategy:
            If called for a multi pair strategy
        """
        if self.strategy_universe.get_pair_count() != 1:
            raise InvalidForMultipairStrategy("Strategy universe is multipair - get_default_pair() not available")
        return self.strategy_universe.get_single_pair()



_time_frame_cache = cachetools.Cache(maxsize=SERIES_CACHE_SIZE)

def _calculate_and_cache_candle_width(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    """Get the evenly timestamped index candle/time bar width.

    - Cached for speed - cache size might not make sense for large trading pair use cases
    """

    key = id(index)

    value = _time_frame_cache.get(key)
    if value is None:
        value = index[-1] - index[-2]
        _time_frame_cache[key] = value

    return value
