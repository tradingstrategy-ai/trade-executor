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
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorResultMap, IndicatorSet, IndicatorKey, IndicatorNotFound, InvalidForMultipairStrategy
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.candle import CandleSampleUnavailable
from tradingstrategy.liquidity import LiquidityDataUnavailable
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.utils.time import get_prior_timestamp, ZERO_TIMEDELTA

logger = logging.getLogger(__name__)


SERIES_CACHE_SIZE = 1024


class IndicatorDataNotFoundWithinDataTolerance(Exception):
    """We try to get forward-filled data, but there is no data within our tolerance."""



@dataclass(slots=True)
class StrategyInputIndicators:
    """Indicator results for the strategy decision.

    A helper class to read and manipulate indicator and price values.
    Thi class wraps the indicator results, both cached and real-time, to a format that has good developer experience
    when accessed from `decide_trades()`.

    - Indicators are prepared in `create_indicators` function
    - The framework takes care of recalculating indicators when needed,
      for backtest and live access
    - For backtests, this class is instiated only once
    - We assume all indicator data is forward-filled and no gaps

    How to use

    - For simple strategies calling :py:meth:`get_indicator_value` should be only required here.
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

    def get_ohlcv(
        self,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
    ) -> pd.DataFrame:
        """Get full OHLCV price feed for a trading pair.

        :return:
            DataFrame with open, high, low, close, volume columns.

        """

        if type(pair) == tuple:
            # Resolve human description
            pair = self.strategy_universe.get_pair_by_human_description(pair)

        if pair is None:
            pair = self.strategy_universe.get_single_pair()

        assert isinstance(pair, TradingPairIdentifier)
        assert pair.internal_id, "pair.internal_id missing - bad unit test data?"

        return self.strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id)

    def get_price(
        self,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription |  None = None,
        data_lag_tolerance=pd.Timedelta(days=7),
        index: int = -1,
        timestamp: pd.Timestamp | None = None,
        column="close",
    ) -> USDollarPrice | None:
        """Read the available close price of a trading pair.

        - Returns the latest available close price.

        - **Does not** return the current price in the decision_cycle,
          because any decision must be made based on the previous price
          to avoid lookahead bias.

        :param pair:
            The trading pair for which we query the price.

            Give as id object or human description tuple format.

            E.g. `(ChainId.centralised_exchange, "binance", "ETH", "USDT")`.

        :param data_lag_tolerance:
            In the case the data has issues (no recent price),
            then accept a price that's this old.

        :param index:
            Access a specific previous timeframe item.

            If not given, always return the previous available value.
            Timeframe = candle bar here.

            Uses Python list access notation.
            - `-1` is the last item (previous time frame value, yesterday).
            - `-2` is the item before previous time frame (the day before yesterday).
            - `0` is looking to the future (the value at the end of the current day that has not yet passed)

        :param timestamp:
            Look price at a specific timestamp.

            Manually calculate lookback. There is no timeshift for this value,
            so unless you are careful you may case lookahead bias.

            `index` parameter is ignored.

        :param column:
            Which column to read from the price series.

            E.g. "volume".

        :return:
            The latest available price.

            ``None`` if no price information is yet available at this point of time for the strategy.
        """
        if timestamp:
            shifted_ts = timestamp
        else:
            assert self.timestamp, f"prepare_decision_cycle() not called - framework missing something somewhere"
            ts = self.timestamp
            time_frame = self.strategy_universe.data_universe.time_bucket.to_pandas_timedelta()
            shifted_ts = ts + time_frame * index

        if type(pair) == tuple:
            # Resolve human description
            pair = self.strategy_universe.get_pair_by_human_description(pair)

        if pair is None:
            pair = self.strategy_universe.get_single_pair()

        assert isinstance(pair, TradingPairIdentifier)
        assert pair.internal_id, "pair.internal_id missing - bad unit test data?"

        try:
            price, when = self.strategy_universe.data_universe.candles.get_price_with_tolerance(
                pair.internal_id,
                shifted_ts,
                tolerance=data_lag_tolerance,
                kind=column,
            )
            return price
        except CandleSampleUnavailable:
            return None

    def get_tvl(
        self,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription |  None = None,
        data_lag_tolerance=pd.Timedelta(days=7),
        index: int = -1,
        timestamp: pd.Timestamp | None = None,
    ) -> USDollarPrice | None:
        """Read the available TVL of a trading pair.

        - Returns the latest available TVL/liquidity sample.

        - **Does not** return the current liquidity in the decision_cycle,
          because any decision must be made based on the previous price
          to avoid lookahead bias.

        See also :py:meth:`get_price`

        :param pair:
            The trading pair for which we query the price.

            Give as id object or human description tuple format.

            E.g. `(ChainId.centralised_exchange, "binance", "ETH", "USDT")`.

        :param data_lag_tolerance:
            In the case the data has issues (no recent price),
            then accept a price that's this old.

        :param index:
            Access a specific previous timeframe item.

            If not given, always return the previous available value.
            Timeframe = candle bar here.

            Uses Python list access notation.
            - `-1` is the last item (previous time frame value, yesterday).
            - `-2` is the item before previous time frame (the day before yesterday).
            - `0` is looking to the future (the value at the end of the current day that has not yet passed)

        :param timestamp:
            Look price at a specific timestamp.

            Manually calculate lookback. There is no timeshift for this value,
            so unless you are careful you may case lookahead bias.

            `index` parameter is ignored.

        :return:
            The latest available TVL.

            ``None`` if no price information is yet available at this point of time for the strategy.
        """
        if timestamp:
            shifted_ts = timestamp
        else:
            assert self.timestamp, f"prepare_decision_cycle() not called - framework missing something somewhere"
            ts = self.timestamp
            time_frame = self.strategy_universe.data_universe.time_bucket.to_pandas_timedelta()
            shifted_ts = ts + time_frame * index

        if type(pair) == tuple:
            # Resolve human description
            pair = self.strategy_universe.get_pair_by_human_description(pair)

        if pair is None:
            pair = self.strategy_universe.get_single_pair()

        assert isinstance(pair, TradingPairIdentifier)
        assert pair.internal_id, "pair.internal_id missing - bad unit test data?"

        try:
            price, when = self.strategy_universe.data_universe.liquidity.get_liquidity_with_tolerance(
                pair.internal_id,
                shifted_ts,
                tolerance=data_lag_tolerance,
            )
            return price
        except LiquidityDataUnavailable:
            return None

    def get_indicator_value(
        self,
        name: str,
        column: str | None = None,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
        index: int = -1,
        clock_shift: pd.Timedelta = pd.Timedelta(hours=0),
        data_delay_tolerance: pd.Timedelta="auto",
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
            Timeframe = candle bar here.

            Uses Python list access notation.
            - `-1` is the last item (previous time frame value, yesterday).
            - `-2` is the item before previous time frame (the day before yesterday).
            - `0` is looking to the future (the value at the end of the current day that has not yet passed)

        :param clock_shift:
            Used in time-shifted backtesting.

        :param data_delay_tolerance:
            If we do not have an exact timestamp match in the data series, look for the previous value.

            Look back max `data_delay_tolerance` days / hours to get a previous value using forward-fill technique.

            We need to do this when there is a mismatch between the indicator timeframe (e.g. daily)
            and decision cycle / price time frame (e.g. 15 minutes).

            Set to `None` to always return indicator value for the exact timestamp match.

            Set to `auto to try to figure out mismatch between indicator data and candle data automatically.s

        :return:
            The latest available indicator value.

            Any NaN, NA or not a number value in the indicator data is translated to Python ``None``.

            Return ``None`` if value not yet available when asked at the current decision moment.

        :raise IndicatorDataNotFoundWithinDataTolerance:
            We asked `data_delay_tolerance` look backwards, but there wasn't any samples within the tolerance.
        """

        series = self.resolve_indicator_data(name, column, pair)
        ts = self.timestamp

        time_frame = _calculate_and_cache_candle_width(series.index)

        if time_frame is None:
            # Bad data.
            # E.g. portfolio data with missing values
            return None

        if data_delay_tolerance == "auto":
            ts = ts.floor(time_frame)
            data_delay_tolerance = time_frame

        shifted_ts = ts + time_frame*index + clock_shift

        # First try direct timestamp hit.
        # This is the case for any normal strategies,
        # where time-series data and decision cycles have the equal indexes
        try:
            value = series[shifted_ts]
        except KeyError:

            if shifted_ts > series.index[-1]:
                # The data series has ended before the timestamp,
                # and there are not going to be new values in the future
                return None

            # Try to check for uneven timeframes
            # E.g. 1d RSI indicator data and 1s decision cycle
            #
            if data_delay_tolerance is not None:
                # TODO: Do we need to cache the indexer... does it has its own storage?
                ffill_indexer = series.index.get_indexer([self.timestamp], method="ffill")
                before_match_iloc = ffill_indexer[0]
                before_match_timestamp = series.index[before_match_iloc]

                if before_match_iloc < 0:
                    # We get -1 if there are no timestamps where the forward fill could start
                    # This means there are not yet any samples available at the timestamp,
                    # because the time series will start after the timestamp
                    return None

                    # first_sample_timestamp = series.index[0]
                    #raise IndicatorDataNotFoundWithinDataTolerance(
                    #    f"Could not find any samples for pair {pair}, indicator {name} at {self.timestamp}\n"
                    #    f"- Series has {len(series)} samples\n"
                    #    f"- First sample is at {first_sample_timestamp}\n"
                    #)
                before_match = series.iloc[before_match_iloc]

                # Internal sanity check
                distance = self.timestamp - before_match_timestamp
                assert distance >= ZERO_TIMEDELTA, f"Somehow we managed to get a indicator timestamp {before_match_timestamp} that is newer than asked {self.timestamp}"

                if distance > data_delay_tolerance:
                    raise IndicatorDataNotFoundWithinDataTolerance(
                        f"Asked indicator {name}. Data delay tolerance is {data_delay_tolerance}, but the delay was longer {distance}.\n"
                        f"Our timestamp {self.timestamp}, fixed timestamp {shifted_ts}, data available at {before_match_timestamp}.\n"
                    )

                value = before_match
            else:
                # No match
                return None

        # The input data was not properly cleaned up and has duplicated values for some dates/times
        assert not isinstance(value, pd.Series), "Duplicate DatetimeIndex entries detected for: {name} {column} {pair}"

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

        if self.timestamp is None:
            # Accessed in backtesting diagnostics
            unlimited = True

        if not unlimited:
            assert self.timestamp is not None, "StrategInputIndicators.timestamp not set for decide_trades(). Call get_indicator_series(unlimited=True) to get all data."

        series = self.resolve_indicator_data(name, column, pair, unlimited=unlimited)

        if unlimited:
            return series

        ts = get_prior_timestamp(series, self.timestamp)
        if ts is None:
            return None

        return series.loc[:ts]

    def get_price_series(
        self,
        column: str = "close",
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
    ) -> pd.Series:
        """Get the whole price series.

        - Use for visualisation and other checks
        - Not useful inside `decide_trades`, as includes future data

        :param column:
            Which column to get, default to "close",

        :return:
            Indicator data.

            Data may contain NaN values.

        """

        if type(pair) == tuple:
            # Resolve human description
            pair = self.strategy_universe.get_pair_by_human_description(pair)

        if pair is None:
            pair = self.strategy_universe.get_single_pair()

        assert isinstance(pair, TradingPairIdentifier)
        assert pair.internal_id, "pair.internal_id missing - bad unit test data?"

        df = self.strategy_universe.data_universe.candles.get_candles_by_pair(
            pair.internal_id,
        )
        return df[column]

    def get_indicator_dataframe(
        self,
        name: str,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None
    ) -> pd.DataFrame:
        """Get the whole raw indicator data for DataFrame-like indicator with multiple columns.

        See also :py:meth:`get_indicator_series`

        :return:
            DataFrame for a multicolumn indicator like Bollinger Bands or ADX
        """
        df = self.resolve_indicator_data(name, "all", pair, unlimited=True)
        assert isinstance(df, pd.DataFrame), f"Not DataFrame indicator: {name}"
        return df

    def resolve_indicator_data(
        self,
        name: str,
        column: str | None = None,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
        unlimited=False,
    ) -> pd.Series | pd.DataFrame:
        """Get access to indicator data series/frame.

        Throw friendly error messages for pitfalls.

        :param name:
            Indicator name

        :param column:
            Column name for multi-column indicators.

            "all" to get the whole DataFrame.

        :param pair:
            Needed when universe contains multiple trading pairs.

            Can be omitted from non-pair indicators.

        :param unlimited:
            Allow loading of past and future data.

        """
        assert type(name) == str
        if column is not None:
            assert type(column) == str, f"Expected string, got {type(column)}: {column}"

        if not unlimited:
            assert self.timestamp, f"StrategyInputIndicators.timestamp is None. prepare_decision_cycle() not called, or you are outside a decide_trades() function."

        indicator = self.available_indicators.get_indicator(name)
        if indicator is None:
            raise IndicatorNotFound(f"Indicator with name '{name}' not defined by create_indicators(). Available indicators are: {self.available_indicators.get_label()}")

        if indicator.source.is_per_pair():

            if pair is None:
                if self.strategy_universe.get_pair_count() != 1:
                    raise InvalidForMultipairStrategy(f"The strategy universe contains multiple pairs. You need to pass pair argument to the function to determine which trading pair you are manipulating.")
                pair = self.strategy_universe.get_single_pair()

            if type(pair) == tuple:
                # Resolve human description
                pair = self.strategy_universe.get_pair_by_human_description(pair)

            assert isinstance(pair, TradingPairIdentifier), f"Expected TradingPairIdentifier instance, got {type(pair)}: {pair}"
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

            if column == "all":
                return data

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

    def is_visualisation_enabled(self) -> bool:
        """Should we render any visualisation or not.

        - Use this function inside `decide_trades()` to figure out if `state.visualisation` should be filled in

        - Disabled for grid seach to optimise grid search speed, as the visualisation results would be likely be discarded

        Example:

        .. code-block:: python

            def decide_trades(input: StrategyInput):

                # ...

                #
                # Visualisations
                #

                if input.is_visualisation_enabled():

                    visualisation = state.visualisation  # Helper class to visualise strategy output

                    visualisation.plot_indicator(
                        timestamp,
                        f"ETH",
                        PlotKind.technical_indicator_detached,
                        current_price[eth_pair],
                        colour="blue",
                    )

                    # Draw BTC + ETH RSI between its trigger zones for this pair of we got a valid value for RSI for this pair

                    # BTC RSI daily
                    if pd.notna(current_rsi_values[btc_pair]):
                        visualisation.plot_indicator(
                            timestamp,
                            f"RSI",
                            PlotKind.technical_indicator_detached,
                            current_rsi_values[btc_pair],
                            colour="orange",
                        )
        """

        # Visuals always enabled for live tradin
        if self.execution_context.mode.is_live_trading():
            return True

        # Grid search disables visual plotting to save speed and space
        return self.execution_context.has_visualisation()


_time_frame_cache = cachetools.Cache(maxsize=SERIES_CACHE_SIZE)

def _calculate_and_cache_candle_width(index: pd.DatetimeIndex | pd.MultiIndex) -> pd.Timedelta | None:
    """Get the evenly timestamped index candle/time bar width.

    - Cached for speed - cache size might not make sense for large trading pair use cases

    :return:
        None of the index is empty and candle width cannot be calculated
    """

    # The original data is in grouped DF
    if isinstance(index, pd.MultiIndex):
        # AssertionError: Got index: MultiIndex([(2854997, '2024-04-04 21:00:00'),
        #        (2854997, '2024-04-04 22:00:00'),
        index = index.get_level_values(1)

    assert isinstance(index, pd.DatetimeIndex), f"Got index: {index}"

    key = id(index)

    value = _time_frame_cache.get(key)
    if value is None:
        if len(index) > 2:
            value = index[-1] - index[-2]
        else:
            value = None
        _time_frame_cache[key] = value

    return value
