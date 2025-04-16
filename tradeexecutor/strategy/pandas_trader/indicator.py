"""Indicator definitions.

- Defining technical analysis and external indicators for :term:`trading strategies <trading strategy>`

- Precalculating indicators and caching the results on a disk

- Continuing indicator calculator in the case of crash

- Support both single backtest and grid search modes

- Support indicators depending on other indicators via dependency order resolution using :py:class:`IndicatorDependencyResolver`

- Support external data sources

To get started with indicators see examples in :py:mod:`tradeexecutor.strategy.pandas_trader.indicator_decorator`.
"""
import datetime
import hashlib
import threading
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from pprint import pformat

from IPython import get_ipython
from skopt.space import Dimension

# Enable pickle patch that allows multiprocessing in notebooks
from tradeexecutor.monkeypatch import cloudpickle_patch

import concurrent
import enum
import inspect
import itertools
import os
import pickle
import shutil
import signal
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from multiprocessing import Process
from pathlib import Path
from types import NoneType
from typing import Callable, Protocol, Any, TypeAlias
import logging

import futureproof
import pandas as pd

from tqdm_loggable.auto import tqdm

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_context import ExecutionContext, notebook_execution_context
from tradeexecutor.strategy.parameters import StrategyParameters, RollingParameter
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, UniverseCacheKey
from tradeexecutor.utils.cpu import get_safe_max_workers_count
from tradeexecutor.utils.python_function import hash_function
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.utils.groupeduniverse import PairCandlesMissing

logger = logging.getLogger(__name__)


#: Where do we keep precalculated indicator Parquet files
#:
DEFAULT_INDICATOR_STORAGE_PATH = Path(os.path.expanduser("~/.cache/indicators"))


class IndicatorCalculationFailed(Exception):
    """We could not calculate the given indicator.

    - Wrap the underlying Python exception to a friendlier error message
    """

class IndicatorFunctionSignatureMismatch(Exception):
    """Given Pythohn function cannot run on the passed parameters."""


class IndicatorNotFound(Exception):
    """Asked for an indicator we do not have.

    See :py:func:`resolve_indicator_data`
    """

class IndicatorDependencyResolutionError(Exception):
    """Something wrong trying to look up data from other indicators.

    """

class IndicatorOrderError(Exception):
    """An indicator in earier dependency resolution order tries to ask data for one that comes later.

    """

class InvalidForMultipairStrategy(Exception):
    """Try to use single trading pair functions in a multipair strategy."""



class IndicatorSource(enum.Enum):
    """The data on which the indicator will be calculated."""

    #: Calculate this indicator based on candle close price
    #:
    #: Example indicators
    #:
    #: - RSI
    #: - Moving overage
    #:
    close_price = "close_price"

    #: Calculate this indicator based on candle open price
    #:
    #: Not used commonly
    #:
    open_price = "open_price"

    #: Calculate this indicator based on multipe data points (open, high, low, close, volume)
    #:
    #: Example indicators
    #:
    #: - Money flow index (MFI) reads close, high, low columns
    #:
    #: The indicator function can take arguments named: open, high, low, close, volume
    #: which all are Pandas US dollar series. If parameters are not present they are discarded.
    #:
    ohlcv = "ohlcv"

    #: Calculate this indicator based open, high, low, close TVL data.
    #:
    #: - Per pair
    #:
    #:
    tvl = "tvl"

    #: This indicator is calculated once per the strategy universe
    #:
    #: These indicators are custom and do not have trading pair set
    #:
    strategy_universe = "strategy_universe"

    #: Data loaded from an external source.
    #:
    #: Per-pair data.
    #:
    external_per_pair = "external_per_pair"

    #: This indicator is based purely on the previously calculated indicators
    #:
    #: - For each pari
    #: - No price data etc. used
    #:
    #: See :py:class:`IndicatorDependencyResolver`
    #:
    dependencies_only_per_pair = "dependencies_only_per_pair"

    #: This indicator is based purely on the previously calculated indicators
    #:
    #: - Once per universe
    #: - No price data etc. used
    #:
    #: See :py:class:`IndicatorDependencyResolver`
    #:
    dependencies_only_universe = "dependencies_only_universe"

    def is_per_pair(self) -> bool:
        """This indicator is calculated to all trading pairs."""
        return self in (
            IndicatorSource.open_price,
            IndicatorSource.close_price,
            IndicatorSource.ohlcv,
            IndicatorSource.external_per_pair,
            IndicatorSource.dependencies_only_per_pair,
            IndicatorSource.tvl,
        )


def _flatten_index(series: pd.Series) -> pd.Series:
    """Ensure that any per-pair series we have has DatetimeIndex, not MultiIndex."""
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    if isinstance(series.index, pd.MultiIndex):
        new_index = series.index.get_level_values(1)  # assume pair id, timestamp tuples
        assert isinstance(new_index, pd.DatetimeIndex)
        series_2 = series.copy()
        series_2.index = new_index
        return series_2
    else:
        raise NotImplementedError(f"Unknown index: {series.index}")


@dataclass(slots=True)
class IndicatorDefinition:
    """A definition for a single indicator.

    - Indicator definitions are static - they do not change between the strategy runs

    - Used as id for the caching the indicator results

    - Definitions are used to calculate indicators for all trading pairs,
      or once over the whole trading universe

    - Indicators are calcualted independently from each other -
      a calculation cannot access cached values of other calculation
    """

    #: Name of this indicator.
    #:
    #: Later in `decide_trades()` you use this name to access the indicator data.
    #:
    name: str

    #: The underlying method we use to
    #:
    #: Same function can part of multiple indicators with different parameters (length).
    #:
    #: Because function pickling issues, this may be set to ``None`` in results.
    #:
    func: Callable | None

    #: Parameters for building this indicator.
    #:
    #: - Each key is a function argument name for :py:attr:`func`.
    #: - Each value is a single value
    #:
    #: - Grid search multiple parameter ranges are handled outside indicator definition
    #:
    parameters: dict

    #: On what trading universe data this indicator is calculated
    #:
    source: IndicatorSource = IndicatorSource.close_price

    #: Dependency resolution order.
    #:
    #: Indicators are calculated in order from the lowest order to the highest.
    #: The order parameter allows lightweight dependency resolution, where later
    #: indicators and read the earlier indicators data.
    #:
    #: By default all indicators are on the same dependency resolution order layer `1`
    #: and cannot access data from other indicators. You need to create indicator
    #: with `order == 2` to be able to access data from indicators where `order == 1`.
    #:
    #: See :py:class:`IndicatorDependencyResolver` for details and examples.
    #:
    dependency_order: int = 1

    #: This indicator needs multiple variations with different parameters for a single run
    variations: bool = False

    _cached_hash: int = None

    def __repr__(self):
        variations_msg = "w/variations" if self.variations else ""
        return f"<Indicator {self.name} using {self.func.__name__ if self.func else '?()'} for {self.parameters} {variations_msg}>"

    def __eq__(self, other):
        return self.name == other.name and self.parameters == other.parameters and self.source == other.source

    def _calculate_cached_hash(self):
        # Optimisation
        if not self._cached_hash:
            def _unlistify(v):
                if isinstance(v, list):
                    return tuple(v)
                return v
            try:
                items = frozenset({k: _unlistify(v) for k, v in self.parameters.items()})
                args = (self.name, items, self.source)
                self._cached_hash = hash(args)
            except Exception as e:
                raise RuntimeError(f"Could not hash {self}.\nIf changing grid search to backtest, remember to change lists to single value.\nIf changing backtest to grid search, do not call create_indicators() directly.\nException is {e}") from e

        return self._cached_hash

    def __hash__(self):
        return self._calculate_cached_hash()

    def __post_init__(self):
        assert type(self.name) == str
        assert type(self.parameters) == dict

        if self.func is not None:
            assert callable(self.func)
            validate_function_kwargs(self.func, self.parameters)

        # Check for common data passing errors
        for k, v in self.parameters.items():
            assert not isinstance(v, Dimension), f"Detected scikit-optimize Dimension as a parameter value: {k}: {v} - did you accidentally pass in optimiser search space to a single backtest"

        # Dict only retains initial order, not fixed sort order.
        # Ensure parameters are always iterated in a stable deterministic order.
        self.parameters = OrderedDict(sorted(self.parameters.items()))

    def get_function_body_hash(self) -> str:
        """Calculate the hash for the function code.

        Allows us to detect if the function body changes.
        """
        return hash_function(self.func, bytecode_only=True)

    def is_needed_for_pair(self, pair: TradingPairIdentifier) -> bool:
        """Currently indicators are calculated for spont pairs only."""
        return pair.is_spot()

    def is_per_pair(self) -> bool:
        return self.source.is_per_pair()

    def calculate_by_pair_external(
        self,
        pair: TradingPairIdentifier,
        resolver: "IndicatorDependencyResolver",
        timestamp: pd.Timestamp | None,
        execution_context: ExecutionContext,
    ) -> pd.DataFrame | pd.Series:
        """Calculate indicator for external data.

        :param pair:
            Trading pair we are calculating for.

        :return:
            Single or multi series data.

            - Multi-value indicators return DataFrame with multiple columns (e.g. BB).
            - Single-value indicators return Series (e.g. RSI, SMA).

        """

        try:
            ret = self.func(**self._fix_parameters_for_function_signature(resolver, pair, timestamp, execution_context))
            output_fixed = _flatten_index(ret)
            return self._check_good_return_value(output_fixed)
        except Exception as e:
            raise IndicatorCalculationFailed(f"Could not calculate external data indicator {self.name} ({self.func}) for parameters {self.parameters}, pair {pair}") from e

    def calculate_by_pair(
        self,
        input: pd.Series,
        pair: TradingPairIdentifier,
        resolver: "IndicatorDependencyResolver",
        timestamp: pd.Timestamp | None,
        execution_context: ExecutionContext,
    ) -> pd.DataFrame | pd.Series:
        """Calculate the underlying indicator value.

        :param input:
            Price series used as input.

        :return:
            Single or multi series data.

            - Multi-value indicators return DataFrame with multiple columns (BB).
            - Single-value indicators return Series (RSI, SMA).

        """
        try:
            input_fixed = _flatten_index(input)
            ret = self.func(input_fixed, **self._fix_parameters_for_function_signature(resolver, pair, timestamp, execution_context))
            return self._check_good_return_value(ret)
        except Exception as e:
            raise IndicatorCalculationFailed(f"Could not calculate indicator {self.name} ({self.func}) for parameters {self.parameters}, input data is {len(input)} rows: {e}, pair is {pair}") from e

    def calculate_dependencies_only_per_pair(
        self,
        pair: TradingPairIdentifier,
        resolver: "IndicatorDependencyResolver",
        timestamp: pd.Timestamp | None,
        execution_context: ExecutionContext,
    ) -> pd.DataFrame | pd.Series:
        """Calculate the indicator value that only takes other indicators as input."""
        assert self.dependency_order > 1, "Dependency-based indicator order cannot be first. Did you forget to declare dependencies?"
        try:
            ret = self.func( **self._fix_parameters_for_function_signature(resolver, pair, timestamp, execution_context))
            return self._check_good_return_value(ret)
        except Exception as e:
            raise IndicatorCalculationFailed(f"Could not calculate indicator {self.name} ({self.func}) for parameters {self.parameters}, exception: {e}, pair is {pair}") from e

    def calculate_dependencies_only_per_universe(
        self,
        resolver: "IndicatorDependencyResolver",
        timestamp: pd.Timestamp | None,
        execution_context: ExecutionContext,
    ) -> pd.DataFrame | pd.Series:
        """Calculate the indicator value that only takes other indicators as input."""
        assert self.dependency_order > 1, "Dependency-based indicator order cannot be first. Did you forget to declare dependencies?"
        try:
            ret = self.func( **self._fix_parameters_for_function_signature(resolver, None, timestamp, execution_context))
            return self._check_good_return_value(ret)
        except Exception as e:
            raise IndicatorCalculationFailed(f"Could not calculate indicator {self.name} ({self.func}) for parameters {self.parameters}, rows: {e}") from e

    def calculate_by_pair_ohlcv(
        self,
        candles: pd.DataFrame,
        pair: TradingPairIdentifier,
        resolver: "IndicatorDependencyResolver",
        timestamp: pd.Timestamp | None,
        execution_context: ExecutionContext,
    ) -> pd.DataFrame | pd.Series:
        """Calculate the underlying OHCLV indicator value.

        Assume function can take parameters: `open`, `high`, `low`, `close`, `volume`,
        or any combination of those.

        :param input:
            Raw OHCLV candles data.

        :return:
            Single or multi series data.

            - Multi-value indicators return DataFrame with multiple columns (BB).
            - Single-value indicators return Series (RSI, SMA).

        """

        assert isinstance(candles, pd.DataFrame), f"OHLCV-based indicator function must be fed with a DataFrame"

        input_fixed = _flatten_index(candles)

        needed_args = ("open", "high", "low", "close", "volume")
        full_kwargs = {}
        func_args = inspect.getfullargspec(self.func).args
        for a in needed_args:
            if a in func_args:
                full_kwargs[a] = input_fixed[a]

        if len(full_kwargs) == 0:
            raise IndicatorCalculationFailed(f"Could not calculate OHLCV indicator {self.name} ({self.func}): does not take any of function arguments from {needed_args}")

        fixed_params = self._fix_parameters_for_function_signature(resolver, pair, timestamp, execution_context)

        full_kwargs.update(fixed_params)

        try:
            ret = self.func(**full_kwargs)
            return self._check_good_return_value(ret)
        except Exception as e:
            raise IndicatorCalculationFailed(f"Could not calculate indicator {self.name} ({self.func}) for parameters {self.parameters}, candles is {len(candles)} rows, {candles.columns} columns\nThe original exception was: {e}\nCalling with args {pformat(full_kwargs)}") from e

    def calculate_universe(
        self,
        input: TradingStrategyUniverse,
        resolver: "IndicatorDependencyResolver",
        timestamp: pd.Timestamp | None,
        execution_context: ExecutionContext,
    ) -> pd.DataFrame | pd.Series:
        """Calculate the underlying indicator value.

        :param input:
            Price series used as input.

        :return:
            Single or multi series data.

            - Multi-value indicators return DataFrame with multiple columns (BB).
            - Single-value indicators return Series (RSI, SMA).

        """
        try:
            ret = self.func(input, **self._fix_parameters_for_function_signature(resolver, None, timestamp, execution_context))
            return self._check_good_return_value(ret)
        except Exception as e:
            raise IndicatorCalculationFailed(f"Could not calculate indicator {self.name} ({self.func}) for parameters {self.parameters}, input universe is {input}.\nException is {e}\n\n To use Python debugger, set `max_workers=1`, and if doing a grid search, also set `multiprocess=False`") from e

    def _check_good_return_value(self, df):
        assert isinstance(df, (pd.Series, pd.DataFrame)), f"Indicator did not return pd.DataFrame or pd.Series: {self.name}, we got {type(df)}\nCheck you are using IndicatorSource correcly e.g. IndicatorSource.close_price when creating indicators"
        return df

    def _fix_parameters_for_function_signature(
        self,
        resolver: "IndicatorDependencyResolver",
        pair: TradingPairIdentifier | None,
        timestamp: pd.Timestamp | None,
        execution_context: ExecutionContext | None,
    ) -> dict:
        """Update parameters to include optional parameter if the indicator needs them.

        - This was a late addon, so we cram it in here.
        """

        parameters = self.parameters.copy()

        func_args = inspect.getfullargspec(self.func).args
        if "dependency_resolver" in func_args:
            parameters["dependency_resolver"] = resolver

        if pair:
            if "pair" in func_args:
                parameters["pair"] = pair

        if "timestamp" in func_args:
            assert isinstance(timestamp, (pd.Timestamp, NoneType)), f"Got {type(timestamp)}, but expected pd.Timestamp"
            parameters["timestamp"] = timestamp

        if execution_context:
            assert isinstance(execution_context, ExecutionContext)
            if "execution_context" in func_args:
                parameters["execution_context"] = execution_context

        return parameters


@dataclass(slots=True, frozen=True)
class IndicatorKey:
    """Cache key used to read indicator results.

    - Used to describe all indicator combinations we need to create

    - Used as the key in the indicator result caching

    """

    #: Trading pair if this indicator is specific to a pair
    #:
    #: `None` if this indicator is for the whole universe, using everything as an input.
    #:
    pair: TradingPairIdentifier | None

    #: The definition of this indicator
    definition: IndicatorDefinition

    def __post_init__(self):
        assert isinstance(self.pair, (TradingPairIdentifier, NoneType))
        assert isinstance(self.definition, IndicatorDefinition)

    def __repr__(self):
        return f"<IndicatorKey {self.get_cache_key()}>"

    def get_pair_cache_id(self) -> str:
        """Get unique value for this trading pair or 'universe' if there isn't one."""
        pair = self.pair
        if pair is not None:
            # TODO: Include 3 letter exchange id to discriminate between uni v2 and uni v3
            if pair.fee != 0.30:
                fee_slug = f"-{int(pair.fee * 10_000)}"
            else:
                fee_slug = ""
            return f"{pair.base.token_symbol}-{pair.quote.token_symbol}{fee_slug}"
        else:
            # Indicator calculated over the universe
            assert not self.definition.source.is_per_pair()
            return "universe"

    def __eq__(self, other):
        return self.pair == other.pair and self.definition == other.definition

    def __hash__(self):
        return hash((self.pair, self.definition))

    def get_cache_key(self) -> str:
        """Get unique key that holds the disk cached file  for the indicator function."""
        slug = self.get_pair_cache_id()

        def norm_value(v: Any):

            if isinstance(v, RollingParameter):
                raise AssertionError(f"Should not happen - rolling parameters must be expanded earlier: {self}: {v}")
                # values = list(v.values)
                # assert len(values) > 0, f"RollingParameter lacks values: {v}"
                # v = ",".join([str(x) for x in values])
            elif isinstance(v, list):
                raise AssertionError(f"Should not happen - parameter received list as a value: {self.definition.name} {type(v)}: {v}")
            elif isinstance(v, set):
                # Used for "set of tags" use cases
                # Set order must be made deterministic.
                v = str(sorted(list(v)))
            elif isinstance(v, enum.Enum):
                v = str(v.value)
            else:
                v = str(v)
            return v

        # Make sort order stable
        entries = list(self.definition.parameters.items())
        entries = sorted(entries, key=lambda t: t[0])

        parameters = ",".join([f"{k}={norm_value(v)}" for k, v in entries])

        if len(parameters) > 80:
            # Parameters are too long to be presented in the filename,
            # so we just take the 8 bytes hash
            parameters = _deterministic_hash(parameters)[-8:]

        return f"{self.definition.name}_{self.definition.get_function_body_hash()}({parameters})-{slug}"


@dataclass(slots=True, frozen=False)
class IndicatorSet:
    """Define the indicators that are needed by a trading strategy.

    - For backtesting, indicators are precalculated

    - For live trading, these indicators are recalculated for the each decision cycle

    - Indicators are calculated for each given trading pair, unless specified otherwise,
      and a separate :py:class:`IndicatorKey` is generated by :py:meth:`generate_combinations`

    - Indicators that have multiple instances of itself with different parameters in the same backtest
      are called variations - mainly useful for strategies where parameters are rollingly adjusted.

    See :py:class:`CreateIndicatorsProtocolV2` for usage.
    """

    indicators: dict[str, IndicatorDefinition] = field(default_factory=dict)
    variation_cache: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    variation_lookup: dict[int, IndicatorDefinition] = field(default_factory=dict)

    def has_indicator(self, name: str) -> bool:
        return name in self.indicators

    def is_varying_indicator(self, name: str) -> bool:
        return len(self.variation_cache[name]) > 0

    @staticmethod
    def make_parameter_key(
        name: str,
        parameters: dict,
    ) -> int:
        assert isinstance(parameters, dict)
        parameters = [("name", name)] + [(key, value) for key, value in parameters.items()]
        parameters.sort(key=lambda x: x[0])
        return hash(tuple(parameters))

    def get_label(self):

        if len(self.indicators) == 0:
            return "<zero indicators defined>"

        return ", ".join(k for k in self.indicators.keys())

    def get_count(self) -> int:
        """How many indicators we have"""
        return len(self.indicators)

    def get_indicator(self, name: str) -> IndicatorDefinition | None:
        """Get a named indicator definition."""
        return self.indicators.get(name)

    def get_variations(self, name) -> list[dict]:
        """Get all parameter variations for an indicator."""
        return self.variation_cache[name]

    def get_indicator_by_name_and_parameters(
        self,
        name: str,
        parameters: dict,
    )-> IndicatorDefinition | None:
        """Get a variation indicator definition."""
        assert type(name) == str
        assert type(parameters) == dict
        key = IndicatorSet.make_parameter_key(name, parameters)
        return self.variation_lookup.get(key)

    def add(
        self,
        name: str,
        func: Callable,
        parameters: dict | None = None,
        source: IndicatorSource=IndicatorSource.close_price,
        order=1,
        variations=False,
    ):
        """Add a new indicator to this indicator set.

        Builds an indicator set for the trading strategy,
        called from `create_indicators`.

        See :py:class:`CreateIndicatorsProtocol` for usage.

        :param name:
            Name of the indicator.

            Human-readable name. If the same function is calculated multiple times, e.g. EMA,
            you can have names like `ema_short` and `ema_long`.

        :param func:
            Python function to be called.

            Function takes arguments from `parameters` dict.
            It must return either :py:class:`pd.DataFrame` or :py:class:`pd.Series`.

        :param parameters:
            Parameters to be passed to the Python function.

            Raw `func` Python arguments.

            You can pass parameters as is from `StrategyParameters`.

        :param source:
            Data source on this indicator is calculated.

            Defaults to the close price for each trading pair.
            To calculate universal indicators set to :py:attr:`IndicatorSource.strategy_universe`.

        :param order:
            Dependency resolution order.

            See :py:attr:`tradingstrategy.strategy.pandas_trader.indicator.IndicatorKey.order` parameter.

        :param variations:
            Set for rolling indicators that need to have their values calculated with several combinations for a backtest run.

            You can add the same variation multiple times and the sequent adds are ignored.
        """
        assert type(name) == str
        assert callable(func), f"{func} is not callable"

        if parameters is None:
            parameters = {}
        assert type(parameters) == dict, f"parameters must be dictionary, we got {parameters.__class__}"
        assert isinstance(source, IndicatorSource), f"Expected IndicatorSource, got {type(source)}"

        # For this indicator, we need eto calculate multiple variations with different parameters
        if variations:
            key = self.make_parameter_key(name, parameters)
            # Use 32-bit hash to separate variations.
            # May be collision prone.
            hashed_name = f"{name}_{hex(key)[-8:]}"
            if hashed_name in self.indicators:
                # add() twice is a legit use case as time series may have the same parameter value twice in different sections
                # f"Hashed name for varying indicator {hashed_name} already defined - hash collision, bug?\nOur parameters: {parameters}\nExisting variations are with parameters: {self.variation_cache[name]}"
                return
            self.indicators[hashed_name] = IndicatorDefinition(name, func, parameters, source, order, variations=True)
            self.variation_cache[name].append(parameters)
            self.variation_lookup[key] = self.indicators[hashed_name]
        else:
            assert name not in self.indicators, f"Indicator {name} already added"
            self.indicators[name] = IndicatorDefinition(name, func, parameters, source, order)

    def iterate(self) -> Iterable[IndicatorDefinition]:
        yield from self.indicators.values()

    def generate_combinations(self, strategy_universe: TradingStrategyUniverse) -> Iterable[IndicatorKey]:
        """Create all individual indicator (per pair) we need to calculate for this trading universe.

        - Because most indicators are per pair, we need to combine the trading pair universe
          with wanted indicators to get the final set of indicators we need to calcuate

        - Note that some indicator are calculated over the whole universe or an
          external source and may not have a trading pair defined

        :return:
            IndicatorKey iterator.

            (Trading pair, indicator) iterator
        """
        for name, indicator in self.indicators.items():
            if indicator.is_per_pair():
                for pair in strategy_universe.iterate_pairs():
                    yield IndicatorKey(pair, indicator)
            else:
                yield IndicatorKey(None, indicator)

    @staticmethod
    def from_indicator_keys(indicator_keys: set["IndicatorKey"]) -> "IndicatorSet":
        """Reconstruct the original indicator set from keys.

        - Used when grid search passes data around processes
        """
        indicator_set = IndicatorSet()
        indicator_set.indicators = {key.definition.name: key.definition for key in indicator_keys}
        return indicator_set


class CreateIndicatorsProtocolV1(Protocol):
    """Call signature for create_indicators function.

    Deprecated. See :py:class:`CreateIndicatorsProtocolV2`.
    """

    def __call__(
        self,
        parameters: StrategyParameters,
        indicators: IndicatorSet,
        strategy_universe: TradingStrategyUniverse,
        execution_context: ExecutionContext,
    ):
        """Build technical indicators for the strategy.

        :param parameters:
            Passed from the backtest / live strategy parametrs.

            If doing a grid search, each paramter is simplified.

        :param indicators:
            Indicator builder helper class.

            Call :py:meth:`IndicatorBuilder.create` to add new indicators to the strategy.

        :param strategy_universe:
            The loaded strategy universe.

            Use to resolve symbolic pair information if needed

        :param execution_context:
            Information about if this is a live or backtest run.

        :return:
            This function does not return anything.

            Instead `indicators.add` is used to attach new indicators to the strategy.

        """


class CreateIndicatorsProtocolV2(Protocol):
    """Call signature for create_indicators function.

    This Protocol class defines `create_indicators()` function call signature.
    Strategy modules and backtests can provide on `create_indicators` function
    to define what indicators a strategy needs. These indicators are precalculated and cached for fast performance.

    - There are multiple indicator types, depending on if they are calculated on pair close price,
      pair OHLCV data or the whole strategy universe. See :py:class:`IndicatorSource`.

    - Uses :py:class`IndicatorSet` class to construct the indicators the strategy can use.

    - To read indicator values in `decide_trades()` function,
      see :py:class:`~tradeexecutor.strategy.strategy_input.StrategyInputIndicators`.

    - For most :py:mod:`pandas_ta` functions. like `pandas_ta.ma`, `pandas_ta.rsi`, `pandas_ta.mfi`, you can pass them directly to
      `indicators.add()` - as those functions have standard argument names like `close`, `high`, `low` that
      are data series provided.

    - If you wish to use data from earlier indicators calculations in later indicator calculations, see :py:class:`IndicatorDependencyResolver`
      for how to do it

    .. note ::

        For new examples of defining indicators, please see :py:mod:`tradeexecutor.strategy.pandas_trader.indicator_decorator`.
        The examples below work too, but with Python decorator-based syntax you can make the code more reaadable
        and shorter.

    Example for creating an Exponential Moving Average (EMA) indicator based on the `close` price.
    This example is for a grid search. Unless specified, indicators are assumed to be
    :py:attr:`IndicatorSource.close_price` type and they only use trading pair close price as input.

    .. code-block:: python

        class Parameters:
            stop_loss_pct = [0.9, 0.95]
            cycle_duration = CycleDuration.cycle_1d
            initial_cash = 10_000

            # Indicator values that are searched in the grid search
            slow_ema_candle_count = 7
            fast_ema_candle_count = [1, 2]


        def create_indicators(
            timestamp: datetime.datetime | None,
            parameters: StrategyParameters,
            strategy_universe: TradingStrategyUniverse,
            execution_context: ExecutionContext
        ):
            indicators = IndicatorSet()
            indicators.add("slow_ema", pandas_ta.ema, {"length": parameters.slow_ema_candle_count})
            indicators.add("fast_ema", pandas_ta.ema, {"length": parameters.fast_ema_candle_count})
            return indicators

    Some indicators may use multiple OHLCV datapoints. In this case, you need to tell the indicator to be :py:attr:`IndicatorSource.ohlcv` type.
    Here is an example for Money Flow Index (MFI) indicator:

    .. code-block:: python

        import pandas_ta

        from tradeexecutor.strategy.parameters import StrategyParameters
        from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource

        class Parameters:
            my_mfi_length = 20

        def create_indicators(
            timestamp: datetime.datetime | None,
            parameters: StrategyParameters,
            strategy_universe: TradingStrategyUniverse,
            execution_context: ExecutionContext
        ):
            indicators = IndicatorSet()
            indicators.add(
                "mfi",
                pandas_ta.mfi,
                parameters={"length": parameters.my_mfi_length},
                source=IndicatorSource.ohlcv,
            )

    Indicators can be custom, and do not need to be calculated per trading pair.
    Here is an example of creating indicators "ETH/BTC price" and "ETC/BTC price RSI with length of 20 bars":

    .. code-block:: python

        def calculate_eth_btc(strategy_universe: TradingStrategyUniverse):
            weth_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WETH", "USDC"))
            wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WBTC", "USDC"))
            btc_price = strategy_universe.data_universe.candles.get_candles_by_pair(wbtc_usdc.internal_id)
            eth_price = strategy_universe.data_universe.candles.get_candles_by_pair(weth_usdc.internal_id)
            series = eth_price["close"] / btc_price["close"]  # Divide two series
            return series

        def calculate_eth_btc_rsi(strategy_universe: TradingStrategyUniverse, length: int):
            weth_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WETH", "USDC"))
            wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WBTC", "USDC"))
            btc_price = strategy_universe.data_universe.candles.get_candles_by_pair(wbtc_usdc.internal_id)
            eth_price = strategy_universe.data_universe.candles.get_candles_by_pair(weth_usdc.internal_id)
            eth_btc = eth_price["close"] / btc_price["close"]
            return pandas_ta.rsi(eth_btc, length=length)

        def create_indicators(parameters: StrategyParameters, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext) -> IndicatorSet:
            indicators = IndicatorSet()
            indicators.add("eth_btc", calculate_eth_btc, source=IndicatorSource.strategy_universe)
            indicators.add("eth_btc_rsi", calculate_eth_btc_rsi, parameters={"length": parameters.eth_btc_rsi_length}, source=IndicatorSource.strategy_universe)
            return indicators

    This protocol class is second (v2) iteration of the function signature.
    """

    def __call__(
        self,
        timestamp: datetime.datetime | None,
        parameters: StrategyParameters,
        strategy_universe: TradingStrategyUniverse,
        execution_context: ExecutionContext,
    ) -> IndicatorSet:
        """Build technical indicators for the strategy.

        :param timestamp:
            The current live execution timestamp.

            Set ``None`` for backtesting, as `create_indicators()` is called only once during the backtest setup.

        :param parameters:
            Passed from the backtest / live strategy parametrs.

            If doing a grid search, each paramter is simplified.

        :param strategy_universe:
            The loaded strategy universe.

            Use to resolve symbolic pair information if needed

        :param execution_context:
            Information about if this is a live or backtest run.

        :return:
            Indicators the strategy is going to need.
        """


#: Use this in function singatures
CreateIndicatorsProtocol: TypeAlias = CreateIndicatorsProtocolV1 | CreateIndicatorsProtocolV2


def call_create_indicators(
    create_indicators_func: Callable,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
    timestamp: datetime.datetime = None,
) -> IndicatorSet:
    """Backwards compatible wrapper for create_indicators().

    - Check `create_indicators_func` version

    - Handle legacy / backwards compat
    """
    assert callable(create_indicators_func)
    args = inspect.getfullargspec(create_indicators_func)
    if "indicators" in args.args:
        # v1 backwards
        indicators = IndicatorSet()
        create_indicators_func(parameters, indicators, strategy_universe, execution_context)
        return indicators

    # v2
    return create_indicators_func(timestamp, parameters, strategy_universe, execution_context)


@dataclass
class IndicatorResult:
    """One result of an indicator calculation we can store on a disk.

    - Allows storing and reading output of a single precalculated indicator

    - Parameters is a single combination of parameters
    """

    #: The universe for which we calculated the result
    #:
    #:
    universe_key: UniverseCacheKey

    #: The pair for which this result was calculated
    #:
    #: Set to ``None`` for indicators without a trading pair, using
    #: :py:attr:`IndicatorSource.strategy_universe`
    #:
    indicator_key: IndicatorKey

    #: Indicator output is one time series, but in some cases can be multiple as well.
    #:
    #: For example BB indicator calculates multiple series from one close price value.
    #:
    #:
    data: pd.DataFrame | pd.Series

    #: Was this indicator result cached or calculated on this run.
    #:
    #: Always cached in a grid search, as indicators are precalculated.
    #:
    cached: bool

    @property
    def pair(self) -> TradingPairIdentifier:
        return self.indicator_key.pair

    @property
    def definition(self) -> IndicatorDefinition:
        return self.indicator_key.definition

    def get_generation_error_message(self) -> str | None:
        """Get error message if we somehow failed to generate this data."""
        return self.data.attrs.get("error")


IndicatorResultMap: TypeAlias = dict[IndicatorKey, IndicatorResult]


class IndicatorStorage(ABC):
    """Base class for cached indicators and live trading indicators."""

    @abstractmethod
    def is_available(self, key: IndicatorKey) -> bool:
        pass

    @abstractmethod
    def load(self, key: IndicatorKey) -> IndicatorResult:
        pass

    @abstractmethod
    def save(self, key: IndicatorKey, df: pd.DataFrame | pd.Series) -> IndicatorResult:
        pass

    @abstractmethod
    def get_disk_cache_path(self) -> Path | None:
        pass

    @abstractmethod
    def get_universe_cache_path(self) -> Path | None:
        pass


class DiskIndicatorStorage(IndicatorStorage):
    """Store calculated indicator results on disk.

    Used in

    - Backtesting

    - Grid seacrh

    Indicators are calculated once and the calculation results can be recycled across multiple backtest runs.

    How to initialise in the notebook:

    .. code-block:: python

        indicator_storage = DiskIndicatorStorage.create_default(strategy_universe)

    TODO: Cannot handle multichain universes at the moment, as serialises trading pairs by their ticker.
    """

    def __init__(self, path: Path, universe_key: UniverseCacheKey):
        assert isinstance(path, Path)
        assert type(universe_key) == str
        self.path = path
        self.universe_key = universe_key

    def __repr__(self):
        return f"<IndicatorStorage at {self.path}>"

    def get_universe_cache_path(self) -> Path:
        return self.path / Path(self.universe_key)

    def get_disk_cache_path(self) -> Path:
        return self.path

    def get_indicator_path(self, key: IndicatorKey) -> Path:
        """Get the Parquet file where the indicator data is stored.

        :return:
            Example `/tmp/.../test_indicators_single_backtes0/ethereum,1d,WETH-USDC-WBTC-USDC,2021-06-01-2021-12-31/sma(length=21).parquet`
        """
        assert isinstance(key, IndicatorKey)
        return self.get_universe_cache_path() / Path(f"{key.get_cache_key()}.parquet")

    def is_available(self, key: IndicatorKey) -> bool:
        return self.get_indicator_path(key).exists()

    def load(self, key: IndicatorKey) -> IndicatorResult:
        """Load cached indicator data from the disk."""
        assert self.is_available(key), f"Data does not exist: {key}, path is: {self.get_indicator_path(key)}"
        path = self.get_indicator_path(key)
        df = pd.read_parquet(path)

        if len(df.columns) == 1 and df.columns[0] == "value":
            # Convert back to series.
            # See save() below.
            df = df[df.columns[0]]

        return IndicatorResult(
            self.universe_key,
            key,
            df,
            cached=True,
        )

    def save(self, key: IndicatorKey, df: pd.DataFrame | pd.Series) -> IndicatorResult:
        """Atomic replacement of the existing data.

        - Avoid leaving partially written files
        """
        assert isinstance(key, IndicatorKey)

        if isinstance(df, pd.Series):
            # For saving, create a DataFrame with a single column "value"
            save_df = pd.DataFrame({"value": df})
        else:
            save_df = df

        assert isinstance(save_df, pd.DataFrame), f"Expected DataFrame, got: {type(df)}"
        path = self.get_indicator_path(key)
        dirname, basename = os.path.split(path)

        os.makedirs(dirname, exist_ok=True)

        temp = tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dirname)
        save_df.to_parquet(temp)
        temp.close()
        # https://stackoverflow.com/a/3716361/315168
        shutil.move(temp.name, path)
        logger.info(
            "Saved: %s, rows: %d, parameters: %s",
            path,
            len(df),
            list(key.definition.parameters.items()),
        )

        assert os.path.exists(path), f"Save failed: {path}"

        return IndicatorResult(
            universe_key=self.universe_key,
            indicator_key=key,
            data=df,
            cached=False,
        )

    @staticmethod
    def create_default(
        universe: TradingStrategyUniverse,
        default_path=DEFAULT_INDICATOR_STORAGE_PATH,
    ) -> "DiskIndicatorStorage":
        """Get the indicator storage with the default cache path."""
        return DiskIndicatorStorage(default_path, universe.get_cache_key())


class MemoryIndicatorStorage(IndicatorStorage):
    """Store calculated indicator results on disk.

    Used in

    - Live trading

    - Indicators are calculated just before `decide_trades()` is called

    - Indicators are recalculated on every decision cycle
    """

    def __init__(self, universe_key: UniverseCacheKey):
        self.universe_key = universe_key
        self.results: dict[IndicatorKey, IndicatorResult] = {}

    def is_available(self, key: IndicatorKey) -> bool:
        return key in self.results

    def load(self, key: IndicatorKey) -> IndicatorResult:
        return self.results[key]

    def save(self, key: IndicatorKey, df: pd.DataFrame | pd.Series) -> IndicatorResult:
        result = IndicatorResult(
            universe_key=self.universe_key,
            indicator_key=key,
            data=df,
            cached=False,
        )

        self.results[key] = result
        return result

    def get_disk_cache_path(self) -> Path | None:
        return None

    def get_universe_cache_path(self) -> Path | None:
        return None

    def get_indicator_path(self, key: IndicatorKey) -> Path:
        # TODO: Return fake path now - abstract memory indicator storage paths away
        return Path("/dev/memory" + key.get_cache_key())


def _serialise_parameters_for_cache_key(parameters: dict) -> str:
    for k, v in parameters.items():
        assert type(k) == str
        assert type(v) not in (list, tuple)  # Don't leak test ranges here - must be a single value
    return "".join([f"{k}={v}" for k, v in parameters.items()])


def _load_indicator_result(storage: DiskIndicatorStorage, key: IndicatorKey) -> IndicatorResult:
    # Cache key is different from indicator key, as we need to truncate
    # paramaeters to make it fit into a filename
    path = storage.get_indicator_path(key)
    logger.info("Loading indicator, indicator key: %s, cache path: %s", key, path)
    assert storage.is_available(key), f"Tried to load indicator that is not in the cache: {key}"
    return storage.load(key)


def _calculate_and_save_indicator_result(
    storage: DiskIndicatorStorage | MemoryIndicatorStorage,
    key: IndicatorKey,
    all_indicators: set[IndicatorKey],
    execution_context: ExecutionContext,
    strategy_cycle_timestamp: datetime.datetime | None,
) -> IndicatorResult | None:
    """Calculate an indicator result.

    - Mark missing data as empty Series

    """
    assert isinstance(storage, IndicatorStorage), f"Got {type(storage)}"
    assert isinstance(key, IndicatorKey), f"Got {type(key)}"
    assert type(all_indicators) == set, f"Got {type(all_indicators)}"
    assert isinstance(execution_context, ExecutionContext), f"Got {type(execution_context)}"
    assert isinstance(strategy_cycle_timestamp, (datetime.datetime, NoneType)), f"Got {type(strategy_cycle_timestamp)}"

    global _universe

    # Picked result
    strategy_universe = _universe

    assert strategy_universe is not None, "Process global _universe not set"

    current_dependency_order = key.definition.dependency_order

    resolver = IndicatorDependencyResolver(
        strategy_universe,
        all_indicators,
        storage,
        current_dependency_order,
    )

    indicator = key.definition

    # Use pd.Timestamp() because we deal with pd.Series()
    if strategy_cycle_timestamp:
        timestamp = pd.Timestamp(strategy_cycle_timestamp)
    else:
        timestamp = None

    if indicator.is_per_pair():
        assert key.pair.internal_id, f"Per-pair indicator lacks pair internal_id: {key.pair}"
        try:
            match indicator.source:
                case IndicatorSource.open_price:
                    column = "open"
                    input = strategy_universe.data_universe.candles.get_samples_by_pair(key.pair.internal_id)[column]
                    data = indicator.calculate_by_pair(
                        input,
                        key.pair,
                        resolver=resolver,
                        timestamp=timestamp,
                        execution_context=execution_context
                    )
                case IndicatorSource.close_price:
                    column = "close"
                    input = strategy_universe.data_universe.candles.get_samples_by_pair(key.pair.internal_id)[column]
                    data = indicator.calculate_by_pair(
                        input,
                        key.pair,
                        resolver=resolver,
                        timestamp=timestamp,
                        execution_context=execution_context
                    )
                case IndicatorSource.ohlcv:
                    input = strategy_universe.data_universe.candles.get_samples_by_pair(key.pair.internal_id)
                    data = indicator.calculate_by_pair_ohlcv(
                        input,
                        key.pair,
                        resolver=resolver,
                        timestamp=timestamp,
                        execution_context=execution_context
                    )
                case IndicatorSource.tvl:
                    assert strategy_universe.data_universe.liquidity is not None, "TVL/liquidity data not loaded, trying to create TVL indicator"
                    input = strategy_universe.data_universe.liquidity.get_samples_by_pair(key.pair.internal_id)
                    data = indicator.calculate_by_pair_ohlcv(
                        input,
                        key.pair,
                        resolver=resolver,
                        timestamp=timestamp,
                        execution_context=execution_context
                    )
                case IndicatorSource.external_per_pair:
                    data = indicator.calculate_by_pair_external(
                        key.pair,
                        resolver=resolver,
                        timestamp=timestamp,
                        execution_context=execution_context
                    )
                case IndicatorSource.dependencies_only_per_pair:
                    data = indicator.calculate_dependencies_only_per_pair(
                        key.pair,
                        resolver=resolver,
                        timestamp=timestamp,
                        execution_context=execution_context,                    )
                case _:
                    raise NotImplementedError(f"Unsupported input source {key.pair} {key.definition} {indicator.source}")

            if data is None:
                error_message = f"Indicator {key.definition.name} generated empty data for pair {key.pair}. Input data length is {len(input)} rows."
                logger.warning(error_message)
                data = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
                data.attrs["error"] = error_message

        except PairCandlesMissing as e:
            logger.info("Indicator data %s not generated for pair %s because of lack of OHLCV data. Exception %s", key.definition.name, key.pair, e)
            data = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
            data.attrs["error"] = str(e)

    else:
        # Calculate indicator over the whole universe
        match indicator.source:
            case IndicatorSource.strategy_universe:
                data = indicator.calculate_universe(
                    strategy_universe,
                    resolver=resolver,
                    timestamp=timestamp,
                    execution_context=execution_context,
                )
            case IndicatorSource.dependencies_only_universe:
                data = indicator.calculate_dependencies_only_per_universe(
                    resolver=resolver,
                    timestamp=timestamp,
                    execution_context=execution_context
                )
            case _:
                raise NotImplementedError(f"Unsupported input source {key.pair} {key.definition} {indicator.source}")

    assert data is not None, f"Indicator function {indicator.name} ({indicator.func}) did not return any result, received Python None instead"

    # Some debug logging
    if data is None:
        diagnostics_text = "<none>"
    else:
        if len(data) >= 1:
            last_value = data.iloc[-1]
            last_value_at = data.index[-1]
            diagnostics_text = f"{len(data)} rows, last value {last_value}, at {last_value_at}"
        else:
            diagnostics_text = f"{data}"

    logger.debug(
        "Indicator calculated: %s, pair: %s, data: %s",
        key.definition.name,
        key.pair.get_ticker() if key.pair else "<universe>",
        diagnostics_text,
    )
    result = storage.save(key, data)
    return result


def load_indicators(
    strategy_universe: TradingStrategyUniverse,
    storage: DiskIndicatorStorage,
    indicator_set: IndicatorSet,
    all_combinations: set[IndicatorKey],
    max_readers=8,
    show_progress=True,
) -> IndicatorResultMap:
    """Load cached indicators.

    - Use a thread pool to speed up IO

    :param all_combinations:
        Load all cached indicators of this set if they are available in the storage.

    :param storage:
        The cache backend we use for the storage

    :param max_readers:
        Number of reader threads we allocate for the task
    """

    task_args = []
    for key in all_combinations:
        if storage.is_available(key):
            task_args.append((storage, key))
        else:
            logger.info(
                "Indicator not available: %s: %s",
                storage.get_indicator_path(key),
                list(key.definition.parameters.items()),
            )

    logger.info(
        "Loading cached indicators, we have %d indicator combinations out of %d available in the cache %s",
        len(task_args),
        len(all_combinations),
        storage.get_universe_cache_path()
    )

    if len(task_args) == 0:
        return {}

    results = {}
    label = indicator_set.get_label()
    key: IndicatorKey

    if show_progress:
        progress_bar = tqdm(total=len(task_args), desc=f"Reading cached indicators {label} for {strategy_universe.get_pair_count()} pairs, using {max_readers} threads")
    else:
        progress_bar = None

    try:
        if max_readers > 1:
            logger.info("Multi-thread reading")

            executor = futureproof.ThreadPoolExecutor(max_workers=max_readers)
            tm = futureproof.TaskManager(executor, error_policy=futureproof.ErrorPolicyEnum.RAISE)

            # Run the checks parallel using the thread pool
            tm.map(_load_indicator_result, task_args)

            # Extract results from the parallel task queue
            for task in tm.as_completed():
                result = task.result
                key = result.indicator_key
                assert key not in results
                results[key] = result
                if progress_bar:
                    progress_bar.update()
        else:
            logger.info("Single-thread reading")
            for result in itertools.starmap(_load_indicator_result, task_args):
                key = result.indicator_key
                assert key not in results
                results[key] = result

        return results
    finally:
        if progress_bar:
            progress_bar.close()



@dataclass
class IndicatorDependencyResolver:
    """A helper class allowing access to the indicators we depend on.

    - Allows you to define indicators that use data from other indicators.

    - Indicators are calculated in the order defined by :py:attr:`IndicatorDefinition.dependency_order`,
      higher dependency order can read data from lower one. You usually specify this with
      `indicator.add(order)` argument.

    - If you add a parameter `dependency_resolver` to your indicator functions,
      the instance of this class is passed and you can use `dependency_resolver`
      to laod and read data from the past indicator calculations.

    - The indicator function still can take its usual values like `close` (for close price series),
      `strategy_universe`, etc. based on :py:class:`IndicatorSource`, even if these values
       are not used in the calculations

    An example where a single-pair indicator uses data from two other indicators:

    .. code-block:: python

        def ma_crossover(
            close: pd.Series,
            pair: TradingPairIdentifier,
            dependency_resolver: IndicatorDependencyResolver,
        ) -> pd.Series:
            # Do cross-over calculation based on other two earlier moving average indicators.
            # Return pd.Series with True/False valeus and DatetimeIndex
            slow_sma: pd.Series = dependency_resolver.get_indicator_data("slow_sma")
            fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma")
            return fast_sma > slow_sma

        indicators = IndicatorSet()
        # Slow moving average
        indicators.add(
            "fast_sma",
            pandas_ta.sma,
            parameters={"length": 7},
            order=1,
        )
        # Fast moving average
        indicators.add(
            "slow_sma",
            pandas_ta.sma,
            parameters={"length": 21},
            order=1,
        )
        # An indicator that depends on both fast MA and slow MA above
        indicators.add(
            "ma_crossover",
            ma_crossover,
            source=IndicatorSource.ohlcv,
            order=2,  # 2nd order indicators can depend on the data of 1st order indicators
        )

    When you use indicator dependency resolution with the grid search, you need to specify indicator parameters you want read,
    as for each named indicator there might be multiple copies with different grid search combinations:

    .. code-block:: python

        class Parameters:
            cycle_duration = CycleDuration.cycle_1d
            initial_cash = 10_000

            # Indicator values that are searched in the grid search
            slow_ema_candle_count = [21, 30]
            fast_ema_candle_count = [7, 12]
            combined_indicator_modes = [1, 2]

        def combined_indicator(close: pd.Series, mode: int, pair: TradingPairIdentifier, dependency_resolver: IndicatorDependencyResolver):
            # An indicator that peeks the earlier grid search indicator calculations
            match mode:
                case 1:
                    # When we look up data in grid search we need to give the parameter of which data we want,
                    # and the trading pair if needed
                    fast_ema = dependency_resolver.get_indicator_data("slow_ema", pair=pair, parameters={"length": 21})
                    slow_ema = dependency_resolver.get_indicator_data("fast_ema", pair=pair, parameters={"length": 7})
                case 2:
                    # Look up one set of parameters
                    fast_ema = dependency_resolver.get_indicator_data("slow_ema", pair=pair, parameters={"length": 30})
                    slow_ema = dependency_resolver.get_indicator_data("fast_ema", pair=pair, parameters={"length": 12})
                case _:
                    raise NotImplementedError()

            return fast_ema * slow_ema * close # Calculate something based on two indicators and price

        def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
            indicators.add("slow_ema", pandas_ta.ema, {"length": parameters.slow_ema_candle_count})
            indicators.add("fast_ema", pandas_ta.ema, {"length": parameters.fast_ema_candle_count})
            indicators.add(
                "combined_indicator",
                combined_indicator,
                {"mode": parameters.combined_indicator_modes},
                source=IndicatorSource.ohlcv,
                order=2,
            )

        combinations = prepare_grid_combinations(
            Parameters,
            tmp_path,
            strategy_universe=strategy_universe,
            create_indicators=create_indicators,
            execution_context=ExecutionContext(mode=ExecutionMode.unit_testing, grid_search=True),
        )
    """

    #: Trading universe
    #:
    #: - Perform additional pair lookups if needed
    #:
    strategy_universe: TradingStrategyUniverse

    #: Available indicators as defined in create_indicators()
    #:
    all_indicators: set[IndicatorKey]

    #: Raw cached indicator results or ones calculated in the memory
    #:
    indicator_storage: IndicatorStorage

    #: The current resolved dependency order level
    current_dependency_order: int = 0

    def match_indicator(
        self,
        name: str | Callable,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
        parameters: dict | None = None,
    ) -> IndicatorKey:
        """Find an indicator key for an indicator.

        - Get the `IndicatorKey` instance that is the look up for loading the indicator data

        - Check by name, pair and parameter

        - Make sure that the indicator defined is on a lower level than the current dependency order level

        :param name:
            The name of the indicator or its defining function.

        :param pair:
            Pair for which we look up the indicator value

        :param parameters:
            Parameters for which this indicator was calculated with.

            We can have several instances of the same indicator in grid search.

        :return:
            IndicatorKey which we can use to look up the indicator data.

        """

        if callable(name):
            # Convert function to its name
            name = name.__name__

        if len(self.all_indicators) > 8:
            all_text = f"We have total {len(self.all_indicators)} indicator keys across all pairs and grid combinations."
        else:
            all_text = f"We have indicators: {self.all_indicators}."

        filtered_by_name = [i for i in self.all_indicators if i.definition.name == name]
        if len(filtered_by_name) == 0:
            raise IndicatorDependencyResolutionError(f"No indicator named {name}. {all_text}")

        if pair is not None:
            filtered_by_pair = [i for i in filtered_by_name if i.pair == pair]
            if len(filtered_by_pair) == 0:
                raise IndicatorDependencyResolutionError(f"No indicator named {name}, for pair {pair}. {all_text}.\nWe have {len(filtered_by_name)} match by {name}: {filtered_by_name if len(filtered_by_name) < 4 else '-'}\nParameters are: {parameters}")
        else:
            filtered_by_pair = filtered_by_name

        if parameters is not None:
            filtered_by_parameters = [i for i in filtered_by_pair if i.definition.parameters == parameters]

            filtered_by_pair_parameters = [i.definition.parameters for i in filtered_by_pair]

            if len(filtered_by_parameters) == 0:
                raise IndicatorDependencyResolutionError(f"No indicator named {name},\nPair {pair},\nParameters {parameters}.\nOther parameter combinations:{filtered_by_pair} with parameters: {filtered_by_pair_parameters}\n{all_text}")
        else:
            filtered_by_parameters = filtered_by_pair

        if len(filtered_by_parameters) != 1:
            raise IndicatorDependencyResolutionError(f"Multiple indicator results for named {name},\n for pair {pair},\n parameters {parameters}.\n{all_text}\nfiltered_by_parameters is: {filtered_by_parameters}.\nDid you forget to specify pair?")

        result = filtered_by_parameters[0]

        if self.current_dependency_order <= result.definition.dependency_order:
            raise IndicatorOrderError(f"The dependency order for {name} is {result.definition.dependency_order}, but we ask data at the current dependency order level {self.current_dependency_order}")

        return result

    def get_indicator_data_pairs_combined(
        self,
        name: str | Callable,
        parameters: dict | None = None,
    ) -> pd.Series:
        """Get a DataFrame that contains indicator data for all pairs combined.

        - Allows to access the indicator data for all pairs as a combined dataframe.

        Example:

        .. code-block:: python

            def regime(
                close: pd.Series,
                pair: TradingPairIdentifier,
                length: int,
                dependency_resolver: IndicatorDependencyResolver,
            ) -> pd.Series:
                fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma", pair=pair, parameters={"length": length})
                return close > fast_sma

            def multipair(universe: TradingStrategyUniverse, dependency_resolver: IndicatorDependencyResolver) -> pd.DataFrame:
                # Test multipair data resolution
                series = dependency_resolver.get_indicator_data_pairs_combined("regime")
                assert isinstance(series.index, pd.MultiIndex)
                assert isinstance(series, pd.Series)
                return series
                # Change from pd.Series to pd.DataFrame with column "value"
                # df = series.to_frame(name='value')
                # assert df.columns == ["value"]
                # return df

            def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
                indicators.add("regime", regime, {"length": parameters.fast_sma}, order=2)
                indicators.add("multipair", multipair, {}, IndicatorSource.strategy_universe, order=3)

        Output:

        .. code-block:: text

            pair_id  timestamp
            1        2021-06-01    False
                     2021-06-02    False
                     2021-06-03    False
                     2021-06-04    False
                     2021-06-05    False
                                   ...
            2        2021-12-27     True
                     2021-12-28     True
                     2021-12-29    False
                     2021-12-30    False
                     2021-12-31    False

        :param name:
            An indicator that was previously calculated by its `order`.

        :return:
            DataFrame with MultiIndex (pair_id, timestamp)
        """

        universal = False
        try:
            indicator = self.match_indicator(name)
            universal = indicator.definition.source == IndicatorSource.strategy_universe
        except Exception as e:
            universal = False

        if universal:
            raise IndicatorCalculationFailed(f"get_indicator_data_pairs_combined() cannot be called if the indicator {name} is not pair based")

        series_map = {pair.internal_id: self.get_indicator_data(name, pair=pair, parameters=parameters) for pair in self.strategy_universe.iterate_pairs()}
        series_list = [s for s in series_map.values() if len(s) > 0]
        pair_ids = list(series_map.keys())

        with warnings.catch_warnings():
            # FutureWarning: The behavior of pd.concat with len(keys) != len(objs) is deprecated. In a future version this will raise instead of truncating to the smaller of the two sequences
            warnings.simplefilter(action='ignore', category=FutureWarning)
            combined = pd.concat(series_list, keys=pair_ids, names=['pair_id', 'timestamp'])
        return combined

    def get_indicator_data(
        self,
        name: str | Callable,
        column: str | None = None,
        pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
        parameters: dict | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Read data from another indicator.

        -The indicator must be prepared in `create_indicators()` earlier,
         and then calculated

        :param name:
            Indicator name, as given to `IndicatorSet.add()`.

        :param parameters:
            If the dependent indicator has multiple versions
            with different parameters, we need to get the specify parameters.

        :param column:
            Column name for multi-column indicators.

            Use when the indicator is :py:class:`pandas.DataFrame` instead
            of :py:class:`pandas.Series`.

            Set to string `all` to get the whole DataFrame.

        :param pair:
            Needed when universe contains multiple trading pairs.

            Can be omitted from non-pair indicators.

        :return:
            The indicator data as is was saved on the disk
        """
        key = self.match_indicator(
            name,
            pair,
            parameters
        )

        indicator_result = self.indicator_storage.load(key)

        if indicator_result is None:
            all_indicators = self.all_indicators
            raise AssertionError(
                f"Indicator results did not contain key {key} for indicator {name}.\n"
                f"Available indicators: {all_indicators}\n"
            )

        data = indicator_result.data
        assert data is not None, f"Indicator pre-calculated values missing for {name} - lookup key {key}"

        if isinstance(data, pd.DataFrame):

            if column == "all":
                return data

            assert column is not None, f"Indicator {name} has multiple available columns to choose from: {data.columns}. Indicator data is {type(data)}"
            assert column in data.columns, f"Indicator {name} subcolumn {column} not in the available columns: {data.columns}"
            series = data[column]
        elif isinstance(data, pd.Series):
            if len(data) == 0:
                # TODO: Special case. Need to figure out how to handle this neatly.
                return data
            assert column != "all", f"Multicolumn indicator requested. Indicator {name} is pd.Series type, not DataFrame type, cannot get multiple columns. Indicator data is {type(data)}"
            series = data
        else:
            raise NotImplementedError(f"Unknown indicator data type {type(data)}")

        return series


#: Indicator multiprocess unit as function parameters
CalculateTaskArguments = tuple[IndicatorStorage, IndicatorKey, set[IndicatorKey]]


def group_indicators(task_args: list[CalculateTaskArguments]) -> dict[int, list[CalculateTaskArguments]]:
    """Split indicator calculations to the groups based on their dependency resolution order.

    :return:
        ordered dict, lowest first
    """

    grouped = {}
    sorted_args = sorted(task_args, key=lambda ta: ta[1].definition.dependency_order)
    for task_args in sorted_args:
        order = task_args[1].definition.dependency_order
        if order not in grouped:
            # Create keys in order
            grouped[order] = []
        grouped[order].append(task_args)
    return grouped


def calculate_indicators(
    strategy_universe: TradingStrategyUniverse,
    storage: IndicatorStorage,
    indicators: IndicatorSet | None,
    execution_context: ExecutionContext,
    remaining: set[IndicatorKey],
    max_workers=8,
    label: str | None = None,
    all_combinations: set[IndicatorKey] | None = None,
    verbose=True,
    strategy_cycle_timestamp: datetime.datetime = None,
) -> IndicatorResultMap:
    """Calculate indicators for which we do not have cached data yet.

    - Use a thread pool to speed up IO

    :param indicators:
        Indicator set we calculate for.

        Can be ``None`` for a grid search, as each individual combination may has its own set.

    :param remaining:
        Remaining indicator combinations for which we do not have a cached rresult

    :param all_combinations:
        All available indicator combinations.

        Only needed if we are doing indicator dependency resolution.

    :param verbose:
        Stdout user printing with helpful messages.
    """

    assert isinstance(strategy_universe, TradingStrategyUniverse)
    assert isinstance(storage, IndicatorStorage)
    assert isinstance(execution_context, ExecutionContext), f"Expected ExecutionContext, got {type(execution_context)}"

    results: IndicatorResultMap

    if label is None:
        if indicators is not None:
            label = indicators.get_label()
        else:
            label = "Indicator calculation"

    logger.info("Calculating indicators: %s", label)

    if len(remaining) == 0:
        logger.info("Nothing to calculate")
        return {}

    task_args: list[CalculateTaskArguments] = []
    for key in remaining:
        task_args.append((storage, key, all_combinations, execution_context, strategy_cycle_timestamp))

    results = {}

    if max_workers > 1:

        # Do a parallel calculation for the maximum speed
        #
        # Set up a futureproof task manager
        #
        # For futureproof usage see
        # https://github.com/yeraydiazdiaz/futureproof

        #
        # Run individual searchers in child processes
        #

        # Copy universe data to child processes only once when the child process is created
        #
        pickled_universe = pickle.dumps(strategy_universe)
        logger.info("Doing a multiprocess indicator calculation, pickled universe is %d bytes", len(pickled_universe))

        # Set up a process pool executing structure
        executor = futureproof.ProcessPoolExecutor(max_workers=max_workers, initializer=_process_init, initargs=(pickled_universe,))
        tm = futureproof.TaskManager(executor, error_policy=futureproof.ErrorPolicyEnum.RAISE)

        # Set up a signal handler to stop child processes on quit
        setup_indicator_multiprocessing(executor)

        # Dependency order resolution.
        # Group is a list of task args where the order is the same
        groups = group_indicators(task_args)

        for order, group_task_args in groups.items():

            # Run the tasks
            tm.map(_calculate_and_save_indicator_result, group_task_args)

            if order != 1:
                desc = f"Calculating indicators {label} using {max_workers} processes, dependency group #{order}"
            else:
                desc = f"Calculating indicators {label} using {max_workers} processes"

            # Track the child process completion using tqdm progress bar
            if verbose:
                progress_bar = tqdm(total=len(task_args), desc=desc)
            else:
                progress_bar = None

            # Extract results from the parallel task queue
            for task in tm.as_completed():
                result = task.result
                results[result.indicator_key] = result
                if progress_bar is not None:
                    progress_bar.update()

            if progress_bar is not None:
                progress_bar.close()

    else:
        # Do single thread - good for debuggers like pdb/ipdb
        #

        global _universe
        _universe = strategy_universe

        logger.info("Doing a single thread indicator calculation")

        # Dependency order resolution.
        # Group is a list of task args where the order is the same
        groups = group_indicators(task_args)

        for order, group_task_args in groups.items():
            iter = itertools.starmap(_calculate_and_save_indicator_result, group_task_args)

            # Force workers to finish
            result: IndicatorResult
            for result in iter:
                results[result.indicator_key] = result

    logger.info("Total %d indicator results calculated", len(results))

    return results


def prepare_indicators(
    create_indicators: CreateIndicatorsProtocol,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
    timestamp: datetime.datetime = None,
):
    """Call the strategy module indicator builder."""

    indicators = call_create_indicators(
        create_indicators,
        parameters,
        strategy_universe,
        execution_context,
        timestamp=timestamp,
    )

    if indicators.get_count() == 0:
        # TODO: Might have legit use cases?
        logger.warning(f"create_indicators() did not create a single indicator")

    return indicators


def calculate_and_load_indicators(
    strategy_universe: TradingStrategyUniverse,
    storage: IndicatorStorage,
    execution_context: ExecutionContext,
    parameters: StrategyParameters | None = None,
    indicators: IndicatorSet | None = None,
    create_indicators: CreateIndicatorsProtocolV1 | None = None,
    max_workers: int | Callable = get_safe_max_workers_count,
    max_readers: int | Callable = get_safe_max_workers_count,
    verbose=True,
    strategy_cycle_timestamp: datetime.datetime = None,
) -> IndicatorResultMap:
    """Precalculate all indicators.

    - Calculate indicators using multiprocessing

    - Display TQDM progress bars for loading cached indicators and calculating new ones

    - Use cached indicators if available

    :param cache_warmup_only:
        Only fill the disk cache, do not load results in the memory.

    :param verbose:
        Stdout printing with heplful messages to the user
    """

    # Resolve CPU count
    if callable(max_workers):
        max_workers = max_workers()

    if callable(max_readers):
        max_readers = max_readers()

    if isinstance(storage, MemoryIndicatorStorage) and (max_workers > 1 or max_readers > 1):
        logger.warning("MemoryIndicatorStorage does not support multiprocessing, setting max_workers and max_readers to 1")
        max_workers = 1
        max_readers = 1

    assert create_indicators or indicators, f"You must give either create_indicators or indicators argument. Got {create_indicators} and {indicators}"

    if create_indicators:
        assert indicators is None, f"Give either indicators or create_indicators, not both"
        assert parameters is not None, f"parameters argument must be given if you give create_indicators"
        indicators = prepare_indicators(create_indicators, parameters, strategy_universe, execution_context, timestamp=strategy_cycle_timestamp)

    assert isinstance(indicators, IndicatorSet), f"Got class {type(indicators)} when IndicatorSet expected"

    all_combinations = set(indicators.generate_combinations(strategy_universe))

    # Safety assert for rolling parameters
    for combination in all_combinations:
        for key, value in combination.definition.parameters.items():
            assert not isinstance(value, list), f"Got {key}: {value}"

    logger.info("Loading indicators %s for the universe %s, storage is %s", indicators.get_label(), strategy_universe.get_cache_key(), storage.get_disk_cache_path())
    cached = load_indicators(
        strategy_universe,
        storage,
        indicators,
        all_combinations,
        max_readers=max_readers,
        show_progress=verbose,  # Don't display progress bars when running as child process under Visual Studio Code
    )

    for key in cached.keys():
        # Check we keyed this right
        assert key in all_combinations, f"Loaded a cached result {key} is not in part of the all combinations we expected"

    if verbose:
        if len(all_combinations) > 0:
            print(f"Using indicator cache {storage.get_universe_cache_path()}")

    calculation_needed = all_combinations - set(cached.keys())
    calculated = calculate_indicators(
        strategy_universe,
        storage,
        indicators,
        execution_context,
        calculation_needed,
        max_workers=max_workers,
        all_combinations=all_combinations,
        verbose=verbose,
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )

    result = cached | calculated

    for key in result.keys():
        # Check we keyed this right
        assert key in all_combinations

    return result


def calculate_and_load_indicators_inline(
    strategy_universe: TradingStrategyUniverse,
    parameters: StrategyParameters,
    indicator_storage_path=DEFAULT_INDICATOR_STORAGE_PATH,
    execution_context: ExecutionContext = notebook_execution_context,
    verbose=True,
    indicator_set: IndicatorSet | None = None,
    create_indicators: CreateIndicatorsProtocol = None,
    storage: IndicatorStorage | None = None,
    max_workers: int | Callable = get_safe_max_workers_count,
    strategy_cycle_timestamp: datetime.datetime = None,
) -> "tradeexecutor.strategy.pandas_trader.strategy_input.StrategyInputIndicators":
    """Calculate indicators in the notebook itself, before starting the backtest.

    - To be used within Jupyter Notebooks

    - Useful for single iteration backtests

    - Useful for accessing indicator data if you do not need a backtest

    Example:

    .. code-block:: python

        # Some example parameters we use to calculate indicators.
        # E.g. RSI length
        class Parameters:
            rsi_length = 20
            sma_long = 200
            sma_short = 12

        # Create indicators.
        # Map technical indicator functions to their parameters, like length.
        # You can also use hardcoded values, but we recommend passing in parameter dict,
        # as this allows later to reuse the code for optimising/grid searches, etc.
        def create_indicators(
            timestamp,
            parameters,
            strategy_universe,
            execution_context,
        ) -> IndicatorSet:
            indicator_set = IndicatorSet()
            indicator_set.add("rsi", pandas_ta.rsi, {"length": parameters.rsi_length})
            indicator_set.add("sma_long", pandas_ta.sma, {"length": parameters.sma_long})
            indicator_set.add("sma_short", pandas_ta.sma, {"length": parameters.sma_short})
            return indicator_set

        # Calculate indicators - will spawn multiple worker processed,
        # or load cached results from the disk
        indicators = calculate_and_load_indicators_inline(
            strategy_universe=strategy_universe,
            parameters=StrategyParameters.from_class(Parameters),
            create_indicators=create_indicators,
        )

        # From calculated indicators, read one indicator (RSI for BTC)
        wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WBTC", "USDC"))
        rsi = indicators.get_indicator_series("rsi", pair=wbtc_usdc)
        assert len(rsi) == 214  # We have series data for 214 days

    """
    assert not parameters.is_grid_search(), "calculate_and_load_indicators_inline() is designed to work only with single backtests"

    # Hack to be able to run notebook with ipython from the command line
    # https://stackoverflow.com/a/39662359/315168
    ipython = get_ipython().__class__.__name__ == "TerminalInteractiveShell"

    # TODO: Eliminate circulates
    from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators

    if storage is None:
        storage = DiskIndicatorStorage(
            indicator_storage_path,
            strategy_universe.get_cache_key()
        )

    if create_indicators:
        assert indicator_set is None, f"Cannot give both indicator_set and create_indicators"
        indicator_set = prepare_indicators(create_indicators, parameters, strategy_universe, execution_context, timestamp=strategy_cycle_timestamp)

    if ipython:
        # Unable to fork when running on Ipython
        logger.warning("IPython detected - forces calculate_and_load_indicators(max_workers=1)")
        max_workers = 1

    indicator_result_map = calculate_and_load_indicators(
        strategy_universe=strategy_universe,
        storage=storage,
        execution_context=execution_context,
        parameters=parameters,
        verbose=verbose,
        indicators=indicator_set,
        max_workers=max_workers,
        max_readers=max_workers,  # TODO: Add another parameter?
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )

    return StrategyInputIndicators(
        strategy_universe=strategy_universe,
        indicator_results=indicator_result_map,
        available_indicators=indicator_set,
        timestamp=None,
    )


def warm_up_indicator_cache(
    strategy_universe: TradingStrategyUniverse,
    storage: DiskIndicatorStorage,
    execution_context: ExecutionContext,
    indicators: set[IndicatorKey],
    max_workers=8,
    all_combinations: set[IndicatorKey] | None = None,
) -> tuple[set[IndicatorKey], set[IndicatorKey]]:
    """Precalculate all indicators.

    - Used for grid search

    - Calculate indicators using multiprocessing

    - Display TQDM progress bars for loading cached indicators and calculating new ones

    - Use cached indicators if available

    - Not need to call from the notebook manuallyt

    :return:
        Tuple (Cached indicators, calculated indicators)
    """

    cached = set()
    needed = set()
    pair_indicators_needed = set()
    universe_indicators_needed = set()

    for key in indicators:
        if storage.is_available(key):
            cached.add(key)
        else:
            needed.add(key)
            if key.pair is None:
                universe_indicators_needed.add(key)
            else:
                pair_indicators_needed.add(key)

    logger.info(
        "warm_up_indicator_cache(), we have %d cached pair-indicators results and need to calculate %d results, of which %d pair indicators and %d universe indicators",
        len(cached),
        len(needed),
        len(pair_indicators_needed),
        len(universe_indicators_needed),
    )

    calculated = calculate_indicators(
        strategy_universe,
        storage,
        None,
        execution_context,
        needed,
        max_workers=max_workers,
        all_combinations=all_combinations,
        label=f"Calculating {len(needed)} indicators for the grid search"
    )

    logger.info(
        "Calculated %d indicator results",
        len(calculated),
    )

    return cached, needed


#: Process global stored universe for multiprocess workers
_universe: TradingStrategyUniverse | None = None

_process_pool: concurrent.futures.ProcessPoolExecutor | None = None

def _process_init(pickled_universe):
    """Child worker process initialiser."""
    # Transfer ove the universe to the child process
    global _universe
    _universe = pickle.loads(pickled_universe)


def _handle_sigterm(*args):
    # TODO: Despite all the effort, this does not seem to work with Visual Studio Code's Interrupt Kernel button
    processes: list[Process] = list(_process_pool._processes.values())
    _process_pool.shutdown()
    for p in processes:
        p.kill()
    sys.exit(1)


def setup_indicator_multiprocessing(executor):
    """Set up multiprocessing for indicators.

    - We use multiprocessing to calculate indicators

    - We want to be able to abort (CTRL+C) gracefully
    """
    global _process_pool_executor
    _process_pool_executor = executor._executor

    # Enable graceful multiprocessing termination only if we run as a backtesting noteboook
    # pytest work around for: test_trading_strategy_engine_v050_live_trading
    # ValueError: signal only works in main thread of the main interpreter
    # https://stackoverflow.com/a/23207116/315168
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_sigterm)


def validate_function_kwargs(func: Callable, kwargs: dict):
    """Check that we can pass the given kwargs to a function.

    Designed to be used with pandas_ta functions -
    many special cases needs to be added.

    :param func:
        TA function

    :param kwargs:
        Parameters we think function can take

    :raise IndicatorFunctionSignatureMismatch:
        You typoed
    """

    # https://stackoverflow.com/a/64504028/315168

    assert callable(func)

    sig = inspect.signature(func)
    allowed_params = sig.parameters

    for our_param, our_value in kwargs.items():
        if our_param not in allowed_params:
            raise IndicatorFunctionSignatureMismatch(f"Function {func} does not take argument {our_param}. Available arguments are: {allowed_params}.")


_empty_series = pd.Series([], dtype="float64")


def wrap_nones(f: Callable) -> Callable:
    """Turn None return values to empty pandsa Series.

    - This is a function decorator that wraps :py:mod:`pandas_ta`
      technical indicator functinos

    - Many of these functions return ``None`` instead of empty
      series when having not enough data

    - We wrap these to empty Pandas series

    - This is used with :py:meth:`IndicatorSet.add`

    :return:
        A function that `returns pd.Series([])` on `None`
    """

    def wrapped(*args, **kwargs) -> pd.Series | pd.DataFrame:
        ret = f(*args, **kwargs)
        if ret is None:
            return _empty_series
        return ret

    # https://stackoverflow.com/a/42422050/315168
    wrapped.__signature__ = inspect.signature(f)

    return wrapped


def _deterministic_hash(input_str: str) -> str:
    """Python hash() is not deterministic across sessions.

    - Make persistent hash
    - This includes multiprocessing
    - This includes notebook reboots
    """
    return hashlib.sha256(input_str.encode()).hexdigest()

