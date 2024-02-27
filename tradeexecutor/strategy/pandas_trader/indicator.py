"""Indicator definitions."""
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
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Callable, Protocol, Any, TypeAlias
import logging

import futureproof
import pandas as pd

from tqdm_loggable.auto import tqdm

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, UniverseCacheKey


logger = logging.getLogger(__name__)


class IndicatorCalculationFailed(Exception):
    """We could not calculate the given indicator.

    - Wrap the underlying Python exception to a friendlier error message
    """


class IndicatorFunctionSignatureMismatch(Exception):
    """Given Pythohn function cannot run on the passed parameters."""


class IndicatorSource(enum.Enum):
    """The data on which the indicator will be calculated."""

    #: Calculate this indicator based on candle close price
    close_price = "close_price"

    #: Calculate this indicator based on candle open price
    open_price = "open_price"



@dataclass(slots=True, frozen=True)
class IndicatorDefinition:
    """A definition for a single indicator.

    - Indicator defintions are static - they do not change between the strategy runs

    - Used as id for the caching the indicator results

    - Definitions are used to calculate indicators for all trading pairs
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
    func: Callable

    #: Parameters for building this indicator.
    #:
    #: - Each key is a function argument name for :py:attr:`func`.
    #: - Each value is a single value
    #:
    parameters: dict

    #: On what trading universe data this indicator is calculated
    #:
    source: IndicatorSource = IndicatorSource.close_price

    def __repr__(self):
        return f"<Indicator {self.name} using {self.func.__name__} for {self.parameters}>"

    def __eq__(self, other):
        return self.name == other.name and self.parameters == other.parameters and self.source == other.source and self.func.__name__ == other.func.__name__

    def __hash__(self):
        # https://stackoverflow.com/a/5884123/315168
        return hash((self.name, frozenset(self.parameters.items()), self.source, self.func.__name__))

    def __post_init__(self):
        assert type(self.name) == str
        assert callable(self.func)
        assert type(self.parameters) == dict

        validate_function_kwargs(self.func, self.parameters)

    def get_cache_key(self) -> str:
        parameters = ",".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.name}({parameters})"

    def is_needed_for_pair(self, pair: TradingPairIdentifier) -> bool:
        """Currently indicators are calculated for spont pairs only."""
        return pair.is_spot()

    def calculate(self, input: pd.Series) -> pd.DataFrame | pd.Series:
        """Calculate the underlying indicator value.

        :param input:
            Price series used as input.

        :return:
            Single or multi series data.

            - Multi-value indicators return DataFrame with multiple columns (BB).
            - Single-value indicators return Series (RSI, SMA).

        """
        try:
            return self.func(input, **self.parameters)
        except Exception as e:
            raise IndicatorCalculationFailed(f"Could not calculate indicator {self.name} ({self.func}) for parameters {self.parameters}, input data is {len(input)} rows") from e


class IndicatorSet:
    """Define the indicators that are needed by a trading strategy.

    - For backtesting, indicators are precalculated

    - For live trading, these indicators are recalculated for the each decision cycle

    - Indicators are calculated for each given trading pair, unless specified otherwise
    """

    def __init__(self):
        self.indicators: dict[str, IndicatorDefinition] = {}

    def has_indicator(self, name: str) -> bool:
        return name in self.indicators

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

    def add(
        self,
        name: str,
        func: Callable,
        parameters: dict,
        source: IndicatorSource=IndicatorSource.close_price,
    ):
        assert type(name) == str
        assert callable(func), f"{func} is not callable"
        assert type(parameters) == dict, f"parameters must be dictionary, we got {parameters.__class__}"
        assert isinstance(source, IndicatorSource), f"Expected IndicatorSource, got {type(source)}"
        self.indicators[name] = IndicatorDefinition(name, func, parameters, source)

    def iterate(self) -> Iterable[IndicatorDefinition]:
        yield from self.indicators.values()



class CreateIndicatorsProtocol(Protocol):
    """Call signature for create_indicators function"""

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

    #: For which indicator this result is
    #:
    definition: IndicatorDefinition

    #: The pair for which this result was calculated
    #:
    #:
    pair: TradingPairIdentifier

    #: Indicator output is one time series, but in some cases can be multiple as well.
    #:
    #: For example BB indicator calculates multiple series from one close price value.
    #:
    #:
    data: pd.DataFrame | pd.Series

    #: Was this indicator result cached or calculated on this run
    #:
    cached: bool



IndicatorResultKey: TypeAlias = tuple[TradingPairIdentifier, IndicatorDefinition]

IndicatorResultMap: TypeAlias = dict[IndicatorResultKey, IndicatorResult]


class IndicatorStorage:
    """Store calculated indicator results on disk.

    TODO: Cannot handle multichain universes at the moment, as serialises trading pairs by their ticker.
    """

    def __init__(self, path: Path, universe_key: UniverseCacheKey):
        assert isinstance(path, Path)
        assert type(universe_key) == str
        self.path = path
        self.universe_key = universe_key

    def __repr__(self):
        return f"<IndicatorStorage at {self.path}>"

    def get_indicator_path(self, ind: IndicatorDefinition, pair: TradingPairIdentifier) -> Path:
        """Get the Parquet file where the indicator data is stored.

        :return:
            Example `/tmp/.../test_indicators_single_backtes0/ethereum,1d,WETH-USDC-WBTC-USDC,2021-06-01-2021-12-31/sma(length=21).parquet`
        """
        return self.path / Path(self.universe_key) / Path(f"{ind.get_cache_key()}-{pair.get_ticker()}.parquet")

    def is_available(self, ind: IndicatorDefinition, pair: TradingPairIdentifier) -> bool:
        return self.get_indicator_path(ind, pair).exists()

    def load(self, ind: IndicatorDefinition, pair: TradingPairIdentifier) -> IndicatorResult:
        """Load cached indicator data from the disk."""
        assert self.is_available(ind, pair)
        path = self.get_indicator_path(ind, pair)
        df = pd.read_parquet(path)

        if len(df.columns) == 1:
            # Convert back to series
            df = df[df.columns[0]]

        return IndicatorResult(
            self.universe_key,
            ind,
            pair,
            df,
            cached=True,
        )

    def save(self, ind: IndicatorDefinition, pair: TradingPairIdentifier, df: pd.DataFrame | pd.Series) -> IndicatorResult:
        """Atomic replacement of the existing data.

        - Avoid leaving partially written files
        """
        assert isinstance(ind, IndicatorDefinition)
        assert isinstance(pair, TradingPairIdentifier)

        if isinstance(df, pd.Series):
            # For saving, create a DataFrame with a single column "value"
            save_df = pd.DataFrame({"value": df})
        else:
            save_df = df

        assert isinstance(save_df, pd.DataFrame), f"Expected DataFrame, got: {type(df)}"
        path = self.get_indicator_path(ind, pair)
        dirname, basename = os.path.split(path)

        os.makedirs(dirname, exist_ok=True)

        temp = tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dirname)

        save_df.to_parquet(temp)
        temp.close()
        # https://stackoverflow.com/a/3716361/315168
        shutil.move(temp.name, path)

        return IndicatorResult(
            universe_key=self.universe_key,
            definition=ind,
            pair=pair,
            data=df,
            cached=False,
        )

    @staticmethod
    def create_default(
        universe: TradingStrategyUniverse,
        default_path=Path(os.path.expanduser("~/.cache/indicators"))
    ) -> "IndicatorStorage":
        """Get the indicator storage with the default cache path."""
        return IndicatorStorage(default_path, universe.get_cache_key())


def _serialise_parameters_for_cache_key(parameters: dict) -> str:

    for k, v in parameters.items():
        assert type(k) == str
        assert type(v) not in (list, tuple)  # Don't leak test ranges here - must be a single value

    return "".join([f"{k}={v}" for k, v in parameters.items()])



def _load_indicator_result(storage: IndicatorStorage, indicator: IndicatorDefinition, pair: TradingPairIdentifier) -> IndicatorResult:
    logger.info("Loading %s %s", pair, indicator)
    assert storage.is_available(indicator, pair), f"Tried to load indicator that is not in the cache {indicator} - {pair}"
    return storage.load(indicator, pair)


def _calculate_and_save_indicator_result(
    strategy_universe: TradingStrategyUniverse,
    storage: IndicatorStorage,
    indicator: IndicatorDefinition,
    pair: TradingPairIdentifier,
) -> IndicatorResult:

    match indicator.source:
        case IndicatorSource.open_price:
            column = "open"
        case IndicatorSource.close_price:
            column = "close"
        case _:
            raise AssertionError(f"Unsupported input source {pair} {indicator} {indicator.source}")

    input = strategy_universe.data_universe.candles.get_samples_by_pair(pair.internal_id)[column]
    data = indicator.calculate(input)

    assert data is not None, f"Indicator function {indicator.name} ({indicator.func}) did not return any result, received Python None instead"

    result = storage.save(indicator, pair, data)
    return result


def load_indicators(
    strategy_universe: TradingStrategyUniverse,
    storage: IndicatorStorage,
    indicator_set: IndicatorSet,
    all_combinations: set[IndicatorResultKey],
    max_readers=8,
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
    for pair, ind in all_combinations:
        if storage.is_available(ind, pair):
            task_args.append((storage, ind, pair))

    logger.info("Loading cached indicators indicators, we have %d combinations available in the cache %s", len(task_args), storage.path)

    if len(task_args) == 0:
        return {}

    results = {}
    label = indicator_set.get_label()
    key: IndicatorResultKey

    with tqdm(total=len(task_args), desc=f"Reading cached indicators {label} for {strategy_universe.get_pair_count()} pairs, {indicator_set.get_count()} indicators, using {max_readers} threads, total {len(task_args)} cached available") as progress_bar:

        if max_readers > 1:
            logger.info("Multi-thread reading")

            executor = futureproof.ThreadPoolExecutor(max_workers=max_readers)
            tm = futureproof.TaskManager(executor, error_policy=futureproof.ErrorPolicyEnum.RAISE)

            # Run the checks parallel using the thread pool
            tm.map(_load_indicator_result, task_args)

            # Extract results from the parallel task queue
            for task in tm.as_completed():
                result = task.result
                key = (result.pair, result.definition)
                results[key] = result
                progress_bar.update()
        else:
            logger.info("Single-thread reading")
            for result in itertools.starmap(_load_indicator_result, task_args):
                key = (result.pair, result.definition)
                results[key] = result

    return results


def calculate_indicators(
    strategy_universe: TradingStrategyUniverse,
    storage: IndicatorStorage,
    indicators: IndicatorSet,
    execution_context: ExecutionContext,
    remaining: set[IndicatorResultKey],
    max_workers=8,
) -> IndicatorResultMap:
    """Calculate indicators for which we do not have cached data yet.

    - Use a thread pool to speed up IO

    :param remaining:
        Remaining indicator combinations for which we do not have a cached rresult
    """

    assert isinstance(execution_context, ExecutionContext), f"Expected ExecutionContext, got {type(execution_context)}"

    results: IndicatorResultMap

    label = indicators.get_label()
    logger.info("Calculating indicators %s", label)

    if len(remaining) == 0:
        logger.info("Nothing to calculate")
        return {}

    task_args = []
    for pair, ind in remaining:
        task_args.append((strategy_universe, storage, ind, pair))

    results = {}

    if max_workers > 1:

        # Do a parallel scan for the maximum speed
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
        logger.info("Doing a multiprocess indicator calculation, picked universe is %d bytes", len(pickled_universe))

        # Set up a process pool executing structure
        executor = futureproof.ProcessPoolExecutor(max_workers=max_workers, initializer=_process_init, initargs=(pickled_universe,))
        tm = futureproof.TaskManager(executor, error_policy=futureproof.ErrorPolicyEnum.RAISE)

        # Set up a signal handler to stop child processes on quit
        _process_pool_executor = executor._executor
        signal.signal(signal.SIGTERM, _handle_sigterm)

        # Run the tasks
        tm.map(_calculate_and_save_indicator_result, task_args)

        # Track the child process completion using tqdm progress bar
        with tqdm(total=len(task_args), desc=f"Grid searching using {max_workers} processes: {label}") as progress_bar:
            # Extract results from the parallel task queue
            for task in tm.as_completed():
                result = task.result
                results[(result.pair, result.definition)] = result
                progress_bar.update()

    else:
        # Do single thread - good for debuggers like pdb/ipdb
        #

        _universe = strategy_universe

        logger.info("Doing a single thread indicator calculation")
        iter = itertools.starmap(_calculate_and_save_indicator_result, task_args)

        # Force workers to finish
        result: IndicatorResult
        for result in iter:
            results[(result.pair, result.definition)] = result

    logger.info("Total %d indicator results calculated", len(results))

    return results


def prepare_indicators(
    create_indicators: CreateIndicatorsProtocol,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,

):
    """Call the strategy module indicator builder."""
    indicators = IndicatorSet()
    create_indicators(parameters, indicators, strategy_universe, execution_context)
    if indicators.get_count() == 0:
        # TODO: Might have legit use cases?
        logger.warning(f"create_indicators() did not create a single indicator")
    import ipdb ; ipdb.set_trace()
    return indicators


def calculate_and_load_indicators(
    strategy_universe: TradingStrategyUniverse,
    storage: IndicatorStorage,
    execution_context: ExecutionContext,
    indicators: IndicatorSet | None = None,
    parameters: StrategyParameters | None = None,
    create_indicators: CreateIndicatorsProtocol | None = None,
    max_workers=8,
    max_readers=8,
) -> IndicatorResultMap:
    """Precalculate all indicators.

    - Calculate indicators using multiprocessing

    - Display TQDM progress bars for loading cached indicators and calculating new ones

    - Use cached indicators if available
    """

    assert create_indicators or indicators, "You must give either create_indicators or indicators argument"

    if create_indicators:
        assert indicators is None, f"Give either indicators or create_indicators, not both"
        assert parameters is not None, f"parameters argument must be given if you give create_indicators"
        indicators = prepare_indicators(create_indicators, parameters, strategy_universe, execution_context)

    assert isinstance(indicators, IndicatorSet), f"Got {type(indicators)}"

    all_combinations: set[IndicatorResultKey] = set()
    for pair in strategy_universe.iterate_pairs():
        for ind in indicators.iterate():
            all_combinations.add((pair, ind))

    logger.info("Loading indicators %s for the universe %s, storage is %s", indicators.get_label(), strategy_universe.get_cache_key(), storage.path)
    cached = load_indicators(strategy_universe, storage, indicators, all_combinations, max_readers=max_readers)

    for key in cached.keys():
        # Check we keyed this right
        assert key in all_combinations, f"Loaded a cached result {key} is not in part of the all combinations we expected"

    calculation_needed = all_combinations - set(cached.keys())
    calculated = calculate_indicators(
        strategy_universe,
        storage,
        indicators,
        execution_context,
        calculation_needed,
        max_workers=max_workers,
    )

    result = cached | calculated

    for key in result.keys():
        # Check we keyed this right
        assert key in all_combinations
        assert isinstance(key[0], TradingPairIdentifier)
        assert isinstance(key[1], IndicatorDefinition)

    return result


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

