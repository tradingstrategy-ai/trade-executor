"""Perform a grid search ove strategy parameters to find optimal parameters."""
import tempfile
from _decimal import Decimal

import numpy

# Enable pickle patch that allows multiprocessing in notebooks
from tradeexecutor.monkeypatch import cloudpickle_patch  

import datetime
import enum
import itertools
import logging
import os
import pickle
import shutil
import signal
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from inspect import isclass
from multiprocessing import Process
from pathlib import Path
from typing import Protocol, Dict, List, Tuple, Any, Optional, Collection, Callable, Iterable
import concurrent.futures.process
from packaging import version

import numpy as np
import pandas as pd
import futureproof

from tradeexecutor.utils.cpu import get_safe_max_workers_count
from tradeexecutor.utils.jupyter_notebook_name import get_notebook_name

try:
    from tqdm_loggable.auto import tqdm
except ImportError:
    # tqdm_loggable is only available at the live execution,
    # but fallback to normal TQDM auto mode
    from tqdm.auto import tqdm

from tradeexecutor.strategy.engine_version import TradingStrategyEngineVersion
from tradeexecutor.strategy.execution_context import ExecutionContext, grid_search_execution_context, standalone_backtest_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, CreateIndicatorsProtocolV1, DiskIndicatorStorage, warm_up_indicator_cache, \
    IndicatorKey, DEFAULT_INDICATOR_STORAGE_PATH, CreateIndicatorsProtocol, call_create_indicators, IndicatorStorage
from tradeexecutor.strategy.universe_model import UniverseOptions


from tradeexecutor.analysis.advanced_metrics import calculate_advanced_metrics, AdvancedMetricsMode
from tradeexecutor.analysis.trade_analyser import TradeSummary, build_trade_analysis
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarAmount, Percent
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol, DecideTradesProtocol2, StrategyParameters, DecideTradesProtocol3, DecideTradesProtocol4
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns


logger = logging.getLogger(__name__)


class GridSearchDataRetention(enum.Enum):
    """What grid search data we generate and load.

    - We want to discard unneeded data to save memory

    See :py:class:`GridSearchResult`.
    """

    #: Pass all grid search data to the parent notebook process
    #:
    #: Includes full state of the backtest results
    #:
    all = "all"

    #: Discard state
    metrics_only = "metrics_only"


def _hide_warnings(func):
    """Function wrapper to suppress warnings caused by quantstats and numpy functions.

    Otherwise these warnings pollute notebook output.
    """

    # Hidden warnings include:

    # In perform_grid_search:
    # /home/.cache/pypoetry/virtualenvs/trade-executor-xSh0vQvh-py3.10/lib/python3.10/site-packages/numpy/lib/function_base.py:2854:
    # RuntimeWarning: invalid value encountered in divide
    # c /= stddev[:, None]

    # In perform_grid_search:
    # /home/alex/.cache/pypoetry/virtualenvs/trade-executor-xSh0vQvh-py3.10/lib/python3.10/site-packages/scipy/stats/_distn_infrastructure.py:2351:
    # RuntimeWarning: invalid value encountered in multiply
    # lower_bound = _a * scale + loc

    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


@dataclass
class GridParameter:
    """One value in grid search matrix."""

    #: Name e.g. `rsi_low`
    name: str

    #: Value e.g 0.8
    value: Any

    #: Was this parameter part of the grid search space, or is it a single parameter.
    #:
    #: Also true for empty lists
    #:
    single: bool

    #: Is this parameter a search space point in an optimiser
    #:
    optimise: bool = False

    def __post_init__(self):
        assert type(self.name) == str

    def __hash__(self):
        return hash((self.name, self.value))

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def is_searchable(self) -> bool:
        return self.optimise or not self.single

    def get_computable_value(self) -> float | int | bool | str:
        """Handle use of rounded Decimals in optimiser."""
        if isinstance(self.value, Decimal):
            return float(self.value)
        return self.value

    def to_path(self) -> str:
        """"""
        value = self.value

        if isinstance(value, Enum):
            return f"{self.name}={self.value.value}"
        elif type(value) == bool:
            return f"{self.name}={self.value.lower()}"
        elif isinstance(value, (numpy.float64, numpy.float32, numpy.int64)):
            # scikit-optimise values
            return f"{self.name}={self.value}"
        elif isinstance(value, Decimal):
            # scikit-optimise values
            # where space.Real is rounded to accuracy
            # that can fit into a filename
            return f"{self.name}={self.value}"
        elif type(value) in (float, int, str):
            return f"{self.name}={self.value}"
        if value is None:
            return f"{self.name}=none"
        else:
            raise NotImplementedError(f"We do not support filename conversion for value {type(value)}={value}")


@dataclass(slots=True)
class GridCombination:
    """One combination line in grid search."""

    #: How many of nth grid combinations this is
    #:
    index: int

    #: In which folder we store the result files of all grid search runs
    #:
    #: Each individual combination will have its subfolder based on its parameter.
    result_path: Path

    #: Alphabetically sorted list of parameters
    #:
    #: Each parameter can have 0...n values.]
    #: If parameter is not "single", i.e. single value, then it is searchable.
    #:
    parameters: Tuple[GridParameter]

    #: Indicators for this combination.
    #:
    #: create_indicators() is called with the :py:attr:`parameters` and it
    #: yields the result of indicators we need to calculate for this grid combination.
    #: Only avaiable if trading_strategy_engine_version > 0.5.
    #:
    #: - One key entry for each trading pair if pair specific indicators are used
    #:
    indicators: set[IndicatorKey] | None = None

    def __post_init__(self):
        assert len(self.parameters) > 0
        assert isinstance(self.result_path, Path), f"Expected Path, got {type(self.result_path)}"
        assert self.result_path.exists() and self.result_path.is_dir(), f"Not a dir: {self.result_path}"

    def __hash__(self):
        return hash(self.parameters)

    def __eq__(self, other):
        return self.parameters == other.parameters

    def __repr__(self):
        buf = f"<GridCombination #{self.index}\n"
        for p in self.parameters:
            buf += f"   {p.name}={p.value}\n"
        buf += ">"
        return buf

    @property
    def searchable_parameters(self) -> List[GridParameter]:
        """Get all parameters that are searchable.

        Searchable parameters have two or more values.
        """
        return [p for p in self.parameters if p.is_searchable()]

    def get_relative_result_path(self) -> Path:
        """Get the path where the resulting state file is stored.

        Try to avoid messing with 256 character limit on filenames, thus break down as folders.
        """
        path_parts = [p.to_path() for p in self.searchable_parameters]
        return Path(os.path.join(*path_parts))

    def get_full_result_path(self) -> Path:
        """Get the path where the resulting state file is stored."""
        return self.result_path.joinpath(self.get_relative_result_path())

    def validate(self):
        """Check arguments can be serialised as fs path."""
        assert len(self.searchable_parameters) > 0, f"Grid search combination does not have any parameters that would have multiple values to search: {self.parameters}. Add parameters to search or use normal backtesting instead. Also make sure your parameter class is not called StrategyParameters, as it is reserved for grid search."
        assert isinstance(self.get_relative_result_path(), Path)

    def as_dict(self) -> dict:
        """Get as kwargs mapping."""
        return {p.name: p.value for p in self.parameters}

    def get_label(self) -> str:
        """Human-readable label for this combination.

        See also :py:meth:`get_all_parameters_label`.
        """
        return f"#{self.index}, " + ", ".join([f"{p.name}={p.value}" for p in self.searchable_parameters])

    def get_all_parameters_label(self) -> str:
        """Get label which includes single value parameters as well.

        See also :py:meth:`get_label`.
        """
        return f"#{self.index}, " + ", ".join([f"{p.name}={p.value}" for p in self.parameters])

    def destructure(self) -> List[Any]:
        """Open parameters dict.

        This will return the arguments in the same order you pass them to :py:func:`prepare_grid_combinations`.
        """
        return [p.value for p in self.parameters]

    def to_strategy_parameters(self) -> StrategyParameters:
        return StrategyParameters(self.as_dict())

    def get_parameter(self, name: str) -> object:
        """Get a parameter value.

        :param name:
            Parameter name

        :raise ValueError:
            If parameter is missing.
        """
        for p in self.parameters:
            if p.name == name:
                return p.value

        raise ValueError(f"No parameter: {name}")

    @staticmethod
    def get_all_indicators(combinations: Iterable["GridCombination"]) -> set[IndicatorKey]:
        """Get all defined indicators that need to be calculated, across all grid search combinatios.

        Duplications are merged.
        """
        indicators = set()
        for c in combinations:
            if c.indicators:
                for i in c.indicators:
                    indicators.add(i)
        return indicators


@dataclass(slots=True, frozen=False)
class GridSearchResult:
    """Result for one grid combination.

    - Result for one grid search combination

    - Calculate various statistics and curves ready in a multiprocess worker

    - Results can be cached on a disk, as a pickle

    - Some of the data might not be available or discarded as per :py:class:`GridSearchDataRetention`
    """

    #: For which grid combination this result is
    combination: GridCombination

    #: The full back test state
    #:
    #: By the default, grid search execution drops these,
    #: as saving and loading them takes extra time, space,
    #: and state is not used to compare grid search results.
    #:
    state: State | None

    #: Calculated trade summary
    #:
    #: Internal stats calculated about trades
    #:
    summary: TradeSummary

    #: Performance metrics
    #:
    #: Use QuantStats lib to calculate these stats.
    #:
    metrics: pd.DataFrame

    #: Needed for visualisations
    #:
    equity_curve: pd.Series

    #: Needed for visualisations
    #:
    returns: pd.Series

    #: What backtest data range we used
    #:
    universe_options: UniverseOptions

    #: Was this result read from the earlier run save
    cached: bool = False

    #: Child process that created this result.
    #:
    #: Only applicable to multiprocessing
    process_id: int = None

    #: Initial cash from the state.
    #:
    #: Copied here from the state, as it is needed to draw equity curves.
    #: Not available in legacy data.
    #:
    initial_cash: USDollarAmount | None = None

    def __hash__(self):
        return self.combination.__hash__()

    def __eq__(self, other):
        return self.combination == other.combination

    def __repr__(self) -> str:
        cagr = self.get_cagr()
        sharpe = self.get_sharpe()
        max_drawdown = self.get_max_drawdown()
        return f"<GridSearchResult\n  {self.combination.get_all_parameters_label()}\n  CAGR: {cagr*100:.2f}% Sharpe: {sharpe:.2f} Max drawdown:{max_drawdown*100:.2f}%\n>"

    def get_label(self) -> str:
        """Get name for this result for charts.

        - Label is grid search parameter key values

        - Includes only searched parameters as label
        """
        return self.combination.get_label()

    def get_metric(self, name: str) -> float:
        """Get a performance metric from quantstats.

        A shortcut method.

        Example:

        .. code-block:: python

            grid_search_results = perform_grid_search(
                decide_trades,
                strategy_universe,
                combinations,
                max_workers=8,
                trading_strategy_engine_version="0.4",
                multiprocess=True,
            )

            print("Sharpe of the first result", grid_search_results[0].get_metric("Sharpe")

        :param name:
            See quantstats for examples

        :return:
            Performance metrics value
        """

        series = self.metrics["Strategy"]
        assert name in self.metrics.index, f"Metric {name} not available. We have: {series.index}"
        return series[name]

    def get_cagr(self) -> Percent:
        return self.get_metric("CAGR﹪")

    def get_sharpe(self) -> float:
        """Get the Sharpe ratio of this grid search result.

        :return:
            0 if not available (the strategy made no trades).
        """
        sharpe = self.get_metric("Sharpe")
        if pd.notna(sharpe) and sharpe != "-":
            assert type(sharpe) in (float, int), f"Got {type(sharpe)} {sharpe}"
            return sharpe
        return 0.0

    def get_max_drawdown(self) -> Percent:
        return self.get_metric("Max Drawdown")

    def get_parameter(self, name) -> object:
        """Get a combination parameter value used to produce this search.

        Useful in filtering.

        .. code-block:: python

            filtered_results = [r for r in grid_search_results if r.combination.get_parameter("regime_filter_ma_length") is None]
            print(f"Grid search results without regime filter: {len(filtered_resutls)}")

        :param name:
            Parameter name

        :raise ValueError:
            If parameter is missing.
        """
        return self.combination.get_parameter(name)

    def get_trade_count(self) -> int:
        """How many trades this strategy made."""
        return self.summary.total_trades

    @staticmethod
    def has_result(combination: GridCombination):
        base_path = combination.result_path
        return base_path.joinpath(combination.get_full_result_path()).joinpath("result.pickle").exists()

    @staticmethod
    def load(combination: GridCombination):
        """Deserialised from the cached Python pickle."""

        base_path = combination.get_full_result_path()

        with open(base_path.joinpath("result.pickle"), "rb") as inp:
            result: GridSearchResult = pickle.load(inp)

        result.cached = True
        return result

    def save(self):
        """Serialise as Python pickle."""
        base_path = self.combination.get_full_result_path()
        base_path.mkdir(parents=True, exist_ok=True)

        # TODO:
        # Fails to pickle functions, but we do not need these in results,
        # so we just shortcut and clear out those functions
        if self.combination.indicators is not None:
            for ind in self.combination.indicators:
                ind.definition.func = None

        # Do atomic replacement to avoid partial pickles,
        # as they cause subsequent test runs to fail
        # https://stackoverflow.com/a/3716361/315168
        final_file = base_path.joinpath("result.pickle")
        temp = tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=base_path)
        pickle.dump(self, temp)
        temp.close()
        shutil.move(temp.name, final_file)


class GridSearchWorker(Protocol):
    """Define how to create different strategy bodies."""

    def __call__(self, universe: TradingStrategyUniverse, combination: GridCombination) -> GridSearchResult:
        """Run a new decide_trades() strategy body based over the serach parameters.

        :param args:
        :param kwargs:
        :return:
        """

def prepare_grid_combinations(
    parameters: Dict[str, List[Any]] | type,
    result_path: Path,
    clear_cached_results=False,
    marker_file="README-GRID-SEARCH.md",
    create_indicators: CreateIndicatorsProtocol | None = None,
    strategy_universe: TradingStrategyUniverse | None = None,
    execution_context: ExecutionContext = grid_search_execution_context,
) -> List[GridCombination]:
    """Get iterable search matrix of all parameter combinations.

    - Make sure we preverse the original order of the grid search parameters.

    - Set up the folder to store the results

    :param parameters:
        A grid of parameters we will search.

        Can be a dict or a class of which all members will be enumerated.

    :param result_path:
        A folder where resulting state files will be stored.

    :param clear_cached_results:
        Clear any existing result files from the saved result cache.

        You need to do this if you change the strategy logic outside
        the given combination parameters, as the framework will otherwise
        serve you the old cached results.

    :param marker_file:
        Safety to prevent novice users to nuke their hard disk with this command.

    :param create_indicators:
        Pass `create_indicators` function if you want your grid seacrh to use fast cached indicators.

    :param strategy_universe:
        Needed with `create_indicators`

    :param execution_context:
        Tell if we are running unit testing or real backtesting.

    :return:
        List of all combinations we need to search through
    """

    from tradeexecutor.monkeypatch import cloudpickle_patch  # Enable pickle patch that allows multiprocessing in notebooks

    assert isinstance(result_path, Path)

    if create_indicators is not None:
        assert strategy_universe is not None, f"You need to pass both create_indicators and strategy_universe"

    if execution_context is not None:
        assert execution_context.grid_search, f"ExecutionContext.grid_search is not set"

    if isclass(parameters):
        parameters = StrategyParameters.from_class(parameters, grid_search=True)

    logger.info("Preparing %d grid combinations, caching results in %s", len(parameters), result_path)

    if clear_cached_results:
        if result_path.exists():
            assert result_path.joinpath(marker_file).exists(), f"{result_path} does not seem to be grid search folder, it lacks {marker_file}"
            shutil.rmtree(result_path)

    result_path.mkdir(parents=True, exist_ok=True)

    with open(result_path.joinpath(marker_file), "wt") as out:
        print("This is a TradingStrategy.ai grid search result folder", file=out)

    args_lists: List[list] = []
    for name, values in parameters.items():
        assert isinstance(values, Collection), f"For parameter {name}, expected list, got: {values}"
        single = len(values) <= 1
        args = [GridParameter(name, v, single) for v in values]
        args_lists.append(args)

    combinations = itertools.product(*args_lists)

    # Maintain the orignal parameter order over itertools.product()
    order = tuple(parameters.keys())
    def sort_by_order(combination: List[GridParameter]) -> Tuple[GridParameter]:
        temp = {p.name: p for p in combination}
        return tuple([temp[o] for o in order])

    combinations = [GridCombination(index=idx, parameters=sort_by_order(c), result_path=result_path) for idx, c in enumerate(combinations, start=1)]
    for c in combinations:
        c.validate()

        # Determine what indicators this grid search combination needs
        if create_indicators is not None:
            indicators = call_create_indicators(
                create_indicators,
                c.to_strategy_parameters(),
                strategy_universe,
                execution_context
            )
            c.indicators = set(indicators.generate_combinations(strategy_universe))

    return combinations


def _run_v04(
    decide_trades: DecideTradesProtocol3 | DecideTradesProtocol4,
    universe: TradingStrategyUniverse,
    combination: GridCombination,
    trading_strategy_engine_version: TradingStrategyEngineVersion,
    indicator_storage: DiskIndicatorStorage,
):
    """Run decide_trades() with input parameteter.

    - v2 style grid search, combination and argument passing
    """

    parameters = combination.to_strategy_parameters()

    backtest_start = parameters.get("backtest_start")
    # assert backtest_start, f"Strategy parameters lack backtest_start, we have {list(input_object.keys())}"

    backtest_end = parameters.get("backtest_end")
    # assert backtest_end, f"Strategy parameters lack backtest_end, we have {list(input_object.keys())}"

    cycle_duration = parameters.get("cycle_duration")
    assert cycle_duration, f"Strategy parameters lack cycle_duration, we have {list(parameters.keys())}"

    initial_cash = parameters.get("initial_cash")
    assert initial_cash, f"Strategy parameters lack initial_cash, we have {list(parameters.keys())}"

    return run_grid_search_backtest(
        combination,
        decide_trades,
        universe,
        start_at=backtest_start,
        end_at=backtest_end,
        cycle_duration=cycle_duration,
        trading_strategy_engine_version=trading_strategy_engine_version,
        parameters=parameters,
        initial_deposit=initial_cash,
        indicator_storage=indicator_storage,
    )


def run_grid_combination_threaded(
    grid_search_worker: GridSearchWorker | DecideTradesProtocol3,
    universe: TradingStrategyUniverse,
    combination: GridCombination,
    trading_strategy_engine_version: TradingStrategyEngineVersion,
    data_retention: GridSearchDataRetention,
    indicator_storage_path: Path | None = None,
):
    """Threared runner.

    Universe is passed as argument.
    """

    # Make sure we always get a indicator storage
    if version.parse(trading_strategy_engine_version) >= version.parse("0.5"):
        assert indicator_storage_path is not None, "indicator_storage_path must be passed"

    if indicator_storage_path is not None:
        assert isinstance(indicator_storage_path, Path)
        indicator_storage = DiskIndicatorStorage(indicator_storage_path, universe.get_cache_key())
    else:
        indicator_storage = None

    if GridSearchResult.has_result(combination):
        result = GridSearchResult.load(combination)
        return result

    if version.parse(trading_strategy_engine_version) >= version.parse("0.4"):
        # New style runner
        result = _run_v04(grid_search_worker, universe, combination, trading_strategy_engine_version, indicator_storage)
    else:
        # Legacy path
        result = grid_search_worker(universe, combination)

    if data_retention != GridSearchDataRetention.all:
        result.state = None

    # Cache result for the future runs
    result.save()

    return result


def run_grid_combination_multiprocess(
    grid_search_worker: GridSearchWorker | DecideTradesProtocol3,
    combination: GridCombination,
    trading_strategy_engine_version: TradingStrategyEngineVersion,
    data_retention: GridSearchDataRetention,
    indicator_storage_path = DEFAULT_INDICATOR_STORAGE_PATH,
):
    """Mutltiproecss runner.

    Universe is passed as process global.

    :param indicator_storage_path:
        Override for unit testing
    """

    from tradeexecutor.monkeypatch import cloudpickle_patch  # Enable pickle patch that allows multiprocessing in notebooks

    assert isinstance(combination, GridCombination)
    assert isinstance(indicator_storage_path, Path)

    global _universe

    universe = _universe

    # Make sure we always get a indicator storage
    if version.parse(trading_strategy_engine_version) >= version.parse("0.5"):
        assert indicator_storage_path is not None, "indicator_storage_path must be passed"

    if indicator_storage_path is not None:
        assert isinstance(indicator_storage_path, Path)
        indicator_storage = DiskIndicatorStorage(indicator_storage_path, universe.get_cache_key())
    else:
        indicator_storage = None

    if GridSearchResult.has_result(combination):
        result = GridSearchResult.load(combination)
        return result

    if version.parse(trading_strategy_engine_version) >= version.parse("0.4"):
        # New style runner
        result = _run_v04(grid_search_worker, universe, combination, trading_strategy_engine_version, indicator_storage)
    else:
        # Legacy path
        result = grid_search_worker(universe, combination)

    result.process_id = os.getpid()

    if data_retention != GridSearchDataRetention.all:
        result.state = None

    # Cache result for the future runs
    result.save()

    return result


def _read_combination_result(c: GridCombination, data_retention: GridSearchDataRetention) -> GridSearchResult:
    assert GridSearchResult.has_result(c), f"Did not have cached result for {c}, but was added to the cache read queue"
    result = GridSearchResult.load(c)
    assert result.cached

    if data_retention != GridSearchDataRetention.all:
        result.state = None

    return result


def warm_up_grid_search_indicator_cache(
    strategy_universe: TradingStrategyUniverse,
    combinations: List[GridCombination],
    indicator_storage: DiskIndicatorStorage,
    max_workers: int = 8,
    execution_context: ExecutionContext = grid_search_execution_context,
):
    """Prepare indicators used in the grid search.

    - Search all possible indicator combinations through grid search combinations

    - Check if we already have them

    - Calculate indicator if not

    - Store on a disk cache

    - Later on `run_backtest()`, in a separate process, will
      read this indicator result off the disk
    """

    # Build an indicator set of all possible indicators needed
    indicators: set[IndicatorKey] = set()
    for c in combinations:
        for ind in c.indicators:
            indicators.add(ind)

    # Will display TQDM progress bar for filling the cache
    warm_up_indicator_cache(
        strategy_universe,
        indicator_storage,
        execution_context=execution_context,
        indicators=indicators,
        max_workers=max_workers,
        all_combinations=indicators,
    )


def _read_cached_results(
    combinations: List[GridCombination],
    data_retention: GridSearchDataRetention,
    reader_pool_size=16,
) -> Dict[GridCombination, GridSearchResult]:
    """Read grid search results that are available on a disk from previous run.

    - Picked results

    - Multiprocess worker would read, unpickle, pickle, send, unpickle

    - Instead, we do a threaded reader

    - cPickle is C code, does not lock GIL
    """

    # Set up a process pool executing structure
    executor = futureproof.ThreadPoolExecutor(max_workers=reader_pool_size)
    tm = futureproof.TaskManager(executor, error_policy=futureproof.ErrorPolicyEnum.RAISE)
    task_args = [(c, data_retention) for c in combinations if GridSearchResult.has_result(c)]

    # Run the checks parallel using the thread pool
    tm.map(_read_combination_result, task_args)

    if len(task_args) == 0:
        return {}

    results = {}

    # Label too long for Datalore
    label = ", ".join(p.name for p in combinations[0].searchable_parameters)
    print(f"Using grid search cache {combinations[0].result_path}, for indicators {label}")
    with tqdm(total=len(task_args), desc=f"Reading cached search results w/ {reader_pool_size} threads") as progress_bar:
        # Extract results from the parallel task queue
        for task in tm.as_completed():
            results[task.args[0]] = task.result
            progress_bar.update()

    return results


@_hide_warnings
def perform_grid_search(
    grid_search_worker: GridSearchWorker | DecideTradesProtocol3 | DecideTradesProtocol4,
    universe: TradingStrategyUniverse,
    combinations: List[GridCombination],
    max_workers: int | Callable=get_safe_max_workers_count,
    reader_pool_size: int | Callable=get_safe_max_workers_count,
    multiprocess=False,
    trading_strategy_engine_version: TradingStrategyEngineVersion="0.3",
    data_retention: GridSearchDataRetention = GridSearchDataRetention.metrics_only,
    execution_context: ExecutionContext = grid_search_execution_context,
    indicator_storage: DiskIndicatorStorage | None = None,
) -> List[GridSearchResult]:
    """Search different strategy parameters over a grid.

    - Run using parallel processing via threads.
      `Numoy should release GIL for threads <https://stackoverflow.com/a/40630594/315168>`__.

    - Save the resulting state files to a directory structure
      for invidual run analysis

    - If a result exists, do not perform the backtest again.
      However we still load the summary

    - Trading Strategy Universe is shared across threads to save memory.

    :param combinations:
        Prepared grid combinations.

        See :py:func:`prepare_grid_combinations`

    :param stats:
        If passed, collect run-time and unit testing statistics to this dictionary.

    :param multiprocess:
        Perform the search using multiple CPUs and Python's multiprocessing.

        If not set, use threaded approach.

        Scales much better, but disabled by default, as it does not work with Jupyter Notebooks very well.

    :param trading_strategy_engine_version:
        Which version of engine we are using.

    :return:
        Grid search results for different combinations.

    """

    from tradeexecutor.monkeypatch import cloudpickle_patch  # Enable pickle patch that allows multiprocessing in notebooks

    global _process_pool_executor

    start = datetime.datetime.utcnow()

    # Resolve CPU count
    if callable(max_workers):
        max_workers = max_workers()

    if callable(reader_pool_size):
        reader_pool_size = reader_pool_size()

    logger.info(
        "Performing a grid search over %s combinations, with %d threads, data retention policy is %s",
        len(combinations),
        max_workers,
        data_retention.name,
    )

    if indicator_storage is None:
        indicator_storage = DiskIndicatorStorage.create_default(universe)
        print(f"Using indicator cac he {indicator_storage.get_universe_cache_path()}")

    # First calculate indicators if create_indicators() protocol is used
    # (engine version = 0.5, DecideTradesProtocolV4)
    if any(c for c in combinations if c.indicators is not None):
        warm_up_grid_search_indicator_cache(
            universe,
            combinations,
            indicator_storage,
            max_workers=max_workers,
            execution_context=execution_context,
        )

    # Load grid search results we have already completed before crash / break / previous strategy parameters
    cached_results = _read_cached_results(combinations, data_retention, reader_pool_size)
    logger.info("Read %d cached results", len(cached_results))

    if len(cached_results) == len(combinations):
        print("All results were cached, grid search skipped")
        return list(cached_results.values())
    
    if len(cached_results) == 0:
        print("No cached grid search results found from previous runs")

    if max_workers > 1:

        # Do a parallel scan for the maximum speed
        #
        # Set up a futureproof task manager
        #
        # For futureproof usage see
        # https://github.com/yeraydiazdiaz/futureproof

        if multiprocess:
            #
            # Run individual searchers in child processes
            #

            # Copy universe data to child processes only once when the child process is created
            #
            pickled_universe = pickle.dumps(universe)
            logger.info("Doing a multiprocess grid search, picked universe is %d bytes", len(pickled_universe))

            # Set up a process pool executing structure
            executor = futureproof.ProcessPoolExecutor(max_workers=max_workers, initializer=_process_init, initargs=(pickled_universe,))
            tm = futureproof.TaskManager(executor, error_policy=futureproof.ErrorPolicyEnum.RAISE)
            task_args = [(grid_search_worker, c, trading_strategy_engine_version, data_retention, indicator_storage.path) for c in combinations if c not in cached_results]

            # Set up a signal handler to stop child processes on quit
            _process_pool_executor = executor._executor
            signal.signal(signal.SIGTERM, _handle_sigterm)

            # Run the tasks
            tm.map(run_grid_combination_multiprocess, task_args)

            # Track the child process completion using tqdm progress bar
            results = []

            # Too wide for Datalore notebooks
            # label = ", ".join(p.name for p in combinations[0].searchable_parameters)
            with tqdm(total=len(task_args), desc=f"Searching") as progress_bar:
                progress_bar.set_postfix({"processes": max_workers})
                # Extract results from the parallel task queue
                for task in tm.as_completed():
                    results.append(task.result)
                    progress_bar.update()
        else:
            #
            # Run individual searchers threads
            #
            logger.warning("Doing a multithread grid search - you should not really use this, pass multiprocessing=True instead")

            task_args = [(grid_search_worker, universe, c, trading_strategy_engine_version, data_retention, indicator_storage.path) for c in combinations if c not in cached_results]

            executor = futureproof.ThreadPoolExecutor(max_workers=max_workers)
            tm = futureproof.TaskManager(executor, error_policy=futureproof.ErrorPolicyEnum.RAISE)

            # Run the checks parallel using the thread pool
            tm.map(run_grid_combination_threaded, task_args)

            # Extract results from the parallel task queue
            results = [task.result for task in tm.as_completed()]

    else:
        # Do single thread - good for debuggers like pdb/ipdb
        #

        logger.info("Doing a single thread grid search")
        task_args = [(grid_search_worker, universe, c, trading_strategy_engine_version, data_retention, indicator_storage.path) for c in combinations]
        iter = itertools.starmap(run_grid_combination_threaded, task_args)

        # Force workers to finish
        results = list(iter)

    duration = datetime.datetime.utcnow() - start
    logger.info("Grid search finished in %s, calculated %d new results", duration, len(results))

    results = list(cached_results.values()) + results
    logger.info("Total %d results", len(results))

    return results


def run_grid_search_backtest(
    combination: GridCombination,
    decide_trades: DecideTradesProtocol | DecideTradesProtocol2 | DecideTradesProtocol4,
    universe: TradingStrategyUniverse,
    create_indicators: CreateIndicatorsProtocolV1 | None = None,
    cycle_duration: Optional[CycleDuration] = None,
    start_at: Optional[datetime.datetime | pd.Timestamp] = None,
    end_at: Optional[datetime.datetime | pd.Timestamp] = None,
    initial_deposit: USDollarAmount = 5000.0,
    trade_routing: Optional[TradeRouting] = None,
    data_delay_tolerance: Optional[pd.Timedelta] = None,
    name: Optional[str] = None,
    routing_model: Optional[TradingStrategyEngineVersion] = None,
    trading_strategy_engine_version: Optional[str] = None,
    cycle_debug_data: dict | None = None,
    parameters: StrategyParameters | None = None,
    indicator_storage: IndicatorStorage | None = None,
    execution_context=standalone_backtest_execution_context,
) -> GridSearchResult:
    assert isinstance(universe, TradingStrategyUniverse), f"Received {universe}"

    if name is None:
        name = combination.get_label()

    if cycle_debug_data is None:
        cycle_debug_data = {}

    universe_range = universe.data_universe.candles.get_timestamp_range()
    if not start_at:
        start_at = universe_range[0]

    if not end_at:
        end_at = universe_range[1]

    if isinstance(start_at, datetime.datetime):
        start_at = pd.Timestamp(start_at)

    if isinstance(end_at, datetime.datetime):
        end_at = pd.Timestamp(end_at)

    if not cycle_duration:
        cycle_duration = CycleDuration.from_timebucket(universe.data_universe.candles.time_bucket)
    else:
        assert isinstance(cycle_duration, CycleDuration)

    if not routing_model:
        routing_model = BacktestRoutingIgnoredModel(universe.get_reserve_asset().address)

    # Run the test
    try:
        state, universe, debug_dump = run_backtest_inline(
            name=name,
            start_at=start_at.to_pydatetime(),
            end_at=end_at.to_pydatetime(),
            client=None,
            cycle_duration=cycle_duration,
            decide_trades=decide_trades,
            create_trading_universe=None,
            create_indicators=create_indicators,
            indicator_combinations=combination.indicators,
            universe=universe,
            initial_deposit=initial_deposit,
            reserve_currency=None,
            trade_routing=TradeRouting.user_supplied_routing_model,
            routing_model=routing_model,
            allow_missing_fees=True,
            data_delay_tolerance=data_delay_tolerance,
            engine_version=trading_strategy_engine_version,
            parameters=parameters,
            indicator_storage=indicator_storage,
            grid_search=True,
            execution_context=execution_context,
        )
    except Exception as e:
        # Report to the notebook which of the grid search combinations is a problematic one
        raise
        raise RuntimeError(f"Running a grid search combination failed:\n{combination}\nThe original exception was: {e}") from e

    # Portfolio performance
    equity = calculate_equity_curve(state)
    returns = calculate_returns(equity)
    metrics = calculate_advanced_metrics(
        returns,
        mode=AdvancedMetricsMode.full,
        periods_per_year=cycle_duration.get_yearly_periods(),
        convert_to_daily=True,
        display=False,
    )

    # Trade stats
    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()

    res = GridSearchResult(
        combination=combination,
        state=state,
        summary=summary,
        metrics=metrics,
        universe_options=universe.options,
        equity_curve=equity,
        returns=returns,
        initial_cash=state.portfolio.get_initial_cash(),
    )

    # Double check we have not broken QuantStats again
    # and somehow outputting string values
    assert type(res.get_cagr()) in (float, int), f"We got {type(res.get_cagr())} for {res.get_cagr()}"
    assert type(res.get_sharpe()) in (float, int)

    return res


def pick_grid_search_result(results: List[GridSearchResult], **kwargs) -> Optional[GridSearchResult]:
    """Pick one combination in the results.

    Example:

    .. code-block:: python

        # Pick a result of a single grid search combination
        # and examine its trading metrics
        sample = pick_grid_search_result(
            results,
            stop_loss_pct=0.9,
            slow_ema_candle_count=7,
            fast_ema_candle_count=2)
        assert sample.summary.total_positions == 2

    :param result:
        Output from :py:func:`perform_grid_search`

    :param kwargs:
        Grid parameters to match

    :return:
        The grid search result with the matching parameters or None if not found

    """

    for r in results:
        # Check if this result matches all the parameters
        match = all([p.value == kwargs[p.name] for p in r.combination.parameters])
        if match:
            return r

    return None


def pick_best_grid_search_result(
        results: List[GridSearchResult],
        key: Callable=lambda r: r.summary.return_percent,
        highest=True,
) -> Optional[GridSearchResult]:
    """Pick the best combination in the results based on one metric.

    Use trading metrics or performance metrics for the selection.

    Example:


    .. code-block:: python

        sample = pick_best_grid_search_result(
            results,
            key=lambda r: r.metrics.loc["Max Drawdown"][0])
            assert sample is not None

    :param result:
        Output from :py:func:`perform_grid_search`

    :param key:
        Lambda function to extract the value to compare from the data.

        If not given use cumulative return.

    :param highest:
        If true pick the highest value, otherwise lowest.

    :return:
        The grid search result with the matching parameters or None if not found

    :return:
        The grid search result with the matching parameters or None if not found

    """

    current_best = -10**27 if highest else 10**27
    match = None

    for r in results:
        # Check if this result matches all the parameters
        value = key(r)

        if value in (None, np.NaN):
            # No result for this combination
            continue

        if highest:
            if value > current_best:
                match = r
                current_best = value
        else:
            if value < current_best:
                match = r
                current_best = value

    return match


def save_forked_multiprocess_strategy_universe(strategy_universe):
    """Prepare handing over the strategy universe for the child processes."""
    global _universe
    _universe = strategy_universe


def load_multiprocess_strategy_universe():
    """Pop the strategy universe data from the parent process."""
    global _universe
    assert _universe, "The strategy_universe process global is not set - did you set this variable before forking from the parent process"
    return _universe


def save_disk_multiprocess_strategy_universe(strategy_universe: TradingStrategyUniverse):
    """Prepare handing over the strategy universe for the child processes.

    For joblib Loki backend that lacks process initialisers.

    See https://github.com/joblib/joblib/pull/1525

    There is no clean up, this is just a workaround of joblib's shortcomings.
    """
    temp_dir = tempfile.mkdtemp()
    fname = os.path.join(temp_dir, strategy_universe.get_cache_key()) + ".pickle"
    with open(fname, "wb") as out:
        pickle.dump(strategy_universe, out)
    return fname


def initialise_multiprocess_strategy_universe_from_disk(fname: str) -> TradingStrategyUniverse:
    """Pop the strategy universe data from the parent process.

    - See :py:func:`save_disk_multiprocess_strategy_universe`

    - Only load once per child process

    - Load the universe from disk and cache in-process
    """

    assert type(fname) == str, f"Got {fname}"

    global _universe

    if _universe is None:
        with open(fname, "rb") as inp:
            _universe = pickle.load(inp)

    return _universe

#: Process global stored universe for multiprocess workers
#:
#: See :py:func:`save_forked_multiprocess_strategy_universe` and :py:func:`load_forked_multiprocess_strategy_universe`
#:
_universe: Optional[TradingStrategyUniverse] = None

_process_pool: concurrent.futures.process.ProcessPoolExecutor | None = None

def _process_init(pickled_universe):
    """Child worker process initialiser."""
    # Transfer ove the universe to the child process
    global _universe
    _universe = pickle.loads(pickled_universe)


def _handle_sigterm(*args):
    # TODO: Despite all the effort, this does not seem to work with Visual Studio Code's Interrupt Kernel button
    processes: List[Process] = list(_process_pool._processes.values())
    _process_pool.shutdown()
    for p in processes:
        p.kill()
    sys.exit(1)


def get_grid_search_result_path(
    #notebook_name: str | Callable = get_notebook_name,
    notebook_name: str | Callable | None = None,
) -> Path:
    """Get a path where to stget_grid_search_result_pathore the grid seach results.

    - Used in grid search notebooks

    - Each notebook gets its own storage folder for grid search results somewhere

    - Have some logic to get a persistent path in Datalore notebook environments

    :param notebook_name:

        Override the name.

        If not given try to resolve automatically, but this is fragile.

        The name of your notebook file.
        E.g. "v19-candle-search".

    """

    if callable(notebook_name):
        notebook_name = notebook_name()
        notebook_name = os.path.basename(notebook_name)

    assert type(notebook_name) == str
    path = os.path.expanduser(f"~/.cache/trading-strategy/grid-search/{notebook_name}")
    os.makedirs(path, exist_ok=True)
    return Path(path)
