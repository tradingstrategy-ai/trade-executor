"""Perform a grid search ove strategy parameters to find optimal parameters."""
import concurrent
import datetime
import itertools
import logging
import os
import pickle
import shutil
import signal
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from inspect import isclass
from multiprocessing import Process
from pathlib import Path
from typing import Protocol, Dict, List, Tuple, Any, Optional, Collection, Callable
import concurrent.futures.process
from packaging import version

import numpy as np
import pandas as pd
import futureproof
from web3.datastructures import ReadableAttributeDict

from tradeexecutor.strategy.engine_version import TradingStrategyEngineVersion
from tradeexecutor.strategy.universe_model import UniverseOptions

try:
    from tqdm_loggable.auto import tqdm
except ImportError:
    # tqdm_loggable is only available at the live execution,
    # but fallback to normal TQDM auto mode
    from tqdm.auto import tqdm

from tradeexecutor.analysis.advanced_metrics import calculate_advanced_metrics, AdvancedMetricsMode
from tradeexecutor.analysis.trade_analyser import TradeSummary, build_trade_analysis
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol, DecideTradesProtocol2, StrategyParameters, DecideTradesProtocol3
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns


logger = logging.getLogger(__name__)


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

    def __post_init__(self):
        pass

    def __hash__(self):
        return hash((self.name, self.value))

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def to_path(self) -> str:
        """"""
        value = self.value

        if isinstance(value, Enum):
            return f"{self.name}={self.value.value}"
        elif type(value) in (float, int, str):
            return f"{self.name}={self.value}"
        if value is None:
            return f"{self.name}=none"
        else:
            raise NotImplementedError(f"We do not support filename conversion for value {type(value)}={value}")


@dataclass()
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

    def __post_init__(self):
        assert len(self.parameters) > 0
        assert isinstance(self.result_path, Path), f"Expected Path, got {type(self.result_path)}"
        assert self.result_path.exists() and self.result_path.is_dir(), f"Not a dir: {self.result_path}"

    def __hash__(self):
        return hash(self.parameters)

    def __eq__(self, other):
        return self.parameters == other.parameters

    @property
    def searchable_parameters(self) -> List[GridParameter]:
        """Get all parameters that are searchable.

        Searchable parameters have two or more values.
        """
        return [p for p in self.parameters if not p.single]

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
        assert isinstance(self.get_relative_result_path(), Path)

    def as_dict(self) -> dict:
        """Get as kwargs mapping."""
        return {p.name: p.value for p in self.parameters}

    def get_label(self) -> str:
        """Human readable label for this combination"""
        return f"#{self.index}, " + ", ".join([f"{p.name}={p.value}" for p in self.searchable_parameters])

    def destructure(self) -> List[Any]:
        """Open parameters dict.

        This will return the arguments in the same order you pass them to :py:func:`prepare_grid_combinations`.
        """
        return [p.value for p in self.parameters]

    def to_strategy_parameters(self) -> StrategyParameters:
        return StrategyParameters(self.as_dict())


@dataclass(slots=True, frozen=False)
class GridSearchResult:
    """Result for one grid combination."""

    #: For which grid combination this result is
    combination: GridCombination

    #: The full back test state
    state: State

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

    def get_label(self) -> str:
        """Get name for this result for charts."""
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
        with open(base_path.joinpath("result.pickle"), "wb") as out:
            pickle.dump(self, out)


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

    :return:
        List of all combinations we need to search through
    """

    assert isinstance(result_path, Path)

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
        assert isinstance(values, Collection), f"Expected list, got: {values}"
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
    return combinations




def _run_v04(
    decide_trades: DecideTradesProtocol3,
    universe: TradingStrategyUniverse,
    combination: GridCombination,
    trading_strategy_engine_version: TradingStrategyEngineVersion,
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
    assert initial_cash, f"Strategy parameters lack initial_deposit, we have {list(parameters.keys())}"

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
    )


def run_grid_combination_threaded(
    grid_search_worker: GridSearchWorker | DecideTradesProtocol3,
    universe: TradingStrategyUniverse,
    combination: GridCombination,
    trading_strategy_engine_version: TradingStrategyEngineVersion,
):
    """Threared runner.

    Universe is passed as argument.
    """
    if GridSearchResult.has_result(combination):
        result = GridSearchResult.load(combination)
        return result

    if version.parse(trading_strategy_engine_version) >= version.parse("0.4"):
        # New style runner
        result = _run_v04(grid_search_worker, universe, combination, trading_strategy_engine_version)
    else:
        # Legacy path
        result = grid_search_worker(universe, combination)


    # Cache result for the future runs
    result.save()

    return result


def run_grid_combination_multiprocess(
    grid_search_worker: GridSearchWorker | DecideTradesProtocol3,
    combination: GridCombination,
    trading_strategy_engine_version: TradingStrategyEngineVersion,
):
    """Mutltiproecss runner.

    Universe is passed as process global.
    """

    from tradeexecutor.monkeypatch import cloudpickle_patch  # Enable pickle patch that allows multiprocessing in notebooks

    global _universe

    universe = _universe

    if GridSearchResult.has_result(combination):
        result = GridSearchResult.load(combination)
        return result

    if version.parse(trading_strategy_engine_version) >= version.parse("0.4"):
        # New style runner
        result = _run_v04(grid_search_worker, universe, combination, trading_strategy_engine_version)
    else:
        # Legacy path
        result = grid_search_worker(universe, combination)

    result.process_id = os.getpid()

    # Cache result for the future runs
    result.save()

    return result


@_hide_warnings
def perform_grid_search(
    grid_search_worker: GridSearchWorker | DecideTradesProtocol3,
    universe: TradingStrategyUniverse,
    combinations: List[GridCombination],
    max_workers=16,
    clear_cached_results=False,
    stats: Optional[Counter] = None,
    multiprocess=False,
    trading_strategy_engine_version: TradingStrategyEngineVersion="0.3",
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

    logger.info("Performing a grid search over %s combinations, with %d threads",
                len(combinations),
                max_workers,
                )

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
            task_args = [(grid_search_worker, c, trading_strategy_engine_version) for c in combinations]

            # Set up a signal handler to stop child processes on quit
            _process_pool_executor = executor._executor
            signal.signal(signal.SIGTERM, _handle_sigterm)

            # Run the tasks
            tm.map(run_grid_combination_multiprocess, task_args)

            # Track the child process completion using tqdm progress bar
            results = []
            label = ", ".join(p.name for p in combinations[0].searchable_parameters)
            with tqdm(total=len(task_args), desc=f"Grid searching using {max_workers} processes: {label}") as progress_bar:
                # Extract results from the parallel task queue
                for task in tm.as_completed():
                    results.append(task.result)
                    progress_bar.update()

        else:
            #
            # Run individual searchers threads
            #

            task_args = [(grid_search_worker, universe, c, trading_strategy_engine_version) for c in combinations]
            logger.info("Doing a multithread grid search")
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
        task_args = [(grid_search_worker, universe, c, trading_strategy_engine_version) for c in combinations]
        iter = itertools.starmap(run_grid_combination_threaded, task_args)

        # Force workers to finish
        results = list(iter)

    duration = datetime.datetime.utcnow() - start
    logger.info("Grid search finished in %s", duration)

    return results


def run_grid_search_backtest(
    combination: GridCombination,
    decide_trades: DecideTradesProtocol | DecideTradesProtocol2,
    universe: TradingStrategyUniverse,
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
    state, universe, debug_dump = run_backtest_inline(
        name=name,
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=cycle_duration,
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=universe,
        initial_deposit=initial_deposit,
        reserve_currency=None,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        allow_missing_fees=True,
        data_delay_tolerance=data_delay_tolerance,
        engine_version=trading_strategy_engine_version,
        parameters=parameters,
    )

    analysis = build_trade_analysis(state.portfolio)
    equity = calculate_equity_curve(state)
    returns = calculate_returns(equity)
    metrics = calculate_advanced_metrics(
        returns,
        mode=AdvancedMetricsMode.full,
        periods_per_year=cycle_duration.get_yearly_periods(),
    )
    summary = analysis.calculate_summary_statistics()

    return GridSearchResult(
        combination=combination,
        state=state,
        summary=summary,
        metrics=metrics,
        universe_options=universe.options,
        equity_curve=equity,
        returns=returns,
    )


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



#: Process global stored universe for multiprocess workers
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


