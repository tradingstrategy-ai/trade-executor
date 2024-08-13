"""Machine learning based optimiser for the strategy parameters.

- Users scikit-optimize library

- Similar as grid seacrh

- Instead of searching all parameter combinations, search only some based on an algorithm

"""
import dataclasses
import datetime
import inspect
import logging
import os
import typing
import warnings
from _decimal import Decimal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type

import psutil
from joblib import Parallel, delayed
from skopt import space, Optimizer
from skopt.space import Dimension

from tqdm_loggable.auto import tqdm


from tradeexecutor.backtest.grid_search import GridCombination, GridSearchDataRetention, GridSearchResult, save_disk_multiprocess_strategy_universe, initialise_multiprocess_strategy_universe_from_disk, run_grid_search_backtest, \
    get_grid_search_result_path, GridParameter
from tradeexecutor.cli.log import setup_notebook_logging
from tradeexecutor.strategy.engine_version import TradingStrategyEngineVersion
from tradeexecutor.strategy.execution_context import ExecutionContext, grid_search_execution_context, scikit_optimizer_context
from tradeexecutor.strategy.pandas_trader.indicator import DiskIndicatorStorage, CreateIndicatorsProtocol, IndicatorStorage
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol4
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.cpu import get_safe_max_workers_count


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OptimiserSearchResult:
    """Single optimiser search result value.

    This is used in different contextes

    - Return value from :py:class:`SearchFunction`

    - Passing data from child worker to the parent process

    - Passing data from :py:func:`perform_optimisation` to the notebook
    """

    #: The raw value of the search function we are optimising
    value: float

    #: Did we flip this value to negative because we are looking for a minimised result
    negative: bool

    #: For which grid combination this result was
    #:
    #: This is the key to load the child worker produced data from the disk in the parent process
    #:
    combination: GridCombination | None = None

    # We do not pass the result directly to child process but let the parent process to read it from the disk
    #:
    #: Call :py:meth:`hydrate` to make this data available.
    #:
    result: GridSearchResult | None = None

    #: Did we filter out this result
    #:
    #: See `result_filter` in :py:func:`perform_optimisation`
    #:
    filtered: bool = False

    def __repr__(self):
        return f"<OptimiserSearchResult {self.combination} = {self.get_original_value()}>"

    # Allow to call min(list[OptimiserSearchResult]) to find the best value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def get_original_value(self) -> float:
        """Get the original best search value.

        - Flip off the extra minus sign if we had to add it
        """
        if self.negative:
            return -self.value
        else:
            return self.value

    def hydrate(self):
        """Load the grid search result data for this result from the disk."""
        self.result = GridSearchResult.load(self.combination)

    def is_valid(self) -> bool:
        """Could not calculate a value."""
        return self.value is not None


class SearchFunction(typing.Protocol):
    """The function definition for the optimiser search function.

    - The search function extracts the result variable we want to optimise
      for from the :py:class:`GridSearchResult`

    Example:

    .. code-block:: python

        # Search for the best CAGR value.
        def optimise_profit(result: GridSearchResult) -> OptimiserSearchResult:
            return OptimiserSearchResult(-result.get_cagr(), negative=True)
    """

    def __call__(self, result: GridSearchResult) -> OptimiserSearchResult:
        """The search function extracts the parm

        :param result:
            The latest backtest result as grid search result object.

        :return:
            Value of the function to optimise.

            Smaller is better. E.g. if you want to optimise profit, return negative profit.
        """


class ResultFilter(typing.Protocol):
    """Apply to drop bad optimiser result.

    - E.g. by :py:class:`MinTradeCountFilter`
    """
    def __call__(self, result: GridSearchResult) -> bool:
        """Return true if the optimiser result satifies our filter criteria."""
        pass

@dataclass(frozen=True, slots=True)
class OptimiserResult:
    """The outcome of the optimiser run.

    - Contains all the grid search results we generated during the run
    """

    #: The parameters we searched
    #:
    #: Both fixed and search space.
    #:
    parameters: StrategyParameters

    # Where we store the grid search results data
    result_path: Path

    #: Different grid search results
    #:
    #: Sortd from the best to the worst.
    #:
    results: list[OptimiserSearchResult]

    #: Where did we store precalculated indicator files.
    #:
    #: Allows to peek into raw indicator data if we need to.
    indicator_storage: DiskIndicatorStorage

    @staticmethod
    def determine_result_path(result_path: Path | None, parameters: StrategyParameters) -> Path:
        """Generate result path or use the exiting."""
        if result_path:
            assert isinstance(result_path, Path)
            return result_path

        name_hint = parameters.get("id") or parameters.get("name")
        assert name_hint, f"Cannot determine parameter id or name for StrategyParameters, needed for optimiser search result storage: {parameters}"

        return get_grid_search_result_path(name_hint)

    def get_combination_count(self) -> int:
        """How many combinations we searched in this optimiser run."""
        return len(self.results)

    def get_results_as_grid_search_results(self) -> list[GridSearchResult]:
        """Get all search results as grid search results list for the analysis.
        
        - Any results that are marked as filtered away are not returned
        """
        return [r.result for r in self.results if not r.filtered]

    def get_cached_count(self) -> int:
        """How many of the results were directly read from the disk and not calculated on this run."""
        return len([r for r in self.results if r.result.cached])

    def get_failed_count(self) -> int:
        """How many backtest runs failed with an exception."""
        return len([r for r in self.results if r.result.exception is not None])

    def get_filtered_count(self) -> int:
        """How many of the results were filtered out by result filter.

        See :py:func:`perform_optimisation`
        """
        return len([r for r in self.results if r.filtered])


class ObjectiveWrapper:
    """Middleware between Optimiser and TS frameworks.

    - Passes the fixed data to the child workers
    """

    def __init__(
        self,
        pickled_universe_fname: str,
        search_func: SearchFunction,
        parameters: StrategyParameters,
        trading_strategy_engine_version: str,
        decide_trades: DecideTradesProtocol4,
        create_indicators: CreateIndicatorsProtocol,
        indicator_storage: IndicatorStorage,
        result_path: Path,
        search_space: list[Dimension],
        real_space_rounding: Decimal,
        log_level: int,
        result_filter: ResultFilter,
        draw_visualisation: bool,
        ignore_wallet_errors: bool,
    ):
        self.search_func = search_func
        self.pickled_universe_fname = pickled_universe_fname
        self.parameters = parameters
        self.decide_trades = decide_trades
        self.create_indicators = create_indicators
        self.indicator_storage = indicator_storage
        self.trading_strategy_engine_version = trading_strategy_engine_version
        self.result_path = result_path
        self.search_space = search_space
        self.real_space_rounding = real_space_rounding
        self.log_level = log_level
        self.result_filter = result_filter
        self.filtered_result_value = 0
        self.draw_visualisation = draw_visualisation
        self.ignore_wallet_errors = ignore_wallet_errors

    def __call__(
        self,
        result_index: int,
        args: list[str | int | float],
    ):
        """This function is at the entry point of a child worker process.

        - Sets up a backtest within a child process,
          so that the results are stored as :py:class:`GridSearchResult`

        :param args:
            The current search space values from the optimiser.
        """

        assert type(result_index) == int
        assert type(args) == list, f"Expected list of args, got {args}"

        if self.log_level:
            setup_notebook_logging(self.log_level, show_process=True)

        logger.info("Starting optimiser batch %d in child worker %d", result_index, os.getpid())

        strategy_universe = initialise_multiprocess_strategy_universe_from_disk(self.pickled_universe_fname)

        combination = create_grid_combination(
            self.result_path,
            self.search_space,
            result_index,
            args,
            self.real_space_rounding,
        )

        if GridSearchResult.has_result(combination):
            # We have run this search point before and can load from the cache
            result = GridSearchResult.load(combination)
        else:
            # We are running this search point for the first time

            # Merge the current search values with the fixed parameter
            merged_parameters = StrategyParameters.from_dict(self.parameters)
            merged_parameters.update({p.name: p.get_computable_value() for p in combination.parameters})

            # Make sure we drag the engine version along
            execution_context = dataclasses.replace(scikit_optimizer_context)
            execution_context.engine_version = self.trading_strategy_engine_version
            execution_context.force_visualisation = self.draw_visualisation

            result = run_grid_search_backtest(
                combination,
                decide_trades=self.decide_trades,
                universe=strategy_universe,
                create_indicators=self.create_indicators,
                parameters=merged_parameters,
                indicator_storage=self.indicator_storage,
                trading_strategy_engine_version=self.trading_strategy_engine_version,
                execution_context=execution_context,
                max_workers=1,  # Don't allow this child process to create its own worker pool for indicator calculations
                initial_deposit=merged_parameters["initial_cash"],
                ignore_wallet_errors=self.ignore_wallet_errors,
            )
            
            result.save(include_state=True)

        if getattr(result, "exception", None) is None:  # Legacy pickle compat
            opt_result = self.search_func(result)

            # Apply result filter and zero out the value for optimiser if needed
            if not self.result_filter(result):
                opt_result.value = self.filtered_result_value  # Zero out the actual value so optimiser knows this is a bad path
                opt_result.filtered = True  # Tell the caller that this is a filtered result and should not be displayed by default

        else:
            # The backtest crashed with an exception,
            # likely OutOfBalance
            opt_result = OptimiserSearchResult(self.filtered_result_value, negative=False)

        opt_result.combination = combination
        logger.info("Optimiser for combination %s resulted to %s, exception is %s, exiting child process", combination, opt_result.value, result.exception)
        return opt_result


def get_optimised_dimensions(parameters: StrategyParameters) -> list[space.Dimension]:
    """Get all dimensions we are going to search."""
    return [p for p in parameters.values() if isinstance(p, space.Dimension)]



def prepare_optimiser_parameters(
    param_class: Type,
) -> StrategyParameters:
    """Optimised parameters must be expressed using scikit-optimise internals.

    - The parameters class must contain at least one parameter that is subclass of :py:class:`Space`

    - Unlike in grid search, `Space` parameters are passed as is, and not preprocessed
      for the number of combinations, as we use iterator count to search through the search space

    - Assign a name to each dimension

    :param param_class:
        Parameters class in your notebook

    param warn_float:
        Warn about unbounded float in the parameters, as it will hurt the performance, because we cannot cache results for arbitrary values.

    :return:
        Validated parameters.
    """

    assert inspect.isclass(param_class), f"Expected class, got {type(param_class)}"

    parameters = StrategyParameters.from_class(param_class)
    assert any([isinstance(v, space.Dimension) for v in parameters.values()]), f"Not a single scikit-optimiser Dimemsion instance detected in {parameters}"

    # Assign names to the dimensions
    for key, value in parameters.iterate_parameters():
        if isinstance(value, space.Dimension):
            value.name = key

    return parameters


def create_grid_combination(
    result_path: Path,
    search_space: list[Dimension],
    index: int,
    search_args: list[str | float | int],
    real_space_rounding: Decimal,
) -> GridCombination:
    """Turn scikit-optimise search arguments to a grid combination.

    GridCombination allows us to store the results on the disk
    and match the data for the later analysis and caching.
    """

    # Convert scikit-optimise determined search space parameters
    # for this round to GridParameters, which are used as a cache key

    assert type(search_space) == list
    assert type(search_args) == list
    assert len(search_space) == len(search_args), f"Got {len(search_space)}, {len(search_args)}"

    parameters = []
    for dim, val in zip(search_space, search_args):

        # Round real numbers in the search space
        # to some manageable values we can use in filenames
        if isinstance(dim, space.Real):
            val = Decimal(val).quantize(real_space_rounding)

        assert dim.name, f"Dimension unnamed: {dim}. Did you call prepare_optimiser_parameters()? Do not call StrategyParameters.from_class()."

        p = GridParameter(name=dim.name, value=val, single=True, optimise=True)
        parameters.append(p)

    combination = GridCombination(
        index=index,
        result_path=result_path,
        parameters=tuple(parameters),
    )
    return combination



class MinTradeCountFilter:
    """Have a minimum threshold of a created trading position count.

    Avoid strategies that have a single or few random successful open and close.
    """

    def __init__(self, min_trade_count):
        self.min_trade_count = min_trade_count

    def __call__(self, result: GridSearchResult) -> bool:
        """Return true if the trade count threshold is reached."""
        return result.get_trade_count() > self.min_trade_count


def perform_optimisation(
    iterations: int,
    search_func: SearchFunction,
    decide_trades: DecideTradesProtocol4,
    create_indicators: CreateIndicatorsProtocol,
    strategy_universe: TradingStrategyUniverse,
    parameters: StrategyParameters,
    max_workers: int | Callable = get_safe_max_workers_count,
    trading_strategy_engine_version: TradingStrategyEngineVersion = "0.5",
    data_retention: GridSearchDataRetention = GridSearchDataRetention.metrics_only,
    execution_context: ExecutionContext = grid_search_execution_context,
    indicator_storage: DiskIndicatorStorage | None = None,
    result_path: Path | None = None,
    min_batch_size=4,
    real_space_rounding=Decimal("0.01"),
    timeout: float = 10 * 60,
    log_level: int | None=None,
    result_filter:ResultFilter = MinTradeCountFilter(50),
    bad_result_value=0,
    draw_visualisation=False,
    ignore_wallet_errors=True,
) -> OptimiserResult:
    """Search different strategy parameters using an optimiser.

    - Use scikit-optimize to find the optimal strategy parameters.

    - The results of previous runs are cached on a disk using the same cache as grid search,
      though the cache is not as effective as each optimise run walks randomly around.

    - Unlike in grid search, indicators are calculated in the child worker processes,
      because we do not know what indicator values we are going to search upfront.
      There might a race condition between different child workers to calculate and save
      indicator data series, but it should not matter as cache writes are atomic.

    - This will likely consume gigabytes of disk space

    :param iterations:
        How many iteratiosn we will search

    :param search_func:
        The function that will rank the optimise iteration results.

        See :py:func:`optimise_profit` and :py:func:`optimise_sharpe`,
        but can be any of your custom functions.

    :param parameters:
        Prepared search space and fixed parameters.

        See :py:func:`prepare_grid_combinations`

    :param trading_strategy_engine_version:
        Which version of engine we are using.

    :param result_path:
        Where to store the grid search results

    :param real_space_rounding:
        For search dimensions that are Real numbers, round to this accuracy.

        We need to write float values as cache filename parameters and too high float accuracy causes
        too long strings breaking filenames.

    :param min_batch_size:
        How many points we ask for the batch processing from the scikit-optimiser once.

        You generally do not need to care about this.

    :param timeout:
        Maximum timeout for a joblib.Parallel for a single worker process for a single iteration.

        Will interrupt hung child processes if a backtest takes forever.

        Increase the number if you are getting issues.

    :param log_level:
        Control for the diagnostics.

        E.g. set to `logging.INFO`.

    :param result_filter:
        Filter bad strategies.

        Try to avoid strategies that open too few trades or are otherwise not viable.

        Strategies that do not pass may be still returned, but the result `filtered` flag is set on.

    :param bad_result_value:
        What placeholder value we use for the optimiser when `result_filter` does not like the outcome.

    :param draw_visualisation:
        Draw and collect visualisation data during the backtest execution when optimising.

        This will slow down optimisation: use this only if you really want to collect the data.

        In `decide_trades()` function, `input.is_visualisation_enabled()` will return True.

    :param ignore_wallet_errors:
        Ignore `OutOfBalance` exceptions.

        In the case if our backtest fails due to making trades for which we do not have money,
        instead of crashing the whole strategy run, mark down these backtest results
        as zero profit.

    :return:
        Grid search results for different combinations.

    """

    assert iterations > 0
    assert isinstance(parameters, StrategyParameters), f"Bad parameters: {type(parameters)}"
    assert callable(search_func), f"Search function is not callable: {search_func}"

    if log_level is not None:
        setup_notebook_logging(log_level)

    start = datetime.datetime.utcnow()

    # Resolve CPU count
    if callable(max_workers):
        max_workers = max_workers()

    search_space = get_optimised_dimensions(parameters)

    result_path = OptimiserResult.determine_result_path(result_path, parameters)

    if indicator_storage is None:
        indicator_storage = DiskIndicatorStorage.create_default(strategy_universe)

    logger.info(
        "Performing an optimisation of %d iterations on %d dimensions, with %d workers, data retention policy is %s",
        iterations,
        len(search_space),
        max_workers,
        data_retention.name,
    )

    # Set up scikit-optimizer for Gaussian Process (Bayes)
    optimizer = Optimizer(
        dimensions=search_space,
        random_state=1,
        base_estimator='gp'  # gp stands for Gaussian Process
    )

    # Set up a joblib processor using multiprocessing (forking)
    # With hot fix https://stackoverflow.com/a/67860638/315168
    current_process = psutil.Process()
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    worker_processor = Parallel(
        n_jobs=max_workers,
        backend="loky",
        timeout=timeout,
        max_nbytes=40*1024*1024,  # Allow passing 40 MBytes for child processes
    )

    batch_size = max(min_batch_size, max_workers)  # Make sure we have some batch size even if running single CPU

    # We do some disk saving trickery to avoid pickling super large
    # trading universe data as as function argument every time a new search is performed
    pickled_universe_fname = save_disk_multiprocess_strategy_universe(strategy_universe)

    objective = ObjectiveWrapper(
        pickled_universe_fname=pickled_universe_fname,
        search_func=search_func,
        parameters=parameters,
        trading_strategy_engine_version=trading_strategy_engine_version,
        decide_trades=decide_trades,
        indicator_storage=indicator_storage,
        result_path=result_path,
        create_indicators=create_indicators,
        search_space=search_space,
        real_space_rounding=real_space_rounding,
        result_filter=result_filter,
        log_level=log_level,
        draw_visualisation=draw_visualisation,
        ignore_wallet_errors=ignore_wallet_errors,
    )

    name = parameters.get("name") or parameters.get("id") or "backtest"

    result_index = 1

    all_results: list[OptimiserSearchResult] = []

    print(f"Optimiser search result cache is {result_path}\nIndicator cache is {indicator_storage.get_disk_cache_path()}")

    with tqdm(total=iterations, desc=f"Optimising {name}, search space is {len(search_space)} variables, using {max_workers} CPUs") as progress_bar:
        for i in range(0, iterations):
            with warnings.catch_warnings():
                # Ignore warning when we too close to optimal:
                # UserWarning: The objective has been evaluated at point [7, 10] before, using random point [23, 21]
                warnings.simplefilter("ignore")
                x = optimizer.ask(n_points=batch_size)  # x is a list of n_points points

            # Prepare a patch of search space params to be send to the worker processes
            if max_workers > 1:
                batch = []
                for args in x:
                    batch.append(delayed(objective)(result_index, args))  # Funny joblib way to construct parallel task
                    result_index += 1

                y = worker_processor(batch)  # evaluate points in parallel
                y = list(y)
            else:
                # Single thread path
                y = []
                for args in x:
                    y.append(objective(result_index, args))
                    logger.info("Got result for %s", args)

                logger.info("Iteration %d multiprocessing complete", i)

            filtered_y = [result.value for result in y]

            # Tell optimiser how well the last batch did
            # by matching each x points to their raw optimise y value,
            # and unpack our data structure
            optimizer.tell(x, filtered_y)

            result: OptimiserSearchResult
            for result in y:
                result.hydrate()  # Load grid search result data from the disk
                all_results.append(result)

            best_so_far: OptimiserSearchResult = min([r for r in all_results if not r.filtered], default=None)  # Get the best value for "bull days matched"
            progress_bar.set_postfix({"Best so far": best_so_far.get_original_value() if best_so_far else "-"})
            progress_bar.update()

    # Cleans up Loky backend hanging in pytest
    # https://stackoverflow.com/a/77130505/315168
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    terminate_set  = subproc_after - subproc_before
    logger.info("Terminating bad Loky workers %d", len(terminate_set))
    for subproc in terminate_set:
        psutil.Process(subproc).terminate()

    logger.info("The best result for the optimiser value was %s", best_so_far)

    all_results.sort()
    duration = datetime.datetime.utcnow() - start
    logger.info("Optimiser search finished in %s, calculated %d results", duration, len(all_results))
    return OptimiserResult(
        parameters=parameters,
        result_path=result_path,
        results=all_results,
        indicator_storage=indicator_storage,
    )
