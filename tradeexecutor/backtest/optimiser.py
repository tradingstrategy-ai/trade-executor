"""Machine learning based optimiser for the strategy parameters.

- Users scikit-optimize library

- Similar as grid seacrh

- Instead of searching all parameter combinations, search only some based on an algorithm

"""
import datetime
import inspect
import logging
import typing
import warnings
from _decimal import Decimal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type

from joblib import Parallel, delayed
from skopt import space, Optimizer
from skopt.space import Dimension

from tqdm_loggable.auto import tqdm


from tradeexecutor.backtest.grid_search import GridSearchWorker, GridCombination, GridSearchDataRetention, GridSearchResult, save_disk_multiprocess_strategy_universe, initialise_multiprocess_strategy_universe_from_disk, run_grid_search_backtest, \
    get_grid_search_result_path, GridParameter
from tradeexecutor.strategy.engine_version import TradingStrategyEngineVersion
from tradeexecutor.strategy.execution_context import ExecutionContext, grid_search_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import DiskIndicatorStorage, CreateIndicatorsProtocol, IndicatorStorage
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol4
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.cpu import get_safe_max_workers_count

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Single optimiser search result value"""

    #: The raw value
    value: float

    #: Did we flip this value to negative  because we are looking for a minimised result
    negative: bool

    #: Full result data (serialised)
    result: GridSearchResult

    def __repr__(self):
        return f"<SearchResult {self.result.get_label()} = {self.get_original_value()}>"

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def get_original_value(self) -> float:
        if self.negative:
            return -self.value
        else:
            return self.value


class SearchFunction(typing.Protocol):
    """The function definition for the optimiser search function.

    - The search function extracts the result variable we want to optimise
      for from the :py:class:`GridSearchResult`
    """

    def __call__(self, result: GridSearchResult) -> SearchResult:
        """The search function extracts the parm

        :param result:
            The latest backtest result as grid search result object.

        :return:
            Value of the function to optimise.

            Smaller is better. E.g. if you want to optimise profit, return negative profit.
        """


@dataclass(frozen=True, slots=True)
class OptimiserResult:
    """The outcome of the optimiser run."""

    #: The parameters we searched
    #:
    #: Both fixed and search space.
    #:
    parameters: StrategyParameters

    # The best found optimised value
    best_result: SearchResult

    # Where we store the grid search results
    result_path: Path

    #: Different grid search results
    #:
    #: From the best to the worst
    #:
    results: list[GridSearchResult]

    # Where did we store precalculated indicator files
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

            # Merge the current search values with the fixed parameter
            merged_parameters = StrategyParameters.from_dict(self.parameters)
            merged_parameters.update({p.name: p.value for p in combination.parameters})

            # We are running this search point for the first time
            result = run_grid_search_backtest(
                combination,
                decide_trades=self.decide_trades,
                universe=strategy_universe,
                create_indicators=self.create_indicators,
                parameters=merged_parameters,
                indicator_storage=self.indicator_storage,
            )

            result.save()

        return self.search_func(result)


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

    parameters = []
    for dim, val in zip(search_space, search_args):

        # Round real numbers in the search space
        # to some manageable values we can use in filenames
        if isinstance(dim, space.Real):
            val = Decimal(val).quantize(real_space_rounding)

        p = GridParameter(name=dim.name, value=val, single=True, optimise=True)
        parameters.append(p)

    combination = GridCombination(
        index=index,
        result_path=result_path,
        parameters=tuple(parameters),
    )
    return combination


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
    real_space_rounding=Decimal("0.00001"),
) -> OptimiserResult:
    """Search different strategy parameters using an optimiser.

    Use scikit-optimize to find the optimal strategy parameters.

    :param combinations:
        Prepared grid combinations.

        See :py:func:`prepare_grid_combinations`

    :param stats:
        If passed, collect run-time and unit testing statistics to this dictionary.

    :param multiprocess:
        Perform the search using multiple CPUs and Python's multiprocessing.

        Set `1` to debug in a single thread.

    :param trading_strategy_engine_version:
        Which version of engine we are using.

    :param result_path:
        Where to store the grid search results

    :param real_space_rounding:
        For search dimensions that are Real numbers, round to this accuracy.

        We need to write float values as cache filename parameters and too high float accuracy causes
        too long strings breaking filenames.

    :return:
        Grid search results for different combinations.

        Sorted so that the first result is the best optimised,
        then decreaseing.

    """

    assert iterations > 0
    assert isinstance(parameters, StrategyParameters), f"Bad parameters: {type(parameters)}"
    assert callable(search_func), f"Search function is not callable: {search_func}"

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
    worker_processor = Parallel(
        n_jobs=max_workers,
        backend="loky",
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
    )

    name = parameters.get("name") or parameters.get("id") or "backtest"

    result_index = 1

    with tqdm(total=iterations, desc=f"Searching the best parameters for {name}") as progress_bar:
        for i in range(0, iterations):
            with warnings.catch_warnings():
                # Ignore warning when we too close to optimal:
                # UserWarning: The objective has been evaluated at point [7, 10] before, using random point [23, 21]
                warnings.simplefilter("ignore")
                x = optimizer.ask(n_points=batch_size)  # x is a list of n_points points

            # Prepare a patch of search space params to be send to the worker processes
            batch = []
            for args in x:
                batch.append(delayed(objective)(result_index, args))  # Funny joblib way to construct parallel task
                result_index += 1

            y = worker_processor(batch)  # evaluate points in parallel
            optimizer.tell(x, y)  # Tell optimiser how well the last batch did

            best_so_far = min(optimizer.yi)  # Get the best value for "bull days matched"
            progress_bar.set_postfix({"Best so far": best_so_far})
            progress_bar.update()

    logger.info("The best result for the optimiser value was %s", best_so_far)

    duration = datetime.datetime.utcnow() - start
    logger.info("Grid search finished in %s, calculated %d new results", duration, len(results))

    results = list(cached_results.values()) + results
    logger.info("Total %d results", len(results))

    return results


def optimise_profit(result: GridSearchResult) -> SearchResult:
    """Search for the best CAGR value."""
    return SearchResult(-result.get_cagr(), negative=True, result=result)