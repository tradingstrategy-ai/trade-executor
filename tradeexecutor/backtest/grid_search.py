"""Perform a grid search ove strategy parameters to find optimal parameters."""
import datetime
import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Dict, List, Tuple, Any

import futureproof

from tradeexecutor.analysis.trade_analyser import TradeSummary
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.state import State
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


@dataclass
class GridParameter:
    name: str
    value: Any

    def __post_init__(self):
        pass

    def __hash__(self):
        return hash((self.name, self.value))

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def to_path(self) -> str:
        """"""
        value = self.value
        if type(value) in (float, int, str):
            return f"{self.name}={self.value}"
        else:
            raise NotImplementedError(f"We do not support filename conversion for value {type(value)}={value}")


@dataclass()
class GridCombination:
    """One combination line in grid search."""

    #: Alphabetically sorted list of parameters
    parameters: Tuple[GridParameter]

    def __hash__(self):
        return hash(self.parameters)

    def __eq__(self, other):
        return self.parameters == other.parameters

    def __post_init__(self):
        """Always sort parameters alphabetically"""
        self.parameters = tuple(sorted(self.parameters, key=lambda p: p.name))

    def get_state_path(self) -> Path:
        """Get the path where the resulting state file is stored."""
        return Path(os.path.join(*[p.to_path() for p in self.parameters]))

    def validate(self):
        """Check arguments can be serialised as fs path."""
        assert isinstance(self.get_state_path(), Path)

    def as_dict(self) -> dict:
        """Get as kwargs mapping."""
        return {p.name: p.value for p in self.parameters}



@dataclass()
class GridSearchResult:
    """Result for one grid combination."""

    combination: GridCombination

    state: State

    summary: TradeSummary



class GridSearcWorker(Protocol):
    """Define how to create different strategy bodies."""


    def __call__(self, universe: TradingStrategyUniverse, **kwargs) -> State:
        """Run a new decide_trades() strategy body based over the serach parameters.

        :param args:
        :param kwargs:
        :return:
        """


def prepare_grid_combinations(parameters: Dict[str, List[Any]]) -> List[GridCombination]:
    """Get iterable search matrix of all parameter combinations."""

    args_lists: List[list] = []
    for name, values in parameters.items():
        args = [GridParameter(name, v) for v in values]
        args_lists.append(args)

    #
    combinations = itertools.product(*args_lists)

    combinations = [GridCombination(c) for c in combinations]
    for c in combinations:
        c.validate()
    return combinations


def run_grid_combination(
        grid_search_worker: GridSearcWorker,
        universe: TradingStrategyUniverse,
        combination: GridCombination,
        result_path: Path,
):

    state_file = result_path.joinpath(combination.get_state_path()).joinpath("state.json")
    if state_file.exists():
        with open(state_file, "rt") as inp:
            data = inp.read()
            state = State.from_json(data)
    else:
        pass

    state = grid_search_worker(
        universe,
        **combination.as_dict())




def perform_grid_search(
        grid_search_worker: GridSearcWorker,
        universe: TradingStrategyUniverse,
        combinations: List[GridCombination],
        result_path: Path,
        max_workers=16,
) -> Dict[GridCombination, GridSearchResult]:
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

    :param result_path:
        A folder where resulting state files will be stored.

    :return
    """

    assert result_path.exists() and result_path.is_dir(), f"Not a dir: {result_path}"

    start = datetime.datetime.utcnow()

    logger.info("Performing a grid search over %s combinations, storing results in %s, with %d threads",
                len(combinations),
                result_path,
                max_workers,
                )

    task_args = [(decide_trades_factory, universe, c, result_path) for c in combinations]

    if max_workers > 1:

        logger.info("Doing a multiprocess grid search")
        # Do a parallel scan for the maximum speed
        #
        # Set up a futureproof task manager
        #
        # For futureproof usage see
        # https://github.com/yeraydiazdiaz/futureproof
        executor = futureproof.ThreadPoolExecutor(max_workers=max_workers)
        tm = futureproof.TaskManager(executor, error_policy=futureproof.ErrorPolicyEnum.RAISE)

        # Run the checks parallel using the thread pool
        tm.map(run_grid_combination, task_args)

        # Extract results from the parallel task queue
        results = [task.result for task in tm.as_completed()]

    else:
        logger.info("Doing a single thread grid search")
        # Do single thread - good for debuggers like pdb/ipdb
        #
        iter = itertools.starmap(run_grid_combination, task_args)

        # Force workers to finish
        results = list(iter)

    duration = datetime.datetime.utcnow() - start
    logger.info("Grid search finished in %s", duration)
