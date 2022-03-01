import datetime
import runpy
from contextlib import AbstractContextManager
from pathlib import Path
import logging
from typing import Callable, Optional

from tradeexecutor.strategy.runner import StrategyRunner, Dataset
from tradingstrategy.client import Client
from tradingstrategy.universe import Universe

logger = logging.getLogger(__name__)


#: What is our Python entry point for strategies
FACTORY_VAR_NAME = "strategy_executor_factory"


class BadStrategyFile(Exception):
    pass


def import_strategy_file(path: Path) -> Callable:
    """Loads a strategy module and returns its factor function."""
    logger.info("Importing strategy %s", path)

    assert isinstance(path, Path)

    strategy_exports = runpy.run_path(path)

    strategy_runner = strategy_exports.get(FACTORY_VAR_NAME)
    if strategy_runner is None:
        raise BadStrategyFile(f"{path} Python module does not declare {FACTORY_VAR_NAME} module variable")

    return strategy_runner


def bootstrap_strategy(
        client: Client,
        timed_task_context_manager: AbstractContextManager,
        path: Path,
        now_: datetime.datetime,
        lookback: Optional[datetime.timedelta]=None, **kwargs) -> [Dataset, Universe, StrategyRunner]:
    """Bootstrap a strategy to the point it can accept its first tick.

    Returns an initialized strategy.

    :param lookback: How much old data we load on bootstrap
    :param kwargs: Random arguments one can pass to factory / StrategyRunner constructor.
    :param now_: Override the current clock for testing

    :raises BadStrategyFile:
    :raises PreflightCheckFailed:

    :return: Tuple of loaded and initialized elements
    """
    factory = import_strategy_file(path)
    runner: StrategyRunner = factory(timed_task_context_manager=timed_task_context_manager, **kwargs)
    time_bucket = runner.get_strategy_time_frame()
    dataset = runner.load_data(time_bucket, client, lookback)
    universe = runner.setup_universe_timed(dataset)
    runner.preflight_check(client, universe, now_)
    return [dataset, universe, runner]

