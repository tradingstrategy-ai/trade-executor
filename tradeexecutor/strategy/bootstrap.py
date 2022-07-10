"""Loading Python strategy modules.

.. warning ::

    Deprecated.

See :py:mod:`strategy_module` instead.

"""
import logging
import runpy
from contextlib import AbstractContextManager
from pathlib import Path

from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.factory import StrategyFactory, make_runner_for_strategy_mod

logger = logging.getLogger(__name__)


#: What is our Python entry point for strategies
FACTORY_VAR_NAME = "strategy_factory"


class BadStrategyFile(Exception):
    pass


def import_strategy_file(path: Path) -> StrategyFactory:
    """Loads a strategy module and returns its factor function."""
    logger.info("Importing strategy %s", path)

    assert isinstance(path, Path)

    strategy_exports = runpy.run_path(path)

    # Strategy v0.1 loading
    if "trading_strategy_engine_version" in strategy_exports:
        return make_runner_for_strategy_mod(strategy_exports)

    # Legacy path
    strategy_runner = strategy_exports.get(FACTORY_VAR_NAME)
    if strategy_runner is None:
        raise BadStrategyFile(f"{path} Python module does not declare {FACTORY_VAR_NAME} module variable")

    return strategy_runner


def bootstrap_strategy(
        timed_task_context_manager: AbstractContextManager,
        path: Path,
        **kwargs) -> StrategyExecutionDescription:
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
    description = factory(
        timed_task_context_manager=timed_task_context_manager, **kwargs)
    return description

