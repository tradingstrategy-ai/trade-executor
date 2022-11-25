"""Loading Python strategy modules.

.. warning ::

    Deprecated.

See :py:mod:`strategy_module` instead.

"""
import logging
import runpy
from contextlib import AbstractContextManager
from pathlib import Path

from tradingstrategy.client import Client

from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.factory import StrategyFactory, make_factory_from_strategy_mod
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.strategy.valuation import ValuationModelFactory

logger = logging.getLogger(__name__)


#: What is our Python entry point for strategies
FACTORY_VAR_NAME = "strategy_factory"


class BadStrategyFile(Exception):
    pass


def import_strategy_file(path: Path) -> StrategyFactory:
    """Loads a strategy module and returns its factor function.

    All exports will be lowercased for further processing,
    so we do not care if constant variables are written in upper or lowercase.
    """
    logger.info("Importing strategy %s", path)
    assert isinstance(path, Path)
    mod = read_strategy_module(path)
    return make_factory_from_strategy_mod(mod)


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

