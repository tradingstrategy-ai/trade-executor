import runpy
from pathlib import Path
import logging

from tradeexecutor.strategy.runner import StrategyRunner


logger = logging.getLogger(__name__)


class BadStrategyFile(Exception):
    pass


def import_strategy_file(path: Path) -> StrategyRunner:
    logger.info("Importing strategy %s", path)
    strategy_exports = runpy.run_path(path)

    strategy_runner = strategy_exports.get("strategy_runner")
    if strategy_runner is None:
        raise BadStrategyFile(f"{path} Python module does not declare strategy_runner module variable")

    if not isinstance(strategy_runner, StrategyRunner):
        raise BadStrategyFile(f"{path} strategy_runner is not instance of StrategyRunner class")

    return strategy_runner
