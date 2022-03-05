import datetime
import time
from queue import Queue
from typing import Optional, Callable

import logging
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyRunDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.client import Client

logger = logging.getLogger(__name__)


def run_main_loop(
        *ignore,
        command_queue: Queue,
        execution_model: ExecutionModel,
        sync_method: SyncMethod,
        approval_model: ApprovalModel,
        pricing_model_factory: PricingModelFactory,
        revaluation_method: RevaluationMethod,
        store: StateStore,
        client: Optional[Client],
        strategy_factory: Callable,
        reset=False,
        max_cycles: Optional[int]=None,
        sleep=1.0,
    ):
    """Runs the main loop of the strategy executor.

    :param reset: Start the state from the scratch.
    """

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    timed_task_context_manager = timed_task

    if reset:
        state = store.create()
    else:
        state = store.load()

    execution_model.initialize()
    execution_model.preflight_check()

    run_description: StrategyRunDescription = strategy_factory(
        execution_model=execution_model,
        timed_task_context_manager=timed_task_context_manager,
        sync_method=sync_method,
        revaluation_method=revaluation_method,
        pricing_model_factory=pricing_model_factory,
        approval_model=approval_model,
        client=client,
    )

    runner: StrategyRunner = run_description.runner

    universe_constructor = run_description.universe_constructor

    cycle = 1
    while True:

        # Reload the trading data
        ts = datetime.datetime.utcnow()
        logger.info("Starting strategy executor main loop cycle %d, UTC is %s", cycle, ts)

        universe = universe_constructor.construct_universe(ts)

        runner.pretick_check(ts, universe)

        runner.tick(ts, universe, state)
        store.sync()

        cycle += 1

        # Termination in integration testing
        if max_cycles is not None:
            if cycle >= max_cycles:
                break

        time.sleep(sleep)