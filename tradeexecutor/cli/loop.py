import datetime
import time
from queue import Queue
from typing import Optional, Callable

import logging
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.client import Client

logger = logging.getLogger(__name__)


def run_main_loop(
        command_queue: Queue,
        execution_model: ExecutionModel,
        sync_method: SyncMethod,
        approval_model: ApprovalModel,
        store: StateStore,
        client: Optional[Client],
        strategy_factory: Callable,
        reset=False,
        max_cycles: Optional[int]=None,
        sleep=1,
    ):
    """Runs the main loop of the strategy executor.

    :param reset: Start the state from the scratch.
    """

    if reset:
        state = store.create()
    else:
        state = store.load()

    execution_model.initialize()
    execution_model.preflight_check()



    strategy_runner = strategy_factory(
        timed_task_context_manager=timed_task,
        execution_model=execution_model,
        approval_model=approval_model,
        revaluation_method=revaluation_method,
        sync_method=sync_method,
        pricing_method=pricing_method,
        reserve_assets=supported_reserves)
    )

    cycle = 1
    while True:
        ts = datetime.datetime.utcnow()
        logger.info("Starting strategy executor main loop cycle %d, UTC is %s", cycle, ts)
        runner.tick(ts, state)
        store.sync()
        time.sleep(sleep)
        cycle += 1

        # Termination in integration testing
        if max_cycles is not None:
            if cycle >= max_cycles:
                break