"""Main trade executor loop."""

import datetime
import json
import time
from pathlib import Path
from queue import Queue
from typing import Optional, Callable

import logging
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
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
        debug_dump_file: Optional[Path]=None,
    ):
    """The main loop of trade executor."""

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

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=execution_model,
        timed_task_context_manager=timed_task_context_manager,
        sync_method=sync_method,
        revaluation_method=revaluation_method,
        pricing_model_factory=pricing_model_factory,
        approval_model=approval_model,
        client=client,
    )

    # Deconstruct strategy input
    runner: StrategyRunner = run_description.runner
    universe_constructor = run_description.universe_model

    # Debug details from every cycle
    debug_dump_state = {}

    cycle = 1
    while True:

        # This Python dict collects internal debugging data through this cycle.
        # Any submodule of strategy execution can add internal information here for
        # unit testing and manual diagnostics. Any data added must be JSON serializable.
        debug_details = {"cycle": cycle}

        # Reload the trading data
        ts = datetime.datetime.utcnow()
        logger.info("Starting strategy executor main loop cycle %d, UTC is %s", cycle, ts)

        # Refresh the trading universe for this cycle
        universe = universe_constructor.construct_universe(ts)

        # Run cycle checks
        runner.pretick_check(ts, universe)

        # Execute the strategy tick and trades
        runner.tick(ts, universe, state, debug_details)

        # Store the current state to disk
        store.sync()

        # Record and write out the internal debug state
        if debug_dump_file is not None:
            debug_dump_state[cycle] = debug_details
            with open(debug_dump_file, "wt") as out:
                json.dump(out, debug_dump_state)

        # Advance to the next tick
        cycle += 1

        # Check for termination in integration testing
        if max_cycles is not None:
            if cycle >= max_cycles:
                break

        time.sleep(sleep)