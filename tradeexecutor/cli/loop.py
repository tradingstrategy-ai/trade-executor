"""Trade executor main loop."""

import logging
import datetime
import pickle
import random
import time
from pathlib import Path
from queue import Queue
from typing import Optional, Callable

from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.mode import ExecutionMode
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.strategy.tick import TickSize, snap_to_next_tick
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.client import Client


logger = logging.getLogger(__name__)


def run_main_loop(
        *ignore,
        name: str,
        command_queue: Queue,
        execution_model: ExecutionModel,
        sync_method: SyncMethod,
        approval_model: ApprovalModel,
        pricing_model_factory: PricingModelFactory,
        revaluation_method: RevaluationMethod,
        store: StateStore,
        client: Optional[Client],
        strategy_factory: Callable,
        tick_size: TickSize,
        max_data_delay: Optional[datetime.timedelta]=None,
        reset=False,
        max_cycles: Optional[int]=None,
        debug_dump_file: Optional[Path]=None,
        backtest_start: Optional[datetime.datetime]=None,
        backtest_end: Optional[datetime.datetime]=None,
        tick_offset: datetime.timedelta=datetime.timedelta(minutes=0),
        trade_immediately=False,
    ):
    """The main loop of trade executor."""

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    if backtest_end or backtest_start:
        assert backtest_start and backtest_end, f"If backtesting both start and end must be given, we have {backtest_start} - {backtest_end}"
        mode = ExecutionMode.backtest
        live = False
    else:
        mode = ExecutionMode.live_trade
        live = True

    timed_task_context_manager = timed_task

    if reset:
        state = store.create()
    else:
        if store.is_empty():
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
    universe_model = run_description.universe_model

    # Debug details from every cycle
    debug_dump_state = {}

    cycle = 1

    if backtest_start:
        # Walk through backtesting range
        ts = backtest_start
        logger.info("Strategy is executed in backtesting mode, starting at %s", ts)
    else:
        # Live trading
        ts = datetime.datetime.utcnow()
        logger.info("Strategy is executed in live mode, now is %s", ts)

    logger.trade("Starting trade execution loop for %s", name)

    # The first trade will be execute immediately, despite the time offset or tick
    if trade_immediately:
        ts = datetime.datetime.now()
    else:
        next_tick = snap_to_next_tick(datetime.datetime.now() + datetime.timedelta(seconds=1), tick_size, tick_offset)
        wait = next_tick - datetime.datetime.utcnow()
        logger.info("Sleeping %s until the first tick at %s UTC", wait, next_tick)
        time.sleep(wait.total_seconds())
        ts = datetime.datetime.utcnow()

    while True:

        # This Python dict collects internal debugging data through this cycle.
        # Any submodule of strategy execution can add internal information here for
        # unit testing and manual diagnostics. Any data added must be JSON serializable.
        debug_details = {
            "cycle": cycle,
            "timestamp": ts,
        }

        logger.trade("Starting strategy cycle %d, UTC is %s", cycle, ts)

        # Refresh the trading universe for this cycle
        universe = universe_model.construct_universe(ts, live)

        # Check if our data is stagnated and we cannot execute the strategy
        if max_data_delay is not None:
            universe_model.check_data_age(ts, universe, max_data_delay)

        # Run cycle checks
        runner.pretick_check(ts, universe)

        # Execute the strategy tick and trades
        runner.tick(ts, universe, state, debug_details)

        # Store the current state to disk
        store.sync(state)

        # Store debug trace
        if debug_dump_file is not None:
            debug_dump_state[cycle] = debug_details

            # Record and write out the internal debug states after every tick
            with open(debug_dump_file, "wb") as out:
                pickle.dump(debug_dump_state, out)

        # Check for termination in integration testing
        if max_cycles is not None:
            if cycle >= max_cycles:
                logger.info("Max test cycles reached")
                break

        # Advance to the next tick
        cycle += 1

        if mode == ExecutionMode.backtest:
            # Backtesting
            next_tick = snap_to_next_tick(ts + datetime.timedelta(seconds=1), tick_size, tick_offset)

            if next_tick >= backtest_end:
                # Backteting has ended
                logger.info("Terminating backesting. Backtest end %s, current timestamp %s", backtest_end, next_tick)
                break

            # Add some fuzziness to gacktesting timestamps
            ts = next_tick + datetime.timedelta(minutes=random.randint(0, 4))
        else:
            # Live trading
            next_tick = snap_to_next_tick(ts + datetime.timedelta(seconds=1), tick_size, tick_offset)
            wait = next_tick - datetime.datetime.utcnow()
            logger.info("Sleeping %s until the next tick at %s UTC", wait, next_tick)
            time.sleep(wait.total_seconds())
            ts = datetime.datetime.utcnow()
