"""Trade executor main loop."""

import logging
import datetime
import pickle
import random
from pathlib import Path
from queue import Queue
from typing import Optional, Callable

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler

from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.state import State
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.strategy.tick import TickSize, snap_to_next_tick, snap_to_previous_tick
from tradeexecutor.strategy.universe_model import UniverseModel
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.client import Client


logger = logging.getLogger(__name__)


class ExecutionLoop:
    """Live or backtesting trade execution loop."""

    def __init__(
            self,
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
            stats_refresh_frequency: datetime.timedelta,
            max_data_delay: Optional[datetime.timedelta]=None,
            reset=False,
            max_cycles: Optional[int]=None,
            debug_dump_file: Optional[Path]=None,
            backtest_start: Optional[datetime.datetime]=None,
            backtest_end: Optional[datetime.datetime]=None,
            tick_offset: datetime.timedelta=datetime.timedelta(minutes=0),
            trade_immediately=False,
        ):
        """See main.py for details."""

        if ignore:
            # https://www.python.org/dev/peps/pep-3102/
            raise TypeError("Only keyword arguments accepted")

        args = locals().copy()
        args.pop("self")
        self.__dict__.update(args)

        self.timed_task_context_manager = timed_task

        self.runner: Optional[StrategyRunner] = None
        self.universe_model: Optional[UniverseModel] = None

        # cycle -> dump mappings
        self.debug_dump_state = {}

    def init_state(self) -> State:
        """Initialize the state for this run."""
        if self.reset:
            state = self.store.create()
        else:
            if self.store.is_empty():
                state = self.store.create()
            else:
                state = self.store.load()

        # Check that we did not corrupt the state while writing it to the disk
        state.perform_integrity_check()

        return state

    def init_execution_model(self):
        self.execution_model.initialize()
        self.execution_model.preflight_check()
        logger.trade("Preflight checks ok")

    def tick(self, unrounded_timestamp: datetime.datetime, state: State, cycle: int, live: bool):
        """Run one trade execution tick."""

        assert isinstance(unrounded_timestamp, datetime.datetime)
        assert isinstance(state, State)

        ts = snap_to_previous_tick(unrounded_timestamp, self.tick_size)

        # This Python dict collects internal debugging data through this cycle.
        # Any submodule of strategy execution can add internal information here for
        # unit testing and manual diagnostics. Any data added must be JSON serializable.
        debug_details = {
            "cycle": cycle,
            "unrounded_timestamp": unrounded_timestamp,
            "timestamp": ts,
        }

        logger.trade("Performing strategy tick #%d for timestamp %s, unrounded time is %s, live trading is %s", cycle, ts, unrounded_timestamp, live)

        # Refresh the trading universe for this cycle
        universe = self.universe_model.construct_universe(ts, live)

        # Check if our data is stagnated and we cannot execute the strategy
        if self.max_data_delay is not None:
            self.universe_model.check_data_age(ts, universe, self.max_data_delay)

        # Run cycle checks
        self.runner.pretick_check(ts, universe)

        # Execute the strategy tick and trades
        self.runner.tick(ts, universe, state, debug_details)

        # Check that state is good before writing it to the disk
        state.perform_integrity_check()

        # Store the current state to disk
        self.store.sync(state)

        # Store debug trace
        if self.debug_dump_file is not None:
            self.debug_dump_state[cycle] = debug_details

            # Record and write out the internal debug states after every tick
            with open(self.debug_dump_file, "wb") as out:
                pickle.dump(self.debug_dump_state, out)

    def create_stats_entry(self, clock: datetime.datetime, state: State):
        """Revalue positions and update statistics.

        A new statistics entry is calculated for portfolio and all of its positions
        and added to the state.

        :param clock: Real-time or historical clock
        """
        update_statistics(clock, state.stats, state.portfolio)

        # Check that state is good before writing it to the disk
        state.perform_integrity_check()

        # Store the current state to disk
        self.store.sync(state)

    def run_backtest(self, state: State):
        """Backtest loop."""

        ts = self.backtest_start

        logger.info("Strategy is executed in backtesting mode, starting at %s", ts)

        cycle = 1
        while True:

            self.tick(ts, state, cycle, live=False)

            self.create_stats_entry(ts, state)

            # Check for termination in integration testing
            if self.max_cycles is not None:
                if cycle >= self.max_cycles:
                    logger.info("Max test cycles reached")
                    break

            # Advance to the next tick
            cycle += 1

            # Backtesting
            next_tick = snap_to_next_tick(ts + datetime.timedelta(seconds=1), self.tick_size, self.tick_offset)

            if next_tick >= self.backtest_end:
                # Backteting has ended
                logger.info("Terminating backesting. Backtest end %s, current timestamp %s", self.backtest_end, next_tick)
                break

            # Add some fuzziness to gacktesting timestamps
            ts = next_tick + datetime.timedelta(minutes=random.randint(0, 4))

    def run_live(self, state: State):
        """Run live trading cycle."""

        ts = datetime.datetime.utcnow()
        logger.info("Strategy is executed in live mode, now is %s", ts)

        cycle = 1

        # The first trade will be execute immediately, despite the time offset or tick
        if self.trade_immediately:
            ts = datetime.datetime.now()
            self.tick(ts, state, cycle, live=True)

        def live_cycle():
            nonlocal cycle
            cycle += 1
            ts = datetime.datetime.now()
            self.tick(ts, state, cycle, live=True)

        def live_positions():
            ts = datetime.datetime.now()
            self.create_stats_entry(ts, state)

        # Set up live trading tasks using APScheduler
        executors = {
            'default': ThreadPoolExecutor(1),
        }
        start_time = datetime.datetime(1970, 1, 1)
        scheduler = BlockingScheduler(executors=executors, timezone=datetime.timezone.utc)
        scheduler.add_job(live_cycle, 'interval', seconds=self.tick_size.to_timedelta().total_seconds(), start_date=start_time + self.tick_offset)
        scheduler.add_job(live_positions, 'interval', seconds=self.stats_refresh_frequency.total_seconds(), start_date=start_time)
        scheduler.start()

    def run(self):
        """The main loop of trade executor.

        Main entry point to the loop.

        - Chooses between live and backtesting execution mode

        - Loads or creates the initial state

        - Sets up strategy runner
        """

        assert self.tick_size is not None

        if self.backtest_end or self.backtest_start:
            assert self.backtest_start and self.backtest_end, f"If backtesting both start and end must be given, we have {self.backtest_start} - {self.backtest_end}"

        state = self.init_state()

        self.init_execution_model()

        run_description: StrategyExecutionDescription = self.strategy_factory(
            execution_model=self.execution_model,
            timed_task_context_manager=self.timed_task_context_manager,
            sync_method=self.sync_method,
            revaluation_method=self.revaluation_method,
            pricing_model_factory=self.pricing_model_factory,
            approval_model=self.approval_model,
            client=self.client,
        )

        # Deconstruct strategy input
        self.runner: StrategyRunner = run_description.runner
        self.universe_model = run_description.universe_model

        if self.backtest_start:
            # Walk through backtesting range
            return self.run_backtest(state)
        else:
            return self.run_live(state)
