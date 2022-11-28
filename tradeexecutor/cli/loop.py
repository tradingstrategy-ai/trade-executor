"""Trade executor main loop."""

import logging
import datetime
import pickle
import random
from pathlib import Path
from queue import Queue
from typing import Optional, Callable, List

import pandas as pd

try:
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.schedulers.blocking import BlockingScheduler
except ImportError:
    # apscheduler is only required in live trading
    pass

try:
    from tqdm_loggable.auto import tqdm
except ImportError:
    # tqdm_loggable is only available at the live execution,
    # but fallback to normal TQDM auto mode
    from tqdm.auto import tqdm

from tradeexecutor.backtest.backtest_pricing import BacktestSimplePricingModel
from tradeexecutor.state.state import State
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.validator import validate_state_serialisation
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.factory import StrategyFactory
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.strategy.cycle import CycleDuration, snap_to_next_tick, snap_to_previous_tick, round_datetime_up
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import UniverseModel, StrategyExecutionUniverse, UniverseOptions
from tradeexecutor.strategy.valuation import ValuationModelFactory
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


logger = logging.getLogger(__name__)


class ExecutionLoop:
    """Live or backtesting trade execution loop.

    This is the main loop of any strategy execution.

    - Run scheduled tasks for different areas (trade cycle, position revaluation, stop loss triggers)

    - Call :py:class:`ExecutionModel` to perform ticking through the strategy

    - Manage the persistent state of the strategy
    """

    def __init__(
            self,
            *ignore,
            name: str,
            command_queue: Queue,
            execution_model: ExecutionModel,
            execution_context: ExecutionContext,
            sync_method: SyncMethod,
            approval_model: ApprovalModel,
            pricing_model_factory: PricingModelFactory,
            valuation_model_factory: ValuationModelFactory,
            store: StateStore,
            client: Optional[Client],
            strategy_factory: Optional[StrategyFactory],
            cycle_duration: CycleDuration,
            stats_refresh_frequency: Optional[datetime.timedelta],
            position_trigger_check_frequency: Optional[datetime.timedelta],
            max_data_delay: Optional[datetime.timedelta] = None,
            reset=False,
            max_cycles: Optional[int] = None,
            debug_dump_file: Optional[Path] = None,
            backtest_start: Optional[datetime.datetime] = None,
            backtest_end: Optional[datetime.datetime] = None,
            backtest_setup: Optional[Callable[[State], None]] = None,
            backtest_candle_time_frame_override: Optional[TimeBucket] = None,
            backtest_stop_loss_time_frame_override: Optional[TimeBucket] = None,
            stop_loss_check_frequency: Optional[TimeBucket] = None,
            tick_offset: datetime.timedelta=datetime.timedelta(minutes=0),
            trade_immediately=False,
    ):
        """See main.py for details."""

        if ignore:
            # https://www.python.org/dev/peps/pep-3102/
            raise TypeError("Only keyword arguments accepted")

        self.cycle_duration = cycle_duration
        self.stop_loss_check_frequency = stop_loss_check_frequency

        args = locals().copy()
        args.pop("self")

        assert "execution_context" in args, "execution_context required"

        # TODO: Spell out individual variables for type hinting support
        self.__dict__.update(args)

        self.timed_task_context_manager = self.execution_context.timed_task_context_manager

        self.runner: Optional[StrategyRunner] = None
        self.universe_model: Optional[UniverseModel] = None

        # cycle -> dump mappings
        self.debug_dump_state = {}

        # Hook in any overrides for strategy cycles
        self.universe_options = UniverseOptions(
            candle_time_bucket_override=self.backtest_candle_time_frame_override,
            stop_loss_time_bucket_override=self.backtest_stop_loss_time_frame_override,
        )

    def is_live_trading_unit_test(self) -> bool:
        """Are we attempting to test live trading functionality in unit tests.

        See `test_cli_commands.py`
        """
        return self.max_cycles == 0

    def init_state(self) -> State:
        """Initialize the state for this run.

        - If we are doing live trading, load the last saved state

        - In backtesting the state is always reset.
          We do not support resumes for crashed backetsting.

        """
        if self.reset:
            # Create empty state and save it
            state = self.store.create()
            state.name = self.name
            self.store.sync(state)
        else:
            if self.store.is_pristine():
                # Create empty state and save it
                state = self.store.create()
                state.name = self.name
                self.store.sync(state)
            else:
                state = self.store.load()

        # Check that we did not corrupt the state while writing it to the disk
        state.perform_integrity_check()

        return state

    def init_execution_model(self):
        """Initialise the execution.

        Perform preflight checks e.g. to see if our trading accounts look sane.
        """
        self.execution_model.initialize()
        if not self.is_live_trading_unit_test():
            self.execution_model.preflight_check()
            logger.trade("Preflight checks ok")

    def init_simulation(
            self,
            universe_model: UniverseModel,
            runner: StrategyRunner,
        ):
        """Set up running on a simulated blockchain.

        Used with :py:mod:`tradeexecutor.testing.simulated_execution_loop`
        to allow fine granularity manipulation of in-memory blockchain
        to simulate trigger conditions in testing.
        """
        assert self.execution_context.mode == ExecutionMode.simulated_trading
        self.init_execution_model()
        self.universe_model = universe_model
        self.runner = runner

    def tick(self,
             unrounded_timestamp: datetime.datetime,
             cycle_duration: CycleDuration,
             state: State,
             cycle: int,
             live: bool,
             backtesting_universe: Optional[StrategyExecutionUniverse]=None) -> StrategyExecutionUniverse:
        """Run one trade execution tick.

        :parma unrounded_timestamp:
            The approximately time when this ticket was triggered.
            Alawys after the tick timestamp.
            Will be rounded to the nearest cycle duration timestamps.

        :param state:
            The current state of the strategy

        :param cycle:
            The number ofthis cycle

        :param cycle_duration:
            Cycle duration for this cycle. Either from the strategy module,
            or a backtest override.

        :param backtesting_universe:
            If passed, use this universe instead of trying to download
            and filter new one. This is shortcut for backtesting
            where the universe does not change between cycles
            (as opposite to live trading new pairs pop in to the existince).

        """

        assert isinstance(unrounded_timestamp, datetime.datetime)
        assert isinstance(state, State)
        assert isinstance(cycle_duration, CycleDuration)

        ts = snap_to_previous_tick(unrounded_timestamp, cycle_duration)

        # This Python dict collects internal debugging data through this cycle.
        # Any submodule of strategy execution can add internal information here for
        # unit testing and manual diagnostics. Any data added must be JSON serializable.
        debug_details = {
            "cycle": cycle,
            "unrounded_timestamp": unrounded_timestamp,
            "timestamp": ts,
        }

        logger.trade("Performing strategy tick #%d for timestamp %s, cycle length is %s, unrounded time is %s, live trading is %s",
                     cycle,
                     ts,
                     cycle_duration.value,
                     unrounded_timestamp,
                     live)

        if backtesting_universe is None:

            # We are running backtesting and the universe is not yet loaded.
            # Unlike live trading, we do not need to reconstruct the universe between
            # trade cycles.

            # Refresh the trading universe for this cycle
            universe = self.universe_model.construct_universe(
                ts,
                self.execution_context.mode,
                self.universe_options,
            )

            # Check if our data is stagnated and we cannot execute the strategy
            if self.max_data_delay is not None:
                self.universe_model.check_data_age(ts, universe, self.max_data_delay)

        else:
            # Recycle the universe instance
            logger.info("Reusing universe from the previous tick")
            universe = backtesting_universe

        # Run cycle checks
        self.runner.pretick_check(ts, universe)

        if cycle == 1 and self.backtest_setup is not None:
            # The hook to set up backtest initial balance
            self.backtest_setup(state, universe, self.sync_method)

        # Execute the strategy tick and trades
        self.runner.tick(ts, universe, state, debug_details)

        # Check that state is good before writing it to the disk
        state.perform_integrity_check()

        # Store the current state to disk
        self.store.sync(state)

        # Store debug trace
        self.debug_dump_state[cycle] = debug_details

        if self.debug_dump_file is not None:
            # Record and write out the internal debug states after every tick
            with open(self.debug_dump_file, "wb") as out:
                pickle.dump(self.debug_dump_state, out)

        # Assume universe stays static between cycles
        # for hourly revaluations
        return universe

    def update_position_valuations(self, clock: datetime.datetime, state: State, universe: StrategyExecutionUniverse, execution_mode: ExecutionMode):
        """Revalue positions and update statistics.

        A new statistics entry is calculated for portfolio and all of its positions
        and added to the state.

        :param clock: Real-time or historical clock
        """

        # Set up the execution to perform the valuation

        routing_state, pricing_model, valuation_method = self.runner.setup_routing(universe)

        with self.timed_task_context_manager("revalue_portfolio_statistics"):
            logger.info("Updating position valuations")
            self.runner.revalue_portfolio(clock, state, valuation_method)

        with self.timed_task_context_manager("update_statistics"):
            logger.info("Updating position statistics")
            update_statistics(clock, state.stats, state.portfolio, execution_mode)

        # Check that state is good before writing it to the disk
        state.perform_integrity_check()

        # Store the current state to disk
        self.store.sync(state)

    def check_position_triggers(self,
                          ts: datetime.datetime,
                          state: State,
                          universe: TradingStrategyUniverse) -> List[TradeExecution]:
        """Run stop loss price checks.

        Used for live stop loss check; backtesting
        uses optimised :py:meth:`run_backtest_stop_loss_checks`.

        :param ts:
            Timestamp of this check cycle

        param universe:
            Trading universe containing price data for stoploss checks.

        :return:
            List of generated trigger trades
        """

        logger.info("Starting stop loss checks at %s", ts)

        routing_state, pricing_model, valuation_method = self.runner.setup_routing(universe)

        # Do stop loss checks for every time point between now and next strategy cycle
        trades = self.runner.check_position_triggers(
            ts,
            state,
            universe,
            pricing_model,
            routing_state
        )

        # Check that state is good before writing it to the disk
        state.perform_integrity_check()

        # Store the current state to disk
        self.store.sync(state)

        return trades

    def warm_up_backtest(self):
        """Load backtesting trading universe.

        Display progress bars for data downloads.
        """
        logger.info("Warming up backesting")

        self.universe_model.preload_universe(self.universe_options)

    def run_backtest_stop_loss_checks(self,
                                      start_ts: datetime.datetime,
                                      end_ts: datetime.datetime,
                                      state: State,
                                      universe: TradingStrategyUniverse):
        """Generate stop loss price checks.

        Backtests may use finer grade data for stop loss signals,
        to be more realistic with actual trading.

        Here we use the finer grade data to check the stop losses
        on a given time period.

        :param start_ts:
            When to start testing (exclusive).
            We test for the next available timestamp.

        :param end_ts:
            When to stop testing (exclusive).

        param universe:
            Trading universe containing price data for stoploss checks.
        """

        assert universe.backtest_stop_loss_candles is not None

        # What is the granularity of our price feed
        # for stop loss checks.
        tick_size = universe.backtest_stop_loss_time_bucket

        logger.info("run_backtest_stop_loss_checks with frequency of %s", tick_size.value)

        assert tick_size.to_pandas_timedelta() > pd.Timedelta(0), f"Cannot do stop loss checks, because no stop loss cycle duration was given"

        # Hop to the next tick
        ts = round_datetime_up(start_ts, tick_size.to_timedelta())

        routing_state, pricing_model, valuation_model = self.runner.setup_routing(universe)
        assert pricing_model, "Routing did not provide pricing_model"

        stop_loss_pricing_model = BacktestSimplePricingModel(universe.backtest_stop_loss_candles, self.runner.routing_model)

        # Do stop loss checks for every time point between now and next strategy cycle
        while ts < end_ts:
            logger.debug("Backtesting stop loss at %s", ts)
            self.runner.check_position_triggers(
                ts,
                state,
                universe,
                stop_loss_pricing_model,
                routing_state
            )
            ts += tick_size.to_timedelta()

    def run_backtest(self, state: State) -> dict:
        """Backtest loop."""

        if self.backtest_end or self.backtest_start:
            assert self.backtest_start and self.backtest_end, f"If backtesting both start and end must be given, we have {self.backtest_start} - {self.backtest_end}"

        ts = self.backtest_start

        logger.info("Strategy is executed in backtesting mode, starting at %s, cycle duration is %s", ts, self.cycle_duration.value)

        cycle = 1
        universe = None

        range = self.backtest_end - self.backtest_start

        ts_format = "%Y-%m-%d"

        friendly_start = self.backtest_start.strftime(ts_format)
        friendly_end = self.backtest_end.strftime(ts_format)

        seconds = int(range.total_seconds())

        self.warm_up_backtest()

        # Allow backtest step to be overwritten from the command line
        if self.universe_options.candle_time_bucket_override:
            backtest_step = CycleDuration.from_timebucket(self.universe_options.candle_time_bucket_override)
        else:
            backtest_step = self.cycle_duration

        cycle_name = backtest_step.value

        with tqdm(total=seconds) as progress_bar:
            while True:

                ts = snap_to_previous_tick(ts, self.cycle_duration)

                # Bump progress bar forward and update backtest status
                progress_bar.update(int(backtest_step.to_timedelta().total_seconds()))
                friedly_ts = ts.strftime(ts_format)
                trade_count = len(list(state.portfolio.get_all_trades()))
                progress_bar.set_description(f"Backtesting {self.name}, {friendly_start}-{friendly_end} at {friedly_ts} ({cycle_name}), total {trade_count:,} trades")

                # Decide trades and everything for this cycle
                universe: TradingStrategyUniverse = self.tick(
                    ts,
                    backtest_step,
                    state,
                    cycle,
                    live=False,
                    backtesting_universe=universe)

                # Revalue our portfolio
                self.update_position_valuations(ts, state, universe, self.execution_context.mode)

                # Check for termination in integration testing.
                # TODO: Get rid of this and only support date ranges to run tests
                if self.max_cycles is not None:
                    if cycle >= self.max_cycles:
                        logger.info("Max backtest cycles reached")
                        break

                # Advance to the next tick
                cycle += 1

                # Backtesting
                next_tick = snap_to_next_tick(ts + datetime.timedelta(seconds=1), backtest_step, self.tick_offset)

                if next_tick >= self.backtest_end:
                    # Backteting has ended
                    logger.info("Terminating backtesting. Backtest end %s, current timestamp %s", self.backtest_end, next_tick)
                    break

                # If we have stop loss checks enabled on a separate price feed,
                # run backtest stop loss checks until the next time
                if universe.backtest_stop_loss_candles is not None:
                    self.run_backtest_stop_loss_checks(
                        ts,
                        next_tick,
                        state,
                        universe,
                    )

                # Add some fuzziness to gacktesting timestamps
                ts = next_tick + datetime.timedelta(minutes=random.randint(0, 4))

            # Validate the backtest state at the end.
            # We want to avoid situation where we have stored
            # non-serialisable types in the state
            validate_state_serialisation(state)

            return self.debug_dump_state

    def run_live(self, state: State):
        """Run live trading cycle."""

        ts = datetime.datetime.utcnow()
        logger.info("Strategy is executed in live mode, now is %s", ts)

        if self.is_live_trading_unit_test():
            # Test app initialisation.
            # Do not start any background threads.
            logger.info("Unit test live trading checkup test detected - aborting.")
            return self.debug_dump_state

        cycle = 1
        universe: Optional[StrategyExecutionUniverse] = None

        # The first trade will be execute immediately, despite the time offset or tick
        if self.trade_immediately:
            ts = datetime.datetime.now()
            self.tick(ts, self.cycle_duration, state, cycle, live=True)

        def live_cycle():
            nonlocal cycle
            nonlocal universe
            try:
                cycle += 1
                ts = datetime.datetime.now()
                universe = self.tick(ts, self.cycle_duration, state, cycle, live=True)
            except Exception as e:
                logger.exception(e)
                scheduler.shutdown(wait=False)
                raise

        def live_positions():
            nonlocal universe

            # We cannot update valuations until we know
            # trading pair universe, because to know the valuation of the position
            # we need to know the route how to sell the token of the position
            if universe is None:
                logger.info("Universe not yet downloaded")
                return

            try:
                ts = datetime.datetime.now()
                self.update_position_valuations(ts, state, universe)
            except Exception as e:
                logger.exception(e)
                scheduler.shutdown(wait=False)
                raise

        def live_trigger_checks():
            nonlocal universe

            # We cannot update valuations until we know
            # trading pair universe, because to know the valuation of the position
            # we need to know the route how to sell the token of the position
            if universe is None:
                logger.info("Universe not yet downloaded")
                return

            try:
                ts = datetime.datetime.now()
                self.check_position_triggers(ts, state, universe)
            except Exception as e:
                logger.exception(e)
                scheduler.shutdown(wait=False)
                raise

        # Set up live trading tasks using APScheduler
        executors = {
            'default': ThreadPoolExecutor(1),
        }
        start_time = datetime.datetime(1970, 1, 1)
        scheduler = BlockingScheduler(executors=executors, timezone=datetime.timezone.utc)
        scheduler.add_job(live_cycle, 'interval', seconds=self.cycle_duration.to_timedelta().total_seconds(), start_date=start_time + self.tick_offset)

        if self.stats_refresh_frequency not in (datetime.timedelta(0), None):
            scheduler.add_job(
                live_positions,
                'interval',
                seconds=self.stats_refresh_frequency.total_seconds(),
                start_date=start_time)

        if self.position_trigger_check_frequency not in (datetime.timedelta(0), None):
            scheduler.add_job(
                live_trigger_checks,
                'interval',
                seconds=self.position_trigger_check_frequency.total_seconds(),
                start_date=start_time)

        try:
            scheduler.start()
        except KeyboardInterrupt:
            # https://github.com/agronholm/apscheduler/issues/338
            scheduler.shutdown(wait=False)
            raise
        logger.info("Scheduler finished - down the live trading loop")

        return self.debug_dump_state

    def run(self) -> dict:
        """The main loop of trade executor.

        Main entry point to the loop.

        - Chooses between live and backtesting execution mode

        - Loads or creates the initial state

        - Sets up strategy runner

        :return:
            Debug state where each key is the cycle number
        """

        state = self.init_state()
        self.init_execution_model()

        run_description: StrategyExecutionDescription = self.strategy_factory(
            execution_model=self.execution_model,
            execution_context=self.execution_context,
            timed_task_context_manager=self.timed_task_context_manager,
            sync_method=self.sync_method,
            valuation_model_factory=self.valuation_model_factory,
            pricing_model_factory=self.pricing_model_factory,
            approval_model=self.approval_model,
            client=self.client,
            routing_model=None,  # Assume strategy factory produces its own routing model
        )

        # Deconstruct strategy input
        self.runner: StrategyRunner = run_description.runner
        self.universe_model = run_description.universe_model

        # Load cycle_duration from v0.1 strategies,
        # if not given from the command line to override backtesting data
        if run_description.cycle_duration and not self.cycle_duration:
            self.cycle_duration = run_description.cycle_duration

        assert self.cycle_duration is not None, "Did not get strategy cycle duration from constructor or strategy run description"

        if self.backtest_start:
            # Walk through backtesting range
            return self.run_backtest(state)
        else:
            return self.run_live(state)

