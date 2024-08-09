"""Trade executor main loop.

TODO: This execution loop needs to be re-architect to two separate subclasses,

- One for backtesting

- One for live execution
"""

import logging
import datetime
import pickle
import random
from pathlib import Path
from queue import Queue
from typing import Optional, Callable, List, cast, Tuple

import pandas as pd
from apscheduler.events import EVENT_JOB_ERROR

from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.cli.watchdog import create_watchdog_registry, register_worker, mark_alive, start_background_watchdog, \
    WatchdogMode
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.wallet import perform_gas_level_checks
from tradeexecutor.state.metadata import Metadata
from tradeexecutor.statistics.in_memory_statistics import refresh_run_state
from tradeexecutor.statistics.statistics_table import serialise_long_short_stats_as_json_table
from tradeexecutor.strategy.account_correction import check_accounts, UnexpectedAccountingCorrectionIssue
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.pandas_trader.decision_trigger import wait_for_universe_data_availability_jsonl
from tradeexecutor.strategy.pandas_trader.indicator import CreateIndicatorsProtocol
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.strategy_cycle_trigger import StrategyCycleTrigger
from tradingstrategy.candle import GroupedCandleUniverse

try:
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.schedulers.blocking import BlockingScheduler
except ImportError:
    # apscheduler is only required in live trading,
    # not in backtesting
    pass

try:
    from tqdm_loggable.auto import tqdm
except ImportError:
    # tqdm_loggable is only available at the live execution,
    # but fallback to normal TQDM auto mode
    from tqdm.auto import tqdm

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.state.state import State, BacktestData
from tradeexecutor.state.store import StateStore
from tradeexecutor.strategy.sync_model import SyncMethodV0, SyncModel
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.validator import validate_state_serialisation
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.statistics.statistics_table import StatisticsTable
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
from tradingstrategy.client import Client, BaseClient
from tradingstrategy.timebucket import TimeBucket


logger = logging.getLogger(__name__)


class LiveSchedulingTaskFailed(Exception):
    """Main loop dies uncleanly.

    Any of live trading looop scheduled tasks can die with an exception.
    Raise this and wrap the underlying exception if we need to crash the trading loop.
    """


class ExecutionTestHook:
    """A test helper to allow to hook into backtest execution to inject events.

    Mostly used to simulate deposits/redemptions.
    """

    def on_before_cycle(
            self,
            cycle: int,
            cycle_st: datetime.datetime,
            state: State,
            sync_model: SyncModel,
    ):
        """Called before entering the strategy tick."""


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
            sync_model: SyncModel,
            approval_model: ApprovalModel,
            pricing_model_factory: PricingModelFactory,
            valuation_model_factory: ValuationModelFactory,
            store: StateStore,
            client: Optional[BaseClient],
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
            backtest_strategy_indicators: Optional[StrategyInputIndicators] = None,
            stop_loss_check_frequency: Optional[TimeBucket] = None,
            tick_offset: datetime.timedelta=datetime.timedelta(minutes=0),
            trade_immediately=False,
            run_state: Optional[RunState]=None,
            strategy_cycle_trigger: StrategyCycleTrigger = StrategyCycleTrigger.cycle_offset,
            routing_model: Optional[RoutingModel] = None,
            execution_test_hook: Optional[ExecutionTestHook] = None,
            metadata: Optional[Metadata] = None,
            check_accounts: Optional[bool] = None,
            minimum_data_lookback_range: Optional[datetime.timedelta] = None,
            universe_options: Optional[UniverseOptions] = None,
            sync_treasury_on_startup=False,
            create_indicators: CreateIndicatorsProtocol = None,
            parameters: StrategyParameters = None,
            visulisation: bool = True,
    ):
        """See main.py for details."""

        #
        # TODO: Initialisation needs a major rewrite
        #

        if ignore:
            # https://www.python.org/dev/peps/pep-3102/
            raise TypeError("Only keyword arguments accepted")

        assert isinstance(sync_model, SyncModel)
        self.sync_model = sync_model

        self.cycle_duration = cycle_duration
        self.stop_loss_check_frequency = stop_loss_check_frequency
        self.strategy_factory = strategy_factory
        self.reset = reset
        self.routing_model = routing_model
        self.execution_model = execution_model
        self.execution_test_hook = execution_test_hook
        self.metadata = metadata
        self.check_accounts = check_accounts
        self.execution_context = execution_context
        self.sync_treasury_on_startup = sync_treasury_on_startup
        self.store = store
        self.name = name
        self.trade_immediately = trade_immediately

        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.backtest_strategy_indicators = backtest_strategy_indicators
        self.create_indicators = create_indicators
        self.parameters = parameters

        args = locals().copy()
        args.pop("self")

        assert "execution_context" in args, "execution_context required"

        # TODO: Spell out individual variables for type hinting support
        self.__dict__.update(args)

        self.timed_task_context_manager = self.execution_context.timed_task_context_manager

        self.runner: Optional[StrategyRunner] = None
        self.universe_model: Optional[UniverseModel] = None
        self.strategy_cycle_trigger = strategy_cycle_trigger
        self.max_cycles = max_cycles
        self.max_data_delay = max_data_delay

        # Crash the strategy execution if we get more lag than this.
        # This is how old the last candle can be.
        self.max_live_data_lag_tolerance = datetime.timedelta(minutes=30)

        # cycle -> dump mappings
        self.debug_dump_state = {}

        # Hook in any overrides for strategy cycles
        if universe_options:
            self.universe_options = universe_options
        else:
            self.universe_options = UniverseOptions(
                candle_time_bucket_override=self.backtest_candle_time_frame_override,
                stop_loss_time_bucket_override=self.backtest_stop_loss_time_frame_override,
            )

        self.minimum_data_lookback_range = minimum_data_lookback_range

        # We hide once-downloaded universe here for live loop
        # tests that perform live trading against forked chain in a fast cycle (1s)
        self.unit_testing_universe: StrategyExecutionUniverse | None = None
        self.visulisation = visulisation

    def is_backtest(self) -> bool:
        """Are we doing a backtest execution."""
        return self.backtest_start is not None

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
        store: StateStore = self.store
        if self.reset:
            logger.info("Resetting the existing state file %s", store)
            # Create empty state and save it
            state = store.create(self.name)
            state.name = self.name
            store.sync(state)
        else:
            if store.is_pristine():
                logger.info("State is unwritten, creating new one %s", store)
                # Create empty state and save it
                state = store.create(self.name)
                state.name = self.name
                store.sync(state)
            else:
                logger.info("Loading state file %s", store)
                state = store.load()

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
            logger.info("Preflight checks ok")

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

    def init_live_run_state(self, run_description: StrategyExecutionDescription):
        """Initialise run-state object.

        We do need to do these updates only once on the startup,
        as there run-state variables do not change over the process lifetime.
        """

        # Expose source code to webhook
        if self.run_state:
            self.run_state.source_code = run_description.source_code

    def refresh_live_run_state(
        self,
        state: State,
        visualisation=False,
        universe: TradingStrategyUniverse=None,
        cycle_duration: CycleDuration=None,
    ):
        """Update the in-process strategy context which we serve over the webhook.

        .. note::

            There is still a gap between be able to serve the full run state and the webhook startup.
            This is because for the full run state, we need to have visualisations
            and we do not have those availble until we have loaded the trading universe data,
            which may take some time.

        :param visualisation:
            Also update technical charts
        """

        assert cycle_duration is not None, "CycleDuration is required, got None"

        run_state = self.run_state

        refresh_run_state(
            run_state,
            state,
            self.execution_context,
            visualisation,
            universe,
            self.sync_model,
            self.metadata.backtested_state,
            self.metadata.key_metrics_backtest_cut_off,
            cycle_duration,
        )

        # Mark last refreshed
        run_state.bumb_refreshed()

    def tick(
        self,
        unrounded_timestamp: datetime.datetime,
        cycle_duration: CycleDuration,
        state: State,
        cycle: int,
        live: bool,
        existing_universe: Optional[StrategyExecutionUniverse]=None,
        strategy_cycle_timestamp: Optional[datetime.datetime] = None,
        extra_debug_data: Optional[dict] = None,
        indicators: StrategyInputIndicators | None = None,
        ) -> StrategyExecutionUniverse:
        """Run one trade execution tick.

        :param unrounded_timestamp:
            The approximately time when this ticket was triggered.
            Alawys after the tick timestamp.
            Will be rounded to the nearest cycle duration timestamps.

        :param strategy_cycle_timestamp:
            Precalculated strategy cycle timestamp based on unrounded timestamp

        :param state:
            The current state of the strategy

        :param cycle:
            The number of this cycle

        :param cycle_duration:
            Cycle duration for this cycle. Either from the strategy module,
            or a backtest override.

        :param live:
            We are doing live trading.

        :param existing_universe:
            If passed, use this universe instead of trying to download
            and filter new one. This is shortcut for backtesting
            where the universe does not change between cycles
            (as opposite to live trading new pairs pop in to the existence).

        :param extra_debug_data:
            Extra data to be passed to the debug dump used in unit testing.

        :return:
            The universe where we are trading.
        """

        assert isinstance(unrounded_timestamp, datetime.datetime)
        assert isinstance(state, State)
        assert isinstance(cycle_duration, CycleDuration)

        if strategy_cycle_timestamp:
            ts = strategy_cycle_timestamp
        else:
            ts = snap_to_previous_tick(unrounded_timestamp, cycle_duration)

        # This Python dict collects internal debugging data through this cycle.
        # Any submodule of strategy execution can add internal information here for
        # unit testing and manual diagnostics. Any data added must be JSON serializable.
        debug_details = {
            "cycle": cycle,
            "unrounded_timestamp": unrounded_timestamp,
            "timestamp": ts,
            "strategy_cycle_trigger": self.strategy_cycle_trigger.value,
        }

        logger.trade(
            "Performing strategy tick #%d for timestamp %s, cycle length is %s, trigger time was %s, live trading is %s, trading univese is %s, version %s, max cycles %d",
             cycle,
             ts,
             cycle_duration.value,
             unrounded_timestamp,
             live,
             existing_universe,
            self.execution_context.engine_version,
            self.max_cycles,
        )

        if existing_universe is None:

            # We are running backtesting and the universe is not yet loaded.
            # Unlike live trading, we do not need to reconstruct the universe between
            # trade cycles.

            # Refresh the trading universe for this cycle
            if self.strategy_cycle_trigger == StrategyCycleTrigger.cycle_offset or self.trade_immediately:
                logger.info("Creating new universe from the scratch using create_trading_universe()")
                universe = self.universe_model.construct_universe(
                    ts,
                    self.execution_context.mode,
                    self.universe_options,
                )

                # Check if our data is stagnated and we cannot execute the strategy
                if self.max_data_delay is not None:
                    self.universe_model.check_data_age(ts, universe, self.max_data_delay)
            elif self.strategy_cycle_trigger == StrategyCycleTrigger.trading_pair_data_availability:
                assert existing_universe is not None, "StrategyCycleTrigger.trading_pair_data_availability needs to retain the previous universe"
            else:
                raise NotImplementedError()
        else:
            # Recycle the universe instance
            logger.info("Reusing previously loaded universe: %s", existing_universe)
            universe = existing_universe

        # Run cycle checks
        if live:
            self.runner.pretick_check(ts, universe)

        if cycle == 1 and self.backtest_setup is not None:
            # The hook to set up backtest initial balance.
            # TODO: Legacy - remove.
            logger.info("Performing initial backtest account funding")
            self.backtest_setup(state, universe, self.sync_model)

        # TODO: This setup repeated in tick().
        # Modify tick() to take these as argument
        routing_state, pricing_model, valuation_model = self.runner.setup_routing(universe)

        if self.execution_context.mode.is_live_trading():
            # In live trading, the interest follows clock
            # (chain blocks)
            interest_timestamp = datetime.datetime.utcnow()
            logger.info("Doing live trading interest sync at %s", interest_timestamp)
        else:
            # In backtesting do discreet steps
            interest_timestamp = ts
            logger.info("Doing backtesitng interest sync at %s", interest_timestamp)

        interest_events = self.sync_model.sync_interests(
            interest_timestamp,
            state,
            cast(TradingStrategyUniverse, universe),
            pricing_model,
        )
        logger.info("Generated %d sync interest events", len(interest_events))

        if live:
            long_short_metrics_latest = (
                self.extract_long_short_stats_from_state(state)
            )
        else:
            long_short_metrics_latest = None

        # Execute the strategy tick and trades
        self.runner.tick(
            strategy_cycle_timestamp=ts,
            universe=universe,
            state=state,
            debug_details=debug_details,
            cycle_duration=cycle_duration,
            cycle=cycle,
            store=self.store,
            long_short_metrics_latest=long_short_metrics_latest,
            indicators=indicators,
        )

        # Update portfolio and position historical data tracking.
        #
        # Statistics are updated on live_positions().
        # However if any position was opened,
        # we need at least one good entry with valuation in
        # PositionStatistics().
        #
        # Thus we need to update statistics right after,
        # otherwise a stop loss can close the position
        # and it never get any good samples to position
        # statistics out of it.
        #
        # TODO: We have update_statistics() and
        # calculate_summary_statistics().
        # Rename other to avoid confusion.
        #
        if live:
            # To be extra careful,
            # save the state if we are going to crash
            # in statistics calculations, so we have a trace
            # of broadcasted transactions
            self.store.sync(state)

            # Perform post-execution accounting check
            # only if we had any trades
            trades = debug_details.get("approved_trades")
            if trades:
                # This will raise an exception if there are issues
                # TODO: Handle deposits during trade executoin
                try:
                    self.runner.check_balances_post_execution(
                        universe,
                        state,
                        cycle
                    )
                except UnexpectedAccountingCorrectionIssue as e:
                    raise RuntimeError(f"Execution aborted at cycle {ts} #{cycle} because on-chain balances were different what expected after executing the trades") from e

            assert long_short_metrics_latest, "long_short_metrics_latest cannot be None during live trading"

            update_statistics(
                datetime.datetime.utcnow(),
                state.stats,
                state.portfolio,
                ExecutionMode.real_trading,
                strategy_cycle_or_wall_clock=strategy_cycle_timestamp,
                long_short_metrics_latest=long_short_metrics_latest,
            )

        state.uptime.record_cycle_complete(cycle)

        # Check that state is good before writing it to the disk
        state.perform_integrity_check()

        # Store the current state to disk
        self.store.sync(state)

        if extra_debug_data is not None:
            debug_details.update(extra_debug_data)

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

        if len(state.portfolio.reserves) == 0:
            logger.info("The strategy has no reserves or deposits yet")

        routing_state, pricing_model, valuation_method = self.runner.setup_routing(universe)

        # TODO: this seems to be duplicated in tick()
        with self.timed_task_context_manager("revalue_portfolio_statistics"):
            logger.info("Updating position valuations")
            self.runner.revalue_state(clock, state, valuation_method)

        with self.timed_task_context_manager("update_statistics"):
            logger.info("Updating position statistics after revaluation")

            long_short_metrics_latest = (
                self.extract_long_short_stats_from_state(state)
            )
            update_statistics(clock, state.stats, state.portfolio, execution_mode, long_short_metrics_latest=long_short_metrics_latest)

        # Check that state is good before writing it to the disk
        state.perform_integrity_check()

        # Store the current state to disk
        self.store.sync(state)

    def extract_long_short_stats_from_state(self, state) -> StatisticsTable:
        """Extracts the latest long short metrics from the state and execution loop
        
        :param state: Current state for the strategy
        
        :return: StatisticsTable of the latest long short metrics
        """
        long_short_metrics_latest = None
        
        if self.execution_context.mode.is_live_trading():
            backtested_state = self.metadata.backtested_state if self.metadata else None
            backtest_cutoff = self.metadata.key_metrics_backtest_cut_off if self.metadata else datetime.timedelta(days=90)
            long_short_metrics_latest = serialise_long_short_stats_as_json_table(
                state, backtested_state
            )
            
        return long_short_metrics_latest

    def check_position_triggers(
          self,
          ts: datetime.datetime,
          state: State,
          universe: TradingStrategyUniverse
    ) -> List[TradeExecution]:
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

        if len(state.portfolio.reserves) == 0:
            logger.info("The strategy has no reserves or deposits yet")
            return []

        routing_state, pricing_model, valuation_method = self.runner.setup_routing(universe)
        
        long_short_metrics_latest = (
            self.extract_long_short_stats_from_state(state)
        )

        # Do stop loss checks for every time point between now and next strategy cycle
        trades = self.runner.check_position_triggers(
            ts,
            state,
            universe,
            pricing_model,
            routing_state,
            long_short_metrics_latest=long_short_metrics_latest,
        )

        # Check that state is good before writing it to the disk
        state.perform_integrity_check()

        # Store the current state to disk
        self.store.sync(state)

        return trades

    def warm_up_backtest(self) -> TradingStrategyUniverse:
        """Load backtesting trading universe.

        Display progress bars for data downloads.
        """
        logger.info("Warming up backesting, universe options are %s", self.universe_options)
        return self.universe_model.preload_universe(self.universe_options, self.execution_context)

    def warm_up_live_trading(self) -> TradingStrategyUniverse:
        """Load live trading universe.

        Display progress bars for data downloads.
        """
        logger.info(
            "Warming up live trading universe, universe options are %s, mode is %s",
            self.universe_options,
            self.execution_context,
        )
        assert self.execution_context.mode.is_live_trading()
        universe = self.universe_model.preload_universe(self.universe_options, self.execution_context)
        universe = cast(TradingStrategyUniverse, universe)
        logger.info("Warmed up universe %s", universe)
        return universe

    def run_backtest_trigger_checks(self,
                                    start_ts: datetime.datetime,
                                    end_ts: datetime.datetime,
                                    state: State,
                                    universe: TradingStrategyUniverse) -> Tuple[int, int]:
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

        :param universe:
            Trading universe containing price data for stoploss checks.

        :return:
            Tuple (take profit, stop loss) count triggered
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

        if isinstance(pricing_model, BacktestPricing):
            stop_loss_pricing_model = BacktestPricing(
                universe.backtest_stop_loss_candles,
                self.runner.routing_model,
                time_bucket=universe.backtest_stop_loss_time_bucket,
                allow_missing_fees=pricing_model.allow_missing_fees
            )
        elif isinstance(pricing_model, GenericPricing):
            # TODO: This needs have a test coverage / figured out if correct
            stop_loss_pricing_model = BacktestPricing(
                universe.backtest_stop_loss_candles,
                self.runner.routing_model,
                time_bucket=universe.backtest_stop_loss_time_bucket,
                allow_missing_fees=False
            )
        else:
            raise AssertionError(f"Don't know how to deal with {pricing_model}")

        # Do stop loss checks for every time point between now and next strategy cycle
        tp = 0
        sl = 0
        
        long_short_metrics_latest = (
            self.extract_long_short_stats_from_state(state)
        )
        
        assert long_short_metrics_latest == None, "long_short_metrics_latest must be None during backtesting"

        while ts < end_ts:
            logger.debug("Backtesting take profit/stop loss at %s", ts)
            trades = self.runner.check_position_triggers(
                ts,
                state,
                universe,
                stop_loss_pricing_model,
                routing_state,
                long_short_metrics_latest=long_short_metrics_latest,
            )
            for t in trades:
                if t.is_stop_loss():
                    sl += 1
                elif t.is_take_profit():
                    tp += 1
            ts += tick_size.to_timedelta()

        return tp, sl

    def run_backtest(self, state: State) -> dict:
        """Backtest loop."""

        if not state.name:
            state.name = self.name

        if self.backtest_end or self.backtest_start:
            assert self.backtest_start and self.backtest_end, f"If backtesting both start and end must be given, we have {self.backtest_start} - {self.backtest_end}"

        ts = self.backtest_start

        cycle = state.cycle

        range = self.backtest_end - self.backtest_start

        ts_format = "%Y-%m-%d"

        friendly_start = self.backtest_start.strftime(ts_format)
        friendly_end = self.backtest_end.strftime(ts_format)

        seconds = int(range.total_seconds())

        universe = self.warm_up_backtest()

        logger.info(
            "run_backtest(): Strategy is executed in backtesting mode\n"
            "  starting at %s\n"
            "  cycle duration is %s\n"
            "  execution context is %s\n"
            "  universe is %s\n",
            ts,
            self.cycle_duration.value,
            self.execution_context,
            universe,
        )

        assert universe is not None, "warm_up_backtest(): Failed to load trading universe in backtesting"

        # Allow backtest step to be overwritten from the command line
        if self.universe_options.candle_time_bucket_override:
            backtest_step = CycleDuration.from_timebucket(self.universe_options.candle_time_bucket_override)
        else:
            backtest_step = self.cycle_duration

        cycle_name = backtest_step.value

        assert backtest_step != CycleDuration.cycle_unknown

        assert isinstance(self.backtest_start, datetime.datetime)
        assert not isinstance(self.backtest_start, pd.Timestamp), f"Expected pandas.Timestamp, got {self.backtest_start.__class__}: {self.backtest_start}"
        assert not isinstance(self.backtest_end, pd.Timestamp)
        assert isinstance(self.backtest_end, datetime.datetime)
        assert self.backtest_start < self.backtest_end
        if universe.backtest_stop_loss_candles is not None:
            assert isinstance(universe.backtest_stop_loss_candles, GroupedCandleUniverse), f"Got {universe.backtest_stop_loss_candles.__class__}"

        state.backtest_data = BacktestData(
            start_at=self.backtest_start,
            end_at=self.backtest_end,
            decision_cycle_duration=backtest_step,
        )

        execution_test_hook =  self.execution_test_hook or ExecutionTestHook()

        # Throttle TQDM updates to 1 per second because
        # otherwise we crash PyCharm
        # https://stackoverflow.com/q/43288550/315168
        last_progress_update = datetime.datetime.utcfromtimestamp(0)
        progress_update_threshold = datetime.timedelta(seconds=0.1)
        last_update_ts = None  # The last pushed timestamp to tqdm
        trigger_checks = 0
        stop_losses = take_profits = 0

        def set_progress_bar_postfix(state, progress_bar, trade_count, cycle, take_profits, stop_losses):
            """Set the values for the progress bar."""
            assert progress_bar is not None
            rolling_profit = state.stats.get_naive_rolling_pnl_pct()
            progress_bar.set_postfix({
                "trades": trade_count,
                "cycles": cycle,
                "TPs": take_profits,
                "SLs": stop_losses,
                "PnL": f"{rolling_profit*100:.2f}%",
            })

        if not self.execution_context.grid_search and self.execution_context.progress_bars:
            # In grid search do not display
            # progress bar for individual backtests
            progress_bar = tqdm(total=seconds)
        else:
            # Grid search, do not do progress bar for this backtest
            progress_bar = None

        while True:
            ts = snap_to_previous_tick(ts, backtest_step)

            # Bump progress bar forward and update backtest status
            if datetime.datetime.utcnow() - last_progress_update > progress_update_threshold:
                friedly_ts = ts.strftime(ts_format)
                trade_count = len(list(state.portfolio.get_all_trades()))
                if progress_bar:
                    progress_bar.set_description(f"Backtesting {self.name}, {friendly_start} - {friendly_end} at {friedly_ts} ({cycle_name})")
                    set_progress_bar_postfix(state, progress_bar, trade_count, cycle, take_profits, stop_losses)
                last_progress_update = datetime.datetime.utcnow()
                if last_update_ts:
                    # Push update for the period
                    passed_seconds = (ts - last_update_ts).total_seconds()
                    if progress_bar:
                        progress_bar.update(int(passed_seconds))
                last_update_ts = ts

            execution_test_hook.on_before_cycle(
                cycle,
                ts,
                state,
                self.sync_model,
            )

            # Decide trades and everything for this cycle
            universe: TradingStrategyUniverse = self.tick(
                ts,
                backtest_step,
                state,
                cycle,
                live=False,
                strategy_cycle_timestamp=ts,
                existing_universe=universe,
                indicators=self.backtest_strategy_indicators,
            )

            # Revalue our portfolio
            self.update_position_valuations(ts, state, universe, self.execution_context.mode)

            # Check for termination in integration testing.
            # TODO: Get rid of this and only support date ranges to run tests
            if self.max_cycles is not None:
                if cycle >= self.max_cycles:
                    logger.info("Max backtest cycles reached")
                    break

            # Backtesting
            next_tick = snap_to_next_tick(ts + datetime.timedelta(seconds=1), backtest_step, self.tick_offset)

            # If we have stop loss checks enabled on a separate price feed,
            # run backtest stop loss checks until the next time
            if universe.backtest_stop_loss_candles is not None:
                res = self.run_backtest_trigger_checks(
                    ts,
                    next_tick,
                    state,
                    universe,
                )
                take_profits += res[0]
                stop_losses += res[1]
            
            if next_tick >= self.backtest_end:
                # Backteting has ended
                logger.info("Terminating backtesting. Backtest end %s, current timestamp %s", self.backtest_end, next_tick)
                trade_count = len(list(state.portfolio.get_all_trades()))
                passed_seconds = (ts - last_update_ts).total_seconds()
                if progress_bar:
                    set_progress_bar_postfix(state, progress_bar, trade_count, cycle, take_profits, stop_losses)
                    progress_bar.update(int(passed_seconds))
                break

            # Add some fuzziness to backtesting timestamps
            # TODO: Make this configurable - sub 1h strategies do not work
            ts = next_tick + datetime.timedelta(minutes=random.randint(0, 4))

            cycle += 1

        if progress_bar is not None:
            progress_bar.close()

        # Validate the backtest state at the end.
        # We want to avoid situation where we have stored
        # non-serialisable types in the state
        if not (self.execution_context.grid_search or self.execution_context.optimiser):
            # Save time in grid seach of not doing unnecessary validation
            # (Very unlikely to break)
            validate_state_serialisation(state)

        return self.debug_dump_state

    def run_live(self, state: State):
        """Run live trading cycle.

        :raise LiveSchedulingTaskFailed:
            If any of live trading concurrent tasks crashes with an exception
        """

        logger.info("run_live(): Strategy is executed in live trading mode")

        # Safety checks
        assert not self.is_backtest()
        assert self.backtest_start is None
        assert self.backtest_end is None

        # Start the watchdog process killer
        watchdog_registry = create_watchdog_registry(WatchdogMode.thread_based)
        start_background_watchdog(watchdog_registry)
        # Create a watchdog thread that checks that the live trading cycle
        # has completed for every candle + some tolerance minutes.
        # This will terminate the live trading process if it has hung for a reason or another.
        #T TODO: Added duration * 2 instead of duration * 1 to debug some issues.
        live_cycle_max_delay = (self.cycle_duration.to_timedelta() * 2 + datetime.timedelta(minutes=15)).total_seconds()
        register_worker(
            watchdog_registry,
            "live_cycle",
            live_cycle_max_delay)

        # Do not allow starting a strategy that has unclean state
        state.check_if_clean()

        logger.trade("The execution state was last saved %s", state.last_updated_at)

        if self.is_live_trading_unit_test():
            # Test app initialisation.
            # Do not start any background threads.
            logger.info("Unit test live trading checkup test detected - aborting.")
            return self.debug_dump_state

        cycle = state.cycle
        universe: Optional[TradingStrategyUniverse] = None
        execution_context = self.execution_context
        run_state: RunState = self.run_state
        crash_exception: Optional[Exception] = None

        tick_offset = self.tick_offset

        # We use trading pair data availability endpoint poll
        # to trigger the cycle.
        # Start the cycle warmup immediately,
        # but later wait down in the loop for the data availability.
        if self.strategy_cycle_trigger == StrategyCycleTrigger.trading_pair_data_availability:
            tick_offset = datetime.timedelta(0)

        assert execution_context, "ExecutionContext missing"

        universe = self.warm_up_live_trading()

        if self.sync_treasury_on_startup:
            reserve_assets = list(universe.reserve_assets)
            logger.info("Syncing treasury events for startup")
            self.sync_model.sync_treasury(
                datetime.datetime.utcnow(),
                state,
                reserve_assets,
            )
            self.store.sync(state)

        logger.info("Performing startup accounting check")
        self.runner.check_accounts(
            universe,
            state
        )

        # Store summary statistics in memory before doing anything else
        self.refresh_live_run_state(state, visualisation=self.visulisation, universe=universe, cycle_duration=self.cycle_duration)

        # A test path: do not wait until making the first trade
        # The first trade will be execute immediately, despite the time offset or tick
        if self.trade_immediately:
            ts = datetime.datetime.now()
            logger.info("Trade immediately triggered, using timestamp %s, cycle is %d", ts, cycle)
            universe = self.tick(ts, self.cycle_duration, state, cycle, live=True)

        def die(exc: Exception):
            # Shutdown the scheduler and mark an clean exit
            nonlocal crash_exception
            logger.exception(exc)
            scheduler.shutdown(wait=False)
            crash_exception = exc

        # Timed task to do the live trading cycles
        def live_cycle():
            nonlocal cycle
            nonlocal universe
            try:

                extra_debug_data = {}

                # Wall clock time
                unrounded_timestamp = datetime.datetime.utcnow()
                strategy_cycle_timestamp = snap_to_previous_tick(unrounded_timestamp, self.cycle_duration)

                logger.info("Executing live strategy cycle %d, now is %s, decision slot is %s",
                            cycle,
                            unrounded_timestamp,
                            strategy_cycle_timestamp
                            )

                # If we are in trigger mode, poll until we have data available
                # and then immediately trigger the decision
                if self.strategy_cycle_trigger == StrategyCycleTrigger.trading_pair_data_availability:
                    universe_update_result = wait_for_universe_data_availability_jsonl(
                        strategy_cycle_timestamp,
                        self.client,
                        universe,
                        max_wait=self.max_data_delay,
                    )
                    logger.trade("Strategy cycle %d, universe updated result received: %s", cycle, universe_update_result)
                    universe = universe_update_result.updated_universe
                    extra_debug_data["universe_update_poll_cycles"] = universe_update_result.poll_cycles

                    # Do a data lag check.
                    # This is not 100% fool-proof check for multipair strategies,
                    # as we randomly pick one pair. However it should detect most of market data feed
                    # stale situtations.
                    last_candle_timestamp = universe.data_universe.candles.df.iloc[-1]["timestamp"].to_pydatetime().replace(tzinfo=None)
                    # We allow 30 minutes + time bucket size lag
                    if last_candle_timestamp is not None:
                        max_allowed_lag = self.max_live_data_lag_tolerance + universe.data_universe.time_bucket.to_timedelta()
                        lag = strategy_cycle_timestamp - last_candle_timestamp
                        if lag > max_allowed_lag:
                            logger.error("Aborting and waiting for manual restart after the data feed is fixed")
                            raise RuntimeError(f"Strategy market data lag exceeded.\n"
                                               f"Currently lag to the start of the last candle is {lag}, allowed max lag is {max_allowed_lag}.\n"
                                               f"Last candle is at {last_candle_timestamp}")
                else:
                    # Force universe recreation on every cycle
                    universe = None

                # Shortcut universe downlado in forked mainnet test strategies
                if self.execution_context.mode == ExecutionMode.unit_testing_trading and not isinstance(self.execution_model, DummyExecutionModel):
                    # Dummy execution marks special test_trading_data_availability_based_strategy_cycle_trigger
                    universe = self.unit_testing_universe

                # Run the main strategy logic
                universe = self.tick(
                    unrounded_timestamp,
                    self.cycle_duration,
                    state,
                    cycle=cycle,
                    strategy_cycle_timestamp=strategy_cycle_timestamp,
                    existing_universe=universe,
                    live=True,
                    extra_debug_data=extra_debug_data,
                )

                if self.execution_context.mode == ExecutionMode.unit_testing_trading:
                    self.unit_testing_universe = universe

                logger.info("run_live() tick complete, universe is now %s", universe)

                # Post execution, update our statistics
                try:
                    # TODO: Visualisations are internally refreshed by runner
                    # this is somewhat bad architecture and refreshing run state should be a responsibility
                    # of a single component
                    self.refresh_live_run_state(state, cycle_duration=self.cycle_duration)
                except Exception as e:
                    # Failing to do the performance statistics is not fatal,
                    # because this does not contain any state changing events
                    logger.warning("refresh_live_run_state() failed in the live cycle: %s", e)
                    logger.exception(e)
                    pass

                # Go to sleep and
                # and advance to the next cycle
                cycle += 1
                state.cycle = cycle

            except Exception as e:
                die(e)

            # Unit testing mode
            # Used e.g. test_strategy_cycle_trigger.py
            if self.max_cycles is not None:
                if cycle >= self.max_cycles:
                    logger.info("Max cycles reached. Cycle %d, max %d", cycle, self.max_cycles)
                    scheduler.shutdown(wait=False)

            run_state.completed_cycle = cycle
            run_state.cycles += 1
            self.refresh_live_run_state(state, cycle_duration=self.cycle_duration)

            # Reset the background watchdog timer
            mark_alive(watchdog_registry, "live_cycle")

        # Timed task to update the valuation of open positions and collect statistics
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

                self.update_position_valuations(ts, state, universe, execution_context.mode)

                self.refresh_live_run_state(state, cycle_duration=self.cycle_duration)
            except Exception as e:
                die(e)

            run_state.position_revaluations += 1
            run_state.bumb_refreshed()

        # Timed task to do the stop loss checks
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
                die(e)

            run_state.position_trigger_checks += 1
            run_state.bumb_refreshed()

        # Set up live trading tasks using APScheduler
        executors = {
            'default': ThreadPoolExecutor(1),  # Background executor for core tasks
            'stats': ThreadPoolExecutor(1),  # Background executor for statistics calculations and visualisations
        }
        start_time = datetime.datetime(1970, 1, 1)

        # We use a single thread scheduler to run our various tasks.
        # Any task blocks other tasks - there is no parallerism or multithread support at the moment.
        # Multithread support would need making the architecture more complex with various locks
        # that could then be additional source of bugs.
        scheduler = BlockingScheduler(
            executors=executors,
            timezone=datetime.timezone.utc
        )

        if self.cycle_duration == CycleDuration.cycle_7d and self.max_cycles == None:
            # Assume 7d without offset is Monday midnight
            logger.info("Live cycle set to trigger Monday midnight")
            # https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.htmlgit-ad
            scheduler.add_job(
                live_cycle,
                'cron',  # Use fixed timepoint instead of internal
                day_of_week=0,
                hour=0,
                minute=0,
                second=0,
                misfire_grace_time = None,  # Will always run the job no matter how late it is
            )
        else:
            # Core live trade execution loop
            seconds = self.cycle_duration.to_timedelta().total_seconds()
            logger.info(
                "Live cycle set to trigger seconds %d, start time %s, offset %s",
                seconds,
                start_time,
                tick_offset
            )
            scheduler.add_job(
                live_cycle,
                'interval',
                seconds=seconds,
                start_date=start_time + tick_offset,
                misfire_grace_time = None,  # Will always run the job no matter how late it is
            )

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

        def listen_error(event):
            if event.exception:
                logger.info("Scheduled task received exception. event: %s, execption: %s", event, event.exception)
            else:
                logger.error("Should not happen")

        scheduler.add_listener(listen_error, EVENT_JOB_ERROR)

        # Display version information on the trade log
        version_info = self.run_state.version
        logger.trade(str(version_info))

        single_shot = self.trade_immediately and self.max_cycles == 1

        # Avoid starting a scheduler if we do --run-single-cycle
        if not single_shot:
            try:
                # https://github.com/agronholm/apscheduler/discussions/683
                scheduler.start()
            except KeyboardInterrupt:
                # https://github.com/agronholm/apscheduler/issues/338
                scheduler.shutdown(wait=False)
                raise
            except Exception as e:
                logger.error("Scheduler raised an exception %s", e)
                raise

            logger.info("Scheduler finished - down the live trading loop")

        if crash_exception:
            raise LiveSchedulingTaskFailed("trade-executor closed because one of the scheduled tasks failed") from crash_exception

        return self.debug_dump_state

    def setup(self) -> State:
        """Set up the main loop of trade executor.

        Main entry point to the loop.

        - Chooses between live and backtesting execution mode

        - Loads or creates the initial state

        - Sets up strategy runner

        :return:
            Loaded execution state
        """

        state = self.init_state()

        if not self.is_backtest():
            if not self.sync_model.is_ready_for_live_trading(state):
                raise RuntimeError(f"{self.sync_model} not initialised for live trading - run trade-executor init command first")

        self.init_execution_model()

        run_description: StrategyExecutionDescription = self.strategy_factory(
            execution_model=self.execution_model,
            execution_context=self.execution_context,
            timed_task_context_manager=self.timed_task_context_manager,
            sync_model=self.sync_model,
            valuation_model_factory=self.valuation_model_factory,
            pricing_model_factory=self.pricing_model_factory,
            approval_model=self.approval_model,
            client=self.client,
            routing_model=self.routing_model,
            run_state=self.run_state,
            create_indicators=self.create_indicators,
            parameters=self.parameters,
            visualisation=self.visulisation,
        )

        self.init_live_run_state(run_description)

        # Deconstruct strategy input
        self.runner: StrategyRunner = run_description.runner

        self.universe_model = run_description.universe_model

        # TODO: Do this only when doing backtesting in a notebook
        # self.set_start_and_end()

        # Load cycle_duration from v0.1 strategies,
        # if not given from the command line to override backtesting data
        if run_description.cycle_duration and not self.cycle_duration:
            self.cycle_duration = run_description.cycle_duration

        assert self.cycle_duration is not None, "Did not get strategy cycle duration from constructor or strategy run description"

        return state

    def set_backtest_start_and_end(self):
        """Set up backtesting start and end times. If no start, end, or lookback info provided, will set automatically to the entire available data range."""
        if self.minimum_data_lookback_range is not None:
            assert not self.backtest_start or not self.backtest_end, "You must not give start_at and end_at if you give minimum_data_lookback_range. minimum_data_lookback_range automatically ends at the current time."
            assert isinstance(self.minimum_data_lookback_range, datetime.timedelta), "minimum_data_lookback_range must be a datetime.timedelta"

            self.backtest_end = datetime.datetime.now()
            self.backtest_start = self.backtest_end - self.minimum_data_lookback_range

        # set automatically if not given
        if self.backtest_start is None:
            u = self.universe_model.construct_universe(
                self.backtest_start,
                self.execution_context.mode,
                self.universe_options,
            )

            if u.universe.candles is not None:
                s,e  = u.universe.candles.get_timestamp_range()
                self.backtest_start = s.to_pydatetime().replace(tzinfo=None)
                self.backtest_end = e.to_pydatetime().replace(tzinfo=None)

                logger.info("Automatically using %s - %s for backtest start and end", self.backtest_start, self.backtest_end)

    def run_with_state(self, state: State) -> dict:
        """Start the execution.

        :return:
            Debug state where each key is the cycle number

        :raise:
            Any exception thrown from this function should be considered as live execution error,
            not a start up error.
        """
        # TODO: Refactor
        if self.is_backtest():
            # Walk through backtesting range
            return self.run_backtest(state)
        else:
            return self.run_live(state)

    def run(self):
        """Start the execution.

        .. note::

            Legacy entry point
        """
        # TODO: Refactor
        state = self.setup()
        return self.run_with_state(state)

    def run_and_setup_backtest(self):
        state = self.setup()
        return self.run_backtest(state)

