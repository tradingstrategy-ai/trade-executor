import datetime
import runpy
from contextlib import AbstractContextManager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from queue import Queue
from typing import Optional, Callable, Tuple

from tradeexecutor.backtest.backtest_execution import BacktestExecutionModel
from tradeexecutor.backtest.backtest_pricing import BacktestSimplePricingModel
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_sync import BacktestSyncer
from tradeexecutor.backtest.backtest_valuation import BacktestValuationModel
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.state.state import State
from tradeexecutor.state.store import NoneStore
from tradeexecutor.strategy.approval import UncheckedApprovalModel, ApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionContext
from tradeexecutor.strategy.factory import make_runner_for_strategy_mod
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.strategy_module import parse_strategy_module, StrategyModuleInformation
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, TradingStrategyUniverseModel, \
    DefaultTradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import StaticUniverseModel
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


@dataclass
class BacktestSetup:
    """Describe backtest setup, ready to run."""

    #: Test start
    start_at: datetime.datetime

    #: Test end
    end_at: datetime.datetime

    #: Override trading_strategy_cycle from strategy module
    cycle_duration: Optional[CycleDuration]

    #: Override trading_strategy_cycle from strategy module
    candle_time_frame: Optional[TimeBucket]

    universe: Optional[TradingStrategyUniverse]
    wallet: SimulatedWallet
    state: State
    pricing_model: Optional[BacktestSimplePricingModel]
    routing_model: Optional[BacktestRoutingModel]
    execution_model: BacktestExecutionModel
    sync_method: BacktestSyncer
    strategy_module: StrategyModuleInformation

    def backtest_static_universe_strategy_factory(
            self,
            *ignore,
            execution_model: BacktestExecutionModel,
            execution_context: ExecutionContext,
            sync_method: BacktestSyncer,
            pricing_model_factory: Callable,
            valuation_model_factory: Callable,
            client: Client,
            timed_task_context_manager: AbstractContextManager,
            approval_model: ApprovalModel,
            **kwargs) -> StrategyExecutionDescription:
        """Create a strategy description and runner based on backtest parameters in this setup."""

        assert not execution_context.live_trading, f"This can be only used for backtesting strategies. execution context is {execution_context}"

        runner = PandasTraderRunner(
            timed_task_context_manager=timed_task_context_manager,
            execution_model=execution_model,
            approval_model=approval_model,
            valuation_model_factory=valuation_model_factory,
            sync_method=sync_method,
            pricing_model_factory=pricing_model_factory,
            routing_model=self.routing_model,
            decide_trades=self.strategy_module.decide_trades,
        )

        if self.universe:
            # Trading universe is set by unit tests
            universe_model = StaticUniverseModel(self.universe)
        else:
            # Trading universe is loaded by the strategy script
            universe_model = DefaultTradingStrategyUniverseModel(
                client,
                execution_context,
                self.strategy_module.create_trading_universe,
                candle_time_frame_override=self.candle_time_frame)

        return StrategyExecutionDescription(
            universe_model=universe_model,
            runner=runner,
            trading_strategy_engine_version=self.strategy_module.trading_strategy_engine_version,
            cycle_duration=self.cycle_duration,
        )


def setup_backtest_for_universe(
        strategy_path: Path,
        start_at: datetime.datetime,
        end_at: datetime.datetime,
        cycle_duration: CycleDuration,
        initial_deposit: int,
        universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
        max_slippage=0.01,
        validate_strategy_module=False,
    ):
    """High-level entry point for running a single backtest.

    The trading universe creation from the strategy is skipped,
    instead of you can pass your own universe e.g. synthetic universe.

    :param cycle_duration:
        Override the default strategy cycle duration
    """

    assert initial_deposit > 0

    wallet = SimulatedWallet()

    deposit_syncer = BacktestSyncer(wallet, Decimal(initial_deposit))

    # Create the initial state
    state = State()
    events = deposit_syncer(state.portfolio, start_at, universe.reserve_assets)
    assert len(events) == 1
    token, usd_exchange_rate = state.portfolio.get_default_reserve_currency()
    assert usd_exchange_rate == 1
    assert state.portfolio.get_current_cash() == initial_deposit

    # Set up execution and pricing
    pricing_model = BacktestSimplePricingModel(universe, routing_model)
    execution_model = BacktestExecutionModel(wallet, max_slippage)

    # Load strategy Python file
    strategy_mod_exports: dict = runpy.run_path(strategy_path)
    strategy_module = parse_strategy_module(strategy_mod_exports)

    if validate_strategy_module:
        # Allow partial strategies to be used in unit testing
        strategy_module.validate()

    return BacktestSetup(
        start_at,
        end_at,
        cycle_duration,
        wallet=wallet,
        state=state,
        universe=universe,
        pricing_model=pricing_model,
        execution_model=execution_model,
        routing_model=routing_model,
        sync_method=deposit_syncer,
        strategy_module=strategy_module,
    )


def setup_backtest(
        strategy_path: Path,
        start_at: datetime.datetime,
        end_at: datetime.datetime,
        initial_deposit: int,
        max_slippage=0.01,
        cycle_duration: Optional[CycleDuration]=None,
        candle_time_frame: Optional[TimeBucket]=None,
    ):
    """High-level entry point for running a single backtest.

    The trading universe creation from the strategy is skipped,
    instead of you can pass your own universe e.g. synthetic universe.

    :param cycle_duration:
        Override the default strategy cycle duration

    :param candle_time_frame:
        Override the default strategy candle time bucket
    """

    assert isinstance(strategy_path, Path), f"Got {strategy_path}"
    assert initial_deposit > 0

    wallet = SimulatedWallet()
    deposit_syncer = BacktestSyncer(wallet, Decimal(initial_deposit))

    execution_model = BacktestExecutionModel(wallet, max_slippage)

    # Load strategy Python file
    strategy_mod_exports: dict = runpy.run_path(strategy_path)
    strategy_module = parse_strategy_module(strategy_mod_exports)

    strategy_module.validate()

    return BacktestSetup(
        start_at,
        end_at,
        cycle_duration=cycle_duration or strategy_module.trading_strategy_cycle,  # Pick overridden cycle duration if provided
        candle_time_frame=candle_time_frame,
        wallet=wallet,
        state=State(),
        universe=None,
        pricing_model=None,  # Will be set up later
        execution_model=execution_model,
        routing_model=None, # Will be set up later
        sync_method=deposit_syncer,
        strategy_module=strategy_module,
    )


def run_backtest(setup: BacktestSetup, client: Optional[Client]=None) -> Tuple[State, dict]:
    """Run a strategy backtest.

    Loads strategy file, construct trading universe is real data
    downloaded with Trading Strategy client.

    :return:
        Tuple(the final state of the backtest, debug dump)
    """

    # State is pristine and not used yet
    assert len(list(setup.state.portfolio.get_all_trades())) == 0

    # Create empty state for this backtest
    store = NoneStore(setup.state)

    def pricing_model_factory(execution_model, universe, routing_model):
        return setup.pricing_model

    def valuation_model_factory(pricing_model):
        return BacktestValuationModel(setup.pricing_model)

    def backtest_setup(state: State, universe: TradingStrategyUniverse, deposit_syncer: BacktestSyncer):
        # Create the initial state
        events = deposit_syncer(state.portfolio, setup.start_at, universe.reserve_assets)
        assert len(events) == 1
        token, usd_exchange_rate = state.portfolio.get_default_reserve_currency()
        assert usd_exchange_rate == 1

    main_loop = ExecutionLoop(
        name="backtest",
        command_queue=Queue(),
        execution_model=setup.execution_model,
        sync_method=setup.sync_method,
        approval_model=UncheckedApprovalModel(),
        pricing_model_factory=pricing_model_factory,
        valuation_model_factory=valuation_model_factory,
        store=store,
        client=client,
        strategy_factory=setup.backtest_static_universe_strategy_factory,
        cycle_duration=setup.cycle_duration,
        stats_refresh_frequency=None,
        max_data_delay=None,
        debug_dump_file=None,
        backtest_start=setup.start_at,
        backtest_end=setup.end_at,
        backtest_setup=backtest_setup,
        tick_offset=datetime.timedelta(seconds=1),
        trade_immediately=True,
    )

    debug_dump = main_loop.run()

    return setup.state, debug_dump
