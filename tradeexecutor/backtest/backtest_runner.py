import datetime
import logging
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
from tradeexecutor.cli.log import setup_notebook_logging
from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.ethereum.default_routes import get_routing_model, get_backtest_routing_model
from tradeexecutor.state.state import State
from tradeexecutor.state.store import NoneStore
from tradeexecutor.strategy.approval import UncheckedApprovalModel, ApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.strategy_module import parse_strategy_module,  \
    DecideTradesProtocol, CreateTradingUniverseProtocol, CURRENT_ENGINE_VERSION
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse,  \
    DefaultTradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import StaticUniverseModel, UniverseOptions
from tradeexecutor.utils.timer import timed_task
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
    universe_options: UniverseOptions

    #: Override trading_strategy_cycle from strategy module
    cycle_duration: Optional[CycleDuration]
    universe: Optional[TradingStrategyUniverse]
    wallet: SimulatedWallet
    state: State
    pricing_model: Optional[BacktestSimplePricingModel]
    routing_model: Optional[BacktestRoutingModel]
    execution_model: BacktestExecutionModel
    sync_method: BacktestSyncer

    trading_strategy_engine_version: str
    trade_routing: TradeRouting
    reserve_currency: ReserveCurrency
    decide_trades: DecideTradesProtocol
    create_trading_universe: Optional[CreateTradingUniverseProtocol]

    data_preload: bool = True

    #: Name for this backtest
    name: str = "backtest"

    # strategy_module: StrategyModuleInformation

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

        if self.routing_model:
            # Use passed routing model
            routing_model = self.routing_model
        else:
            # Use routing model from the strategy.
            # The strategy file chooses one of predefined routing models.
            trade_routing = self.trade_routing
            assert trade_routing, "Strategy module did not provide trade_routing"
            routing_model = get_backtest_routing_model(trade_routing, self.reserve_currency)

        runner = PandasTraderRunner(
            timed_task_context_manager=timed_task_context_manager,
            execution_model=execution_model,
            approval_model=approval_model,
            valuation_model_factory=valuation_model_factory,
            sync_method=sync_method,
            pricing_model_factory=pricing_model_factory,
            routing_model=routing_model,
            decide_trades=self.decide_trades,
        )

        if self.universe:
            # Trading universe is set by unit tests
            universe_model = StaticUniverseModel(self.universe)
        else:
            # Trading universe is loaded by the strategy script
            universe_model = DefaultTradingStrategyUniverseModel(
                client,
                execution_context,
                self.create_trading_universe)

        return StrategyExecutionDescription(
            universe_model=universe_model,
            runner=runner,
            trading_strategy_engine_version=self.trading_strategy_engine_version,
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
        candle_time_frame: Optional[TimeBucket]=None,
    ):
    """High-level entry point for setting up a single backtest for a predefined universe.

    The trading universe creation from the strategy is skipped,
    instead of you can pass your own universe e.g. synthetic universe.
    This is useful for running backtests against synthetic universes.

    :param cycle_duration:
        Override the default strategy cycle duration

    :param candle_time_frame:
        Override the default strategy candle time bucket

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
    pricing_model = BacktestSimplePricingModel(universe.universe.candles, routing_model)
    execution_model = BacktestExecutionModel(wallet, max_slippage)

    # Load strategy Python file
    strategy_mod_exports: dict = runpy.run_path(strategy_path)
    strategy_module = parse_strategy_module(strategy_mod_exports)

    if validate_strategy_module:
        # Allow partial strategies to be used in unit testing
        strategy_module.validate()

    universe_options = UniverseOptions(candle_time_bucket_override=candle_time_frame)

    return BacktestSetup(
        start_at=start_at,
        end_at=end_at,
        cycle_duration=cycle_duration,
        universe_options=universe_options,
        wallet=wallet,
        state=state,
        universe=universe,
        pricing_model=pricing_model,
        execution_model=execution_model,
        routing_model=routing_model,
        sync_method=deposit_syncer,
        decide_trades=strategy_module.decide_trades,
        create_trading_universe=None,
        reserve_currency=strategy_module.reserve_currency,
        trade_routing=strategy_module.trade_routing,
        trading_strategy_engine_version=strategy_module.trading_strategy_engine_version,
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
    """High-level entry point for setting up a backtest from a strategy module.

    This function is useful for running backtests for strategies in
    notebooks and tests.

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

    universe_options = UniverseOptions(candle_time_bucket_override=candle_time_frame)

    return BacktestSetup(
        start_at,
        end_at,
        cycle_duration=cycle_duration or strategy_module.trading_strategy_cycle,  # Pick overridden cycle duration if provided
        universe_options=universe_options,
        wallet=wallet,
        state=State(),
        universe=None,
        pricing_model=None,  # Will be set up later
        execution_model=execution_model,
        routing_model=None, # Will be set up later
        sync_method=deposit_syncer,
        decide_trades=strategy_module.decide_trades,
        create_trading_universe=strategy_module.create_trading_universe,
        reserve_currency=strategy_module.reserve_currency,
        trade_routing=strategy_module.trade_routing,
        trading_strategy_engine_version=strategy_module.trading_strategy_engine_version,
    )


def run_backtest(
        setup: BacktestSetup,
        client: Optional[Client]=None) -> Tuple[State, TradingStrategyUniverse, dict]:
    """Run a strategy backtest.

    Loads strategy file, construct trading universe is real data
    downloaded with Trading Strategy client.

    :return:
        Tuple(the final state of the backtest, trading universe, debug dump)
    """

    # State is pristine and not used yet
    assert len(list(setup.state.portfolio.get_all_trades())) == 0

    # Create empty state for this backtest
    store = NoneStore(setup.state)

    # Captured in teh callback
    backtest_universe: TradingStrategyUniverse = None

    def pricing_model_factory(execution_model, universe, routing_model):
        if setup.pricing_model:
            # Use pricing model given inline
            return setup.pricing_model

        # Construct a backtest pricing model
        return BacktestSimplePricingModel(universe, routing_model)

    def valuation_model_factory(pricing_model):
        return BacktestValuationModel(pricing_model)

    if not setup.universe:
        def backtest_setup(state: State, universe: TradingStrategyUniverse, deposit_syncer: BacktestSyncer):
            # Called on the first cycle. Only if the universe is not predefined.
            # Create the initial state of the execution.
            nonlocal backtest_universe
            events = deposit_syncer(state.portfolio, setup.start_at, universe.reserve_assets)
            assert len(events) == 1, f"Did not get 1 initial backtest deposit event, got {len(events)} events"
            token, usd_exchange_rate = state.portfolio.get_default_reserve_currency()
            assert usd_exchange_rate == 1
            backtest_universe = universe
    else:
        backtest_universe = setup.universe
        def backtest_setup(state: State, universe: TradingStrategyUniverse, deposit_syncer: BacktestSyncer):
            pass

    execution_context = ExecutionContext(
        mode=ExecutionMode.backtesting,
        timed_task_context_manager=timed_task,
    )

    main_loop = ExecutionLoop(
        name=setup.name,
        command_queue=Queue(),
        execution_model=setup.execution_model,
        execution_context=execution_context,
        sync_method=setup.sync_method,
        approval_model=UncheckedApprovalModel(),
        pricing_model_factory=pricing_model_factory,
        valuation_model_factory=valuation_model_factory,
        store=store,
        client=client,
        strategy_factory=setup.backtest_static_universe_strategy_factory,
        cycle_duration=setup.cycle_duration,
        stats_refresh_frequency=None,
        position_trigger_check_frequency=None,
        max_data_delay=None,
        debug_dump_file=None,
        backtest_start=setup.start_at,
        backtest_end=setup.end_at,
        backtest_setup=backtest_setup,
        tick_offset=datetime.timedelta(seconds=1),
        trade_immediately=True,
    )

    debug_dump = main_loop.run()

    return setup.state, backtest_universe, debug_dump


def run_backtest_inline(
    *ignore,
    start_at: datetime.datetime,
    end_at: datetime.datetime,
    client: Optional[Client],
    decide_trades: DecideTradesProtocol,
    cycle_duration: CycleDuration,
    initial_deposit: float,
    reserve_currency: ReserveCurrency,
    trade_routing: Optional[TradeRouting],
    create_trading_universe: Optional[CreateTradingUniverseProtocol]=None,
    universe: Optional[TradingStrategyUniverse]=None,
    routing_model: Optional[BacktestRoutingModel]=None,
    max_slippage=0.01,
    candle_time_frame: Optional[TimeBucket]=None,
    log_level=logging.WARNING,
    data_preload=True,
    name: str="backtest",
) -> Tuple[State, TradingStrategyUniverse, dict]:
    """Run backtests for given decide_trades and create_trading_universe functions.

    Does not load strategy from a separate .py file.
    Useful for running strategies directly from notebooks.

    :param name:
        Name for this backtest. If not set default to "backtest".

    :param start_at:
        When backtesting starts

    :param end_at:
        When backtesting ends

    :param client:
        You need to set up a Trading Strategy client for fetching the data

    :param decide_trades:
        Trade decider function of your strategy

    :param create_trading_universe:
        Universe creation function of your strategy.
        You must give either create_trading_universe or universe.

    :param universe:
        The pregenerated universe for this backtest.
        You must give either create_trading_universe or universe.

    :param cycle_duration:
        Strategy cycle duration

    :param candle_time_frame:
        Candles we use for this strategy

    :param initial_deposit:
        how much reserve currency we allocate as a capital at the beginning of the backtest

    :param reserve_currency:
        Reserve currency used for the strategy

    :param trade_routing:
        Routing model for trades

    :param routing_model:
        Use a predefined routing model.

    :param max_slippage:
        Max slippage tolerance for trades before execution failure

    :param log_level:
        Python logging level to display log messages during the backtest run.

    :param data_preload:
        Before the backtesting begins, load and cache datasets
        with nice progress bar to the user.

    :return:
        tuple (State of a completely executed strategy, trading strategy universe, debug dump dict)
    """

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    assert isinstance(start_at, datetime.datetime)
    assert isinstance(end_at, datetime.datetime)
    assert initial_deposit > 0

    # Setup our special logging level if not done yet.
    # (Not done when called from notebook)
    setup_notebook_logging(log_level)

    wallet = SimulatedWallet()
    deposit_syncer = BacktestSyncer(wallet, Decimal(initial_deposit))

    stop_loss_data_available = universe.has_stop_loss_data() if universe else False

    execution_model = BacktestExecutionModel(
        wallet,
        max_slippage,
        stop_loss_data_available=stop_loss_data_available,
    )

    if universe:
        if not routing_model:
            assert trade_routing, "You just give either routing_mode or trade_routing"
            assert reserve_currency, "Reserve current must be given to generate routing model"
            routing_model = get_backtest_routing_model(trade_routing, reserve_currency)
        pricing_model = BacktestSimplePricingModel(universe.universe.candles, routing_model)
    else:
        assert create_trading_universe, "Must give create_trading_universe if no universe given"
        pricing_model = None

    universe_options = UniverseOptions(
        candle_time_bucket_override=candle_time_frame,
    )

    backtest_setup = BacktestSetup(
        start_at,
        end_at,
        cycle_duration=cycle_duration,  # Pick overridden cycle duration if provided
        universe_options=universe_options,
        wallet=wallet,
        state=State(name=name),
        universe=universe,
        pricing_model=pricing_model,  # Will be set up later
        execution_model=execution_model,
        routing_model=routing_model,  # Use given routing model if available
        sync_method=deposit_syncer,
        decide_trades=decide_trades,
        create_trading_universe=create_trading_universe,
        reserve_currency=reserve_currency,
        trade_routing=trade_routing,
        trading_strategy_engine_version=CURRENT_ENGINE_VERSION,
        name=name,
        data_preload=data_preload,
    )

    return run_backtest(backtest_setup, client)

