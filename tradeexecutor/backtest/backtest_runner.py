import datetime
import logging
import runpy
from contextlib import AbstractContextManager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from queue import Queue
from typing import Optional, Callable, Tuple, Type
import logging

import pandas as pd

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_generic_router import EthereumBacktestPairConfigurator
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.backtest.legacy_backtest_sync import BacktestSyncer
from tradeexecutor.backtest.backtest_valuation import BacktestValuationModel
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.cli.log import setup_notebook_logging, setup_custom_log_levels, setup_strategy_logging
from tradeexecutor.cli.loop import ExecutionLoop, ExecutionTestHook
from tradeexecutor.ethereum.routing_data import get_routing_model, get_backtest_routing_model
from tradeexecutor.state.state import State
from tradeexecutor.state.store import NoneStore
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.approval import UncheckedApprovalModel, ApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode, standalone_backtest_execution_context
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.indicator import CreateIndicatorsProtocolV1, calculate_and_load_indicators, DiskIndicatorStorage, IndicatorSet, \
    IndicatorKey, load_indicators, CreateIndicatorsProtocol, call_create_indicators, IndicatorResultMap
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.strategy_module import parse_strategy_module, \
    DecideTradesProtocol, CreateTradingUniverseProtocol, CURRENT_ENGINE_VERSION, StrategyModuleInformation, DecideTradesProtocol2, read_strategy_module, \
    StrategyParameters, DecideTradesProtocol3, DecideTradesProtocol4
from tradeexecutor.strategy.engine_version import TradingStrategyEngineVersion
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse,  \
    DefaultTradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import StaticUniverseModel, UniverseOptions
from tradeexecutor.utils.accuracy import setup_decimal_accuracy
from tradeexecutor.utils.cpu import get_safe_max_workers_count
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class BacktestResult:
    """What we have after backtest"""

    #: The resulting state after the backtest run
    state: State

    #: Universe we loaded
    strategy_universe: TradingStrategyUniverse

    #: Python bag of values for internal testing
    diagnostics_data: dict

    #: Indicators we calculated for the backtest
    indicators: StrategyInputIndicators

    def __iter__(self):
        #: Legacy compatibility
        yield self.state
        yield self.strategy_universe
        yield self.diagnostics_data


@dataclass
class BacktestSetup:
    """Describe backtest setup, ready to run."""

    #: Test start
    #:
    #: Legacy. Use `UniverseOptions`.
    #:
    start_at: datetime.datetime | None

    #: Test end
    #:
    #: Legacy. Use `UniverseOptions`.
    #:
    end_at: datetime.datetime | None

    #: Override trading_strategy_cycle from strategy module
    universe_options: UniverseOptions

    #: Override trading_strategy_cycle from strategy module
    cycle_duration: Optional[CycleDuration]
    universe: Optional[TradingStrategyUniverse]
    wallet: SimulatedWallet
    state: State
    pricing_model: Optional[BacktestPricing]
    routing_model: Optional[BacktestRoutingModel]
    execution_model: BacktestExecution
    sync_model: BacktestSyncModel

    trading_strategy_engine_version: str
    trade_routing: TradeRouting
    reserve_currency: ReserveCurrency
    decide_trades: DecideTradesProtocol
    create_trading_universe: Optional[CreateTradingUniverseProtocol]

    #
    # Indicators needed for this backtest
    # - create_indicators() used with a single backtest
    # - indicators used by grid search as indicators have been defined and calculated in a prior step
    #
    create_indicators: Optional[CreateIndicatorsProtocolV1] = None
    indicator_combinations: Optional[set[IndicatorKey]] = None
    indicator_storage: Optional[DiskIndicatorStorage] = None

    data_preload: bool = True

    #: Name for this backtest
    name: str = "backtest"

    minimum_data_lookback_range: Optional[datetime.timedelta] = None

    # strategy_module: StrategyModuleInformation
    pair_configurator: Optional[EthereumBacktestPairConfigurator] = None

    #: Backtest/grid parameters
    #:
    #: When trading_strategy_engine_version >= 0.4
    #:
    parameters: StrategyParameters | None = None

    #: Is this backtest part a grid saerch
    #:
    grid_search: bool = False

    #: What's the execution mode of this backtest run
    #:
    mode: ExecutionMode = ExecutionMode.backtesting

    #: How many workers to use for indicator calculation
    #:
    max_workers: int = 8

    def backtest_static_universe_strategy_factory(
            self,
            *ignore,
            execution_model: BacktestExecution,
            execution_context: ExecutionContext,
            sync_model: BacktestSyncModel,
            pricing_model_factory: Callable,
            valuation_model_factory: Callable,
            client: Client,
            timed_task_context_manager: AbstractContextManager,
            approval_model: ApprovalModel,
            **kwargs) -> StrategyExecutionDescription:
        """Create a strategy description and runner based on backtest parameters in this setup."""

        logger.info("backtest_static_universe_strategy_factory(), engine version is %s", execution_context.engine_version)

        assert not execution_context.live_trading, f"This can be only used for backtesting strategies. execution context is {execution_context}"

        if self.universe:
            # Trading universe is set by unit tests
            universe_model = StaticUniverseModel(self.universe)
        else:
            # Trading universe is loaded by the strategy script
            universe_model = DefaultTradingStrategyUniverseModel(
                client,
                execution_context,
                self.create_trading_universe)

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
            sync_model=sync_model,
            pricing_model_factory=pricing_model_factory,
            routing_model=routing_model,
            decide_trades=self.decide_trades,
            execution_context=execution_context,
            parameters=self.parameters,
        )

        return StrategyExecutionDescription(
            universe_model=universe_model,
            runner=runner,
            trading_strategy_engine_version=self.trading_strategy_engine_version,
            cycle_duration=self.cycle_duration,
        )

    def prepare_indicators(self, execution_context: ExecutionContext) -> StrategyInputIndicators:
        """Prepare indicators for this backtest run.

        - Calculate and cache the indicator results

        - Display TQDM progress bar about reading cached results and calculating new indicators
        """
        if self.create_indicators is None:
            # Legacy - create_indicators() not defined
            # Empty indicator set
            return StrategyInputIndicators(
                self.universe,
                IndicatorSet(),
                {}
            )

        assert self.universe is not None, "You need to pass backtest_universe if you want to use create_indicators"
        assert self.parameters is not None, "You need to pass backtest parameters if you want to use create_indicators"
        assert self.indicator_storage, f"indicator_storage missing"

        storage = self.indicator_storage

        indicator_set = call_create_indicators(
            self.create_indicators,
            parameters=self.parameters,
            strategy_universe=self.universe,
            execution_context=execution_context,
        )

        assert indicator_set is not None, "create_indicators(): must return IndicatorSet object"
        assert isinstance(indicator_set, IndicatorSet)

        indicator_results = calculate_and_load_indicators(
            strategy_universe=self.universe,
            storage=storage,
            execution_context=execution_context,
            indicators=indicator_set,
            parameters=self.parameters,
            max_workers=self.max_workers,
            verbose=execution_context.progress_bars,
        )

        strategy_input_indicators = StrategyInputIndicators(
            self.universe,
            indicator_results=indicator_results,
            available_indicators=indicator_set,
        )

        return strategy_input_indicators

    def load_indicators(self) -> StrategyInputIndicators:
        """Load indicators for this backtest.

        - Applies for grid search execution path

        - All indicators must be precalculated in cache warm up
        """
        assert self.indicator_combinations is not None
        assert type(self.indicator_combinations) == set
        assert self.indicator_storage, f"indicator_storage missing"
        storage = self.indicator_storage
        available_indicators = IndicatorSet.from_indicator_keys(self.indicator_combinations)

        indicator_results = load_indicators(
            self.universe,
            storage,
            available_indicators,
            self.indicator_combinations,
            show_progress=False,  # Don't mess main grid search notebook progress bars
        )

        logger.info(
            "BacktestSetup.load_indicators() - loaded %d indicator result out of %d total results, %d indicators defined",
            len(indicator_results),
            len(self.indicator_combinations),
            available_indicators.get_count(),
        )

        if len(indicator_results) != len(self.indicator_combinations):
            for key in self.indicator_combinations:
                if key not in indicator_results:
                    cache_path = storage.get_indicator_path(key)
                    raise RuntimeError(f"Indicator key {key} could not be loaded - but is assumed to be on available. Cache path is {cache_path}")

        strategy_input_indicators = StrategyInputIndicators(
            self.universe,
            indicator_results=indicator_results,
            available_indicators=available_indicators,
        )

        return strategy_input_indicators


def setup_backtest_for_universe(
    strategy: Path | StrategyModuleInformation,
    start_at: datetime.datetime,
    end_at: datetime.datetime,
    cycle_duration: CycleDuration,
    initial_deposit: int | float,
    universe: TradingStrategyUniverse,
    routing_model: Optional[BacktestRoutingModel] = None,
    max_slippage=0.01,
    validate_strategy_module=False,
    candle_time_frame: Optional[TimeBucket]=None,
    allow_missing_fees=False,
    name: Optional[str] = None,
    universe_options: Optional[UniverseOptions] = None,
    create_indicators: CreateIndicatorsProtocol | None = None,
    parameters: StrategyParameters | None = None,
    indicator_storage: DiskIndicatorStorage | None = None,
    max_workers: int | None = None,
):
    """High-level entry point for setting up a single backtest for a predefined universe.

    The trading universe creation from the strategy is skipped,
    instead of you can pass your own universe e.g. synthetic universe.
    This is useful for running backtests against synthetic universes.

    :param cycle_duration:
        Override the default strategy cycle duration

    :param allow_missing_fees:
        Legacy workaround

    :param candle_time_frame:
        Override the default strategy candle time bucket

    """

    assert initial_deposit >= 0, f"Got initial deposit amount: {initial_deposit}"

    wallet = SimulatedWallet()

    # deposit_syncer = BacktestSyncer(wallet, Decimal(initial_deposit))
    sync_model = BacktestSyncModel(wallet, Decimal(initial_deposit))

    # Create the initial state
    state = State()
    events = sync_model.sync_treasury(start_at, state, universe.reserve_assets)
    # assert len(events) == 1
    token, usd_exchange_rate = state.portfolio.get_default_reserve_asset()
    assert usd_exchange_rate == 1
    assert state.portfolio.get_cash() == initial_deposit

    # Load strategy Python file
    if isinstance(strategy, Path):
        strategy_path = strategy
        strategy_mod_exports: dict = runpy.run_path(strategy_path)
        strategy_module = parse_strategy_module(strategy_path, strategy_mod_exports)
    elif isinstance(strategy, StrategyModuleInformation):
        strategy_module = strategy
    else:
        raise AssertionError(f"Unsupported {strategy}")

    if validate_strategy_module:
        # Allow partial strategies to be used in unit testing
        strategy_module.validate()
        assert strategy.trading_strategy_engine_version

    trade_routing = strategy_module.trade_routing

    stop_loss_data_available = False
    if universe:
        if universe.backtest_stop_loss_candles is not None:
            stop_loss_data_available = True

    # Check version to avoid issues with legacy code
    if trade_routing == TradeRouting.default and strategy_module.is_version_greater_or_equal_than(0, 3, 0):
        pair_configurator = EthereumBacktestPairConfigurator(universe)
        routing_model = GenericRouting(pair_configurator)
        pricing_model = GenericPricing(pair_configurator)
    else:
        # Set up execution and pricing
        pricing_model = BacktestPricing(universe.data_universe.candles, routing_model, allow_missing_fees=allow_missing_fees)

    execution_model = BacktestExecution(wallet, max_slippage, stop_loss_data_available=stop_loss_data_available)

    if universe_options is None:
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
        sync_model=sync_model,
        decide_trades=strategy_module.decide_trades,
        create_trading_universe=None,
        reserve_currency=strategy_module.reserve_currency,
        trade_routing=strategy_module.trade_routing,
        trading_strategy_engine_version=strategy_module.trading_strategy_engine_version,
        name=name,
        create_indicators=create_indicators,
        parameters=parameters,
        indicator_storage=indicator_storage,
        max_workers=max_workers or get_safe_max_workers_count(),
    )


def setup_backtest(
    strategy_path: Path,
    start_at: Optional[datetime.datetime] = None,
    end_at: Optional[datetime.datetime] = None,
    initial_deposit: Optional[USDollarAmount] = None,
    max_slippage: Optional[float] = 0.01,
    cycle_duration: Optional[CycleDuration]=None,
    candle_time_frame: Optional[TimeBucket]=None,
    strategy_module: Optional[StrategyModuleInformation]=None,
    name: Optional[str] = None,
    minimum_data_lookback_range: Optional[datetime.timedelta] = None,
    universe_options: Optional[UniverseOptions] = None,
    client: Optional[Client] = None,
) -> BacktestSetup:
    """High-level entry point for setting up a backtest from a strategy module.

    - This function is useful for running backtests for strategies in
      notebooks and unit tests

    - Instead of giving strategy and trading universe as direct function arguments,
      this entry point loads a strategy given as a Python file

    .. note ::

        A lot of arguments for this function are optional/
        unit test only/legacy. Only `strategy_path` is needed.

    See also

    - :py:func:`run_backtest_inline`

    :param strategy_path:
        Path to the strategy Python module


    :param end_at:
        Legacy. Use universe_options.

    :param max_slippage:
        Legacy

    :param cycle_duration:
        Override the default strategy cycle duration

    :param candle_time_frame:
        Legacy. Use universe_options.

        Override the default strategy candle time bucket

    :param strategy_module:
        If strategy module was previously loaded

    :param initial_deposit:
        Legacy.

        Override INITIAL_CASH from the strategy module.
    """

    assert max_slippage >= 0, f"You must give max slippage. Got max slippage {max_slippage}"

    assert isinstance(strategy_path, Path), f"Got {strategy_path}"

    # Load strategy Python file
    if strategy_module is None:
        # strategy_mod_exports: dict = runpy.run_path(strategy_path)
        # strategy_module = parse_strategy_module(strategy_path, strategy_mod_exports)
        strategy_module = read_strategy_module(strategy_path)

    if not initial_deposit:
        initial_deposit = strategy_module.initial_cash

    assert initial_deposit, "Initial cash not given as argument or strategy module"
    assert initial_deposit > 0, "Must have money"

    # Just in case we have not done this yet
    setup_custom_log_levels()

    wallet = SimulatedWallet()
    # deposit_syncer = BacktestSyncer(wallet, Decimal(initial_deposit))
    sync_model = BacktestSyncModel(wallet, Decimal(initial_deposit))

    if strategy_module.is_version_greater_or_equal_than(0, 2, 0):
        # Backtest variables were injected later in the development
        strategy_module.validate_backtest()
    else:
        strategy_module.validate()

    if universe_options is None:
        universe_options = UniverseOptions(
            candle_time_bucket_override=candle_time_frame,
            start_at=strategy_module.backtest_start or start_at,
            end_at=strategy_module.backtest_end or end_at,
        )

    if not name:
        name = strategy_module.name or f"Backtest for {strategy_module.path.stem}"

    stop_loss_data_available = False

    if client is not None:
        logger.info("Loading backtesting universe data for %s", universe_options)
        universe = strategy_module.create_trading_universe(
            pd.Timestamp.utcnow(),
            client,
            standalone_backtest_execution_context,
            universe_options,
        )
        stop_loss_data_available = universe.has_stop_loss_data()
    else:
        universe = None

    if universe is not None and strategy_module.trade_routing == TradeRouting.default:
        pair_configurator = EthereumBacktestPairConfigurator(universe)
        routing_model = GenericRouting(pair_configurator)
        pricing_model = GenericPricing(pair_configurator)
    else:
        routing_model = None
        pricing_model = None

    execution_model = BacktestExecution(
        wallet,
        max_slippage,
        stop_loss_data_available=stop_loss_data_available,
    )

    return BacktestSetup(
        universe_options.start_at,
        universe_options.end_at,
        cycle_duration=cycle_duration or strategy_module.trading_strategy_cycle,  # Pick overridden cycle duration if provided
        universe_options=universe_options,
        wallet=wallet,
        state=State(name=name),
        universe=universe,
        pricing_model=pricing_model,
        execution_model=execution_model,
        routing_model=routing_model,
        sync_model=sync_model,
        decide_trades=strategy_module.decide_trades,
        create_trading_universe=strategy_module.create_trading_universe,
        reserve_currency=strategy_module.reserve_currency,
        trade_routing=strategy_module.trade_routing,
        trading_strategy_engine_version=strategy_module.trading_strategy_engine_version,
        name=name,
        minimum_data_lookback_range=minimum_data_lookback_range,
    )


def run_backtest(
    setup: BacktestSetup,
    client: Optional[Client]=None,
    allow_missing_fees=False,
    execution_test_hook: Optional[ExecutionTestHook] = None,
    execution_context: ExecutionContext | None = None,
) -> BacktestResult:
    """Run a strategy backtest.

    Loads strategy file, construct trading universe is real data
    downloaded with Trading Strategy client.

    :param allow_missing_fees:
        Legacy workaround

    :return:
        Tuple(the final state of the backtest, trading universe, debug dump)
    """

    # State is pristine and not used yet
    assert len(list(setup.state.portfolio.get_all_trades())) == 0

    # Create empty state for this backtest
    store = NoneStore(setup.state)

    # Captured in teh callback
    backtest_universe: TradingStrategyUniverse = None

    def pricing_model_factory(execution_model, universe: TradingStrategyUniverse, routing_model):
        if setup.pricing_model:
            # Use pricing model given inline
            return setup.pricing_model

        return BacktestPricing(
            universe,
            routing_model,
            data_delay_tolerance=guess_data_delay_tolerance(universe),
            allow_missing_fees=allow_missing_fees,
        )

    def valuation_model_factory(pricing_model):
        return BacktestValuationModel(pricing_model)

    if not setup.universe:

        def backtest_setup(state: State, universe: TradingStrategyUniverse, sync_model: BacktestSyncModel):
            # Use strategy script create_trading_universe() hook to construct the universe
            # Called on the first cycle. Only if the universe is not predefined.
            # Create the initial state of the execution.
            nonlocal backtest_universe

            # Mark backtest stop loss data being available,
            # after create_trading_universe() has loaded it
            if universe.has_stop_loss_data():
                setup.execution_model.stop_loss_data_available = True

            #events = deposit_syncer(state.portfolio, setup.start_at, universe.reserve_assets)
            #assert len(events) == 1, f"Did not get 1 initial backtest deposit event, got {len(events)} events.\nMake sure you did not call backtest_setup() twice?"

            events = sync_model.sync_treasury(setup.start_at, state, list(universe.reserve_assets))
            # assert len(events) == 1, f"Did not get 1 initial backtest deposit event, got {len(events)} events.\nMake sure you did not call backtest_setup() twice?"

            token, usd_exchange_rate = state.portfolio.get_default_reserve_asset()
            assert usd_exchange_rate == 1
            backtest_universe = universe
    else:
        backtest_universe = setup.universe

        def backtest_setup(state: State, universe: TradingStrategyUniverse, deposit_syncer: BacktestSyncer):
            pass

    if execution_context is None:
        execution_context = ExecutionContext(
            mode=setup.mode,
            timed_task_context_manager=timed_task,
            engine_version=setup.trading_strategy_engine_version,
            parameters=setup.parameters,
            grid_search=setup.grid_search,
        )

    if execution_context.is_version_greater_or_equal_than(0, 5, 0):
        # Needed for DecideTradesProtocolV4
        if setup.create_indicators is not None:
            # Indicators need to be created now
            backtest_strategy_indicators = setup.prepare_indicators(execution_context)
        elif setup.indicator_combinations is not None:
            # Grid search
            # Indicators were created earlier, load now
            backtest_strategy_indicators = setup.load_indicators()
        else:
            raise AssertionError(f"run_backtest(): You must give either create_indicators or indicator_combinations argument")
    else:
        # Legacy
        backtest_strategy_indicators = None

    main_loop = ExecutionLoop(
        name=setup.name,
        command_queue=Queue(),
        execution_model=setup.execution_model,
        execution_context=execution_context,
        sync_model=setup.sync_model,
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
        backtest_candle_time_frame_override=setup.universe_options.candle_time_bucket_override,
        tick_offset=datetime.timedelta(seconds=1),
        trade_immediately=True,
        execution_test_hook=execution_test_hook,
        minimum_data_lookback_range=setup.minimum_data_lookback_range,
        universe_options=setup.universe_options,
        backtest_strategy_indicators=backtest_strategy_indicators,
    )

    diagnostics_data = main_loop.run_and_setup_backtest()

    # Expose to the caller through non-API.
    # Don't serialise this in grid search to make it faster/save space
    if not execution_context.grid_search:
        diagnostics_data["indicators"] = backtest_strategy_indicators

    # We are no longer in an active timeframe,
    # prevent using any stale timestamp we have.
    # STrategyInputIndicators has some internal asserts to detect timestamp = None,
    if backtest_strategy_indicators:
        backtest_strategy_indicators.timestamp = None

    result = BacktestResult(
        state=setup.state,
        strategy_universe=backtest_universe,
        diagnostics_data=diagnostics_data,
        indicators=backtest_strategy_indicators
    )

    return result


def run_backtest_inline(
    *ignore,
    start_at: Optional[datetime.datetime] = None,
    end_at: Optional[datetime.datetime] = None,
    minimum_data_lookback_range: Optional[datetime.timedelta] = None,
    client: Optional[Client],
    decide_trades: DecideTradesProtocol | DecideTradesProtocol2 | DecideTradesProtocol3 | DecideTradesProtocol4,
    create_trading_universe: CreateTradingUniverseProtocol = None,
    create_indicators: CreateIndicatorsProtocol = None,
    indicator_combinations: set[IndicatorKey] | None = None,
    indicator_storage: DiskIndicatorStorage | None = None,
    cycle_duration: CycleDuration | None = None,
    initial_deposit: float | None = None,
    reserve_currency: ReserveCurrency | None = None,
    trade_routing: Optional[TradeRouting] | None = None,
    universe: Optional[TradingStrategyUniverse] = None,
    routing_model: Optional[BacktestRoutingModel] = None,
    max_slippage=0.01,
    candle_time_frame: Optional[TimeBucket] = None,
    log_level=logging.WARNING,
    data_preload=True,
    data_delay_tolerance: Optional[pd.Timedelta] = None,
    name: str="backtest",
    allow_missing_fees=False,
    engine_version: Optional[TradingStrategyEngineVersion] = None,
    strategy_logging=False,
    parameters: Type | StrategyParameters | None = None,
    mode: ExecutionMode = ExecutionMode.backtesting,
    max_workers=8,
    grid_search=False,
    execution_context=standalone_backtest_execution_context,
) -> BacktestResult:
    """Run backtests for given decide_trades and create_trading_universe functions.

    Does not load strategy from a separate .py file.
    Useful for running strategies directly from notebooks.

    :param name:
        Name for this backtest. If not set default to "backtest".

    :param start_at:
        When backtesting starts.

        If not given take the date of the first candle.

    :param end_at:
        When backtesting ends.

        If not given take the date of the last candle.

    :param minimum_data_lookback_range:
        If start_at and end_at are not given, use this range to determine the backtesting period. Cannot be used with start_at and end_at. Automatically ends at the current time.

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

    :param data_delay_tolerance:
        What is the maximum hours/days lookup we allow in the backtesting when we ask for the latest price of an asset.

        The asset price fetch might fail due to sparse candle data - trades have not been made or the blockchain was halted during the price look-up period.
        Because there are no trades we cannot determine what was the correct asset price using {data_delay_tolerance} data tolerance delay.

        The default value `None` tries to guess the value based on the univerity candle timeframe,
        but often this guess is incorrect as only analysing every pair data gives a correct answer.

        The workarounds include ignoring assets in your backtest that might not have price data (look up they have enough valid candles
        at the decide_trades timestamp) or simply increasing this parameter.

        This parameter is passed to :py:class:`tradeexecutor.backtest.backtest_pricing.BacktestSimplePricingModel`.

    :param allow_missing_fees:
        Allow synthetic data to lack fee information.

        Only set in legacy backtests.

    :param engine_version:
        The used TS engine version/

        See :py:mod:`tradeexecutor.strategy.engine_version`.

    :param strategy_logging:
        Enable PositionManager log output.

        Set `True` to display output.
        See :py:meth:`tradeexecutor.strategy.pandas_trading.position_manager.PositionManager.log` for usage.

    :param parameters:
        Strategy parameters as a singleton class.

        Allows using the same parameter structure for single backtest runs and gridtesting.

        See :py:class:`tradeexecutor.strategy.parameters.StrategyParameters`.

    :param max_workers:
        Maximum number of worker processes to use to calculate indicators.

        Set to `1` to disable multiprocessing / debug.

    :return:
        tuple (State of a completely executed strategy, trading strategy universe, debug dump dict)
    """

    from tradeexecutor.monkeypatch import cloudpickle_patch  # Enable pickle patch that allows multiprocessing in notebooks 

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    if parameters is not None:
        if type(parameters) == type:
            # Class like definition
            parameters = StrategyParameters.from_class(parameters, grid_search=False)
        else:
            assert isinstance(parameters, StrategyParameters)

        parameters.validate_backtest()

    if start_at is None and end_at is None:
        if parameters and hasattr(parameters, "backtest_start") and hasattr(parameters, "backtest_end"):
            start_at = parameters.backtest_start
            end_at = parameters.backtest_end
        else:
            start_at, end_at = universe.data_universe.candles.get_timestamp_range()
            start_at = start_at.to_pydatetime()
            end_at = end_at.to_pydatetime()

    if start_at:
        assert isinstance(start_at, datetime.datetime)
        assert end_at, "You must give end_at if you give start_at"
    
    if end_at:
        assert isinstance(end_at, datetime.datetime)
        assert start_at, "You must give start_at if you give end_at"

    if universe:
        assert isinstance(universe, TradingStrategyUniverse)

    if trade_routing is None:
        trade_routing = TradeRouting.default

    if trade_routing == TradeRouting.default:
        assert universe is not None, "Cannot do generic routing in backtesting without universe"

    if cycle_duration is None:
        assert parameters, f"You need to give either cycle_duration or parameters argument"
        cycle_duration = parameters.cycle_duration

    if initial_deposit is None:
        assert parameters, f"You need to give either initial_deposit or parameters argument"
        initial_deposit = parameters.initial_cash

    assert initial_deposit > 0

    # Setup our special logging level if not done yet.
    # (Not done when called from notebook)
    if strategy_logging:
        setup_strategy_logging()
    else:
        setup_notebook_logging(log_level)

    # Make sure no rounding bugs
    setup_decimal_accuracy()

    wallet = SimulatedWallet()
    # deposit_syncer = BacktestSyncer(wallet, Decimal(initial_deposit))
    sync_model = BacktestSyncModel(wallet, Decimal(initial_deposit))

    stop_loss_data_available = universe.has_stop_loss_data() if universe else False

    execution_model = BacktestExecution(
        wallet,
        max_slippage,
        stop_loss_data_available=stop_loss_data_available,
    )

    if universe:

        if data_delay_tolerance is None:
            data_delay_tolerance = guess_data_delay_tolerance(universe)

        pair_configurator = EthereumBacktestPairConfigurator(universe, data_delay_tolerance=data_delay_tolerance)

        if trade_routing == TradeRouting.default:
            routing_model = GenericRouting(pair_configurator)

        elif not routing_model:
            assert trade_routing, "You just give either routing_mode or trade_routing"
            routing_model = get_backtest_routing_model(trade_routing, reserve_currency)

        if trade_routing == TradeRouting.default:
            pricing_model = GenericPricing(
                pair_configurator,
            )
        else:
            pricing_model = BacktestPricing(
                universe.data_universe.candles,
                routing_model,
                data_delay_tolerance=data_delay_tolerance,
                allow_missing_fees=allow_missing_fees,
            )
    else:
        assert create_trading_universe, "Must give create_trading_universe if no universe given"
        pricing_model = None

    universe_options = UniverseOptions(
        candle_time_bucket_override=candle_time_frame,
        start_at=start_at,
        end_at=end_at,
    )

    # Create default storage
    if indicator_storage is None and universe is not None:
        indicator_storage = DiskIndicatorStorage.create_default(universe)

    # Backwardd compatibility for the code that never set execution context
    if execution_context.engine_version != engine_version:
        execution_context.engine_version = engine_version

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
        sync_model=sync_model,
        decide_trades=decide_trades,
        create_trading_universe=create_trading_universe,
        indicator_combinations=indicator_combinations,
        indicator_storage=indicator_storage,
        create_indicators=create_indicators,
        reserve_currency=reserve_currency,
        trade_routing=trade_routing,
        trading_strategy_engine_version=engine_version,
        name=name,
        data_preload=data_preload,
        minimum_data_lookback_range=minimum_data_lookback_range,
        parameters=parameters,
        mode=mode,
        max_workers=max_workers,
        grid_search=grid_search,
    )

    result = run_backtest(
        backtest_setup,
        client,
        allow_missing_fees=True,
        execution_context=execution_context,
    )

    result.diagnostics_data["wallet"] = wallet

    #: TODO: Hack to pass the backtest data range to the grid search
    #:
    result.strategy_universe.options = universe_options
    return result


def guess_data_delay_tolerance(universe: TradingStrategyUniverse) -> pd.Timedelta:
    """Try to dynamically be flexible with the backtesting pricing look up.

    This could work around some data quality issues or early historical data.
    """
    if universe.data_universe.time_bucket == TimeBucket.d7:
        data_delay_tolerance = pd.Timedelta("9d")
    else:
        data_delay_tolerance = pd.Timedelta("2d")

    return data_delay_tolerance


