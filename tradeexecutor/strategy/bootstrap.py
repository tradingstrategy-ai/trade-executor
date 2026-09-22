"""Loading Python strategy modules.

.. warning ::

    Deprecated.

See :py:mod:`strategy_module` instead.

"""
import logging
from contextlib import AbstractContextManager
from pathlib import Path

from tradeexecutor.state.types import Percent
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.pandas_trader.indicator import CreateIndicatorsProtocolV1, CreateIndicatorsProtocol
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.routing import RoutingModel
from tradingstrategy.client import Client

from tradeexecutor.ethereum.routing_data import get_routing_model
from tradeexecutor.strategy.sync_model import SyncMethodV0, SyncModel
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.factory import StrategyFactory
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import DefaultTradingStrategyUniverseModel
from tradeexecutor.strategy.valuation import ValuationModelFactory
from tradeexecutor.strategy.recorder.recorder import DecisionRecorder
from tradeexecutor.strategy.recorder.storage import validate_recorder_strategy_id

logger = logging.getLogger(__name__)


def import_strategy_file(path: Path) -> StrategyFactory:
    """Loads a strategy module and returns its factor function.

    All exports will be lowercased for further processing,
    so we do not care if constant variables are written in upper or lowercase.
    """
    logger.info("Importing strategy %s", path)
    assert isinstance(path, Path)
    mod_or_factory = read_strategy_module(path)

    if not isinstance(mod_or_factory, StrategyModuleInformation):
        # Legacy path, see read_strategy_module() comments
        return mod_or_factory

    return make_factory_from_strategy_mod(mod_or_factory)


def bootstrap_strategy(
        timed_task_context_manager: AbstractContextManager,
        path: Path,
        **kwargs) -> StrategyExecutionDescription:
    """Bootstrap a strategy to the point it can accept its first tick.

    Returns an initialized strategy.

    :param lookback: How much old data we load on bootstrap
    :param kwargs: Random arguments one can pass to factory / StrategyRunner constructor.
    :param now_: Override the current clock for testing

    :raises BadStrategyFile:
    :raises PreflightCheckFailed:

    :return: Tuple of loaded and initialized elements
    """
    factory = import_strategy_file(path)
    description = factory(
        timed_task_context_manager=timed_task_context_manager, **kwargs)
    return description


def make_factory_from_strategy_mod(mod: StrategyModuleInformation) -> StrategyFactory:
    """Create the executor factory for a loaded managed-positions strategy.

    :func:`import_strategy_file` calls this after validating a modern strategy
    module. The returned factory is later invoked by execution-loop setup to
    connect ``create_trading_universe()``, ``create_indicators()``, and
    ``decide_trades()`` to their framework models. Keeping this wiring in one
    factory also ensures optional live recording is created with the same state
    path and executor ID used by the CLI.

    :param mod:
        Validated strategy-module metadata and callbacks.
    :return:
        Keyword-only factory consumed by executor bootstrap.
    """

    mod_info = mod

    assert mod_info.trading_strategy_type == StrategyType.managed_positions, "Unsupported strategy tpe"

    assert mod_info, "chain_id blockchain information missing from the strategy module"

    def default_strategy_factory(
            *ignore,
            execution_model: ExecutionModel,
            execution_context: ExecutionContext,
            sync_model: SyncModel,
            pricing_model_factory: PricingModelFactory,
            valuation_model_factory: ValuationModelFactory,
            client: Client,
            timed_task_context_manager: AbstractContextManager,
            approval_model: ApprovalModel,
            run_state: RunState,
            routing_model: RoutingModel | None = None,
            create_indicators: CreateIndicatorsProtocol | None = None,
            parameters: StrategyParameters | None = None,
            visualisation=True,
            max_price_impact: Percent | None = None,
            state_path: Path | None = None,
            strategy_id: str | None = None,
            **kwargs) -> StrategyExecutionDescription:
        """Build the managed-positions runner for one execution loop.

        ``ExecutionLoop.setup()`` calls this factory with execution, pricing,
        universe, and state services. It creates ``DecisionRecorder`` only for
        an opted-in live strategy, then passes that same instance and the
        authoritative state path to ``PandasTraderRunner``. Backtests and
        undecorated strategies retain the ordinary runner path. Diagnostic
        callers without a persistent state path do not create a recorder.

        :param state_path:
            Persistent executor state path. A live strategy that enables
            ``@record_decision`` writes its live recorder database beside it.
        :param strategy_id:
            Executor identifier used for the recorder filename and run metadata.
            The strategy file stem is used when the caller has no identifier.
        """

        # Migration assert
        assert run_state, "run_state needs to be passed for new strategies"

        if ignore:
            # https://www.python.org/dev/peps/pep-3102/
            raise TypeError("Only keyword arguments accepted")

        if sync_model is not None:
            assert isinstance(sync_model, SyncModel), f"SyncModel not good: {sync_model}"

        universe_model = DefaultTradingStrategyUniverseModel(
            client,
            execution_context,
            mod_info.create_trading_universe)

        # Routing model can come with hardcoded Python tables of addresses (see default_routes.py)
        # or it is dynamically generated for any local dev chain.
        # If it is not dynamically generated, here set up one of the default routing models from
        # strategy module's trade_routing var.
        if routing_model is None:
            if mod_info.trade_routing == TradeRouting.default:
                # Left for later until we have the trading strategy universe downloaded
                # See EthereumExecutionModel.setup_routing()
                routing_model = None
            else:
                routing_model = get_routing_model(
                    execution_context,
                    mod_info.trade_routing,
                    mod_info.reserve_currency)

        if parameters is None:
            # Get Parameters instance from the strategy module
            parameters = mod_info.parameters

        create_indicators = mod_info.create_indicators or create_indicators

        recorder = None
        # The execution loop supplies its persistent state path. Diagnostic
        # commands also use live-data modes, but do not run decision cycles and
        # omit this path; they must not open a recorder or require one.
        if state_path is not None and execution_context.mode.is_live_trading() and getattr(mod_info.decide_trades, "__record_decision__", False):
            recorder_id = strategy_id or mod_info.path.stem
            recorder_id = validate_recorder_strategy_id(recorder_id)
            recorder_path = Path(state_path).parent / f"{recorder_id}-record.duckdb"
            logger.info("Strategy input recorder enabled: %s", recorder_path)
            recorder = DecisionRecorder(
                recorder_path,
                recorder_id,
                mod_info.source_code,
                strategy_file=str(mod_info.path),
                executor_revision=getattr(getattr(run_state, "version", None), "commit_hash", None),
            )

        runner = PandasTraderRunner(
            timed_task_context_manager=timed_task_context_manager,
            execution_model=execution_model,
            approval_model=approval_model,
            valuation_model_factory=valuation_model_factory,
            sync_model=sync_model,
            pricing_model_factory=pricing_model_factory,
            routing_model=routing_model,
            decide_trades=mod_info.decide_trades,
            execution_context=execution_context,
            run_state=run_state,
            accounting_checks=kwargs["check_accounts"] if kwargs.get("check_accounts") is not None else execution_context.mode.is_live_trading(),
            unit_testing=execution_context.mode.is_unit_testing(),
            create_indicators=create_indicators,
            parameters=parameters,
            visualisation=visualisation,
            max_price_impact=max_price_impact,
            recorder=recorder,
            state_path=state_path,
        )

        logger.info(
            "Starting strategy runner in execution mode %s:\n%s\nVisualisations are: %s\nmax_price_impact is: %s\nExecution model is: %s",
            execution_context.mode.name,
            runner.__class__.__name__,
            visualisation,
            max_price_impact,
            execution_model,
        )

        return StrategyExecutionDescription(
            universe_model=universe_model,
            runner=runner,
            trading_strategy_engine_version=mod_info.trading_strategy_engine_version,
            cycle_duration=mod_info.trading_strategy_cycle,
            chain_id=mod_info.get_default_chain_id(),
            source_code=mod_info.source_code,
        )

    return default_strategy_factory
