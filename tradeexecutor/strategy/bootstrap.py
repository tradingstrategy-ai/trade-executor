"""Loading Python strategy modules.

.. warning ::

    Deprecated.

See :py:mod:`strategy_module` instead.

"""
import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Optional

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
    """Initialises the strategy script file and hooks it to the executor.

    Assumes the module has two functions

    - `decide_trade`

    - `create_trading_universe`

    Hook this up the strategy execution system.
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
            **kwargs) -> StrategyExecutionDescription:

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
            accounting_checks=execution_context.mode.is_live_trading(),
            unit_testing=execution_context.mode.is_unit_testing(),
            create_indicators=create_indicators,
            parameters=parameters,
        )

        logger.info("Starting strategy runner in execution mode %s:\n%s", execution_context.mode.name, runner.__class__.__name__)

        return StrategyExecutionDescription(
            universe_model=universe_model,
            runner=runner,
            trading_strategy_engine_version=mod_info.trading_strategy_engine_version,
            cycle_duration=mod_info.trading_strategy_cycle,
            chain_id=mod_info.get_default_chain_id(),
            source_code=mod_info.source_code,
        )

    return default_strategy_factory


