"""Strategy initialisation using factory design pattern.

Bind loaded strategies to the execution environment.
"""

from typing import Protocol, Optional

from contextlib import AbstractContextManager

from tradeexecutor.ethereum.default_routes import get_routing_model
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.strategy_module import parse_strategy_module
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import DefaultTradingStrategyUniverseModel
from tradingstrategy.client import Client

from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.valuation import ValuationModelFactory



class StrategyFactory(Protocol):
    """A callable that creates a new strategy when loaded from an external script."""

    # Only accept kwargs as per https://www.python.org/dev/peps/pep-3102/
    def __call__(
        *ignore,
        execution_model: ExecutionModel,
        sync_method: SyncMethod,
        pricing_model_factory: PricingModelFactory,
        valuation_model_factory: ValuationModelFactory,
        client: Optional[Client],
        timed_task_context_manager: AbstractContextManager,
        approval_model: ApprovalModel,
        routing_model: Optional[RoutingModel] = None,
        **kwargs) -> StrategyExecutionDescription:
        """

        :param ignore:

        :param execution_model:
            TODO

        :param sync_method:
            TODO

        :param pricing_model_factory:
            TODO

        :param valuation_model_factory:
            TODO

        :param client:
            TODO

        :param timed_task_context_manager:
            TODO

        :param approval_model:
            TODO

        :param routing_model:
            Strategy factory can create its own routing model, or accept the passed one.
            Passing it here is mainly used in the tests as a shortcut.

        :param kwargs:
        :return:
        """


def make_runner_for_strategy_mod(mod) -> StrategyFactory:
    """Initialises the strategy script file and hooks it to the executor.

    Assumes the module has two functions

    - `decide_trade`

    - `create_trading_universe`

    Hook this up the strategy execution system.
    """

    mod_info = parse_strategy_module(mod)
    mod_info.validate()

    assert mod_info.trading_strategy_type == StrategyType.managed_positions, "Unsupported strategy tpe"

    def default_strategy_factory(
            *ignore,
            execution_model: ExecutionModel,
            execution_context: ExecutionContext,
            sync_method: SyncMethod,
            pricing_model_factory: PricingModelFactory,
            valuation_model_factory: ValuationModelFactory,
            client: Client,
            timed_task_context_manager: AbstractContextManager,
            approval_model: ApprovalModel,
            **kwargs) -> StrategyExecutionDescription:

        if ignore:
            # https://www.python.org/dev/peps/pep-3102/
            raise TypeError("Only keyword arguments accepted")

        universe_model = DefaultTradingStrategyUniverseModel(
            client,
            execution_context,
            mod_info.create_trading_universe)

        routing_model = get_routing_model(mod_info.trade_routing, mod_info.reserve_currency)

        runner = PandasTraderRunner(
            timed_task_context_manager=timed_task_context_manager,
            execution_model=execution_model,
            approval_model=approval_model,
            valuation_model_factory=valuation_model_factory,
            sync_method=sync_method,
            pricing_model_factory=pricing_model_factory,
            routing_model=routing_model,
            decide_trades=mod_info.decide_trades,
        )

        return StrategyExecutionDescription(
            universe_model=universe_model,
            runner=runner,
            trading_strategy_engine_version=mod_info.trading_strategy_engine_version,
            cycle_duration=mod_info.trading_strategy_cycle,
        )

    return default_strategy_factory
