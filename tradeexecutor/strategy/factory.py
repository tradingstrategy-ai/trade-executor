""""Strategy factory is the end point to the loaded strategies."""

import typing

from contextlib import AbstractContextManager

from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradingstrategy.client import Client


# For typing.Protocol see https://stackoverflow.com/questions/68472236/type-hint-for-callable-that-takes-kwargs
class StrategyFactory(typing.Protocol):
    """A callable that creates a new strategy when loaded from an external script."""

    # Only accept kwargs as per https://www.python.org/dev/peps/pep-3102/
    def __call__(
        *ignore,
        execution_model: ExecutionModel,
        sync_method: SyncMethod,
        pricing_model_factory: PricingModelFactory,
        client: typing.Optional[Client],
        timed_task_context_manager: AbstractContextManager,
        approval_model: ApprovalModel,
        **kwargs) -> StrategyExecutionDescription:
        pass
