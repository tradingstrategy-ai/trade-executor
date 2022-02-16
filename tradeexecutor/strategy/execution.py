import abc
from typing import List

from tradeexecutor.state.state import State, TradeExecution


class ExecutionModel(abc.ABC):
    """Define how live trades are executed."""

    @abc.abstractmethod
    def execute_trades(self, state: State, trades: List[TradeExecution]):
        pass