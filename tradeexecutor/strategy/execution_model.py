import abc
import datetime
from typing import List, Tuple

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution


class ExecutionModel(abc.ABC):
    """Define how trades are executed.

    See also :py:class:`tradeexecutor.strategy.mode.ExecutionMode`.
    """

    @abc.abstractmethod
    def preflight_check(self):
        """Check that we can start the trade executor

        :raise: AssertionError if something is a miss
        """

    @abc.abstractmethod
    def initialize(self):
        """Read any on-chain, etc., data to get synced.
        """

    @abc.abstractmethod
    def execute_trades(self, ts: datetime.datetime, state: State, trades: List[TradeExecution]) -> Tuple[List[TradeExecution], List[TradeExecution]]:
        """Execute the trades determined by the algo on a designed Uniswap v2 instance.

        :return: Tuple List of succeeded trades, List of failed trades
        """
        pass