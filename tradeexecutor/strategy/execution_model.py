import abc
import datetime
from typing import List, Tuple

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.universe_model import TradeExecutorTradingUniverse


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
    def execute_trades(self,
                       ts: datetime.datetime,
                       universe: TradeExecutorTradingUniverse,
                       state: State,
                       trades: List[TradeExecution]):
        """Execute the trades determined by the algo on a designed Uniswap v2 instance.

        :param ts:
            Timestamp of the trade cycle.

        :param universe:
            Current trading universe for this cycle.

        :param state:
            State of the trade executor.

        :param trades:
            List of trades decided by the strategy.
            Will be executed and modified in place.

        """
