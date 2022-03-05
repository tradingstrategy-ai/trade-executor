"""Construct the trading universe for the strategy."""
import abc
import datetime
from typing import Callable, List

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.state import Portfolio, AssetIdentifier

#: Create the trading universe for the a strategy runner
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.runner import Dataset
from tradingstrategy.universe import Universe

UniverseConstructionMethod = Callable[[Dataset], Universe]


class TradeExecutorTradingUniverse:
    """Represents whatever data a strategy needs to have in order to make trading decisions."""


class UniverseConstructor(abc.ABC):
    """Create and manage trade universe.

    On a live execution, the trade universe is reconstructor for the every tick,
    by refreshing the trading data from the server.
    """

    @abc.abstractmethod
    def construct_universe(self, ts: datetime.datetime) -> TradeExecutorTradingUniverse:
        pass