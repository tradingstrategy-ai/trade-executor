"""Construct the trading universe for the strategy."""
import abc
import datetime
from dataclasses import dataclass
from typing import List

from tradeexecutor.state.state import AssetIdentifier


@dataclass
class TradeExecutorTradingUniverse:
    """Represents whatever data a strategy needs to have in order to make trading decisions.

    Any strategy specific subclass will handle candle/liquidity datasets.
    """

    #: The list of reserve assets used in this strategy.
    #:
    #: Currently we support only one reserve asset per strategy, though in the
    #: future there can be several.
    #:
    #: Usually return the list of a BUSD/USDC/similar stablecoin.
    reserve_assets: List[AssetIdentifier]


class UniverseModel(abc.ABC):
    """Create and manage trade universe.

    On a live execution, the trade universe is reconstructor for the every tick,
    by refreshing the trading data from the server.
    """

    @abc.abstractmethod
    def construct_universe(self, ts: datetime.datetime) -> TradeExecutorTradingUniverse:
        """On each strategy tick, refresh/recreate the trading universe for the strategy.

        This is called in mainloop before the strategy tick. It needs to download
        any data updates since the last tick.
        """



