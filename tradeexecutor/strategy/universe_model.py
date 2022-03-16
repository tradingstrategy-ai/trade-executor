"""Construct the trading universe for the strategy."""
import abc
import datetime
from dataclasses import dataclass
from typing import List

from tradeexecutor.state.state import AssetIdentifier


class DataTooOld(Exception):
    """We try to execute live trades, but our data is too old for us to work with."""


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
    def construct_universe(self, ts: datetime.datetime, live: bool) -> TradeExecutorTradingUniverse:
        """On each strategy tick, refresh/recreate the trading universe for the strategy.

        This is called in mainloop before the strategy tick. It needs to download
        any data updates since the last tick.

        :param live:
            The strategy is executed in live mode. Any cached data should be ignored.
        """

    def check_data_age(self, ts: datetime.datetime, universe: TradeExecutorTradingUniverse, best_before_duration: datetime.timedelta):
        """Check if our data is up-to-date and we do not have issues with feeds.

        Ensure we do not try to execute live trades with stale data.

        :raise DataTooOld: in the case data is too old to execute.
        """


class StaticUniverseModel(UniverseModel):
    """Universe that never changes and all assets are in in-process memory.

    Only useful for testing, because
    - any real trading pair universe is deemed to change
    - trade executor is deemed to go down and up again
    """

    def __init__(self, universe: TradeExecutorTradingUniverse):
        assert isinstance(universe, TradeExecutorTradingUniverse)
        self.universe = universe

    def construct_universe(self, ts: datetime.datetime, live: bool) -> TradeExecutorTradingUniverse:
        """Always return the same universe copy - there is no refresh."""
        return self.universe

