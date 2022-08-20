"""Construct the trading universe for the strategy."""
import abc
import datetime
from dataclasses import dataclass
from typing import List, Optional

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.execution_context import ExecutionMode


class DataTooOld(Exception):
    """We try to execute live trades, but our data is too old for us to work with."""


@dataclass
class StrategyExecutionUniverse:
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

    def __post_init__(self):
        # Check that reserve assets look good
        for asset in self.reserve_assets:
            assert asset.token_symbol, f"Missing token symbol {asset}"
            assert asset.decimals, f"Missing token decimals {asset}"


class UniverseModel(abc.ABC):
    """Create and manage trade universe.

    On a live execution, the trade universe is reconstructor for the every tick,
    by refreshing the trading data from the server.
    """

    def preload_universe(self):
        """Triggered before backtesting execution.

        - Load all datasets with progress bar display

        - Data is saved in FS cache

        - Not triggered in live trading, as universe changes between cycles
        """

    @abc.abstractmethod
    def construct_universe(self, ts: datetime.datetime, mode: ExecutionMode) -> StrategyExecutionUniverse:
        """On each strategy tick, refresh/recreate the trading universe for the strategy.

        This is called in mainloop before the strategy tick. It needs to download
        any data updates since the last tick.

        :param mode:
            Are we live trading or backtesting.
        """

    def check_data_age(self, ts: datetime.datetime, universe: StrategyExecutionUniverse, best_before_duration: datetime.timedelta):
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

    def __init__(self, universe: StrategyExecutionUniverse):
        assert isinstance(universe, StrategyExecutionUniverse)
        self.universe = universe

    def construct_universe(self, ts: datetime.datetime, live: bool) -> StrategyExecutionUniverse:
        """Always return the same universe copy - there is no refresh."""
        return self.universe

