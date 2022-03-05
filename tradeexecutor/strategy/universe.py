"""Construct the trading universe for the strategy."""

import datetime
from typing import Callable, List

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.state import Portfolio, AssetIdentifier

#: Create the trading universe for the a strategy runner
from tradeexecutor.strategy.runner import Dataset
from tradingstrategy.universe import Universe

UniverseConstructionMethod = Callable[[Dataset], Universe]
