"""Synchrone deposits/withdrawals of the portfolio."""

import datetime
from typing import Callable, List

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier

#: Syncs the external portfolio changes from a (blockchain) source.
#: See ethereum/hotwallet_sync.py for details.
SyncMethod = Callable[[Portfolio, datetime.datetime, List[AssetIdentifier]], List[ReserveUpdateEvent]]
