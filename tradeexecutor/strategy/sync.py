"""Synchrone deposits/withdrawals of the portfolio."""

import datetime
from typing import Callable, List

from tradeexecutor.state.state import Portfolio, AssetIdentifier

SyncMethod = Callable[[Portfolio, datetime.datetime, List[AssetIdentifier]], None]
