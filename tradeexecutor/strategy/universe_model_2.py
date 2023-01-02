"""Construct the trading universe for the strategy."""
import abc
import datetime
import time
from dataclasses import dataclass
import logging
from typing import List, Optional, Set

import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import Exchange
from tradingstrategy.pair import DEXPair
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradingstrategy.types import PrimaryKey


logger = logging.getLogger(__name__)


class TradingDataDescription:
    """Describe the trading data the strategy needs."""

    maximum_data_age: pd.Timedelta

    chains: Set[ChainId]

    exchanges: Set[Exchange]

    pairs: Set[DEXPair]

    candle_bucket: Optional[TimeBucket] = None

    stop_loss_bucket: Optional[TimeBucket] = None

    liquidity_bucket: Optional[TimeBucket] = None


class TriggeredUniverseModel:
    """Trigger a strategy tick as soon as we receive new candle data from the oracle."""

    def __init__(self,
                 client: Client,
                 maximum_wait: pd.Timedelta):
        self.client = client




