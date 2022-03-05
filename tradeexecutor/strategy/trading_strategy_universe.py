import abc
import contextlib
import datetime
import textwrap
from dataclasses import dataclass
from typing import Optional
import logging
import pandas as pd

from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.runner import Dataset
from tradeexecutor.strategy.universe import TradeExecutorTradingUniverse, UniverseConstructor
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Contain raw loaded datasets."""

    time_frame: TimeBucket
    exchanges: ExchangeUniverse
    pairs: pd.DataFrame
    candles: pd.DataFrame
    liquidity: pd.DataFrame


class TradingStrategyUniverse(TradeExecutorTradingUniverse):
    universe: Universe


class TradingStrategyUniverseConstructor(UniverseConstructor):
    """A universe constructor that builds the trading universe data using Trading Strategy client.

    On a live exeuction, trade universe is reconstructor for the every tick,
    by refreshing the trading data from the server.
    """

    def __init__(self, client: Client, timed_task_context_manager: contextlib.AbstractContextManager):
        self.client = client
        self.timed_task_context_manager = timed_task_context_manager

    def log_universe(self, universe: Universe):
        """Log the state of the current universe.]"""
        data_start, data_end = universe.candles.get_timestamp_range()
        logger.info(textwrap.dedent(f"""
                Universe constructed.                    
                
                Time periods
                - Time frame {universe.time_frame.value}
                - Candle data: {data_start} - {data_end}
                
                The size of our trading universe is
                - {len(universe.exchanges)} exchanges
                - {universe.pairs.get_count()} pairs
                - {universe.candles.get_sample_count()} candles
                - {universe.liquidity.get_sample_count()} liquidity samples                
                """))
        return universe

    def load_data(self, time_frame: TimeBucket) -> Dataset:
        """Loads the server-side data using the client.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.

        :param lookback: how long to the past load data e.g. 1 year, 1 month. **Not implemented yet**.

        :return: None if not dataset for the strategy required
        """
        client = self.client
        with self.timed_task_context_manager("load_data", time_frame=time_frame.value):
            exchanges = client.fetch_exchange_universe()
            pairs = client.fetch_pair_universe().to_pandas()
            candles = client.fetch_all_candles(time_frame).to_pandas()
            liquidity = client.fetch_all_liquidity_samples(time_frame).to_pandas()
            return Dataset(
                time_frame=time_frame,
                exchanges=exchanges,
                pairs=pairs,
                candles=candles,
                liquidity=liquidity,
            )





