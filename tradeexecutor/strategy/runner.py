"""Define a live strategy execution model."""

import abc
import datetime
import textwrap
from contextlib import AbstractContextManager
import logging
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
from typing import List, ContextManager, Optional

from tradingstrategy.client import Client

from tradeexecutor.state.state import State, TradeExecution
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


logger = logging.getLogger(__name__)


class PreflightCheckFailed(Exception):
    """Something was wrong with the datafeeds."""


@dataclass
class Dataset:
    """Contain raw loaded datasets."""

    time_frame: TimeBucket
    exchanges: ExchangeUniverse
    pairs: pd.DataFrame
    candles: pd.DataFrame
    liquidity: pd.DataFrame



class StrategyRunner(abc.ABC):
    """A base class for a strategy live trade executor."""

    def __init__(self, timed_task_context_manager: AbstractContextManager):
        self.timed_task_context_manager = timed_task_context_manager

    def load_data(self, time_frame: TimeBucket, client: Client, lookback: Optional[datetime.timedelta]=None) -> Optional[Dataset]:
        """Loads the server-side data using the client.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.

        :param lookback: how long to the past load data e.g. 1 year, 1 month. **Not implemented yet**.

        :return: None if not dataset for the strategy required
        """
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

    def setup_universe_timed(self, dataset: Optional[Dataset]) -> Optional[Universe]:
        """Time the setting up of the universe.

        :return: None if no universe for the strategy required
        """
        if dataset:
            with self.timed_task_context_manager("setup_universe", time_frame=dataset.time_frame.value):
                universe = self.construct_universe(dataset)
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
        else:
            logger.info("Strategy did not load dataset, universe not generated")
            return None

    @abc.abstractmethod
    def construct_universe(self, dataset: Dataset) -> Universe:
        """Sets up pairs, candles and liquidity samples.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.
        :return:
        """

    @abc.abstractmethod
    def preflight_check(self, client: Client, universe: Universe, now_: datetime.datetime):
        """Called when the trade executor instance is started.

        :param client: Trading Strategy client to check server versions etc.

        :param universe: THe currently constructed universe

        :param now_: Real-time clock signal or past clock timestamp in the case of unit testing

        :raise PreflightCheckFailed: In the case we cannot go live
        """
        pass

    @abc.abstractmethod
    def get_strategy_time_frame(self) -> Optional[TimeBucket]:
        """Return the default candle time bucket used in this strategy.

        :return: None if not relevant for the strategy
        """

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, universe: Universe, state: State) -> List[TradeExecution]:
        pass