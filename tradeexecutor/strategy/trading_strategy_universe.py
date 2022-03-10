import contextlib
import textwrap
from dataclasses import dataclass
import logging
import pandas as pd

from tradeexecutor.state.state import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.strategy.universe_model import TradeExecutorTradingUniverse, UniverseModel
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import DEXPair
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


@dataclass
class TradingStrategyUniverse(TradeExecutorTradingUniverse):
    """A trading executor trading universe that using data from TradingStrategy.ai data feeds."""
    universe: Universe


class TradingStrategyUniverseModel(UniverseModel):
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
                - {len(universe.exchanges):,} exchanges
                - {universe.pairs.get_count():,} pairs
                - {universe.candles.get_sample_count():,} candles
                - {universe.liquidity.get_sample_count():,} liquidity samples                
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


def translate_trading_pair(pair: DEXPair) -> TradingPairIdentifier:
    """Translate trading pair from Pandas universe to Trade Executor universe.

    Translates a trading pair presentation from Trading Strategy client Pandas format to the trade executor format.

    Trade executor work with multiple different strategies, not just Trading Strategy client based.
    For example, you could have a completely on-chain data based strategy.
    Thus, Trade Executor has its internal asset format.

    This module contains functions to translate asset presentations between Trading Strategy client
    and Trade Executor.


    This is called when a trade is made: this is the moment when trade executor data format must be made available.
    """
    base = AssetIdentifier(
        chain_id=pair.chain_id.value,
        address=pair.base_token_address,
        token_symbol=pair.base_token_symbol,
        decimals=None
    )
    quote = AssetIdentifier(
        chain_id=pair.chain_id.value,
        address=pair.quote_token_address,
        token_symbol=pair.quote_token_symbol,
        decimals=None
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=pair.address,
        internal_id=pair.pair_id,
    )
