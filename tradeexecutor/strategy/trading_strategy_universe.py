import contextlib
import datetime
import textwrap
from abc import abstractmethod
from dataclasses import dataclass
import logging
from typing import List, Optional

import pandas as pd

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.universe_model import TradeExecutorTradingUniverse, UniverseModel, DataTooOld
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import DEXPair, PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.utils.groupeduniverse import filter_for_pairs

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
    universe: Optional[Universe] = None

    @staticmethod
    def create_single_pair_universe(
        dataset: Dataset,
        chain_id: ChainId,
        exchange_slug: str,
        base_token: str,
        quote_token: str) -> "TradingStrategyUniverse":
        """Filters down the dataset for a single trading pair.

        This is ideal for strategies that only want to trade a single pair.
        """

        # We only trade on Pancakeswap v2
        exchange_universe = dataset.exchanges
        exchange = exchange_universe.get_by_chain_and_slug(chain_id, exchange_slug)
        assert exchange, f"No exchange {exchange_slug} found on chain {chain_id.name}"

        # Create trading pair database
        pair_universe = PandasPairUniverse.create_single_pair_universe(
            dataset.pairs,
            exchange,
            base_token,
            quote_token,
        )

        # Get daily candles as Pandas DataFrame
        all_candles = dataset.candles
        filtered_candles = filter_for_pairs(all_candles, pair_universe.df)
        candle_universe = GroupedCandleUniverse(filtered_candles)

        # Get liquidity candles as Pandas Dataframe
        all_liquidity = dataset.liquidity
        filtered_liquidity = filter_for_pairs(all_liquidity, pair_universe.df)
        liquidity_universe = GroupedLiquidityUniverse(filtered_liquidity)

        pair = pair_universe.get_single()

        # We have only a single pair, so the reserve asset must be its quote token
        trading_pair_identifier = translate_trading_pair(pair)
        reserve_assets = [
            trading_pair_identifier.quote
        ]

        universe = Universe(
            time_frame=dataset.time_frame,
            chains={chain_id},
            pairs=pair_universe,
            exchanges={exchange},
            candles=candle_universe,
            liquidity=liquidity_universe,
        )

        return TradingStrategyUniverse(universe=universe, reserve_assets=reserve_assets)



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

        if universe.liquidity:
            liquidity_start, liquidity_end = universe.liquidity.get_timestamp_range()
        else:
            liquidity_start = liquidity_end = None

        logger.info(textwrap.dedent(f"""
                Universe constructed.                    
                
                Time periods
                - Time frame {universe.time_frame.value}
                - Candle data range: {data_start} - {data_end}
                - Liquidity data range: {liquidity_start} - {liquidity_end}
                
                The size of our trading universe is
                - {len(universe.exchanges):,} exchanges
                - {universe.pairs.get_count():,} pairs
                - {universe.candles.get_sample_count():,} candles
                - {universe.liquidity.get_sample_count():,} liquidity samples                
                """))
        return universe

    def load_data(self, time_frame: TimeBucket, live: bool) -> Dataset:
        """Loads the server-side data using the client.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.
        :param live: If True disable all caches
        :param lookback: how long to the past load data e.g. 1 year, 1 month. **Not implemented yet**.
        :return: None if not dataset for the strategy required
        """
        client = self.client
        with self.timed_task_context_manager("load_data", time_frame=time_frame.value):

            if live:
                # This will force client to redownload the data
                logger.info("Purging trading data caches")
                client.clear_caches()
            else:
                logger.info("Using cached data if available")

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

    def check_data_age(self, ts: datetime.datetime, universe: TradingStrategyUniverse, best_before_duration: datetime.timedelta):
        """Check if our data is up-to-date and we do not have issues with feeds.

        Ensure we do not try to execute live trades with stale data.

        :raise DataTooOld: in the case data is too old to execute.
        """
        max_age = ts - best_before_duration
        universe = universe.universe

        if universe.candles is not None:
            # Convert pandas.Timestamp to executor internal datetime format
            candle_start, candle_end = universe.candles.get_timestamp_range()
            candle_start = candle_start.to_pydatetime().replace(tzinfo=None)
            candle_end = candle_end.to_pydatetime().replace(tzinfo=None)

            if candle_end < max_age:
                diff = max_age - candle_end
                raise DataTooOld(f"Candle data {candle_start} - {candle_end} is too old to work with, we require threshold {max_age}, diff is {diff}")

        if universe.liquidity is not None:
            liquidity_start, liquidity_end = universe.liquidity.get_timestamp_range()
            liquidity_start = liquidity_start.to_pydatetime().replace(tzinfo=None)
            liquidity_end = liquidity_end.to_pydatetime().replace(tzinfo=None)

            if liquidity_end < max_age:
                raise DataTooOld(f"Liquidity data is too old to work with {liquidity_start} - {liquidity_end}")

    @staticmethod
    def create_from_dataset(dataset: Dataset, chains: List[ChainId], reserve_assets: List[AssetIdentifier], pairs_index=True):
        """Create an trading universe from dataset with zero filtering for the data."""

        exchanges = list(dataset.exchanges.exchanges.values())
        logger.debug("Preparing pairs")
        pairs = PandasPairUniverse(dataset.pairs, build_index=pairs_index)
        logger.debug("Preparing candles")
        candle_universe = GroupedCandleUniverse(dataset.candles)
        logger.debug("Preparing liquidity")
        liquidity_universe = GroupedLiquidityUniverse(dataset.liquidity)

        universe = Universe(
            time_frame=dataset.time_frame,
            chains=chains,
            pairs=pairs,
            exchanges=exchanges,
            candles=candle_universe,
            liquidity=liquidity_universe,
        )

        logger.debug("Universe created")
        return TradingStrategyUniverse(universe=universe, reserve_assets=reserve_assets)

    @abstractmethod
    def construct_universe(self, ts: datetime.datetime, live: bool) -> TradingStrategyUniverse:
        pass


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

    # TODO: Add decimals here

    base = AssetIdentifier(
        chain_id=pair.chain_id.value,
        address=pair.base_token_address,
        token_symbol=pair.base_token_symbol,
        decimals=None,
    )
    quote = AssetIdentifier(
        chain_id=pair.chain_id.value,
        address=pair.quote_token_address,
        token_symbol=pair.quote_token_symbol,
        decimals=None,
    )

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=pair.address,
        internal_id=pair.pair_id,
        info_url=pair.get_trading_pair_page_url(),
        exchange_address=pair.exchange_address,
    )
