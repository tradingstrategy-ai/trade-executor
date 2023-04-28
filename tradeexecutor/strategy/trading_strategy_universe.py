"""Trading Strategy oracle data integration.

Define trading universe based on data from :py:mod:`tradingstrategy` and
market data feeds.

See :ref:`trading universe` for more information.
"""

import contextlib
import datetime
import textwrap
from abc import abstractmethod
from dataclasses import dataclass
import logging
from math import isnan
from typing import List, Optional, Callable, Tuple, Set, Dict, Iterable, Collection

import pandas as pd

from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradingstrategy.token import Token

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse, UniverseModel, DataTooOld, UniverseOptions
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client, BaseClient
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.liquidity import GroupedLiquidityUniverse, ResampledLiquidityUniverse
from tradingstrategy.pair import DEXPair, PandasPairUniverse, resolve_pairs_based_on_ticker, \
    filter_for_exchanges, filter_for_quote_tokens, StablecoinFilteringMode, filter_for_stablecoins, \
    HumanReadableTradingPairDescription
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.utils.groupeduniverse import filter_for_pairs


logger = logging.getLogger(__name__)


class TradingUniverseIssue(Exception):
    """Raised in the case trading universe has some bad data etc. issues."""


@dataclass
class Dataset:
    """Contain raw loaded datasets."""

    #: Granularity of our OHLCV data
    time_bucket: TimeBucket

    #: All exchanges
    exchanges: ExchangeUniverse

    #: All trading pairs
    pairs: pd.DataFrame

    #: Candle data for all pairs
    candles: Optional[pd.DataFrame] = None

    #: All liquidity samples
    liquidity: Optional[pd.DataFrame] = None

    #: Liquidity data granularity
    liquidity_time_bucket: Optional[TimeBucket] = None

    #: Granularity of backtesting OHLCV data
    backtest_stop_loss_time_bucket: Optional[TimeBucket] = None

    #: All candles in stop loss time bucket
    backtest_stop_loss_candles: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Check we got good data."""
        candles = self.candles
        if candles is not None:
            assert isinstance(candles, pd.DataFrame), f"Expected DataFrame, got {candles.__class__}"

        liquidity = self.liquidity
        if liquidity is not None:
            assert isinstance(liquidity, pd.DataFrame), f"Expected DataFrame, got {liquidity.__class__}"


@dataclass
class TradingStrategyUniverse(StrategyExecutionUniverse):
    """A trading executor trading universe that using data from TradingStrategy.ai data feeds.

    - Supports generic trading universe definitions

    - Adds special support for reserve currency handling and
      take profit/stop loss backtesting
    """

    #: Trading universe datasets.
    #:
    #: This encapsulates more generic `Universe` class from `tradingstrategy` package.
    universe: Optional[Universe] = None

    #: Are we using special take profit/stop loss trigger data.
    #:
    #: If we are, what is the time granularity of this data.
    backtest_stop_loss_time_bucket: Optional[TimeBucket] = None

    #: Special Pandas data feed for candles used only during the backtesting.
    #:
    #: This allows us to simulate take profit and stop loss trigger orders
    #: that would be based on real-time market data in a live execution environment.
    backtest_stop_loss_candles: Optional[GroupedCandleUniverse] = None

    #: How much historic data is needed by the strategy to make its decision.
    #:
    #: This is the explicit time frame in days or hours for the historical
    #: data before today for the strategy to make a decision.
    #:
    #: This will limit the amount of data downloaded from the oracle
    #: and greatly optimise the strategy decision execution.
    #:
    #: E.g. for 90 days you can use `datetime.timedelta(days=90)`
    required_history_period: Optional[datetime.timedelta] = None

    def __repr__(self):
        pair_count = self.universe.pairs.get_count()
        if pair_count <= 3:
            pair_tickers = [f"{p.base_token_symbol}-{p.quote_token_symbol}" for p in self.universe.pairs.iterate_pairs()]
            return f"<TradingStrategyUniverse for {', '.join(pair_tickers)}>"
        else:
            return f"<TradingStrategyUniverse for {self.universe.pairs.get_count()} pairs>"

    def clone(self) -> "TradingStrategyUniverse":
        """Create a copy of this universe.

        Any dataframes are now copied,
        but set by reference.
        """
        u = self.universe
        new_universe = Universe(
            time_bucket=u.time_bucket,
            chains=u.chains,
            exchanges=u.exchanges,
            pairs=u.pairs,
            candles=u.candles,
            liquidity=u.liquidity,
        )
        return TradingStrategyUniverse(
            universe=new_universe,
            backtest_stop_loss_time_bucket=self.backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=self.backtest_stop_loss_candles,
            reserve_assets=self.reserve_assets,
            required_history_period=self.required_history_period,
        )

    def get_pair_count(self) -> int:
        return self.universe.pairs.get_count()

    def is_empty(self) -> bool:
        """This is an empty universe

        - without trading pairs

        - ...or without candles
        """
        candles = self.universe.candles.df if self.universe.candles else []
        return self.universe.pairs.get_count() == 0 or len(candles) == 0

    def is_single_pair_universe(self) -> bool:
        """Is this trading universe made for a single pair trading.

        Note that even a single pair universe may contain two trading pairs,
        if we need intermediate routing pairs. E.g. AAVE -> BNB -> BUSD/
        """

        # TODO: Make a stupid assumption here
        # as our strategies currently have 1 or 2 pairs for single pair trading.
        # Convert this to a proper flag later.
        return self.universe.pairs.get_count() <= 2

    def has_stop_loss_data(self) -> bool:
        """Do we have data available to determine trade stop losses.

        Note that this applies for backtesting only - when
        doing production trade execution, stop loss data is not part of the universe
        but real time pricing comes directly from the exchange using real-time
        side channels.
        """
        return (self.backtest_stop_loss_candles is not None) and \
               (self.backtest_stop_loss_time_bucket is not None)

    def validate(self):
        """Check that the created universe looks good.

        :raise TradingUniverseIssue:
            In the case of detected issues
        """
        if len(self.reserve_assets) != 1:
            raise TradingUniverseIssue(f"Only single reserve asset strategies supported for now, got {self.reserve_assets}")

        for a in self.reserve_assets:
            if a.decimals == 0:
                raise TradingUniverseIssue(f"Reserve asset lacks decimals {a}")

    def get_pair_by_human_description(self, desc: HumanReadableTradingPairDescription) -> TradingPairIdentifier:
        """Get pair by its human-readable description.

        See :py:meth:`tradingstrategy.pair.PandasPairUniverse.get_pair_by_human_description`

        The trading pair must be loaded in the exchange universe.

        :return:
            The trading pair on the exchange.

            Highest volume trading pair if multiple matches.

        :raise NoPairFound:
            In the case input data cannot be resolved.

        """
        assert self.universe.exchange_universe, "You must set universe.exchange_universe to be able to use this method"
        pair = self.universe.pairs.get_pair_by_human_description(self.universe.exchange_universe, desc)
        return translate_trading_pair(pair)

    def create_single_pair_universe(
        dataset: Dataset,
        chain_id: Optional[ChainId] = None,
        exchange_slug: Optional[str] = None,
        base_token: Optional[str] = None,
        quote_token: Optional[str] = None,
        pair: Optional[HumanReadableTradingPairDescription] = None
    ) -> "TradingStrategyUniverse":
        """Filters down the dataset for a single trading pair.

        This is ideal for strategies that only want to trade a single pair.

        :param pair:
            Give the pair we create the universe for

        :param exchange_slug:
            Legacy.

            This or `pair`.

        :param chain_id:
            Legacy.

            This or `pair`.

        :param base_token:
            Legacy.

            This or `pair`.

        :param quote_token:
            Legacy.

            This or `pair`.
        """

        if pair:
            chain_id, exchange_slug, base_token, quote_token, *fees = pair

            if len(fees) > 0:
                raise NotImplementedError("This method still lacks fee filtering")

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
        if dataset.candles is not None:
            all_candles = dataset.candles
            filtered_candles = filter_for_pairs(all_candles, pair_universe.df)
            candle_universe = GroupedCandleUniverse(filtered_candles)
        else:
            candle_universe = None

        # Get liquidity candles as Pandas Dataframe
        if dataset.liquidity is not None and not dataset.liquidity.empty:
            all_liquidity = dataset.liquidity
            filtered_liquidity = filter_for_pairs(all_liquidity, pair_universe.df)
            liquidity_universe = GroupedLiquidityUniverse(filtered_liquidity)
        else:
            liquidity_universe = None

        pair = pair_universe.get_single()

        # We have only a single pair, so the reserve asset must be its quote token
        trading_pair_identifier = translate_trading_pair(pair)
        reserve_assets = [
            trading_pair_identifier.quote
        ]

        universe = Universe(
            time_bucket=dataset.time_bucket,
            chains={chain_id},
            pairs=pair_universe,
            exchanges={exchange},
            candles=candle_universe,
            liquidity=liquidity_universe,
        )

        if dataset.backtest_stop_loss_candles is not None:
            stop_loss_candle_universe = GroupedCandleUniverse(
                dataset.backtest_stop_loss_candles,
                time_bucket=dataset.backtest_stop_loss_time_bucket)
        else:
            stop_loss_candle_universe = None

        # TODO: Not sure if we need to be smarter about stop loss candle
        # data handling here
        return TradingStrategyUniverse(
            universe=universe,
            reserve_assets=reserve_assets,
            backtest_stop_loss_time_bucket=dataset.backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candle_universe,
        )

    @staticmethod
    def create_limited_pair_universe(
        dataset: Dataset,
        chain_id: ChainId,
        exchange_slug: str,
        pairs: Set[Tuple[str, str]],
        reserve_asset_pair_ticker: Optional[Tuple[str, str]] = None) -> "TradingStrategyUniverse":
        """Filters down the dataset for couple trading pair.

        This is ideal for strategies that only want to trade few pairs,
        or a single pair using three-way trading on a single exchange.

        The university reserve currency is set to the quote token of the first pair.

        :param dataset:
            Datasets downloaded from the server

        :param pairs:
            List of trading pairs as ticket tuples. E.g. `[ ("WBNB, "BUSD"), ("Cake", "WBNB") ]`

        :param reserve_asset_pair_ticker:
            Choose the quote token of this trading pair as a reserve asset.
            This must be given if there are several pairs (Python set order is unstable).

        """

        # We only trade on Pancakeswap v2
        exchange_universe = dataset.exchanges
        exchange = exchange_universe.get_by_chain_and_slug(chain_id, exchange_slug)
        assert exchange, f"No exchange {exchange_slug} found on chain {chain_id.name}"

        # Create trading pair database
        pair_universe = PandasPairUniverse.create_limited_pair_universe(
            dataset.pairs,
            exchange,
            pairs,
        )

        # Get daily candles as Pandas DataFrame
        all_candles = dataset.candles
        filtered_candles = filter_for_pairs(all_candles, pair_universe.df)
        candle_universe = GroupedCandleUniverse(filtered_candles)

        # Get liquidity candles as Pandas Dataframe
        if dataset.liquidity:
            all_liquidity = dataset.liquidity
            filtered_liquidity = filter_for_pairs(all_liquidity, pair_universe.df)
            liquidity_universe = GroupedLiquidityUniverse(filtered_liquidity)
        else:
            # Liquidity data loading skipped
            liquidity_universe = None

        if reserve_asset_pair_ticker:
            reserve_pair = pair_universe.get_one_pair_from_pandas_universe(
                exchange.exchange_id,
                reserve_asset_pair_ticker[0],
                reserve_asset_pair_ticker[1],
            )
        else:
            assert len(pairs) == 1, "Cannot automatically determine reserve asset if there are multiple trading pairs."
            first_ticker = next(iter(pairs))
            reserve_pair = pair_universe.get_one_pair_from_pandas_universe(
                exchange.exchange_id,
                first_ticker[0],
                first_ticker[1],
            )
        # We have only a single pair, so the reserve asset must be its quote token
        trading_pair_identifier = translate_trading_pair(reserve_pair)
        reserve_assets = [
            trading_pair_identifier.quote
        ]

        universe = Universe(
            time_bucket=dataset.time_bucket,
            chains={chain_id},
            pairs=pair_universe,
            exchanges={exchange},
            candles=candle_universe,
            liquidity=liquidity_universe,
        )

        if dataset.backtest_stop_loss_candles is not None:
            stop_loss_candle_universe = GroupedCandleUniverse(
                dataset.backtest_stop_loss_candles,
                time_bucket=dataset.backtest_stop_loss_time_bucket)
        else:
            stop_loss_candle_universe = None

        # TODO: Not sure if we need to be smarter about stop loss candle
        # data handling here
        return TradingStrategyUniverse(
            universe=universe,
            reserve_assets=reserve_assets,
            backtest_stop_loss_time_bucket=dataset.backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candle_universe,
        )

    def get_pair_by_address(self, address: str) -> Optional[TradingPairIdentifier]:
        """Get a trading pair data by a smart contract address."""
        pair = self.universe.pairs.get_pair_by_smart_contract(address)
        if not pair:
            return None
        return translate_trading_pair(pair)

    def get_single_pair(self) -> TradingPairIdentifier:
        """Get the single trading pair in this universe."""
        pair = self.universe.pairs.get_single()
        return translate_trading_pair(pair)

    @staticmethod
    def create_multipair_universe(
        dataset: Dataset,
        chain_ids: Iterable[ChainId],
        exchange_slugs: Iterable[str],
        quote_tokens: Iterable[str],
        reserve_token: str,
        factory_router_map: Dict[str, tuple],
        liquidity_resample_frequency: Optional[str] = None,
    ) -> "TradingStrategyUniverse":
        """Create a trading universe where pairs match a filter conditions.

        These universe may contain thousands of trading pairs.
        This is for strategies that trade across multiple pairs,
        like momentum strategies.

        :param dataset:
            Datasets downloaded from the oracle

        :param chain_ids:
            Allowed blockchains

        :param exchange_slugs:
            Allowed exchanges

        :param quote_tokens:
            Allowed quote tokens as smart contract addresses

        :param reserve_token:
            The token addresses that are used as reserve assets.

        :param factory_router_map:
            Ensure we have a router address for every exchange we are going to use.
            TODO: In the future this information is not needed.

        :param liquidity_resample_frequency:
            Create a resampled liquidity universe instead of accurate one.

            If given, this will set `Universe.resampled_liquidity` attribute.

            Using :py:class:`ResampledLiquidityUniverse` will greatly
            speed up backtests that estimate trading pair liquidity,
            by trading off sample accuracy for code execution speed.

            Must be a Pandas frequency string value, like `1d`.

            Note that resamping itself takes a lot of time upfront,
            so you want to use this only if the backtest takes lont time.
        """

        assert type(chain_ids) == list or type(chain_ids) == set
        assert type(exchange_slugs) == list or type(exchange_slugs) == set
        assert type(quote_tokens) == list or type(quote_tokens) == set
        assert type(reserve_token) == str
        assert reserve_token.startswith("0x")

        for t in quote_tokens:
            assert t.startswith("0x")

        time_bucket = dataset.time_bucket

        # Normalise input parameters
        chain_ids = set(chain_ids)
        exchange_slugs = set(exchange_slugs)
        quote_tokens = set(q.lower() for q in quote_tokens)
        factory_router_map = {k.lower(): v for k, v in factory_router_map.items()}

        x: Exchange
        avail_exchanges = dataset.exchanges.exchanges
        our_exchanges = {x for x in avail_exchanges.values() if (x.chain_id in chain_ids) and (x.exchange_slug in exchange_slugs)}
        exchange_universe = ExchangeUniverse.from_collection(our_exchanges)

        # Check we got all exchanges in the dataset
        for x in our_exchanges:
            assert x.address.lower() in factory_router_map, f"Could not find router for a exchange {x.exchange_slug}, factory {x.address}, router map is: {factory_router_map}"

        # Choose all trading pairs that are on our supported exchanges and
        # with our supported quote tokens
        pairs_df = filter_for_exchanges(dataset.pairs, list(our_exchanges))
        pairs_df = filter_for_quote_tokens(pairs_df, quote_tokens)

        # Remove stablecoin -> stablecoin pairs, because
        # trading between stable does not make sense for our strategies
        pairs_df = filter_for_stablecoins(pairs_df, StablecoinFilteringMode.only_volatile_pairs)

        # Create the trading pair data for this specific strategy
        pairs = PandasPairUniverse(
            pairs_df,
            exchange_universe=exchange_universe,
        )

        # We do a bit detour here as we need to address the assets by their trading pairs first
        reserve_token_info = pairs.get_token(reserve_token)
        assert reserve_token_info, f"Reserve token {reserve_token} missing the trading pairset"
        reserve_assets = [
            translate_token(reserve_token_info)
        ]

        # Get daily candles as Pandas DataFrame
        all_candles = dataset.candles
        filtered_candles = filter_for_pairs(all_candles, pairs_df)
        candle_universe = GroupedCandleUniverse(filtered_candles, time_bucket=time_bucket)

        # Get liquidity candles as Pandas Dataframe
        all_liquidity = dataset.liquidity
        filtered_liquidity = filter_for_pairs(all_liquidity, pairs_df)

        if liquidity_resample_frequency is None:
            liquidity_universe = GroupedLiquidityUniverse(filtered_liquidity, time_bucket=dataset.liquidity_time_bucket)
            resampled_liquidity = None
        else:
            liquidity_universe = None
            # Do just a print notification now, consider beta
            # Optimally we want to resample, then store on a local disk cache,
            # so that we do not need to run resample at the start of each backtest
            print(f"Resamping liquidity data to {liquidity_resample_frequency}, this may take a long time")
            resampled_liquidity = ResampledLiquidityUniverse(filtered_liquidity, resample_period=liquidity_resample_frequency)

        if dataset.backtest_stop_loss_candles is not None:
            backtest_stop_loss_time_bucket = dataset.backtest_stop_loss_time_bucket
            filtered_candles = filter_for_pairs(dataset.backtest_stop_loss_candles, pairs_df)
            backtest_stop_loss_candles = GroupedCandleUniverse(filtered_candles, time_bucket=dataset.backtest_stop_loss_time_bucket)
        else:
            backtest_stop_loss_time_bucket = None
            backtest_stop_loss_candles = None

        universe = Universe(
            time_bucket=dataset.time_bucket,
            chains=chain_ids,
            pairs=pairs,
            exchanges=our_exchanges,
            candles=candle_universe,
            liquidity=liquidity_universe,
            resampled_liquidity=resampled_liquidity,
            exchange_universe=exchange_universe,
        )

        return TradingStrategyUniverse(
            universe=universe,
            reserve_assets=reserve_assets,
            backtest_stop_loss_time_bucket=backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=backtest_stop_loss_candles,
        )

    @staticmethod
    def create_multichain_universe_by_pair_descriptions(
        dataset: Dataset,
        pairs: Collection[HumanReadableTradingPairDescription],
        reserve_token_symbol: str,
    ) -> "TradingStrategyUniverse":
        """Create a trading universe based on list of (exchange, pair tuples)

        This is designed for backtesting multipe pairs across different chains.
        The created universe do not have any routing options and thus
        cannot make any trades.

        :param dataset:
            Datasets downloaded from the oracle

        :param pairs:
            List of trading pairs to filter down.

            The pair set is desigend to be small, couple of dozens of pairs max.

            See :py:data:`HumanReadableTradingPairDescription`.

        :param reserve_token_symbol:
            The token symbol of the reverse asset.

            Because we do not support routing, we just pick the first matching token.
        """

        time_bucket = dataset.time_bucket

        # Create trading pair database
        pair_universe = PandasPairUniverse(dataset.pairs)

        # Filter pairs first and then rest by the resolved pairs
        our_pairs = {pair_universe.get_pair_by_human_description(dataset.exchanges, d) for d in pairs}
        chain_ids = {d[0] for d in pairs}
        pair_ids = {p.pair_id for p in our_pairs}
        exchange_ids = {p.exchange_id for p in our_pairs}
        our_exchanges = {dataset.exchanges.get_by_id(id) for id in exchange_ids}
        filtered_pairs_df = dataset.pairs.loc[dataset.pairs["pair_id"].isin(pair_ids)]

        # Recreate universe again, now with limited pairs
        pair_universe = PandasPairUniverse(filtered_pairs_df)

        # We do a bit detour here as we need to address the assets by their trading pairs first
        reserve_token = None
        for p in pair_universe.iterate_pairs():
            if p.quote_token_symbol == reserve_token_symbol:
                translated_pair = translate_trading_pair(p)
                reserve_token = translated_pair.quote

        assert reserve_token, f"Reserve token {reserve_token_symbol} missing the pair quote tokens"
        reserve_assets = [
            reserve_token
        ]

        # Get daily candles as Pandas DataFrame
        all_candles = dataset.candles
        filtered_candles = filter_for_pairs(all_candles, filtered_pairs_df)
        candle_universe = GroupedCandleUniverse(filtered_candles, time_bucket=time_bucket)

        universe = Universe(
            time_bucket=dataset.time_bucket,
            chains=chain_ids,
            pairs=pair_universe,
            exchanges=our_exchanges,
            candles=candle_universe,
            exchange_universe=ExchangeUniverse.from_collection(our_exchanges),
        )

        if dataset.backtest_stop_loss_candles is not None:
            stop_loss_candle_universe = GroupedCandleUniverse(dataset.backtest_stop_loss_candles)
        else:
            stop_loss_candle_universe = None

        return TradingStrategyUniverse(
            universe=universe,
            reserve_assets=reserve_assets,
            backtest_stop_loss_time_bucket=dataset.backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candle_universe,
        )


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
                - Time frame {universe.time_bucket.value}
                - Candle data range: {data_start} - {data_end}
                - Liquidity data range: {liquidity_start} - {liquidity_end}
                
                The size of our trading universe is
                - {len(universe.exchanges):,} exchanges
                - {universe.pairs.get_count():,} pairs
                - {universe.candles.get_sample_count():,} candles
                - {universe.liquidity.get_sample_count():,} liquidity samples                
                """))
        return universe

    def load_data(self,
                  time_frame: TimeBucket,
                  mode: ExecutionMode,
                  backtest_stop_loss_time_frame: Optional[TimeBucket]=None) -> Dataset:
        """Loads the server-side data using the client.

        :param client:
            Client instance. Note that this cannot be stable across ticks, as e.g.
            API keys can change. Client is recreated for every tick.

        :param mode:
            Live trading or vacktesting

        :param backtest_stop_loss_time_frame:
            Load more granular data for backtesting stop loss

        :return:
            None if not dataset for the strategy required
        """

        assert isinstance(mode, ExecutionMode), f"Expected ExecutionMode, got {mode}"

        client = self.client

        with self.timed_task_context_manager("load_data", time_bucket=time_frame.value):

            if mode.is_fresh_data_always_needed():
                # This will force client to redownload the data
                logger.info("Execution mode %s, purging trading data caches", mode)
                client.clear_caches()
            else:
                logger.info("Execution mode %s, not live trading, Using cached data if available", mode)

            exchanges = client.fetch_exchange_universe()
            pairs = client.fetch_pair_universe().to_pandas()
            candles = client.fetch_all_candles(time_frame).to_pandas()
            liquidity = client.fetch_all_liquidity_samples(time_frame).to_pandas()

            if backtest_stop_loss_time_frame:
                backtest_stop_loss_candles = client.fetch_all_candles(backtest_stop_loss_time_frame).to_pandas()
            else:
                backtest_stop_loss_candles = None

            return Dataset(
                time_bucket=time_frame,
                backtest_stop_loss_time_bucket=backtest_stop_loss_time_frame,
                exchanges=exchanges,
                pairs=pairs,
                candles=candles,
                liquidity=liquidity,
                backtest_stop_loss_candles=backtest_stop_loss_candles,
            )

    def check_data_age(self, ts: datetime.datetime, universe: TradingStrategyUniverse, best_before_duration: datetime.timedelta) -> datetime.datetime:
        """Check if our data is up-to-date and we do not have issues with feeds.

        Ensure we do not try to execute live trades with stale data.

        :raise DataTooOld:
            in the case data is too old to execute.

        :return:
            The data timestamp
        """
        max_age = ts - best_before_duration
        universe = universe.universe
        candle_end = None

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

            return min(liquidity_end, candle_end)

        else:
            return candle_end

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

        logger.debug("Preparing backtest stop loss data")
        if dataset.backtest_stop_loss_candles:
            backtest_stop_loss_candles = GroupedCandleUniverse(dataset.backtest_stop_loss_candles)
        else:
            backtest_stop_loss_candles = None

        universe = Universe(
            time_bucket=dataset.time_bucket,
            chains=chains,
            pairs=pairs,
            exchanges=exchanges,
            candles=candle_universe,
            liquidity=liquidity_universe,
        )

        logger.debug("Universe created")
        return TradingStrategyUniverse(
            universe=universe,
            reserve_assets=reserve_assets,
            backtest_stop_loss_candles=backtest_stop_loss_candles,
            backtest_stop_loss_time_bucket=dataset.backtest_stop_loss_time_bucket)

    @abstractmethod
    def construct_universe(self,
                           ts: datetime.datetime,
                           mode: ExecutionMode,
                           options: UniverseOptions) -> TradingStrategyUniverse:
        pass


class DefaultTradingStrategyUniverseModel(TradingStrategyUniverseModel):
    """Shortcut for simple strategies.

    - Assumes we have a strategy that fits to :py:mod:`tradeexecutor.strategy_module` definiton

    - At the start of the backtests or at each cycle of live trading, call
      the `create_trading_universe` callback of the strategy

    - Validate the output of the function
    """

    def __init__(self,
                 client: Optional[BaseClient],
                 execution_context: ExecutionContext,
                 create_trading_universe: Callable):
        """

        :param candle_time_frame_override:
            Use this candle time bucket instead one given in the strategy file.
            Allows to "speedrun" strategies.

        :param stop_loss_time_frame_override:
            Use this stop loss frequency instead one given in the strategy file.
            Allows to "speedrun" strategies.

        """
        assert isinstance(client, BaseClient) or client is None
        assert isinstance(execution_context, ExecutionContext), f"Got {execution_context}"
        assert isinstance(create_trading_universe, Callable), f"Got {create_trading_universe}"
        self.client = client
        self.execution_context = execution_context
        self.create_trading_universe = create_trading_universe

    def preload_universe(self, universe_options: UniverseOptions):
        """Triggered before backtesting execution.

        - Load all datasets with progress bar display.

        - Not triggered in live trading, as universe changes between cycles
        """
        # TODO: Circular imports
        from tradeexecutor.backtest.data_preload import preload_data
        with self.execution_context.timed_task_context_manager(task_name="preload_universe"):
            return preload_data(
                self.client,
                self.create_trading_universe,
                universe_options=universe_options)

    def construct_universe(self,
                           ts: datetime.datetime,
                           mode: ExecutionMode,
                           options: UniverseOptions) -> TradingStrategyUniverse:
        with self.execution_context.timed_task_context_manager(task_name="create_trading_universe"):
            universe = self.create_trading_universe(
                ts,
                self.client,
                self.execution_context,
                options)
            assert isinstance(universe, TradingStrategyUniverse), f"Expected TradingStrategyUniverse, got {universe.__class__}"
            universe.validate()
            return universe


def translate_token(token: Token, require_decimals=True) -> AssetIdentifier:
    """Translate Trading Strategy token data definition to trade executor.

    Trading Strategy client uses compressed columnar data for pairs and tokens.

    Creates `AssetIdentifier` based on data coming from
    Trading Strategy :py:class:`tradingstrategy.pair.PandasPairUniverse`.

    :param require_decimals:
        Most tokens / trading pairs are non-functional without decimals information.
        Assume decimals is in place. If not then raise AssertionError.
        This check allows some early error catching on bad data.

    """

    if require_decimals:
        assert token.decimals, f"Bad token: {token}"
        assert token.decimals > 0, f"Bad token: {token}"

    return AssetIdentifier(
        token.chain_id.value,
        token.address,
        token.symbol,
        token.decimals
    )


def translate_trading_pair(pair: DEXPair) -> TradingPairIdentifier:
    """Translate trading pair from client download to the trade executor.

    Trading Strategy client uses compressed columnar data for pairs and tokens.

    Translates a trading pair presentation from Trading Strategy client Pandas format to the trade executor format.

    Trade executor work with multiple different strategies, not just Trading Strategy client based.
    For example, you could have a completely on-chain data based strategy.
    Thus, Trade Executor has its internal asset format.

    This module contains functions to translate asset presentations between Trading Strategy client
    and Trade Executor.


    This is called when a trade is made: this is the moment when trade executor data format must be made available.
    """

    assert isinstance(pair, DEXPair), f"Expected DEXPair, got {type(pair)}"
    assert pair.base_token_decimals is not None, f"Base token missing decimals: {pair}"
    assert pair.quote_token_decimals is not None, f"Quote token missing decimals: {pair}"

    base = AssetIdentifier(
        chain_id=pair.chain_id.value,
        address=pair.base_token_address,
        token_symbol=pair.base_token_symbol,
        decimals=pair.base_token_decimals,
    )
    quote = AssetIdentifier(
        chain_id=pair.chain_id.value,
        address=pair.quote_token_address,
        token_symbol=pair.quote_token_symbol,
        decimals=pair.quote_token_decimals,
    )

    if pair.fee and isnan(pair.fee):
        # Repair some broken data
        fee = None
    else:
        # Convert DEXPair.fee BPS to %
        # So, after this, fee can either be multiplier or None
        if pair.fee is not None:
            # If BPS fee is set it must be more than 1 BPS.
            # Allow explicit fee = 0 in testing.
            # if pair.fee != 0:
            #     assert pair.fee > 1, f"DEXPair fee must be in BPS, got {pair.fee}"
            
            # can receive fee in bps or multiplier, but not raw form
            if pair.fee > 1:
                fee = pair.fee/10_000
                
            else:
                fee = pair.fee
            
            # highest fee tier is currently 1% and lowest in 0.01%
            if fee != 0:
                assert 0.0001 <= fee <= 0.01, "bug in converting fee to multiplier, make sure bps"
        else:
            fee = None

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=pair.address,
        internal_id=pair.pair_id,
        info_url=pair.get_trading_pair_page_url(),
        exchange_address=pair.exchange_address,
        fee=fee,
    )


def create_pair_universe_from_code(chain_id: ChainId, pairs: List[TradingPairIdentifier]) -> "PandasPairUniverse":
    """Create the trading universe from handcrafted data.

    Used in unit testing.
    """
    data = []
    for idx, p in enumerate(pairs):
        assert p.base.decimals
        assert p.quote.decimals
        assert p.internal_exchange_id, f"All trading pairs must have internal_exchange_id set, did not have it set {p}"
        assert p.internal_id
        dex_pair = DEXPair(
            pair_id=p.internal_id,
            chain_id=chain_id,
            exchange_id=p.internal_exchange_id,
            address=p.pool_address,
            exchange_address=p.exchange_address,
            dex_type=ExchangeType.uniswap_v2,
            base_token_symbol=p.base.token_symbol,
            quote_token_symbol=p.quote.token_symbol,
            token0_symbol=p.base.token_symbol,
            token1_symbol=p.quote.token_symbol,
            token0_address=p.base.address,
            token1_address=p.quote.address,
            token0_decimals=p.base.decimals,
            token1_decimals=p.quote.decimals,
            fee=int(p.fee * 10_000) if p.fee else None,  # Convert to bps according to the documentation
        )
        data.append(dex_pair.to_dict())
    df = pd.DataFrame(data)
    return PandasPairUniverse(df)


def load_all_data(
        client: BaseClient,
        time_frame: TimeBucket,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
        with_liquidity=True,
        liquidity_time_frame: Optional[TimeBucket] = None,
        stop_loss_time_frame: Optional[TimeBucket] = None,
) -> Dataset:
    """Load all pair, candle and liquidity data for a given time bucket.

    - Backtest data is never reloaded

    - Live trading purges old data fields and reloads data

    :param client:
        Trading Strategy client instance

    :param time_frame:
        Candle time bucket of which granularity to data to load.

        Set to `TimeBucket.not_applicable` to downlaod only exchange and pair data,
        as used in unit testing.

    :param execution_context:
        Defines if we are live or backtesting

    :param with_liquidity:
        Load liquidity data.

        Note that all pairs may not have liquidity data available.

    :param stop_loss_time_frame:
        Load more granular candle data for take profit /tstop loss backtesting.

    :param liquidity_time_frame:
        Enable downloading different granularity of liquidity data.

        If not given default to `time_frame`.

    :return:
        Dataset that covers all historical data.

        This dataset is big and you need to filter it down for backtests.
    """

    assert isinstance(client, BaseClient)
    assert isinstance(time_frame, TimeBucket)
    assert isinstance(execution_context, ExecutionContext)

    # Apply overrides
    time_frame = universe_options.candle_time_bucket_override or time_frame

    assert universe_options.stop_loss_time_bucket_override is None, "Not supported yet"

    live = execution_context.live_trading
    with execution_context.timed_task_context_manager("load_data", time_bucket=time_frame.value):
        if live:
            # This will force client to redownload the data
            logger.info("Purging trading data caches")
            client.clear_caches()
        else:
            logger.info("Using cached data if available")

        exchanges = client.fetch_exchange_universe()
        pairs = client.fetch_pair_universe().to_pandas()

        candles = liquidity = stop_loss_candles = None

        if time_frame != TimeBucket.not_applicable:
            candles = client.fetch_all_candles(time_frame).to_pandas()

            if with_liquidity:
                if not liquidity_time_frame:
                    liquidity_time_frame = time_frame
                liquidity = client.fetch_all_liquidity_samples(liquidity_time_frame).to_pandas()

            if stop_loss_time_frame:
                stop_loss_candles = client.fetch_all_candles(stop_loss_time_frame).to_pandas()


        return Dataset(
            time_bucket=time_frame,
            liquidity_time_bucket=liquidity_time_frame,
            exchanges=exchanges,
            pairs=pairs,
            candles=candles,
            liquidity=liquidity,
            backtest_stop_loss_time_bucket=stop_loss_time_frame,
            backtest_stop_loss_candles=stop_loss_candles,
        )


def load_pair_data_for_single_exchange(
        client: BaseClient,
        execution_context: ExecutionContext,
        time_bucket: TimeBucket,
        chain_id: Optional[ChainId] = None,
        exchange_slug: Optional[str] = None,
        pair_tickers: Set[Tuple[str, str]] | Collection[HumanReadableTradingPairDescription] | None = None,
        universe_options: UniverseOptions = None,
        liquidity=False,
        stop_loss_time_bucket: Optional[TimeBucket] = None,
        required_history_period: Optional[datetime.timedelta] = None,
) -> Dataset:
    """Load pair data for a single decentralised exchange.

    If you are not trading the full trading universe,
    this function does a much smaller dataset download than
    :py:func:`load_all_data`.

    - This function uses optimised JSONL loading
      via :py:meth:`~tradingstrategy.client.Client.fetch_candles_by_pair_ids`.

    - Backtest data is never reloaded.
      Furthermore, the data is stored in :py:class:`Client`
      disk cache for the subsequent notebook and backtest runs.

    - Live trading purges old data fields and reloads data

    Example:

    ... code-block:: python

        TRADING_PAIR = (ChainId.avalanche, "trader-joe", "WAVAX", "USDC")

        dataset = load_pair_data_for_single_exchange(
            client,
            pair=TRADING_PAIR,
            execution_context=execution_context,
            universe_options=universe_options,
        )

    Example (old):

    .. code-block:: python

        # Time bucket for our candles
        candle_time_bucket = TimeBucket.d1

        # Which chain we are trading
        chain_id = ChainId.bsc

        # Which exchange we are trading on.
        exchange_slug = "pancakeswap-v2"

        # Which trading pair we are trading
        trading_pairs = {
            ("WBNB", "BUSD"),
            ("Cake", "WBNB"),
        }

        # Load all datas we can get for our candle time bucket
        dataset = load_pair_data_for_single_exchange(
            client,
            execution_context,
            candle_time_bucket,
            chain_id,
            exchange_slug,
            trading_pairs,
        )

    :param client:
        Trading Strategy client instance

    :param time_bucket:
        The candle time frame

    :param chain_id:
        Which blockchain hosts our exchange.

        Legacy. Give this or `pair`.

    :param exchange_slug:
        Which exchange hosts our trading pairs

        Legacy. Give this or `pair`.

    :param exchange_slug:
        Which exchange hosts our trading pairs

        Legacy. Give this or `pair`.

    :param pair_tickers:
        List of trading pair tickers as base token quote token tuples.
        E.g. `[('WBNB', 'BUSD'), ('Cake', 'BUSD')]`.

    :param liquidity:
        Set true to load liquidity data as well

    :param stop_loss_time_bucket:
        If set load stop loss trigger
        data using this candle granularity.

    :param execution_context:
        Defines if we are live or backtesting

    :param universe_options:
        Override values given the strategy file.
        Used in testing the framework.

    :param required_history_period:
        How much historical data we need to load.

        Depends on the strategy. Defaults to load all data.
    """

    assert isinstance(client, Client)
    assert isinstance(time_bucket, TimeBucket)
    assert isinstance(execution_context, ExecutionContext)
    if chain_id is not None:
        assert isinstance(chain_id, ChainId)

    if exchange_slug is not None:
        assert isinstance(exchange_slug, str)

    assert isinstance(universe_options, UniverseOptions)

    # Apply overrides
    stop_loss_time_bucket = universe_options.stop_loss_time_bucket_override or stop_loss_time_bucket
    time_bucket = universe_options.candle_time_bucket_override or time_bucket

    live = execution_context.live_trading
    with execution_context.timed_task_context_manager("load_pair_data_for_single_exchange", time_bucket=time_bucket.value):
        if live:
            # This will force client to redownload the data
            logger.info("Purging trading data caches")
            client.clear_caches()
        else:
            logger.info("Using cached data if available")

        exchanges = client.fetch_exchange_universe()
        pairs_df = client.fetch_pair_universe().to_pandas()

        # Resolve full pd.Series for each pair
        # we are interested in
        our_pairs = resolve_pairs_based_on_ticker(
            pairs_df,
            chain_id,
            exchange_slug,
            pair_tickers,
        )

        assert len(our_pairs) > 0, f"Pair data not found {chain_id.name}, {exchange_slug}, {pair_tickers}"

        assert len(our_pairs) == len(pair_tickers), f"Pair resolution failed. Wanted to have {len(pair_tickers)} pairs, but after pair id resolution ended up with {len(our_pairs)} pairs"

        our_pair_ids = set(our_pairs["pair_id"])

        if len(our_pair_ids) > 1:
            desc = f"Loading OHLCV data for {exchange_slug}"
        else:
            pair = pair_tickers[0]
            desc = f"Loading OHLCV data for {pair[0]}-{pair[1]}"

        if required_history_period is not None:
            start_time = datetime.datetime.utcnow() - required_history_period
        else:
            start_time = None

        candles = client.fetch_candles_by_pair_ids(
            our_pair_ids,
            time_bucket,
            progress_bar_description=desc,
            start_time=start_time,
        )

        if stop_loss_time_bucket:
            stop_loss_desc = f"Loading granular price data for stop loss/take profit for {exchange_slug}"
            stop_loss_candles = client.fetch_candles_by_pair_ids(
                our_pair_ids,
                stop_loss_time_bucket,
                progress_bar_description=stop_loss_desc,
                start_time=start_time,
            )
        else:
            stop_loss_candles = None

        if liquidity:
            raise NotImplemented("Partial liquidity data loading is not yet supported")

        return Dataset(
            time_bucket=time_bucket,
            exchanges=exchanges,
            pairs=our_pairs,
            candles=candles,
            liquidity=None,
            backtest_stop_loss_time_bucket=stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candles,
        )



def load_partial_data(
        client: BaseClient,
        execution_context: ExecutionContext,
        time_bucket: TimeBucket,
        pairs: Collection[HumanReadableTradingPairDescription],
        universe_options: UniverseOptions,
        liquidity=False,
        stop_loss_time_bucket: Optional[TimeBucket]=None,
        required_history_period: Optional[datetime.timedelta] = None,
        start_at: Optional[datetime.datetime] = None,
        end_at: Optional[datetime.datetime] = None,
        name: Optional[str] = None,
) -> Dataset:
    """Load pair data for named trading pairs.a

    A loading function designed to load data for 2-20 pairs.
    Instead of loading all pair data over Parquet datasets,
    load only specific pair data from their corresponding JSONL endpoints.

    :param client:
        Trading Strategy client instance

    :param time_bucket:
        The candle time frame

    :param chain_id:
        Which blockchain hosts our exchange

    :param exchange_slug:
        Which exchange hosts our trading pairs

    :param exchange_slug:
        Which exchange hosts our trading pairs

    :param pair_tickers:
        List of trading pair tickers as base token quote token tuples.
        E.g. `[('WBNB', 'BUSD'), ('Cake', 'BUSD')]`.

    :param liquidity:
        Set true to load liquidity data as well

    :param stop_loss_time_bucket:
        If set load stop loss trigger
        data using this candle granularity.

    :param execution_context:
        Defines if we are live or backtesting

    :param universe_options:
        Override values given the strategy file.
        Used in testing the framework.

    :param required_history_period:
        How much historical data we need to load.

        Depends on the strategy. Defaults to load all data.

    :param start_at:
        Load data for a specific backtesting data range.

    :param end_at:
        Load data for a specific backtesting data range.

    :param name:
        The loading operation name used in progress bars

    :return:
        Datataset containing the requested data

    """

    assert isinstance(client, Client)
    assert isinstance(time_bucket, TimeBucket)
    assert isinstance(execution_context, ExecutionContext)
    assert isinstance(universe_options, UniverseOptions)

    # Apply overrides
    stop_loss_time_bucket = universe_options.stop_loss_time_bucket_override or stop_loss_time_bucket
    time_bucket = universe_options.candle_time_bucket_override or time_bucket

    assert start_at and end_at, "Current implementatation is designed for backtest use only and needs both start_at and end_at timestamps"

    with execution_context.timed_task_context_manager("load_partial_pair_data", time_bucket=time_bucket.value):

        exchange_universe = client.fetch_exchange_universe()

        pairs_df = client.fetch_pair_universe().to_pandas()
        pair_universe = PandasPairUniverse(pairs_df)

        # Filter pairs first and then rest by the resolved pairs
        our_pairs = {pair_universe.get_pair_by_human_description(exchange_universe, d) for d in pairs}
        our_pair_ids = {p.pair_id for p in our_pairs}
        exchange_ids = {p.exchange_id for p in our_pairs}
        our_exchanges = {exchange_universe.get_by_id(id) for id in exchange_ids}
        our_exchange_universe = ExchangeUniverse.from_collection(our_exchanges)

        # Eliminate the pairs we are not interested in from the database
        filtered_pairs_df = pairs_df.loc[pairs_df["pair_id"].isin(our_pair_ids)]

        # Autogenerate names by the pair count
        if not name:
            name = f"{len(filtered_pairs_df)} pairs"

        progress_bar_desc = f"Loading OHLCV data for {name}"
        candles = client.fetch_candles_by_pair_ids(
            our_pair_ids,
            time_bucket,
            progress_bar_description=progress_bar_desc,
            start_time=start_at,
            end_time=end_at,
        )

        if stop_loss_time_bucket:
            stop_loss_desc = f"Loading stop loss/take granular data for {name}"
            stop_loss_candles = client.fetch_candles_by_pair_ids(
                our_pair_ids,
                stop_loss_time_bucket,
                progress_bar_description=stop_loss_desc,
                start_time=start_at,
                end_time=end_at,
            )
        else:
            stop_loss_candles = None

        if liquidity:
            raise NotImplemented("Partial liquidity data loading is not yet supported")

        return Dataset(
            time_bucket=time_bucket,
            exchanges=our_exchange_universe,
            pairs=filtered_pairs_df,
            candles=candles,
            liquidity=None,
            backtest_stop_loss_time_bucket=stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candles,
        )
