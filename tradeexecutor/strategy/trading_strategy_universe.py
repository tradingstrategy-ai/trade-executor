"""Trading Strategy oracle data integration.

Define trading universe based on data from :py:mod:`tradingstrategy` and
market data feeds.

See :ref:`trading universe` for more information.
"""

import contextlib
import datetime
import pickle
import textwrap
from abc import abstractmethod
from dataclasses import dataclass, field
import logging
from math import isnan
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Set, Dict, Iterable, Collection, TypeAlias

import pandas as pd
from tabulate import tabulate

from tradingstrategy.lending import LendingReserveUniverse, LendingReserveDescription, LendingCandleType, LendingCandleUniverse, UnknownLendingReserve, LendingProtocolType, LendingReserve
from tradingstrategy.token import Token
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client, BaseClient
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.liquidity import GroupedLiquidityUniverse, ResampledLiquidityUniverse
from tradingstrategy.pair import DEXPair, PandasPairUniverse, resolve_pairs_based_on_ticker, \
    filter_for_exchanges, filter_for_quote_tokens, StablecoinFilteringMode, filter_for_stablecoins, \
    HumanReadableTradingPairDescription, filter_for_chain, filter_for_base_tokens, filter_for_exchange, filter_for_trading_fee
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.token_metadata import TokenMetadata
from tradingstrategy.transport.cache import OHLCVCandleType
from tradingstrategy.types import TokenSymbol, NonChecksummedAddress
from tradingstrategy.universe import Universe
from tradingstrategy.utils.groupeduniverse import filter_for_pairs, NoDataAvailable
from tradingstrategy.utils.token_extra_data import load_extra_metadata
from tradingstrategy.utils.token_filter import add_base_quote_address_columns
from tradingstrategy.vault import VaultMetadata
from tradingstrategy.alternative_data.vault import load_multiple_vaults, load_vault_price_data, convert_vault_prices_to_candles

from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind, AssetType
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse, UniverseModel, DataTooOld, UniverseOptions, default_universe_options
from tradeexecutor.state.types import JSONHexAddress, Percent


logger = logging.getLogger(__name__)


#: Unique hash string for each universe.
#:
#: Semi-human readable, is used for filenames on a disk.
#:
UniverseCacheKey: TypeAlias = str


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

    #: All lending reserves
    lending_reserves: Optional[LendingReserveUniverse] = None

    #: Candle data for all pairs
    candles: Optional[pd.DataFrame] = None

    #: All liquidity samples
    liquidity: Optional[pd.DataFrame] = None

    #: All lendinds candles
    lending_candles: Optional[LendingCandleUniverse] = None

    #: Liquidity data granularity
    liquidity_time_bucket: Optional[TimeBucket] = None

    #: Granularity of backtesting OHLCV data
    backtest_stop_loss_time_bucket: Optional[TimeBucket] = None

    #: All candles in stop loss time bucket
    backtest_stop_loss_candles: Optional[pd.DataFrame] = None

    #: Data clipping period
    start_at: Optional[datetime.datetime] = None

    #: Data clipping period
    end_at: Optional[datetime.datetime] = None

    #: How much back we looked from today
    history_period: Optional[datetime.timedelta] = None

    def __repr__(self):
        return f"<Dataset pairs:{len(self.pairs)} candles:{len(self.candles)} start:{self.start_at} end:{self.end_at} live history period:{self.history_period}>"

    def get_chain_ids(self) -> Set[ChainId]:
        """Get all chain ids on this dataset."""
        return {e.chain_id for e in self.exchanges.exchanges}

    def __post_init__(self):
        """Check we got good data."""
        candles = self.candles
        if candles is not None:
            assert isinstance(candles, pd.DataFrame), f"Expected DataFrame, got {candles.__class__}"

        liquidity = self.liquidity
        if liquidity is not None:
            assert isinstance(liquidity, pd.DataFrame), f"Expected DataFrame, got {liquidity.__class__}"

        lending_candles = self.lending_candles
        if lending_candles is not None:
            assert isinstance(lending_candles, LendingCandleUniverse), f"Expected LendingCandleUniverse, got {lending_candles.__class__}"

        # For live trading start_at and end_at contains the timebucket rounded data load period
        # if self.history_period:
        #    assert self.start_at is None and self.end_at is None, f"You can only give history_period or backtesting range. We got {self.start_at}, {self.end_at}, {self.history_period}"


@dataclass(slots=True)
class TradingStrategyUniverse(StrategyExecutionUniverse):
    """A trading strategy universe using our own data feeds.

    - Captures both the market data feeds
      and factors needed to make trading decisions,
      like the strategy reserve currency and backtesting tweaks

    """

    #: Trading universe datasets.
    #:
    #: This encapsulates more generic `Universe` class from `tradingstrategy` package.
    data_universe: Optional[Universe] = None

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

    #: What options were applied to this universe
    #:
    #: Could be e.g.
    #:
    #: - Loaded data range
    #: - Backtested data range
    #:
    #: Not set in legacy code paths.
    #: Currently the life cycle of this variable is not well-defined,
    #: mostly used to pass the backtest data range around,
    #: and set before backtest run, not during the universe construction.
    #:
    options: UniverseOptions | None = None

    #: How much lag we allow in the price feed.
    #:
    #: This can be set for a very high value when working with open ended universes where the liquidity pools are
    #: disappear when liquidity providers leave.
    #:
    #: TODO: Use carefully - this is mostly a workaround variable and
    #: will have a more robust TVL/liquidity solution in the future.
    #:
    price_data_delay_tolerance: datetime.timedelta | None = None

    #: Translated asset and trading pair identifier cache.
    #:
    #: - When called `asset.set_tags()` or setting `asset.other_data()` it is stored here.
    #:
    #: - See :py:meth:`warm_up_data`
    #:
    pair_cache: dict = field(default_factory=dict)

    def __repr__(self):
        pair_count = self.data_universe.pairs.get_count()
        if pair_count <= 3:
            pair_tickers = [f"{p.base_token_symbol}-{p.quote_token_symbol}" for p in self.data_universe.pairs.iterate_pairs()]
            return f"<TradingStrategyUniverse for {', '.join(pair_tickers)}>"
        else:
            return f"<TradingStrategyUniverse for {self.data_universe.pairs.get_count()} pairs>"

    def __post_init__(self):
        """Check that we correctly constructed the instance."""

        if self.data_universe is not None:
            assert isinstance(self.data_universe, Universe)

        if self.backtest_stop_loss_candles is not None:
            assert isinstance(self.backtest_stop_loss_candles, GroupedCandleUniverse), f"Expected GroupedCandleUniverse, got {self.backtest_stop_loss_candles.__class__}"
            assert isinstance(self.backtest_stop_loss_time_bucket, TimeBucket)

    def get_cache_key(self) -> UniverseCacheKey:
        """Get semi-human-readable filename id for this universe.

        .. note::

            Currently does not capture all the nuances of the data.
            Must be defined later to produce an accurate hash on Universe.

        """

        assert len(self.data_universe.chains) == 1

        # Currently supports only full date ranges,
        # to keep filenames clean.
        # Easy to support any other range, just add tests.
        # assert self.start_at.hour == 0
        # assert self.end_at.hour == 0
        # assert self.start_at.minute == 0
        # assert self.end_at.minute == 0
        assert self.start_at.second == 0, f"Got start_at: {self.start_at}"
        assert self.end_at.second == 0, f"Got end_at: {self.end_at}"

        time_str = f"{self.start_at.strftime('%Y-%m-%d')}-{self.end_at.strftime('%Y-%m-%d')}"
        if self.get_pair_count() < 5:
            pair_str = "-".join([p.get_ticker() for p in self.data_universe.pairs.iterate_pairs()])
        else:
            pair_str = str(self.get_pair_count())

        chain_str = self.data_universe.get_default_chain().get_slug()
        time_bucket_str = self.data_universe.time_bucket.value

        separator = "_"

        # Add forward fill flag to the universe cache file name
        match self.data_universe.forward_filled:
            case None:
                ff = ""
            case True:
                ff = f"{separator}ff"
            case False:
                ff = f"{separator}nff"

        key = f"{chain_str}{separator}{time_bucket_str}{separator}{pair_str}{separator}{time_str}{ff}"
        assert len(key) < 256, f"Generated very long fname cache key, check the generation logic: {key}"
        return key

    @property
    def start_at(self) -> datetime.datetime:
        """Start timestamp of the data.

        - Valid for backtesting only
        - Based on OHLCV candles
        """
        start, end = self.data_universe.candles.get_timestamp_range()
        return start.to_pydatetime()

    @property
    def end_at(self) -> datetime.datetime:
        """End timestamp of the data.

        - Valid for backtesting only
        - Based on OHLCV candles
        """
        start, end = self.data_universe.candles.get_timestamp_range()
        return end.to_pydatetime()

    @property
    def universe(self):
        """Backwards compatibility method.

        Deprecate in some point.
        """
        return self.data_universe

    def is_open_ended_universe(self) -> bool:
        """Can new trading pairs to be added to this universe over time.

        :return:
            True if the trading universe may contain hundreds or thousands of trading pairs.
        """
        # TODO: In the future strategy modules, make a real flag for this and now we just use this hack
        return self.data_universe.pairs.get_count() > 20

    def has_lending_data(self) -> bool:
        """Is any lending data available.

        .. note ::

            For live trading, lending candles are not available,
            but any lending rates are directly updated from on-chain sources.
        """
        return self.data_universe.lending_reserves is not None

    def has_liquidity_data(self) -> bool:
        """Is any liquidty data available.
        """
        return self.data_universe.liquidity is not None

    def check_has_vault(self, vault_spec: tuple):
        """Check if we have price data loaded for a vault.

        :raise AssertionError:
            A helpful error if the data is missing
        """
        assert isinstance(vault_spec, tuple), f"Expected vault_spec to be a tuple, got {vault_spec.__class__}: {vault_spec}"
        chain_id = vault_spec[0]
        assert isinstance(chain_id, ChainId), f"Expected ChainId, got {chain_id.__class__}: {chain_id}"
        address = vault_spec[1]
        assert address.startswith("0x"), f"Expected address, got {address}"

        # Raises if not found
        vault_pair = self.data_universe.pairs.get_pair_by_smart_contract(
            address,
        )
        assert vault_pair, f"Vault not in pair universe: {vault_spec}"

        candles = self.data_universe.candles.get_candles_by_pair(
            vault_pair,
        )

        assert candles is not None, f"No price data found for vault: {vault_pair}"

    def get_pair_by_id(self, internal_id: int) -> Optional[TradingPairIdentifier]:
        """Get a trading pair data by its internal id (pair_id)."""
        pair = self.data_universe.pairs.get_pair_by_id(internal_id)
        if not pair:
            return None
        return translate_trading_pair(pair, cache=self.pair_cache)

    def get_trading_pair(self, pair: int | DEXPair) -> TradingPairIdentifier:
        """Get a pair by id or by its data description.

        :param pair:
            Trading pair internal id or DEXPair object

        :return:
            Tradind strategy pair definition.

        :raise PairNotFoundError:
            If we have not loaded data for the given pair id.

        """
        if type(pair) == int:
            dex_pair = self.data_universe.pairs.get_pair_by_id(pair)
        else:
            dex_pair = pair

        return translate_trading_pair(dex_pair, cache=self.pair_cache)

    def can_open_spot(
            self,
            timestamp: pd.Timestamp,
            pair: TradingPairIdentifier,
            liquidity_threshold=None,
    ) -> bool:
        """Can we do a spot trade for a trading pair.

        To be used with backtesting. We will
        check a spot market exists at a certain historic point of time.

        :param timestamp:
            When

        :param pair:
            The wanted trading pair

        :param liquidity_threshold:
            Not implemented yet.

        :return:
            True if we can open a spot position.
        """
        raise NotImplementedError("This function is still TBD")

    def has_any_lending_data(self) -> bool:
        """Does this trading universe has any lending data loaded"""
        return self.data_universe.lending_reserves is not None

    def has_lending_market_available(
        self,
        timestamp: pd.Timestamp | datetime.datetime,
        asset: AssetIdentifier,
        liquidity_threshold=None,
        market_metric: LendingCandleType=LendingCandleType.variable_borrow_apr,
        data_lag_tolerance=pd.Timedelta("1w"),
    ) -> bool:
        """Did an asset have a lending market available at certain historic point of time.

        To be used with backtesting. We will
        check a lending market exists at a certain historic point of time.

        :param timestamp:
            When

        :param pair:
            The wanted trading pair

        :param liquidity_threshold:
            Not implemented yet.

        :return:
            True if we can open a spot position.
        """

        if isinstance(timestamp, datetime.datetime):
            timestamp = pd.Timestamp(timestamp)

        assert isinstance(timestamp, pd.Timestamp), f"Expected pd.Timestamp, got {timestamp.__class__}: {timestamp}"

        assert self.data_universe.lending_candles, "Lending market data is not loaded - cannot determine if we can short or not"

        try:
            reserve = self.data_universe.lending_reserves.get_by_chain_and_address(
                ChainId(asset.chain_id),
                asset.address,
            )
        except UnknownLendingReserve:
            # Market not registered, not available in the dataset
            return False

        assert market_metric == LendingCandleType.variable_borrow_apr, f"Not supported yet: {market_metric}"
        candles = self.data_universe.lending_candles.variable_borrow_apr

        try:
            value, drift = candles.get_single_rate(
                reserve,
                timestamp,
                data_lag_tolerance
            )
            return value > 0
        except NoDataAvailable:
            return False

    def warm_up_data(self):
        """Make sure all data is properly

        - Trading pair data must be preprocessed before we can call `set_tags()` or set `other_data`

        - See :py:attr:`pair_cache` for details
        """
        # Loading pair from iterate_pairs() will build a persistent data structure for it
        _ = [p for p in self.iterate_pairs()]

    def can_open_short(
        self,
        timestamp: pd.Timestamp | datetime.datetime,
        pair: TradingPairIdentifier,
        liquidity_threshold=None,
    ) -> bool:
        """Can we do a short trade for a trading pair.

        To be used with backtesting. We will
        check a lending market exists at a certain historic point of time
        for both base and quote asset.

        :param timestamp:
            When

        :param pair:
            The wanted trading pair

        :param liquidity_threshold:
            Not implemented yet.

        :return:
            True if we can open a short position.
        """
        assert isinstance(pair, TradingPairIdentifier), f"Expected TradingPairIdentifier, got: {pair.__class__}: {pair}"
        return self.has_lending_market_available(timestamp, pair.base, liquidity_threshold) \
            and self.has_lending_market_available(timestamp, pair.quote, liquidity_threshold)
    
    def can_open_credit_supply(
        self,
        timestamp: pd.Timestamp | datetime.datetime,
        pair: TradingPairIdentifier,
        liquidity_threshold=None,
    ) -> bool:
        """Can we do a credit supply trade for a trading pair.

        To be used with backtesting. We will
        check a lending market exists at a certain historic point of time
        for both base and quote asset.

        :param timestamp:
            When

        :param pair:
            The wanted trading pair

        :param liquidity_threshold:
            Not implemented yet.

        :return:
            True if we can open a credit supply position.
        """
        assert isinstance(pair, TradingPairIdentifier), f"Expected TradingPairIdentifier, got: {pair.__class}: {pair}"
        return self.has_lending_market_available(timestamp, pair.quote, liquidity_threshold)

    def clone(self) -> "TradingStrategyUniverse":
        """Create a copy of this universe.

        Any dataframes are now copied,
        but set by reference, so do not mutate in place.
        """
        u = self.data_universe
        new_universe = Universe(
            time_bucket=u.time_bucket,
            chains=u.chains,
            exchanges=u.exchanges or set(),
            exchange_universe=u.exchange_universe,
            pairs=u.pairs,
            candles=u.candles,
            liquidity=u.liquidity,
            resampled_liquidity=u.resampled_liquidity,
            lending_candles=u.lending_candles,
        )
        return TradingStrategyUniverse(
            data_universe=new_universe,
            backtest_stop_loss_time_bucket=self.backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=self.backtest_stop_loss_candles,
            reserve_assets=self.reserve_assets,
            required_history_period=self.required_history_period,
        )

    def write_pickle(self, path: Path):
        """Write a pickled ouput of this universe"""
        assert isinstance(path, Path)
        with path.open("wb") as out:
            pickle.dump(self, out)

    @staticmethod
    def read_pickle_dangerous(path: Path) -> "TradingStrategyUniverse":
        """Write a pickled ouput of this universe.

        .. warning::

            Only use for trusted input. Python pickle issue.

        """
        assert isinstance(path, Path)
        with path.open("rb") as inp:
            item = pickle.load(inp)
            assert isinstance(item, TradingStrategyUniverse)
            return item

    def get_pair_count(self) -> int:
        return self.data_universe.pairs.get_count()

    def is_empty(self) -> bool:
        """This is an empty universe

        - without trading pairs

        - ...or without candles
        """
        candles = self.data_universe.candles.df if self.data_universe.candles else []
        return self.data_universe.pairs.get_count() == 0 or len(candles) == 0

    def is_single_pair_universe(self) -> bool:
        """Is this trading universe made for a single pair trading.

        Note that even a single pair universe may contain two trading pairs,
        if we need intermediate routing pairs. E.g. AAVE -> BNB -> BUSD/
        """

        # TODO: Make a stupid assumption here
        # as our strategies currently have 1 or 2 pairs for single pair trading.
        # Convert this to a proper flag later.
        return self.data_universe.pairs.get_count() == 1

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
        assert self.data_universe.exchange_universe, "You must set universe.exchange_universe to be able to use this method"
        try:
            pair = self.data_universe.pairs.get_pair_by_human_description(self.data_universe.exchange_universe, desc)
        except Exception as e:
            # TODO: Have a better exception here
            raise RuntimeError(f"Failed to look up: {desc}") from e
        return translate_trading_pair(pair, cache=self.pair_cache)

    def get_lending_reserve_by_human_description(self, reserve_description: LendingReserveDescription) -> TradingPairIdentifier:
        """Get a trading pair that represents supplying credit to Aave pool.

        :param reserve_description:
            E.g. ("ethereum", "aave", "USDC")

        :return:
            The trading pair on the exchange that represents lending reserve.

            Like ("ethereum", "aave", "aUSDC", "USDC")
        """

        assert len(reserve_description) == 3

        assert self.data_universe.lending_reserves, "You must set universe.lending_reserves to be able to use this method"
        lending_reserve = self.data_universe.lending_reserves.resolve_lending_reserve(reserve_description)
        return translate_credit_reserve(
            lending_reserve,
            strategy_reserve=self.get_reserve_asset()
        )

    def get_pair_by_smart_contract(self, address: JSONHexAddress) -> TradingPairIdentifier:
        """Get pair by its smart contract address.

        - Most useful for vaults
        - See :py:meth:`tradingstrategy.pair.PandasPairUniverse.get_pair_by_smart_contract`

        :return:
            The trading pair object for the vault.

        :raise NoPairFound:
            In the case input data cannot be resolved.
        """
        try:
            pair = self.data_universe.pairs.get_pair_by_smart_contract(address)
        except Exception as e:
            # TODO: Have a better exception here
            raise RuntimeError(f"Failed to look up: {address}") from e
        return translate_trading_pair(pair, cache=self.pair_cache)

    def iterate_pairs(self) -> Iterable[TradingPairIdentifier]:
        """Iterate over all available trading pairs.

        - Different from :py:meth:`tradingstrategy.pair.PandasPairUniverse.iterate_pairs` as this yields `TradingPairIdentifier` instances
        """
        for p in self.data_universe.pairs.iterate_pairs():
            yield translate_trading_pair(p, cache=self.pair_cache)

    def iterate_credit_for_reserve(self) -> Iterable[TradingPairIdentifier]:
        """Iterate over all available credit supply pairs.

        - We can deposit our reserve and get aToken for interest
        """
        strategy_reserve = self.get_reserve_asset()
        for lending_reserve in self.data_universe.lending_reserves.iterate_reserves():
            yield translate_credit_reserve(
                lending_reserve,
                strategy_reserve
            )

    def create_single_pair_universe(
            dataset: Dataset,
            chain_id: Optional[ChainId] = None,
            exchange_slug: Optional[str] = None,
            base_token: Optional[str] = None,
            quote_token: Optional[str] = None,
            pair: Optional[HumanReadableTradingPairDescription] = None,
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

        if not pair:
            if base_token and quote_token:
                # Legacy method of giving pair
                pair = (chain_id, exchange_slug, base_token, quote_token)

                # Limit the pair universe for selected entries
                pair_universe = PandasPairUniverse.create_pair_universe(
                    dataset.pairs,
                    pairs=[pair],
                )
            else:
                # Pair not given, but
                # input has only single pair
                assert len(dataset.pairs) == 1, "You need to give pair argument if input pair universe contains multiple pairs"
                pair_universe = PandasPairUniverse(dataset.pairs)
                pair_data = pair_universe.get_single()
                pair = pair_data.to_human_description()
        else:
            # Pair given
            pair_universe = PandasPairUniverse.create_pair_universe(
                dataset.pairs,
                pairs=[pair],
            )

        chain_id = pair[0]
        exchange_slug = pair[1]

        # We only trade on Pancakeswap v2
        exchange_universe = dataset.exchanges
        exchange = exchange_universe.get_by_chain_and_slug(chain_id, exchange_slug)
        assert exchange, f"No exchange {exchange_slug} found on chain {chain_id.name}"

        # Get daily candles as Pandas DataFrame
        if dataset.candles is not None:
            all_candles = dataset.candles
            filtered_candles = filter_for_pairs(all_candles, pair_universe.df)
            candle_universe = GroupedCandleUniverse(filtered_candles, time_bucket=dataset.time_bucket)
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

        # Legacy code fix
        pair_universe.exchange_universe = exchange_universe

        universe = Universe(
            time_bucket=dataset.time_bucket,
            chains={chain_id},
            pairs=pair_universe,
            exchanges={exchange},
            candles=candle_universe,
            liquidity=liquidity_universe,
            exchange_universe=exchange_universe,
            lending_candles=dataset.lending_candles
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
            data_universe=universe,
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
        reserve_asset_pair_ticker: Optional[Tuple[str, str]] = None,
        forward_fill=False,
    ) -> "TradingStrategyUniverse":
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

        :param forward_fill:
            Forward-fill the data.

            When working with sparse data (gaps in candles), many strategies need
            these gaps to be filled. Setting this parameter `True`
            will automatically forward-fill any data we are loading from the dataset.

            See :term:`forward fill` for more information.

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

        if all_candles is not None:
            filtered_candles = filter_for_pairs(all_candles, pair_universe.df)
            candle_universe = GroupedCandleUniverse(
                filtered_candles,
                time_bucket=dataset.time_bucket,
                forward_fill=forward_fill
            )
        else:
            candle_universe = None

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
            exchange_universe=exchange_universe,
            lending_candles=dataset.lending_candles,
            forward_filled=forward_fill,
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
            data_universe=universe,
            reserve_assets=reserve_assets,
            backtest_stop_loss_time_bucket=dataset.backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candle_universe,
        )

    def get_pair_by_address(self, address: str) -> Optional[TradingPairIdentifier]:
        """Get a trading pair data by a smart contract address."""
        pair = self.data_universe.pairs.get_pair_by_smart_contract(address)
        if not pair:
            return None
        return translate_trading_pair(pair, cache=self.pair_cache)

    def get_asset_by_address(self, address: str, chain_id: ChainId=None) -> Optional[AssetIdentifier]:
        """Get a trading pair data by a smart contract address."""
        if chain_id is None:
            chain_id = self.data_universe.get_default_chain()
        token = self.data_universe.pairs.get_token(address, chain_id=chain_id)
        if not token:
            return None
        return translate_token(token)

    def get_single_pair(self) -> TradingPairIdentifier:
        """Get the single trading pair in this universe.

        :raise Exception:
            If we have more than one trading pair
        """
        pair = self.data_universe.pairs.get_single()
        return translate_trading_pair(pair, cache=self.pair_cache)

    def get_single_chain(self) -> ChainId:
        """Get the single trading pair in this universe.

        :raise Exception:
            If we have more than one chain
        """
        assert len(self.data_universe.chains) == 1
        return next(iter(self.data_universe.chains))

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
            data_universe=universe,
            reserve_assets=reserve_assets,
            backtest_stop_loss_time_bucket=backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=backtest_stop_loss_candles,
        )

    @staticmethod
    def create_from_dataset(
        dataset: Dataset,
        reserve_asset: JSONHexAddress | TokenSymbol=None,
        forward_fill=False,
        forward_fill_until: datetime.datetime | None = None,
    ):
        """Create a universe from loaded dataset.

        Example:

        .. code-block:: python

            def create_trading_universe(
                timestamp: datetime.datetime,
                client: Client,
                execution_context: ExecutionContext,
                universe_options: UniverseOptions,
            ) -> TradingStrategyUniverse:

                # ... a lot of code goes here

                # First phase:
                # Load all data files and JSON API streams
                # from the server needed to construct the trading universe
                dataset = load_partial_data(
                    client=client,
                    time_bucket=Parameters.candle_time_bucket,
                    pairs=pairs_df,
                    execution_context=execution_context,
                    universe_options=universe_options,
                    liquidity=True,
                    liquidity_time_bucket=TimeBucket.d1,
                    liquidity_query_type=OHLCVCandleType.tvl_v2,
                )

                # Second phase
                # Construct trading strategy universe from the dataset.
                # Wrangles/massages data out for the issues
                strategy_universe = TradingStrategyUniverse.create_from_dataset(
                    dataset,
                    # USDC on Base
                    reserve_asset="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    # Because we use sparse candles, we need to fill gaps between candles
                    forward_fill=True,
                    # Because some pairs may have died during real-time trading, we still
                    # need to forward fill until today, not the last day they had a trade
                    forward_fill_until=timestamp,
                )

                return strategy_universe

        For more code examples see :py:func:`load_trading_and_lending_data`.

        :param reserve_asset:
            Which reserve asset to use.

            As the token address or symbol.

            If not given try to guess from the dataset.

            - Assume all trading pairs have the same quote token
              and that is our reserve asset

            - If dataset has trading pairs with different quote tokens,
              aborts

            Examples: 
            
            - `0x2791bca1f2de4661ed88a30c99a7a9449aa84174`  (USDC.e bridged address on Polygon)

            - `0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359`  (USDC native)

        :param forward_fill:
            Forward-fill the data.

            When working with sparse data (gaps in candles), many strategies need
            these gaps to be filled. Setting this parameter `True`
            will automatically forward-fill any data we are loading from the dataset.

            See :term:`forward fill` for more information.
        """

        logger.info(
            "create_from_dataset(), forward fill: %s, until: %s, reserve: %s, dataset pairs %d",
            forward_fill,
            forward_fill_until,
            reserve_asset,
            len(dataset.pairs)
        )

        chain_ids = dataset.pairs["chain_id"].unique()

        assert len(chain_ids) == 1, f"Currently only single chain datasets supported, got chains {chain_ids}"
        chain_id = ChainId(chain_ids[0])

        pairs = PandasPairUniverse(dataset.pairs, exchange_universe=dataset.exchanges)

        if not reserve_asset:
            quote_token = pairs.get_single_quote_token()
            reserve_asset = translate_token(quote_token)
        elif reserve_asset.startswith("0x"):
            reserve_asset_token = pairs.get_token(reserve_asset)
            assert reserve_asset_token, f"Pairs dataset does not contain data for the specified reserve asset: {reserve_asset}.\nThere are {pairs.get_count()} trading pairs loaded."
            reserve_asset = translate_token(reserve_asset_token)
        else:
            reserve_asset_token = pairs.get_token_by_symbol(reserve_asset)
            reserve_asset = translate_token(reserve_asset_token)

        if forward_fill:
            if forward_fill_until is None:
                forward_fill_until = dataset.end_at

            if forward_fill_until is None:
                # Legacy unit test backwards compatibiltiy
                logger.warning("forward_fill_until could not be determined, assuming legacy test path and does the slow method")
                forward_fill_until = dataset.candles["timestamp"].max()

            logger.info("Forward filling trading data to %s", forward_fill_until)
            assert forward_fill_until is not None, f"forward_fill set, not forward_fill_until not set or not available in dataset"

        candle_universe = GroupedCandleUniverse(
            dataset.candles,
            forward_fill=forward_fill,
            time_bucket=dataset.time_bucket,
            forward_fill_until=forward_fill_until,
        )

        if dataset.backtest_stop_loss_candles is not None:
            stop_loss_candle_universe = GroupedCandleUniverse(
                dataset.backtest_stop_loss_candles,
                forward_fill=forward_fill,
                time_bucket=dataset.backtest_stop_loss_time_bucket,
                forward_fill_until=forward_fill_until,
            )
        else:
            stop_loss_candle_universe = None

        if dataset.liquidity is not None:

            if isinstance(dataset.liquidity.index, pd.Index):
                # Unindexed data
                dataset.liquidity = dataset.liquidity.set_index("timestamp", drop=False)
            elif isinstance(dataset.liquidity.index, pd.MultiIndex):
                # The hack we had to do to in CachedHTTPTransport.fetch_tvl_by_pair_ids() because
                # for some reason pd.concat() kept failing on Github.
                # Goes from (pair_id, timestamp) -> (timestamp) index
                dataset.liquidity = dataset.liquidity.reset_index(level=0, drop=True)

            assert isinstance(dataset.liquidity.index, pd.DatetimeIndex), f"Got {dataset.liquidity.index.__class__}"
            liquidity_universe = GroupedLiquidityUniverse(
                dataset.liquidity,
                time_bucket=dataset.liquidity_time_bucket,
                forward_fill=forward_fill,
                forward_fill_until=forward_fill_until,
                index_automatically=False,
            )
            resampled_liquidity = None
        else:
            liquidity_universe = None
            resampled_liquidity = None

        universe = Universe(
            time_bucket=dataset.time_bucket,
            chains={chain_id},
            pairs=pairs,
            candles=candle_universe,
            liquidity=liquidity_universe,
            resampled_liquidity=resampled_liquidity,
            exchange_universe=dataset.exchanges,
            exchanges={e for e in dataset.exchanges.exchanges.values()},
            lending_candles=dataset.lending_candles,
            forward_filled=forward_fill,
        )

        return TradingStrategyUniverse(
            data_universe=universe,
            reserve_assets=[reserve_asset],
            backtest_stop_loss_time_bucket=dataset.backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candle_universe,
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

        assert reserve_token, f"Reserve token {reserve_token_symbol} missing the specified pair quote tokens of {reserve_token}"
        reserve_assets = [
            reserve_token
        ]

        # Get daily candles as Pandas DataFrame
        all_candles = dataset.candles
        filtered_candles = filter_for_pairs(all_candles, filtered_pairs_df)
        candle_universe = GroupedCandleUniverse(filtered_candles, time_bucket=time_bucket)
        exchange_universe=ExchangeUniverse.from_collection(our_exchanges)

        universe = Universe(
            time_bucket=dataset.time_bucket,
            chains=chain_ids,
            pairs=pair_universe,
            exchanges=our_exchanges,
            candles=candle_universe,
            exchange_universe=exchange_universe,
        )

        pair_universe.exchange_universe = exchange_universe

        if dataset.backtest_stop_loss_candles is not None:
            stop_loss_candle_universe = GroupedCandleUniverse(dataset.backtest_stop_loss_candles)
        else:
            stop_loss_candle_universe = None

        return TradingStrategyUniverse(
            data_universe=universe,
            reserve_assets=reserve_assets,
            backtest_stop_loss_time_bucket=dataset.backtest_stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candle_universe,
        )

    def get_credit_supply_pair(self) -> TradingPairIdentifier:
        """Get the credit supply trading pair.

        This trading pair identifies the trades where we move our strategy reserve
        currency to Aave lending pool to gain interest.

        Assume we move our single reserve currency to a single lending reserve we know for it.

        TODO: Currently one pair per universe supported only.

        :return:
            Credit supply trading pair with ticket (aToken for reserve asset, reserve asset)
        """
        reserve_asset = self.get_reserve_asset()

        # Will raise exception if not available
        reserve = self.data_universe.lending_reserves.get_by_chain_and_address(
            ChainId(reserve_asset.chain_id),
            reserve_asset.address,
        )

        # Sanity check
        assert reserve.get_asset().address == reserve_asset.address

        # atoken = translate_token(reserve.get_atoken(), underlying=underlying)
        return translate_credit_reserve(
            reserve,
            reserve_asset,
        )

    def get_shorting_pair(self, pair: TradingPairIdentifier) -> TradingPairIdentifier:
        """Get the shorting pair from trading pair

        This trading pair identifies the trades where we borrow asset from Aave against 
        collateral.
        
        :param pair:
            Trading pair of the asset to be shorted
        
        :return:
            Short pair with ticker (vToken for borrowed asseet, aToken for reserve asset)
        """

        if not self.data_universe.lending_reserves:
            raise TradingUniverseIssue(f"Lending rates data missing. Asking shorting data for trading pair {pair}.\nMake sure you load lending rates data if you want to backtest leveraged trading.")

        assert pair.kind == TradingPairKind.spot_market_hold

        borrow_token = pair.base
        collateral_token = pair.quote
        assert collateral_token == self.get_reserve_asset()

        try:
            # Will raise exception if not available
            borrow_reserve = self.data_universe.lending_reserves.get_by_chain_and_address(
                ChainId(borrow_token.chain_id),
                borrow_token.address,
            )
        except UnknownLendingReserve as e:
            raise UnknownLendingReserve(f"Could not resolve borrowed token {borrow_token}") from e

        try:
            collateral_reserve = self.data_universe.lending_reserves.get_by_chain_and_address(
                ChainId(collateral_token.chain_id),
                collateral_token.address,
            )
        except UnknownLendingReserve as e:
            raise UnknownLendingReserve(f"Could not resolve collateral token {collateral_token}") from e

        vtoken = translate_token(
            borrow_reserve.get_vtoken(),
            underlying=borrow_token,
            type=AssetType.borrowed,
        )
        atoken = translate_token(
            collateral_reserve.get_atoken(),
            underlying=collateral_token,
            type=AssetType.collateral,
            # TODO: this is only the latest liquidation theshold, for historical data
            # for backtesting we neend to plugin some other logic later
            liquidation_threshold=collateral_reserve.additional_details.liquidation_threshold,
        )

        return TradingPairIdentifier(
            vtoken,
            atoken,
            pool_address=borrow_token.address,
            exchange_address=borrow_token.address,
            internal_id=borrow_reserve.reserve_id,
            kind=TradingPairKind.lending_protocol_short,
            underlying_spot_pair=pair,
        )

    def get_trading_broken_reason(
        self,
        pair: TradingPairIdentifier,
        min_candles_required: int,
        min_price=0.00000001,
        max_price=1_000_000,
    ) -> str | None:
        """Can we trade a pair.

        Check if we can trade a particular trading pair.

        - Work around spotty low cap coins

        - Check that we have minimum amout of bars of data

        - Check that price data does not look weird

        :param pair:
            Trading pair

        :param min_candles_required:
            How many bars of adta we need

        :param min_price:
            Avoid low cap tokens with float64 breaking prices

        :param max_pric:
            Avoid low cap tokens with float64 breaking prices

        :return:
            A string why the trading pair is broken.

            `None` if good.
        """
        candles = self.data_universe.candles.get_candles_by_pair(pair.internal_id)

        if candles is None:
            return "No OHLCV candles"

        if len(candles) >= min_candles_required:

            # Get the first opening price
            for column in ("open", "close"):
                pair_min_price = candles[column].min()
                pair_max_price = candles[column].max()

                if pair_min_price < min_price:
                    return f"Avoid pairs with too low price. Pair min price is {pair_min_price}"
                elif pair_max_price > max_price:
                    return f"Avoid pairs with too high price. Pair max price is {pair_max_price}"
            return None
        return f"Not enough OHLCV candles, {min_candles_required} candles needed"
    
    def get_latest_supply_apr(
        self,
        asset: AssetIdentifier | None = None,
        timestamp: datetime.datetime | None = None,
        tolerance: datetime.timedelta = datetime.timedelta(days=2),
    ) -> float:
        """Get the latest supply APR for the lending asset.

        :param asset:
            Specify the asset to get the rate for
            If none, use the reserve asset.

            Must be non-wrapped, not aToken.

        :param timestamp:
            Specify timestamp to get the rate for
            If none, return the latest rate

        :param tolerance:
            Lookback tolerance with stale data.

            Default to 1 days if not given.

        :return:
            Latest supply APR rate
        """
        if not asset:
            asset = self.get_reserve_asset()

        if isinstance(tolerance, (datetime.timedelta, pd.Timedelta)):
            tolerance = pd.Timedelta(tolerance)
        else:
            tolerance = pd.Timedelta(days=2)

        # Will raise exception if not available
        lending_reserve = self.data_universe.lending_reserves.get_by_chain_and_address(
            ChainId(asset.chain_id),
            asset.address,
        )

        if timestamp:
            try:
                rate, _ = self.data_universe.lending_candles.supply_apr.get_single_rate(
                    lending_reserve,
                    timestamp,
                    data_lag_tolerance=tolerance,
                )
                return rate
            except Exception as e:
                raise RuntimeError(f"get_latest_supply_apr() failed, timestamp: {timestamp}, lending reserve: {lending_reserve}, asset: {asset}, tolerance: {tolerance}\nException: {e}") from e

        # get last available rate
        df = self.data_universe.lending_candles.supply_apr.get_rates_by_reserve(lending_reserve)
        last = df.iloc[-1]

        logger.info("Got latest supply APR %s for %s at %s", last, asset, last.index)

        return last["close"]


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
                  backtest_stop_loss_time_frame: Optional[TimeBucket] = None) -> Dataset:
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

    @staticmethod
    def check_data_age(
        ts: datetime.datetime,
        strategy_universe: TradingStrategyUniverse,
        best_before_duration: datetime.timedelta,
        best_before_duration_liquidity: datetime.timedelta = datetime.timedelta(days=2),
    ) -> datetime.datetime:
        """Check if our data is up-to-date and we do not have issues with feeds.

        Ensure we do not try to execute live trades with stale data.

        :param ts:
            Current time

        :param strategy_universe:
            Strategy universe to examine.

        :param best_before_duration:
            Data must be not older than this duration.

        :param best_before_duration_liquidity:
            Because liquidity data is not as critical as candle data, and uses 1d granulariy, it has its own tolerance.

        :raise DataTooOld:
            in the case data is too old to execute.

        :return:
            The data timestamp
        """

        # Avoid circular import
        from tradeexecutor.analysis.pair import display_strategy_universe

        assert isinstance(strategy_universe, TradingStrategyUniverse), f"Expected TradingStrategyUniverse, got {strategy_universe.__class__}"


        data_universe = strategy_universe.data_universe
        candle_end = None

        if data_universe.candles is not None:

            max_age = ts - best_before_duration

            # Convert pandas.Timestamp to executor internal datetime format
            candle_start, candle_end = data_universe.candles.get_timestamp_range(
                exclude_forward_fill=True,
            )
            candle_start = candle_start.to_pydatetime().replace(tzinfo=None)
            candle_end = candle_end.to_pydatetime().replace(tzinfo=None)

            ff_candle_start, ff_candle_end = data_universe.candles.get_timestamp_range(
                exclude_forward_fill=False,
            )

            # Do a very throughful logging on what went wrong with our data
            if candle_end < max_age:
                diff = max_age - candle_end

                universe_dump_df = display_strategy_universe(
                    strategy_universe,
                    show_volume=False,
                    show_tax=False,
                    show_tvl=False,
                    compact=True,
                )

                universe_output_msg = tabulate(
                    universe_dump_df,
                    headers="keys",
                    tablefmt="fancy_grid",
                )

                # universe_output_msg too long for sentry?
                # base-ath  | OSError: [Errno 90] Message too long
                # logger.error(
                #    f"Universe data too old. Non-forward-filled data is:\n{universe_output_msg}",
                #)

                print(
                    f"Universe data too old. Non-forward-filled data is:\n{universe_output_msg}",
                )

                raise DataTooOld(
                    f"Candle data {candle_start} - {candle_end} is too old to work with\n" \
                    f"we require threshold {max_age}, diff is {diff}, asked best before duration is {best_before_duration}\n"
                    f"Forward-filled data is {ff_candle_start} - {ff_candle_end}\n"
                    f"More information in logs"
                )

        if data_universe.liquidity is not None:
            max_age = ts - best_before_duration_liquidity
            liquidity_start, liquidity_end = data_universe.liquidity.get_timestamp_range(
                exclude_forward_fill=True,
            )
            liquidity_start = liquidity_start.to_pydatetime().replace(tzinfo=None)
            liquidity_end = liquidity_end.to_pydatetime().replace(tzinfo=None)

            if liquidity_end < max_age:
                raise DataTooOld(f"Liquidity data is too old to work with {liquidity_start} - {liquidity_end}")

            return min(liquidity_end, candle_end)

        else:
            return candle_end

    @staticmethod
    def create_from_dataset(
            dataset: Dataset,
            chains: List[ChainId],
            reserve_assets: List[AssetIdentifier],
            pairs_index=True):
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
            data_universe=universe,
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

    def preload_universe(
            self,
            universe_options: UniverseOptions,
            execution_context: ExecutionContext | None = None
    ):
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
                universe_options=universe_options,
                execution_context=execution_context,
            )

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


def translate_token(
    token: Token,
    require_decimals=True,
    underlying: AssetIdentifier | None = None,
    type: AssetType | None = AssetType.token,
    liquidation_threshold: float | None = None,
) -> AssetIdentifier:
    """Translate Trading Strategy token data definition to trade executor.

    Trading Strategy client uses compressed columnar data for pairs and tokens.

    Creates `AssetIdentifier` based on data coming from
    Trading Strategy :py:class:`tradingstrategy.pair.PandasPairUniverse`.

    :param underlying:
        Underlying asset for dynamic lending tokens.

    :param require_decimals:
        Most tokens / trading pairs are non-functional without decimals information.
        Assume decimals is in place. If not then raise AssertionError.
        This check allows some early error catching on bad data.

    :param type:
        What kind of asset this is.

    :param liquidation_theshold:
        Aave liquidation threhold for this asset, only collateral type asset can have this.
    """

    if require_decimals:
        assert token.decimals, f"Bad token: {token}"
        assert token.decimals > 0, f"Bad token: {token}"

    if liquidation_threshold:
        assert type == AssetType.collateral, f"Only collateral tokens can have liquidation threshold, got {type}"

    return AssetIdentifier(
        token.chain_id.value,
        token.address,
        token.symbol,
        token.decimals,
        underlying=underlying,
        type=type,
        liquidation_threshold=liquidation_threshold,
    )


def translate_trading_pair(dex_pair: DEXPair, cache: dict | None = None) -> TradingPairIdentifier:
    """Translate trading pair from client download to the trade executor.

    Trading Strategy client uses compressed columnar data for pairs and tokens.

    Translates a trading pair presentation from Trading Strategy client Pandas format to the trade executor format.

    Trade executor work with multiple different strategies, not just Trading Strategy client based.
    For example, you could have a completely on-chain data based strategy.
    Thus, Trade Executor has its internal asset format.

    This module contains functions to translate asset presentations between Trading Strategy client
    and Trade Executor.


    This is called when a trade is made: this is the moment when trade executor data format must be made available.

    :param cache:
        Cache of constructed objects.

        Pair internal id -> TradingPairIdentifier

        See :py:class:`tradingstrategy.state.identifier.AssetIdentifier` for life cycle notes.
    """

    if cache is not None:
        cached = cache.get(dex_pair.pair_id)
        if cached is not None:
            return cached

    assert isinstance(dex_pair, DEXPair), f"Expected DEXPair, got {type(dex_pair)}"
    assert dex_pair.base_token_decimals is not None, f"Base token missing decimals: {dex_pair}"
    assert dex_pair.quote_token_decimals is not None, f"Quote token missing decimals: {dex_pair}"

    base = AssetIdentifier(
        chain_id=dex_pair.chain_id.value,
        address=dex_pair.base_token_address,
        token_symbol=dex_pair.base_token_symbol,
        decimals=dex_pair.base_token_decimals,
    )
    quote = AssetIdentifier(
        chain_id=dex_pair.chain_id.value,
        address=dex_pair.quote_token_address,
        token_symbol=dex_pair.quote_token_symbol,
        decimals=dex_pair.quote_token_decimals,
    )

    if dex_pair.fee and isnan(dex_pair.fee):
        # Repair some broken data
        fee = None
    else:
        # Convert DEXPair.fee BPS to %
        # So, after this, fee can either be multiplier or None
        if dex_pair.fee is not None:
            # If BPS fee is set it must be more than 1 BPS.
            # Allow explicit fee = 0 in testing.
            # if pair.fee != 0:
            #     assert pair.fee > 1, f"DEXPair fee must be in BPS, got {pair.fee}"

            # can receive fee in bps or multiplier, but not raw form
            if dex_pair.fee >= 1:
                fee = dex_pair.fee / 10_000
            else:
                fee = dex_pair.fee

            # highest fee tier is currently 1% and lowest in 0.01%
            if fee != 0:
                assert 0.0001 <= fee <= 0.01, f"bug in converting fee to multiplier, make sure bps, got fee {fee}"
        else:
            fee = None

    if dex_pair.dex_type == ExchangeType.erc_4626_vault:
        # For vaults, exchange_name is set as the vault protocol name e.g. "morpho" or "ipor"
        kind = TradingPairKind.vault
    else:
        kind = TradingPairKind.spot_market_hold

    pair = TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=dex_pair.address,
        internal_id=int(dex_pair.pair_id),
        info_url=dex_pair.get_trading_pair_page_url(),
        exchange_address=dex_pair.exchange_address,
        fee=fee,
        reverse_token_order=dex_pair.token0_symbol != dex_pair.base_token_symbol,
        exchange_name=dex_pair.exchange_name,
        kind=kind,
        internal_exchange_id=dex_pair.exchange_id,
    )

    # Need to be loaded with load_extra_metadata()
    if dex_pair.buy_tax and dex_pair.buy_tax < 900:
        # 900+ are error codes for built-in internal token tax measurer
        # that should be no longer used - don't bring over these error codes from DEXPair
        pair.base.other_data["buy_tax"] = dex_pair.buy_tax
        pair.base.other_data["sell_tax"] = dex_pair.sell_tax

    # Need to be loaded with load_extra_metadata().
    # see _reduce_other_data() for caveats.
    if dex_pair.other_data:
        # Because other_data is very heavy, we should only copy fields we really care.
        # Below are the whitelisted fields.

        pair.other_data = {}

        # Pass and parse TokenMetadata instance
        token_sniffer_data = None
        metadata = dex_pair.other_data.get("token_metadata")

        match metadata:
            case TokenMetadata():
                pair.other_data["token_metadata"] = metadata
                token_sniffer_data = metadata.token_sniffer_data
            case VaultMetadata():
                pair.other_data["token_metadata"] = metadata
                pair.other_data["vault_features"] = metadata.features
                pair.other_data["vault_protocol"] = metadata.protocol_slug
                pair.other_data["vault_name"] = metadata.vault_name
                pair.other_data["vault_performance_fee"] = metadata.performance_fee
                pair.other_data["vault_management_fee"] = metadata.management_fee
            case None:
                pass
            case _:
                raise NotImplementedError(f"Unknown token metadata type {type(metadata)}")

        if token_sniffer_data is None:
            token_sniffer_data = dex_pair.other_data.get("token_sniffer_data")

        if token_sniffer_data:
            # TODO: Legacy, remove. Instead use TradingPairIdentifier.get_xxx() accessor functions.
            pair.other_data.update({
                "token_sniffer_data": {
                    "swap_simulation": token_sniffer_data.get("swap_simulation"),
                    "score": token_sniffer_data.get("score"),
                }
            })

    # if dex_pair.dex_type == ExchangeType.erc_4626_vault:
    #    import ipdb ; ipdb.set_trace()

    if cache is not None:
        cache[pair.internal_id] = pair

    return pair


def translate_credit_reserve(
    lending_reserve: LendingReserve,
    strategy_reserve: AssetIdentifier,
) -> TradingPairIdentifier:
    """Translate lending protocol reserve from client download to the trade executor.

    :param lending_reserve:
        Raw Token data from Trading Strategy Client

    :param reverse:
        The trading universe reserve asset
    """

    assert isinstance(lending_reserve, LendingReserve)
    internal_id = lending_reserve.reserve_id
    atoken = lending_reserve.get_atoken()

    assert isinstance(atoken, Token)
    assert isinstance(strategy_reserve, AssetIdentifier)

    lending_reserve_underlying = translate_token(lending_reserve.get_asset())

    # TODO: This is the hack fix when Polygon renamed
    # token symbol USDC -> USDC.e
    # In this case
    # Reserve asset: AssetIdentifier(chain_id=137, address='0x2791bca1f2de4661ed88a30c99a7a9449aa84174', token_symbol='USDC', decimals=6, internal_id=None, info_url=None, underlying=None, type=None, liquidation_threshold=None)
    # Underlying: AssetIdentifier(chain_id=137, address='0x2791bca1f2de4661ed88a30c99a7a9449aa84174', token_symbol='USDC.e', decimals=6, internal_id=None, info_url=None, underlying=None, type=<AssetType.token: 'token'>, liquidation_threshold=None)
    if lending_reserve.asset_symbol == "USDC.e":
        underlying = reserve_asset = lending_reserve_underlying
    else:
        underlying = strategy_reserve

    atoken = translate_token(atoken, underlying=underlying)

    return TradingPairIdentifier(
        atoken,
        underlying,
        pool_address=strategy_reserve.address,  # TODO: Now using reserve asset
        exchange_address=strategy_reserve.address,  # TODO: Now using reserve asset
        internal_id=internal_id,
        kind=TradingPairKind.credit_supply,
    )


def create_pair_universe_from_code(chain_id: ChainId, pairs: List[TradingPairIdentifier]) -> "PandasPairUniverse":
    """Create the trading universe from handcrafted data.

    Used in unit testing.
    """
    data = []
    used_ids = set()
    for idx, p in enumerate(pairs):
        assert p.base.decimals
        assert p.quote.decimals
        assert p.internal_exchange_id, f"All trading pairs must have internal_exchange_id set, did not have it set {p}"
        assert p.internal_id

        assert p.internal_id not in used_ids, f"Duplicate internal id {p}: {p.internal_id}"

        # TODO: The reverse translate here is incomplete
        if p.is_vault():
            dex_type = ExchangeType.erc_4626_vault
        else:
            dex_type = ExchangeType.uniswap_v2

        other_data = p.other_data

        dex_pair = DEXPair(
            pair_id=p.internal_id,
            chain_id=chain_id,
            exchange_id=p.internal_exchange_id,
            address=p.pool_address,
            exchange_address=p.exchange_address,
            dex_type=dex_type,
            base_token_symbol=p.base.token_symbol,
            quote_token_symbol=p.quote.token_symbol,
            token0_symbol=p.base.token_symbol,
            token1_symbol=p.quote.token_symbol,
            token0_address=p.base.address,
            token1_address=p.quote.address,
            token0_decimals=p.base.decimals,
            token1_decimals=p.quote.decimals,
            fee=int(p.fee * 10_000) if p.fee else None,  # Convert to bps according to the documentation
            other_data=other_data,
        )
        used_ids.add(p.internal_id)
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

    .. warning::

        Does not work in low memory environments due to high amount of trading pairs.
        Use :py:func:`tradeexecutor.strategy.trading_strategy_universe.load_partial_data`.

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
        if live and not execution_context.mode.is_unit_testing():
            # This will force client to redownload the data
            logger.info("Purging trading data caches for %s, mode is %s", client.__class__.__name__, execution_context.mode)
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


def load_partial_data(
    client: BaseClient,
    execution_context: ExecutionContext,
    time_bucket: TimeBucket,
    pairs: Collection[HumanReadableTradingPairDescription] | pd.DataFrame,
    universe_options: UniverseOptions,
    liquidity=False,
    liquidity_time_bucket: TimeBucket | None = None,
    liquidity_query_type: OHLCVCandleType = OHLCVCandleType.tvl_v1,
    preloaded_tvl_df: pd.DataFrame | None = None,
    stop_loss_time_bucket: Optional[TimeBucket] = None,
    required_history_period: datetime.timedelta | None = None,
    lending_reserves: LendingReserveUniverse | Collection[LendingReserveDescription] | None = None,
    lending_candle_types: Collection[LendingCandleType] = (LendingCandleType.supply_apr, LendingCandleType.variable_borrow_apr),
    start_at: datetime.datetime | None = None,
    end_at: datetime.datetime | None = None,
    name: str | None = None,
    candle_progress_bar_desc: str | None = None,
    lending_candle_progress_bar_desc: str | None = None,
    pair_extra_metadata=False,
    vaults: list[tuple[ChainId, JSONHexAddress]] | None = None,
    vault_bundled_price_data: bool | Path=False,
    round_start_end: bool = True,
) -> Dataset:
    """Load pair data for given trading pairs.

    A loading function designed to load OHLCV data up to 1000 pairs.
    Instead of loading all pair data over Parquet datasets, load only specific pair data from their corresponding JSONL endpoints,
    streaming the data and caching them as local files.

    Because or partial data processing,
    this function has less chance to run out of memory than unlike :py:func:`tradeexecutor.strategy.trading_strategy_universe.load_all_data`.

    Example of loading spot-only data:

    ... code-block:: python

        from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
        from tradingstrategy.transport.cache import OHLCVCandleType


        def create_trading_universe(
                ts: datetime.datetime,
                client: Client,
                execution_context: ExecutionContext,
                universe_options: UniverseOptions,
        ) -> TradingStrategyUniverse:

            dataset = load_partial_data(
                client=client,
                time_bucket=Parameters.candle_time_bucket,
                pairs=pairs_df,
                execution_context=execution_context,
                universe_options=universe_options,
                liquidity=True,
                liquidity_time_bucket=TimeBucket.d1,
                liquidity_query_type=OHLCVCandleType.tvl_v2,
            )

            strategy_universe = TradingStrategyUniverse.create_from_dataset(
                dataset,
                reserve_asset="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC on Base
                forward_fill=True,  # We got very gappy data from low liquid DEX coins
            )

            return universe

    Example of loading spot pair and Aave credit pool data:

    .. code-block:: python

        from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
        # TODO
        pass

    :param client:
        Trading Strategy client instance

    :param time_bucket:
        The candle time frame.

    :param pairs:

        List of trading pair tickers.

        Can be

        - Human-readable descriptions, see :py:attr:`tradingstrategy.pair.HumanReadableTradingPairDescription`.
        - Direct :py:class:`pandas.DataFrame` of pairs.

    :param lending_resserves:
        Lending reserves for which you want to download the data.

        Either list of lending pool descriptions or preloaded lending universe.

    :param lending_candle_types:
        What lending data columns to load

    :param liquidity:
        Set true to load liquidity data as well

    :param liquidity_time_bucket:
        Granularity of loaded TVL data.

        If not given use `time_bucket`.

    :param liquidity_query_type:
        Whether to use new-style or old-style data for TVL.

        See :py:class:`OHLCVCandleType` for details.

    :param preloaded_tvl_df:
        Liquidity data was earlier loaded with ``fetch_tvl(min_tvl)`` when constructing the trading universe.

        We do not reload this same data, but use the preloaded DataFrame directly.

    :param lending_reserves:
        Set true to load lending reserve data as well

    :lending_candle_types:
        What lending data columns to load

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

        TODO: Going to be deprecatd. Please use ``universe_options.start_at`` instead.

    :param end_at:
        Load data for a specific backtesting data range.

        TODO: Going to be deprecatd. Please use ``universe_options.end_at`` instead.

    :param name:
        The loading operation name used in progress bars

    :param candle_progress_bar_desc:
        Override the default progress bar message

    :param lending_candle_progress_bar_desc:
        Override the default progress bar message

    :param pair_extra_metadata:
        Load TokenSniffer data, buy/sell tax and other extra metadata.

        Slow and API endpoint severely limited. Use only if you are dealing with a limited number of pairs.

    :param vaults:
        List of (chain, vault address) tuples to load vault data for.

        Vault metadata loeaded from tradingstrategy data bundle.

        Currently does not load any historical data.

    :param vault_bundled_price_data:
        For vaults, also load bundled static price data.

        Use minimal backtest dataset from Trading Strategy package bundle.

        Not applicable for live trading.

    :param round_start_end:
        Round start and end times to the nearest time bucket.

        Avoid polluting cache space with second difference in timestamps.

        Only applicable for live trading. This is especially helpful for making unit testing/integration testing faster.

        If not given live trading tries to load the data from the server until the last second.

        Max 1d rounding. For weekly time bucket you still get daily boundaries.

    :return:
        Datataset containing the requested data

    """

    assert isinstance(client, Client)
    assert isinstance(time_bucket, TimeBucket)
    assert isinstance(execution_context, ExecutionContext)
    assert isinstance(universe_options, UniverseOptions)

    if preloaded_tvl_df is not None:
        assert not liquidity, "Cannot use liquidity argument with preloaded_tvl_df"

    if required_history_period:
        assert isinstance(required_history_period, datetime.timedelta), f"required_history_period: expected timedelta, got {type(required_history_period)}: {required_history_period}"

    # Apply overrides
    stop_loss_time_bucket = universe_options.stop_loss_time_bucket_override or stop_loss_time_bucket
    time_bucket = universe_options.candle_time_bucket_override or time_bucket

    # Some sanity and safety check
    if len(pairs) >= 1000:
        logger.warning("load_partial_data() method is designed to load data for low number or trading pairs, got %d - this might be slow", len(pairs))

    # Legacy compat
    if execution_context.mode.is_backtesting():
        if not start_at:
            start_at = universe_options.start_at

        # Legacy compat
        if not end_at:
            end_at = universe_options.end_at

        assert start_at
        assert end_at
    elif execution_context.mode.is_live_trading():
        if not required_history_period:
            required_history_period = universe_options.history_period
        assert required_history_period, f"Doing live trading {execution_context.mode}, but universe_options.history_period missing: {universe_options}"


    else:
        raise NotImplementedError(f"Cannot determine trading mode: {execution_context.mode}")

    # Where the data loading start can come from the hard backtesting range (start - end)
    # or how many days of historical data we ask for
    data_load_start_at = start_at or (datetime.datetime.utcnow() - required_history_period)

    # Generate a rounded range of the latest data
    if round_start_end and execution_context.mode.is_live_trading():

        # TODO: Special workaround for some legacy tests/weekly data
        flooring_time_bucket = time_bucket if time_bucket < TimeBucket.d7 else TimeBucket.d1

        floored_start = flooring_time_bucket.floor_datetime(data_load_start_at)
        logger.info(
            "Floored start timestamp %s -> %s for bucket %s",
            data_load_start_at,
            floored_start,
            time_bucket.value,
        )

        start_at = data_load_start_at = floored_start
        if not end_at:
            end_at = datetime.datetime.utcnow()
            floored_end = flooring_time_bucket.floor_datetime(end_at)
            logger.info(
                "Floored end timestamp %s -> %s for bucket %s",
                end_at,
                floored_end,
                time_bucket.value,
            )
            end_at = floored_end

    logger.info(
        "load_partial_data(): data_load_start_at: %s, start_at: %s, end_at: %s, required_history_period: %s",
        data_load_start_at,
        start_at,
        end_at,
        required_history_period,
    )

    with execution_context.timed_task_context_manager("load_partial_pair_data", time_bucket=time_bucket.value):

        exchange_universe = client.fetch_exchange_universe()

        if isinstance(pairs, pd.DataFrame):
            # Prefiltered pairs
            assert len(pairs) > 0, "The passed in pairs dataframe was empty"

            # Skip vault data for now
            # as it is not present
            # TODO: Add later when centralised vault data is available
            loadable_pairs = pairs[pairs["dex_type"] != ExchangeType.erc_4626_vault]

            filtered_pairs_df = pairs
            our_pair_ids = loadable_pairs["pair_id"]
            exchange_ids = loadable_pairs["exchange_id"]
            our_exchanges = {exchange_universe.get_by_id(id) for id in exchange_ids}
            our_exchange_universe = ExchangeUniverse.from_collection(our_exchanges)

        else:
            # Load and filter pairs
            pairs_df = client.fetch_pair_universe().to_pandas()

            # We do not build the pair index here,
            # as we assume we filter out the pairs down a bit,
            # and then recontruct a new pair universe with only few selected pairs with full indexes
            # later. The whole purpose of this here is to
            # go around lack of good look up functions of raw DataFrame pairs data.
            pair_universe = PandasPairUniverse(pairs_df, build_index=False, exchange_universe=exchange_universe)

            # Filter pairs first and then rest by the resolved pairs
            our_pairs = {pair_universe.get_pair_by_human_description(exchange_universe, d) for d in pairs}

            our_pair_ids = {p.pair_id for p in our_pairs}
            exchange_ids = {p.exchange_id for p in our_pairs}
            our_exchanges = {exchange_universe.get_by_id(id) for id in exchange_ids}
            our_exchange_universe = ExchangeUniverse.from_collection(our_exchanges)

            # Eliminate the pairs we are not interested in from the database
            filtered_pairs_df = pairs_df.loc[pairs_df["pair_id"].isin(our_pair_ids)]

        if pair_extra_metadata:
            # Load token tax data
            filtered_pairs_df = filtered_pairs_df.copy()
            filtered_pairs_df = add_base_quote_address_columns(filtered_pairs_df)
            filtered_pairs_df = load_extra_metadata(
                pairs_df=filtered_pairs_df,
                client=client,
            )

        # Autogenerate names by the pair count
        if not name:
            name = f"{len(filtered_pairs_df)} pairs"

        if not candle_progress_bar_desc:
            candle_progress_bar_desc = f"Loading OHLCV data for {name}"

        candles_df = client.fetch_candles_by_pair_ids(
            our_pair_ids,
            time_bucket,
            progress_bar_description=candle_progress_bar_desc,
            start_time=data_load_start_at,
            end_time=end_at,
        )

        candles_pairs = set(candles_df["pair_id"].unique())
        asked_pairs = set(our_pair_ids)

        if len(candles_pairs) != len(asked_pairs):
            logger.info(
                "Data missing warning: We asked OHLCV data for %d trading pairs, but only got for %d pairs. This is usually because time period %s - %s does not have OHLCV data for all asked pairs.",
                len(asked_pairs),
                len(candles_pairs),
                data_load_start_at,
                end_at
            )

        if stop_loss_time_bucket:
            stop_loss_desc = f"Loading stop loss/take profit granular trigger data for {name}"
            stop_loss_candles = client.fetch_candles_by_pair_ids(
                our_pair_ids,
                stop_loss_time_bucket,
                progress_bar_description=stop_loss_desc,
                start_time=data_load_start_at,
                end_time=end_at,
            )
        else:
            stop_loss_candles = None

        if liquidity:
            liquidity_time_bucket = liquidity_time_bucket or time_bucket
            liquidity_progress_bar_desc = f"Loading TVL/liquidity data for {name}"
            liquidity_df = client.fetch_tvl_by_pair_ids(
                our_pair_ids,
                liquidity_time_bucket,
                progress_bar_description=liquidity_progress_bar_desc,
                start_time=data_load_start_at,
                end_time=end_at,
                query_type=liquidity_query_type,
            )
        elif preloaded_tvl_df is not None:
            assert liquidity_time_bucket is not None, "load_partial_data(): liquidity_time_bucket must be given with preloaded_tvl_df argument"
            # Different column naming adapter
            preloaded_tvl_df = preloaded_tvl_df.rename(columns={
                "bucket": "timestamp",
            })
            liquidity_df = preloaded_tvl_df
            liquidity_df = liquidity_df.sort_values(by=["timestamp"])
        else:
            liquidity_time_bucket = None
            liquidity_df = None

        if lending_reserves:
            if isinstance(lending_reserves, LendingReserveUniverse):
                lending_reserve_universe = lending_reserves
            else:
                lending_reserve_universe = client.fetch_lending_reserve_universe()
                lending_reserve_universe = lending_reserve_universe.limit(lending_reserves)

            if not lending_candle_progress_bar_desc:
                lending_candle_progress_bar_desc = f"Downloading lending rate data for {lending_reserve_universe.get_count()} assets, until {end_at.strftime('%Y-%m-%d')}"

            lending_candles_map = client.fetch_lending_candles_for_universe(
                lending_reserve_universe,
                bucket=TimeBucket.h1,
                candle_types=lending_candle_types,
                start_time=data_load_start_at,
                end_time=end_at,
                progress_bar_description=lending_candle_progress_bar_desc,
            )
            lending_candles = LendingCandleUniverse(lending_candles_map, lending_reserve_universe)
        else:
            lending_reserve_universe = None
            lending_candles = None

        logger.info(
            "load_partial_data() complete: time bucket: %s",
            time_bucket,
        )

        # Include vault data for designed vaults if asked
        if vaults:
            logger.info("Including vaults: %s", vaults)
            vault_exchanges, vault_pairs_df = load_multiple_vaults(vaults)
            our_exchange_universe.add(vault_exchanges)
            filtered_pairs_df = pd.concat([filtered_pairs_df, vault_pairs_df])

        if vault_bundled_price_data:
            assert vaults, "Vaults must be given to load bundled price data"
            assert not execution_context.mode.is_live_trading(), "Cannot load bundled price data in live trading"

            assert isinstance(vault_bundled_price_data, (bool, Path)), "vault_bundled_price_data must be bool or Path"
            if isinstance(vault_bundled_price_data, Path):
                vault_prices_bundle_path = vault_bundled_price_data
            else:
                vault_prices_bundle_path = DEFAULT_VAULT_PRICE_BUNDLE

            vault_prices_df = load_vault_price_data(
                vault_pairs_df,
                prices_path=vault_prices_bundle_path,    
            )
            offset = time_bucket.to_frequency()
            freq_string = f"{offset.n}{offset.name.lower()}"
            vault_candle_df, vault_liquidity_df = convert_vault_prices_to_candles(vault_prices_df, freq_string)
            candles_df = pd.concat([candles_df, vault_candle_df])
            if liquidity_df is not None:
                liquidity_df = pd.concat([liquidity_df, vault_liquidity_df])

        # Collect some debug data for the first 5 pairs
        # to diagnose data loding problems
        if execution_context.mode.is_live_trading():
            for pair_id in list(our_pair_ids)[0:5]:
                pair_candles = candles_df[candles_df["pair_id"] == pair_id]
                if len(pair_candles) > 0:
                    first_at = min(pair_candles["timestamp"])
                    last_at = max(pair_candles["timestamp"])
                    duration = last_at - first_at
                    duration_weeks = duration / pd.Timedelta(days=7)
                    logger.info(
                        "Pair id: %d, candle count: %d, first: %s, last: %s, duration: %s, duration (weeks): %s",
                        pair_id,
                        len(pair_candles),
                        first_at,
                        last_at,
                        duration,
                        duration_weeks
                    )
                else:
                    logger.warning("Pair %d no data", pair_id)

        return Dataset(
            time_bucket=time_bucket,
            exchanges=our_exchange_universe,
            pairs=filtered_pairs_df,
            candles=candles_df,
            liquidity=liquidity_df,
            liquidity_time_bucket=liquidity_time_bucket,
            backtest_stop_loss_time_bucket=stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candles,
            lending_reserves=lending_reserve_universe,
            lending_candles=lending_candles,
            start_at=start_at,
            end_at=end_at,
            history_period=required_history_period,
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
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
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

    :param start_time:
        Timestamp when to start loding data (inclusive)

    :param end_time:
        Timestamp when to end loding data (inclusive)
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

        assert len(our_pairs) > 0, f"Pair data not found chain: {chain_id}, exchange: {exchange_slug}, tickers: {pair_tickers}, pair dataset len: {len(pairs_df):,}"

        assert len(our_pairs) == len(pair_tickers), f"Pair resolution failed. Wanted to have {len(pair_tickers)} pairs, but after pair id resolution ended up with {len(our_pairs)} pairs"

        our_pair_ids = set(our_pairs["pair_id"])

        if len(our_pair_ids) > 1:
            desc = f"Loading OHLCV data for {exchange_slug}"
        else:
            pair = pair_tickers[0]
            desc = f"Loading OHLCV data for {pair[0]}-{pair[1]}"

        if required_history_period is not None:
            assert start_time is None, "You cannot give both start_time and required_history_period"
            start_time = datetime.datetime.utcnow() - required_history_period

        candles = client.fetch_candles_by_pair_ids(
            our_pair_ids,
            time_bucket,
            progress_bar_description=desc,
            start_time=start_time,
            end_time=end_time,
        )

        stop_loss_candles = None
        if stop_loss_time_bucket:
            if execution_context.live_trading:
                logger.info(f"Loading granular price data for stop loss/take profit skipped in live trading as live price events from the JSON-RPC endpoint are used")
            else:
                stop_loss_desc = f"Loading granular price data for stop loss/take profit for {exchange_slug}"
                stop_loss_candles = client.fetch_candles_by_pair_ids(
                    our_pair_ids,
                    stop_loss_time_bucket,
                    progress_bar_description=stop_loss_desc,
                    start_time=start_time,
                    end_time=end_time,
                )

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
            start_at=start_time,
            end_at=end_time,
        )


def load_trading_and_lending_data(
    client: BaseClient,
    execution_context: ExecutionContext,
    chain_id: ChainId,
    time_bucket: TimeBucket = TimeBucket.d1,
    universe_options: UniverseOptions = default_universe_options,
    *,
    exchange_slugs: Set[str] | str | None = None,
    lending_protocol: LendingProtocolType | None = None,
    liquidity: bool = False,
    stop_loss_time_bucket: TimeBucket | None = None,
    asset_ids: Set[TokenSymbol] | None = None,
    reserve_assets: Set[TokenSymbol | NonChecksummedAddress] = frozenset({"USDC"}),
    name: str | None = None,
    volatile_only=False,
    trading_fee: Percent | None = None,
    any_quote=False,
):
    """Load trading and lending market for a single chain for all long/short pairs.

    - A shortcut method for constructing trading universe for multipair long/short strategy

    - Gets all supported lending pairs on a chain

    - Discards trading pairs that do not have a matching lending reserve
      with a quote token ``reserve_assset_symbol``

    - Will log output regarding the universe construction for diagnostics

    More information

    - For parameter documentation see :py:func:`load_partial_data`.

    - See also :py:meth:`TradingStrategyUniverse.create_from_dataset`

    Example for historical data:

    .. code-block:: python

        start_at = datetime.datetime(2023, 9, 1)
        end_at = datetime.datetime(2023, 10, 1)

        # Load all trading and lending data on Polygon
        # for all lending markets on a relevant time period
        dataset = load_trading_and_lending_data(
            client,
            execution_context=unit_test_execution_context,
            universe_options=UniverseOptions(start_at=start_at, end_at=end_at),
            chain_id=ChainId.polygon,
            exchange_slug="uniswap-v3",
        )

        strategy_universe = TradingStrategyUniverse.create_from_dataset(dataset)
        data_universe = strategy_universe.data_universe

        # Check one loaded reserve metadata
        usdc_reserve = data_universe.lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "USDC")

        # Check the historical rates
        lending_candles = data_universe.lending_candles.variable_borrow_apr
        rates = lending_candles.get_rates_by_reserve(usdc_reserve)

        assert rates["open"][pd.Timestamp("2023-09-01")] == pytest.approx(3.222019)
        assert rates["open"][pd.Timestamp("2023-10-01")] == pytest.approx(3.446714)

    Example for current data:

    .. code-block: python

        # Load all trading and lending data on Polygon
        # for all lending markets on a relevant time period
        dataset = load_trading_and_lending_data(
            client,
            execution_context=unit_test_execution_context,
            universe_options=UniverseOptions(history_period=datetime.timedelta(days=7)),
            chain_id=ChainId.polygon,
            exchange_slug="uniswap-v3",
        )

        assert dataset.history_period == datetime.timedelta(days=7)

        strategy_universe = TradingStrategyUniverse.create_from_dataset(dataset)
        data_universe = strategy_universe.data_universe

        # Check one loaded reserve metadata
        usdc_reserve = data_universe.lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "USDC")

        # Check the historical rates
        lending_candles = data_universe.lending_candles.variable_borrow_apr
        rates = lending_candles.get_rates_by_reserve(usdc_reserve)

        trading_pair = (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
        pair = data_universe.pairs.get_pair_by_human_description(trading_pair)
        price_feed = data_universe.candles.get_candles_by_pair(pair.pair_id)

        two_days_ago = pd.Timestamp(datetime.datetime.utcnow() - datetime.timedelta(days=2)).floor("D")
        assert rates["open"][two_days_ago] > 0
        assert rates["open"][two_days_ago] < 10  # Erdogan warnings
        assert price_feed["open"][two_days_ago] > 0
        assert price_feed["open"][two_days_ago] < 10_000  # To the moon warning

    :param asset_ids:
        Load only these lending reserves.

        If not given load all lending reserves available on a chain.

    :param trading_fee:
        Loan only trading pairs on a specific fee tier.

        For example set to ``0.0005`` to load only 5 BPS Uniswap pairs.

    :param reserve_assets:
        In which currency, the trading pairs must be quoted for the lending pool.

        The reserve asset data is read from the lending reserve universe.

        This will affect the shape of the trading universe.

        For trading, we need to have at least one trading pair with this quote token.
        The best fee is always picked.

    :param volatile_only:
        If set to False, ignore stablecoin-stablecoin trading pairs.

        TODO: Does not work correctly at the moment.

    :param any_quote:
        Include ETH, MATIC, etc. quoted trading pairs and three-legged trades.

    """

    assert isinstance(client, Client)
    assert isinstance(time_bucket, TimeBucket)
    assert isinstance(execution_context, ExecutionContext)
    assert isinstance(chain_id, ChainId)

    assert len(reserve_assets) == 1, f"Currently only one reserve asset is supported, got {reserve_assets}"
    (reserve_asset_id,) = reserve_assets

    if exchange_slugs is not None:
        if type(exchange_slugs) == str:
            exchange_slugs = {exchange_slugs}

        assert isinstance(exchange_slugs, set)

    lending_reserves = client.fetch_lending_reserve_universe()
    lending_reserves = lending_reserves.limit_to_chain(chain_id)

    if lending_protocol:
        lending_reserves = lending_reserves.limit_to_protocol(lending_protocol)

    if asset_ids is None:
        asset_ids = set()

    all_assets = asset_ids | {reserve_asset_id}

    if asset_ids:
        lending_reserves = lending_reserves.limit_to_assets(all_assets)

    assert lending_reserves.get_count() > 0, f"No lending reserves found for {asset_ids}"

    # Use addrress based lookups for certainty
    if reserve_asset_id.startswith("0x"):
        reserve_asset = lending_reserves.get_by_chain_and_address(
            chain_id,
            reserve_asset_id
        )
    else:
        reserve_asset = lending_reserves.get_by_chain_and_symbol(
            chain_id,
            reserve_asset_id
        )

    assert reserve_asset, f"Reserve asset not in the lending reserve universe: {reserve_asset_id}"

    pairs_df = client.fetch_pair_universe().to_pandas()

    pairs_df = filter_for_chain(pairs_df, chain_id)
    pairs_df = filter_for_stablecoins(pairs_df, StablecoinFilteringMode.only_volatile_pairs)
    pairs_df = filter_for_base_tokens(pairs_df, lending_reserves.get_asset_addresses())

    if not any_quote:
        pairs_df = filter_for_quote_tokens(pairs_df, {reserve_asset.asset_address})

    if trading_fee:
        pairs_df = filter_for_trading_fee(pairs_df, trading_fee)

    if exchange_slugs:
        pairs_df = filter_for_exchange(pairs_df, exchange_slugs)

    if not name:
        symbols = pairs_df["base_token_symbol"].unique()
        name = "Trading and lending universe for " + ", ".join(symbols)

    logger.info(
        "Setting up trading and lending universe on %s using %s as reserve asset, total %d pairs, range is %s",
        chain_id.get_name(),
        reserve_asset_id,
        len(pairs_df),
        universe_options.get_range_description(),
        )

    assert len(pairs_df) > 0, f"load_trading_and_lending_data(): No trading pairs left after loading and filtering given inputs. Check that you are passing a good confirmation in create_universe()."

    # We do not build the pair index here,
    # as we assume we filter out the pairs down a bit,
    # and then recontruct a new pair universe with only few selected pairs with full indexes
    # later. The whole purpose of this here is to
    # go around lack of good look up functions of raw DataFrame pairs data.
    dataset = load_partial_data(
        client=client,
        execution_context=execution_context,
        time_bucket=time_bucket,
        pairs=pairs_df,
        universe_options=universe_options,
        liquidity=liquidity,
        stop_loss_time_bucket=stop_loss_time_bucket,
        lending_candle_types=(LendingCandleType.supply_apr, LendingCandleType.variable_borrow_apr,),
        lending_reserves=lending_reserves,
        name=name,
        candle_progress_bar_desc=f"Downloading OHLCV data for {len(pairs_df)} trading pairs",
        lending_candle_progress_bar_desc=f"Downloading interest rate data for {lending_reserves.get_count()} assets",
        )

    return dataset

