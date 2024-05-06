"""Token to trading pair mapping helpers."""
import logging

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import JSONHexAddress, USDollarAmount
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse, filter_for_base_tokens
from tradingstrategy.timebucket import TimeBucket


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TokenTuple:
    """A token you want to trade.

    - Presented as chain id, ERC-20 address tuple
    """
    chain_id: ChainId
    address: JSONHexAddress

    def __eq__(self, other):
        return other.chain_id == self.chain_id and other.address == self.address

    def __hash__(self):
        return hash((self.chain_id, self.address))

    def __post_init__(self):
        # Type check and transform input
        assert isinstance(self.chain_id, ChainId)
        assert type(self.address) == str
        assert self.address.startswith("0x")
        assert self.address == self.address.lower()


def find_best_pairs_for_tokens(
    pair_universe: PandasPairUniverse,
    tokens: Iterable[TokenTuple],
    reserve_token: JSONHexAddress,
    intermediate_token: JSONHexAddress | None = None,
    volume_30d_threshold_today: USDollarAmount = 0,
) -> Iterable[TradingPairIdentifier]:
    """Find the best DEXes and trading pairs to trade tokens.

    - Find the best trading pairs for a list of tokens

    - If the token has both /USDC and /WETH /WMATIC e.g.
      pairs try to pick the best one

    - USDC is always preferred if it has enough volume

    :param tokens:
        A list of tokens

    :param pair_universe:
        The available trading pair universe.

    :param reserve_currency:
        Token symbol which we are trading against.

    :param intermediate_token:
        Allow routing tokens trades through this token.

        If you are trading USDC treasury you want to access
        /WETH nominated pairs, then this is WETH.

    :param volume_30d_threshold_today:
        Drop trading pairs that are too small.

        Based on the latest snapshotted volume.

    :return:
        Iterable of trading pairs that match criteria.
    """

    assert isinstance(pair_universe, PandasPairUniverse), f"Expected PandasPairUniverse, got {type(pair_universe)}"
    df = pair_universe.df

    assert isinstance(df, pd.DataFrame)

    if intermediate_token:
        assert intermediate_token.startswith("0x")
        intermediate_token = intermediate_token.lower()

    assert reserve_token.startswith("0x")
    reserve_token = reserve_token.lower()

    for token in tokens:
        assert isinstance(token, TokenTuple)

        # No self pairs
        if token.address in (intermediate_token, reserve_token):
            continue

        # Take a subset of raw pair DataFrame where we have the current token as a base token
        matching_pair_ids = filter_for_base_tokens(df, {token.address})
        pair_matches = [pair_universe.get_pair_by_id(pair_id) for pair_id in matching_pair_ids["pair_id"]]

        # We have now several trading pairs for the token.
        # Try to find the best one.

        # Get the best volume first
        pair_matches.sort(key=lambda p: p.volume_30d, reverse=True)

        logger.info("Token %s has %d potential pairs", token.address, len(pair_matches))

        for p in pair_matches:
            # First checkc USDC (reserve currency) volume
            if p.quote_token_address == reserve_token:
                if p.volume_30d >= volume_30d_threshold_today:
                    logger.info("Pair %s matches reserve token %s", p, reserve_token)
                    yield translate_trading_pair(p)
                    break
            # Then check for WMATIC/WETH (intermediate token) volume
            elif p.quote_token_address == intermediate_token:
                if p.volume_30d >= volume_30d_threshold_today:
                    logger.info("Pair %s matches intermediate token %s", p, intermediate_token)
                    yield translate_trading_pair(p)
                    break
            else:
                logger.info("Token %s pair %s discarded, no reserve or intermediate match", token.address, p)


def create_trading_universe_for_tokens(
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
    time_bucket: TimeBucket,
    tokens: Iterable[TokenTuple],
    reserve_token: JSONHexAddress,
    intermediate_token: JSONHexAddress | None = None,
    volume_30d_threshold_today: USDollarAmount = 0,
    stop_loss_time_bucket: TimeBucket | None = None,
    name: str | None = None,
) -> TradingStrategyUniverse:
    """Create a trading universe based on a list of ERC-20 tokens addresses only.

    - Takes a full trading universe and a list of ERC-20 addresses input,
      and returns a new trading pair universe with the best match for the tradeable tokens

    - Display TQDM progress bar for the load

    :param client:
        Trading Strategy data client

    :param execution_context:
        Needed to know if backtesting or live trading

    :param universe_options:
        Backtesting date range or historical live trading look back needed.

    :param time_bucket:
        Candle time bucket to use.

        E.g. `TimeBucket.d1`.

    :param stop_loss_time_bucket:
        Backtest stop loss simulation time bucket.

        Optional.

    :param tokens:
        Tokens we want to load

    :param reserve_token:
        The reserve currency of a strategy.

        E.g. USDC on Polygon ``.

    :param intermediate_token:
        Intermediate token which we trade through.

        E.g. WMATIC on Polygon ``.

    :param volume_30d_threshold_today:
        Volume filter threshold.

    :param name:
        Optional name for this trading universe.

        Autogenerated if not given.

    """

    tokens = list(tokens)

    logger.info("Creating trading universe for %d tokens. Preloading all exchange and pair data", len(tokens))
    exchange_universe = client.fetch_exchange_universe()
    pairs_df = client.fetch_pair_universe().to_pandas()
    pair_universe = PandasPairUniverse(
        pairs_df,
        build_index=True,
        exchange_universe=exchange_universe
    )

    # Create a set of trading pairs based on the filtering conditions
    our_pairs = {p for p in find_best_pairs_for_tokens(pair_universe, tokens, reserve_token, intermediate_token, volume_30d_threshold_today)}
    assert len(our_pairs) > 0, "Zero tokens left after filtering"

    pair_ids = [p.internal_id for p in our_pairs]
    filtered_pairs_df = pairs_df.loc[pairs_df.pair_id.isin(pair_ids)]

    chain_ids = {p.chain_id for p in our_pairs}
    assert len(chain_ids) == 1, f"Multiple chain_ids in the source: {chain_ids}"

    chain = ChainId(next(iter(chain_ids)))

    if not name:
        name = f"{len(tokens)} tokens on chain {chain.name}"

    # Reload all data
    logger.info("Loading final trading universe data with candles")
    dataset = load_partial_data(
        client=client,
        execution_context=execution_context,
        time_bucket=time_bucket,
        pairs=filtered_pairs_df,
        universe_options=universe_options,
        liquidity=False,
        stop_loss_time_bucket=stop_loss_time_bucket,
        name=name,
        candle_progress_bar_desc=f"Downloading OHLCV data for {len(filtered_pairs_df)} trading pairs",
        )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(dataset, reserve_asset=reserve_token)
    logger.info("Created strategy universe with %d pairs", strategy_universe.get_pair_count())

    bad_pairs = set()

    for p in our_pairs:
        candles = strategy_universe.data_universe.candles.get_candles_by_pair(p.internal_id)
        if candles is None or len(candles) == 0:
            bad_pairs.add(p)

    if bad_pairs:
        logger.warning(
            f"We have no %s OHCLV candles for the following %d trading pairs.\n"
            f"Double check the data at the TradingStrategy.ai search.\n"
            f"This might be because these tokens are not well established and only saw very limited trading.\n"
            f"Remove tokens with no data from the input address set.",
            time_bucket,
            len(bad_pairs),
        )
        for p in bad_pairs:
            logger.warning("Pair %s, base token %s", p, p.base.address)

    return strategy_universe
