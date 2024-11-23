"""Universe reverse loading.

Load trading universe from candles, not vice versa.

"""
import datetime
import enum
from typing import Set, Tuple

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client

from tradeexecutor.state.state import State
from tradeexecutor.state.types import ZeroExAddress
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.exchange import Exchange, ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


class DataRangeMode(enum.Enum):
    """How to get the candle data range from a trading state."""

    #: Use the first and last executed rade
    trades = "trades"

    #: Use the technical indicator range.
    #:
    #: State updates its technical indicator plots even
    #: even if the strategy is not making any trades.
    indicators = "indicators"


def reverse_trading_universe_from_state(
    state: State,
    client: Client,
    time_bucket: TimeBucket,
    overlook_period: datetime.timedelta = datetime.timedelta(days=7),
    data_range_mode: DataRangeMode = DataRangeMode.trades,
) -> TradingStrategyUniverse:
    """Reverse-engineer trading universe from an existing execution state.

    - Exchanges are filtered down the set that the trading execution state already has

    - Pairs are filtered down the set that the trading execution state already has

    - We may leak extra pairs, because we do not do strict (chain, pair address)
      tuple checks and some pairs have duplicate address across chains

    - Exchanges are not filtered, but contain the whole set of available exchanges

    - No backtest stop loss data is loaded

    - No liquidity data is loaded

    .. note ::

        Trading data granularity, or time bucket, may be different that the strategy
        originally used.

    :param state:
        The trade executor state

    :param client:
        Client used to downlaod the data

    :param time_bucket:
        Granularity of the candle data

    :param overlook_period:
        We load candle data for the duration of trades in the portfolio.

        We add +/- `overlook_period` to the data range.

    :param data_range_mode:
        Is this for visualising the latest technical indicators,
        or old executed trades.

    :return:
        A trading universe containing data for all trading pairs
    """

    assert len(list(state.portfolio.get_all_positions())) > 0, "Portfolio contains no positions"

    chains: Set[int] = set()

    pair_addresses: Set[ZeroExAddress] = set()

    # TODO: Remove pair_id usage here
    pair_ids = set()

    start = datetime.datetime(2099, 1, 1)
    end = datetime.datetime(1970, 1, 1)

    for trade in state.portfolio.get_all_trades():
        pair = trade.pair
        chains.add(pair.chain_id)
        pair_addresses.add(pair.pool_address)
        pair_ids.add(pair.internal_id)

        if data_range_mode == DataRangeMode.trades:
            start = min(trade.started_at or trade.opened_at or trade.executed_at, start)
            
            if trade.executed_at:
                end = max(end, trade.executed_at)

    if data_range_mode == DataRangeMode.indicators:
        # Get the data from the latest technical indicators.
        start, end = state.visualisation.get_timestamp_range()

    exchange_universe = client.fetch_exchange_universe()
    pairs = client.fetch_pair_universe().to_pandas()

    # Filter down pairs to what we have in the existing state
    pairs = pairs.loc[
        pairs["chain_id"].isin(chains)
    ]

    pairs = pairs.loc[
        pairs["address"].isin(pair_addresses)
    ]

    # TODO: Add (chain_id, address) look up method
    candles = client.fetch_candles_by_pair_ids(
        pair_ids,
        time_bucket,
        start_time=start - overlook_period,
        end_time=end + overlook_period,
    )

    candle_universe = GroupedCandleUniverse(candles)

    pair_universe = PandasPairUniverse(pairs)

    # Filter down exchanges
    exchanges: Set[Exchange] = set()
    for p in pair_universe.iterate_pairs():
        exchanges.add(exchange_universe.get_by_id(p.exchange_id))

    universe = Universe(
        chains={ChainId(id) for id in chains},
        time_bucket=time_bucket,
        exchanges=exchanges,
        pairs=pair_universe,
        candles=candle_universe,
    )

    reserve_assets = {reserve_position.asset for reserve_position in state.portfolio.reserves.values()}

    ts_universe = TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=reserve_assets,
    )

    return ts_universe


def create_universe_from_trading_pair_identifiers(
    pairs: list[TradingPairIdentifier],
    exchange_universe: ExchangeUniverse,
) -> PandasPairUniverse:
    """Create pair universe from a list of pair identifiers.

    - Used in tests to set up pair universes when we cannot yet downlaod data from the oracle
    """

    counter = 0
    def allocate_pair_id():
        nonlocal counter
        counter += 1
        return counter

    data = [
        {
            "pair_id": allocate_pair_id(),
            "chain_id": t.chain_id,
            "exchange_id": exchange_universe.get_by_chain_and_factory(ChainId(t.chain_id), t.exchange_address).exchange_id,
            "address": t.pool_address,
            "token0_address": t.base.address,
            "token1_address": t.quote.address,
            "token0_symbol": t.base.token_symbol,
            "token1_symbol": t.quote.token_symbol,
            "base_token_symbol": t.base.token_symbol,
            "quote_token_symbol": t.quote.token_symbol,
            "dex_type": exchange_universe.get_by_chain_and_factory(ChainId(t.chain_id), t.exchange_address).exchange_type.value,
            "token0_decimals": t.base.decimals,
            "token1_decimals": t.quote.decimals,
            "exchange_slug": exchange_universe.get_by_chain_and_factory(ChainId(t.chain_id), t.exchange_address).exchange_slug,
            "exchange_address": t.exchange_address,
            "pair_slug": f"{t.base.token_symbol}-{t.quote.token_symbol}",
        } for t in pairs
    ]
    df = pd.DataFrame(data)
    universe = PandasPairUniverse(df, exchange_universe=exchange_universe)
    return universe


