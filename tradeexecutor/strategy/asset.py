"""Asset management helpers.

Figure how to map different tokens related to their trading positions.
"""

from typing import List, Collection, Set

from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair



def _is_open_ended_universe(pair_universe: PandasPairUniverse):
    # TODO: Have this properly defined in a strategy module
    return pair_universe.get_count() > 20


def get_relevant_assets(
        pair_universe: PandasPairUniverse,
        reserve_assets: Collection[AssetIdentifier],
        state: State,
) -> Set[AssetIdentifier]:
    """Get list of tokens that are relevant for the straegy.

    We need to know the list of tokens we need to scan for the strategy
    to do the accounting checks.

    A token is relevant if it

    - Can be used in a trading position

    - Can be used as a reserve currency

    For open-ended trading universes we only consider trading pairs that have been traded
    at least once.

    :return:
        A list of tokens of which balances we need to check when doing accounting
    """

    assert isinstance(pair_universe, PandasPairUniverse)
    assert isinstance(state, State)

    assets = set()

    for asset in reserve_assets:
        assets.add(asset)

    if _is_open_ended_universe(pair_universe):
        # For open ended universe we can have thousands of assets
        # so we cannot query them all
        for p in state.portfolio.get_all_positions():
            assets.add(p.pair.base)
    else:
        for p in pair_universe.iterate_pairs():
            pair = translate_trading_pair(p)
            assets.add(pair.base)

    return assets


def map_onchain_asset_to_position(
        asset: AssetIdentifier,
        state: State,
) -> TradingPosition | ReservePosition | None:
    """Map an on-chain found asset to a trading position.

    - Any reserve currency deposits go to the reserve

    - Any trading position assets go to their respective open trading
      and frozen position

    - If there are trading position assets and no position is open,
      then panic

    :param asset:
        On-chain read token we should make

    :param state:
        The current strategy state

    :return:
        The position we think the asset belongs to.

        None if there is no reserve, open or frozen
        positions we know of.
    """

    for p in state.portfolio.open_positions.values():
        if asset == p.pair.base:
            return p

    # Frozen position should be considered open positions,
    # when trying to assign tokens to a position they need to be in
    for p in state.portfolio.frozen_positions.values():
        if asset == p.pair.base:
            return p

    r: ReservePosition
    for r in state.portfolio.reserves.values():
        if asset == r.asset:
            return r

    return None
