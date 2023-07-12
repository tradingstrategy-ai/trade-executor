"""Asset management helpers.

Figure how to map different tokens related to their trading positions.
"""

from typing import List

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair


def get_relevant_assets(
        universe: TradingStrategyUniverse,
        state: State,
) -> List[AssetIdentifier]:
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

    assets = []

    for asset in universe.reserve_assets:
        assets.append(asset)

    if universe.is_open_ended_universe():
        for p in state.portfolio.get_all_positions():
            assets.append(p.pair.base)
    else:
        for p in universe.universe.pairs.iterate_pairs():
            pair = translate_trading_pair(p)
            assets.append(pair.base)

    return assets


def map_onchain_asset_to_position(
        asset: AssetIdentifier,
        state: State,
) -> TradingPosition | ReservePosition | None:
    """Map an on-chain found asset to a trading position.

    - Any reserve currency deposits go to the reserve

    - Any trading position assets go to their respective open trading position

    - If there are trading position assets and no position is open,
      then panic
    """

    for p in state.portfolio.get_all_positions():
        if asset == p.pair.base:
            return p

    r: ReservePosition
    for r in state.portfolio.reserves.values():
        if asset == r.asset:
            return r

    return None
