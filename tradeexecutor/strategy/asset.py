"""Asset management helpers.

Figure how to map different tokens related to their trading positions.
"""
import logging
from dataclasses import field, dataclass
from decimal import Decimal
from typing import List, Collection, Set, Dict, Tuple

from tradeexecutor.state.generic_position import GenericPosition
from tradeexecutor.state.portfolio import Portfolio
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair


logger = logging.getLogger(__name__)


@dataclass
class AssetToPositionsMapping:
    """Tell us which positions hold the asset in a portfolio."""

    #: Token we are checking
    asset: AssetIdentifier

    #: Positions using this token
    positions: Set[GenericPosition] = field(default_factory=set)

    #: Expected amount of tokens we will find on chain
    #:
    #: This is the quantity across all positions.
    #:
    quantity: Decimal = Decimal(0)

    def is_one_to_one_asset_to_position(self) -> bool:
        return len(self.positions) == 1

    def is_for_reserve(self) -> bool:
        return len(self.positions) == 1 and isinstance(self.get_only_position(), ReservePosition)

    def get_only_position(self) -> GenericPosition:
        assert len(self.positions) == 1
        return next(iter(self.positions))

    def get_first_position(self) -> GenericPosition:
        return next(iter(self.positions))


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
            assert pair.is_spot(), f"Can only match spot positions, got {pair}"
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

    - Always check reserve first

    - If multiple positions are sharing the asset e.g. collateral
      return the firs position

    :param asset:
        On-chain read token we should make

    :param state:
        The current strategy state

    :return:
        The position we think the asset belongs to.

        None if there is no reserve, open or frozen
        positions we know of.
    """

    r: ReservePosition
    for r in state.portfolio.reserves.values():
        if asset == r.asset:
            return r

    for p in state.portfolio.get_open_and_frozen_positions():
        if asset == p.pair.base:
            return p
        if asset == p.pair.quote:
            return p

    return None


def get_asset_amounts(p: TradingPosition) -> List[Tuple[AssetIdentifier, Decimal]]:
    """What tokens this position should hold in a wallet."""
    if p.is_spot() or p.is_vault():
        return [(p.pair.base, p.get_quantity())]
    elif p.is_short():
        return [
            (p.pair.base, p.loan.get_borrowed_quantity()),
            (p.pair.quote, p.loan.get_collateral_quantity()),
        ]
    if p.is_credit_supply():
        # Some frozen positions might not have loan
        if not p.loan:
            return []
        return [
            (p.pair.base, p.loan.get_collateral_quantity()),
        ]
    else:
        raise NotImplementedError()


def get_onchain_assets(pair: TradingPairIdentifier) -> List[AssetIdentifier]:
    if pair.is_spot():
        return [pair.base]
    elif pair.is_short():
        return [pair.base, pair.quote]
    else:
        raise NotImplementedError()


def build_expected_asset_map(
    portfolio: Portfolio,
    pair_universe: PandasPairUniverse = None,
    universe_enumaration_threshold=20,
    ignore_reserve=False,
) -> dict[AssetIdentifier, AssetToPositionsMapping]:
    """Get list of tokens that the portfolio should hold.

    - Open and frozen positions have :py:class:`AssetToPositionsMapping` set to the executed balance

    - Closed positions have :py:class:`AssetToPositionsMapping` set to zero balance

    :param portfolio:
        Current portfolio

    :param pair_universe:
        If given, enumerate all pairs here as well.

        We might have balance on an asset we have not traded yet,
        causing accounting incorrectness.

    :param universe_enumaration_threshold:
        Max pairs per universe before we do auto enumation.

        Prevent denial of service on open-ended universes > 100 pairs.

    :param ignore_reserve:
        Do not include reserve asset in the set

    :return:
        Token -> (Amount, positions hold across mappings)
    """

    mappings: Dict[AssetIdentifier, AssetToPositionsMapping] = {}

    if not ignore_reserve:
        r: ReservePosition
        for r in portfolio.reserves.values():
            if r.asset not in mappings:
                mappings[r.asset] = AssetToPositionsMapping(asset=r.asset)

            mappings[r.asset].positions.add(r)
            mappings[r.asset].quantity += r.quantity

    for p in portfolio.get_open_and_frozen_positions():
        for asset, amount in get_asset_amounts(p):
            if asset not in mappings:
                mappings[asset] = AssetToPositionsMapping(asset=asset)
            mappings[asset].positions.add(p)
            mappings[asset].quantity += amount
            if p.is_frozen():
                position_type = "Frozen"
            else:
                position_type = "Open"
            logger.info("%s position #%d has asset %s for %f", position_type, p.position_id, asset.token_symbol, amount)

    # Map closed positions as expected 0 asset amount
    #
    # Closed positions appear in the position list only if there is not existing position,
    # otherwise all closed positions would get enumerated one by one
    #
    closed_positions_lifo = list(portfolio.closed_positions.values())
    closed_positions_lifo.reverse()
    for p in closed_positions_lifo:
        for asset, amount in get_asset_amounts(p):
            if asset not in mappings:
                mappings[asset] = AssetToPositionsMapping(asset=asset)
                mappings[asset].positions.add(p)
                logger.debug("Closed position #%d touched asset %s", p.position_id, asset.token_symbol)

    if pair_universe is not None:
        assert isinstance(pair_universe, PandasPairUniverse)
        if pair_universe.get_count() < universe_enumaration_threshold:
            for dex_pair in pair_universe.iterate_pairs():
                p = translate_trading_pair(dex_pair)
                # Catch some bad pairs
                assert p.base.token_symbol, f"Token symbol missing: {p}"
                for asset in get_onchain_assets(p):
                    if asset not in mappings:
                        mappings[asset] = AssetToPositionsMapping(asset=asset)
                        logger.info("Discovered asset %s in the pair universe", asset.token_symbol)

        else:
            logger.info(
                "Universe is not added to the accounting checks, because it is %d pairs and threshold is %d pairs",
                pair_universe.get_count(),
                universe_enumaration_threshold,
            )

    return mappings
