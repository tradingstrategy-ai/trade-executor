"""Logic for managing reserve update events.

- Sync between chain, reserve position and portfolio inflow/outflow events
"""

import datetime
from typing import List, Iterable

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent, logger
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.repair import close_position_with_empty_trade
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.strategy.account_correction import calculate_account_corrections
from tradeexecutor.strategy.asset import build_expected_asset_map, AssetToPositionsMapping
from tradeexecutor.strategy.sync_model import OnChainBalance
from tradingstrategy.pair import PandasPairUniverse


def apply_reserve_update_events(
    state: State,
    reserve_update_events: List[ReserveUpdateEvent],
    default_price=1.0,
) -> List[BalanceUpdate]:
    """Apply deposit and withdraws on reserves in the portfolio.

    - Updates :py:class:`ReservePosition` instance to reflect the latest available balance

    - Generates balance update events needed to calculate inflows/outflows

    - Marks the last treasury updated at

    TODO: This needs to be refactored as is partially the old treasury sync code.

    :param default_price: Set the reserve currency price for new reserves.
    """

    assert isinstance(state, State)

    portfolio = state.portfolio
    treasury_sync = state.sync.treasury
    balance_update_events = []

    logger.info(
        "Converting %d reserve update events to balance update events",
        len(reserve_update_events),
    )

    for evt in reserve_update_events:

        res_pos = portfolio.reserves.get(evt.asset.get_identifier())
        if res_pos is not None:
            # Update existing
            res_pos.quantity = evt.new_balance
            res_pos.last_sync_at = evt.updated_at
            logger.info("Portfolio reserve synced. Asset: %s", evt.asset)
        else:
            # Initialise new reserve position
            res_pos = ReservePosition(
                asset=evt.asset,
                quantity=evt.new_balance,
                last_sync_at=evt.updated_at,
                reserve_token_price=default_price,
                last_pricing_at=evt.updated_at,
                initial_deposit_reserve_token_price=default_price,
                initial_deposit=evt.new_balance,
            )
            portfolio.reserves[res_pos.get_identifier()] = res_pos
            logger.info(
                "Portfolio initial ReservePosition created. Asset: %s, identifier %s",
                evt.asset,
                evt.asset.get_identifier(),
            )

        # Generate related balance events
        event_id = portfolio.next_balance_update_id
        portfolio.next_balance_update_id += 1

        asset = evt.asset
        quantity = evt.change
        cause = BalanceUpdateCause.deposit if quantity > 0 else BalanceUpdateCause.redemption

        # TODO: Assume stablecoins are 1:1 with dollar
        usd_value = float(quantity)

        bu = BalanceUpdate(
            balance_update_id=event_id,
            position_type=BalanceUpdatePositionType.reserve,
            cause=cause,
            asset=asset,
            block_mined_at=evt.mined_at,  # There is
            strategy_cycle_included_at=evt.updated_at,
            chain_id=asset.chain_id,
            old_balance=evt.past_balance,
            usd_value=usd_value,
            quantity=quantity,
            position_id=None,
        )


        # Store balance update event on the reserve position
        res_pos.balance_updates[bu.balance_update_id] = bu

        ref = BalanceEventRef.from_balance_update_event(bu)
        balance_update_events.append(bu)
        treasury_sync.balance_update_refs.append(ref)

    treasury_sync.last_updated_at = datetime.datetime.utcnow()
    return balance_update_events


def perform_balance_update(
    timestamp: datetime.datetime,
    state: State,
    position: TradingPosition,
    ab: OnChainBalance,
    mapping: AssetToPositionsMapping,
):
    """

    - Very similar to `calculate_account_corrections`
    """

    logger.info("Applying balance update fo%s  position %s", ab, position)

    assert isinstance(position, TradingPosition)
    assert position.is_spot()

    strategy_timestamp = timestamp
    event_timestamp = ab.timestamp
    actual_amount = ab.amount
    expected_amount = mapping.quantity
    diff = actual_amount - expected_amount

    portfolio = state.portfolio
    asset = ab.asset
    block_number = ab.block_number

    event_id = portfolio.next_balance_update_id
    portfolio.next_balance_update_id += 1

    usd_value = position.calculate_quantity_usd_value(diff)

    evt = BalanceUpdate(
        balance_update_id=event_id,
        position_type=BalanceUpdatePositionType.open_position,
        cause=BalanceUpdateCause.correction,
        asset=asset,
        block_mined_at=event_timestamp,
        strategy_cycle_included_at=strategy_timestamp,
        chain_id=asset.chain_id,
        old_balance=actual_amount,
        usd_value=usd_value,
        quantity=diff,
        owner_address=None,
        tx_hash=None,
        log_index=None,
        position_id=position.position_id,
        block_number=block_number,
        notes=f"Flow event at block {block_number:,}",
    )

    ref = BalanceEventRef.from_balance_update_event(evt)
    accounting = state.sync.accounting
    accounting.balance_update_refs.append(ref)

    position.balance_updates[evt.balance_update_id] = evt

    # The position has gone to zero
    if position.can_be_closed():
        # In a lot of places we assume that a position with 1 trade cannot be closed
        # Make a 0-sized trade so that we know the position is closed
        t = close_position_with_empty_trade(portfolio, position)
        logger.info("Position %s closed with a trade %s", position, t)
        assert position.is_closed()


def apply_balance_update_events(
    timestamp: datetime.datetime,
    state: State,
    asset_balances: Iterable[OnChainBalance],
    asset_to_position: dict[AssetIdentifier, AssetToPositionsMapping],
):
    """Apply generic balance change events.

    - Used for Velvet in-kind deposit/withdrawal

    - Reserve position is handled separately by :py:func:`apply_reserve_update_events`
    """

    block_number = None
    for ab in asset_balances:

        asset = ab.asset
        mapping = asset_to_position[asset]

        if len(mapping.positions) == 0:
            # This asset does not have open our closed positions,
            # but is present in the trading universe
            logger.info("No mapping for asset: %s", ab.asset)
            continue
        elif mapping.is_one_to_one_asset_to_position():
            position = mapping.get_only_position()
            perform_balance_update(
                timestamp,
                state,
                position,
                ab,
                mapping,
            )
        else:
            raise NotImplementedError(f"Has multiple positions per asset: {ab}")

        block_number = ab.block_number

    if block_number:
        accounting = state.sync.accounting
        accounting.last_updated_at = datetime.datetime.utcnow()
        accounting.last_block_scanned = block_number
