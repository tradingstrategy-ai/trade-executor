"""Logic for managing reserve update events.

- Sync between chain, reserve position and portfolio inflow/outflow events
"""

import datetime
from typing import List

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent, logger
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef


def apply_sync_events(
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
