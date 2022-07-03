"""Synchrone deposits/withdrawals of the portfolio.

Syncs the external portfolio changes from a (blockchain) source.
See ethereum/hotwallet_sync.py for details.
"""

import datetime
from typing import Callable, List

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent, logger
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.reserve import ReservePosition


SyncMethod = Callable[[Portfolio, datetime.datetime, List[AssetIdentifier]], List[ReserveUpdateEvent]]


def apply_sync_events(portfolio: Portfolio, new_reserves: List[ReserveUpdateEvent], default_price=1.0):
    """Apply deposit and withdraws on reserves in the portfolio.

    :param default_price: Set the reserve currency price for new reserves.
    """

    for evt in new_reserves:

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
            logger.info("Portfolio reserve created. Asset: %s", evt.asset)