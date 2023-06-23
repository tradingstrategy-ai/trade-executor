"""Synchrone deposits/withdrawals of the portfolio.

Syncs the external portfolio changes from a (blockchain) source.
See ethereum/hotwallet_sync.py for details.
"""

import datetime
from decimal import Decimal
from typing import List

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent, logger
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef


class DummyWalletSyncer:
    """Simulate a wallet events with a fixed balance set in the beginning."""

    def __init__(self, initial_deposit_amount: Decimal = Decimal(0)):
        assert isinstance(initial_deposit_amount, Decimal)
        self.initial_deposit_amount = initial_deposit_amount
        self.initial_deposit_processed_at = None

    def __call__(self, portfolio: Portfolio, ts: datetime.datetime, supported_reserves: List[AssetIdentifier]) -> List[ReserveUpdateEvent]:
        """Process the backtest initial deposit.

        The backtest wallet is credited once at the start.
        """

        if not self.initial_deposit_processed_at:
            self.initial_deposit_processed_at = ts

            assert len(supported_reserves) == 1

            reserve_token = supported_reserves[0]

            # Generate a deposit event
            evt = ReserveUpdateEvent(
                asset=reserve_token,
                past_balance=Decimal(0),
                new_balance=self.initial_deposit_amount,
                updated_at=ts
            )

            # Update state
            apply_sync_events(portfolio, [evt])

            return [evt]
        else:
            return []


def apply_sync_events(state: State, new_reserves: List[ReserveUpdateEvent], default_price=1.0) -> List[BalanceUpdate]:
    """Apply deposit and withdraws on reserves in the portfolio.

    - Updates :yp:class:`ReservePosition` instance to reflect the latest available balance

    - Generates balance update events needed to calculate inflows/outflows

    - Marks the last treasury updated at

    TODO: This needs to be refactored as is partially the old treasury sync code.

    :param default_price: Set the reserve currency price for new reserves.
    """

    assert isinstance(state, State)

    portfolio = state.portfolio
    treasury_sync = state.sync.treasury
    balance_update_events = []

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
        res_pos.balance_updates[bu.balance_update_id] = bu
        ref = BalanceEventRef(
            balance_event_id=bu.balance_update_id,
            strategy_cycle_included_at=bu.strategy_cycle_included_at,
            cause=bu.cause,
            position_type=bu.position_type,
            position_id=bu.position_id,
            usd_value=bu.usd_value,
        )

        balance_update_events.append(bu)
        treasury_sync.balance_update_refs.append(ref)

    treasury_sync.last_updated_at = datetime.datetime.utcnow()
    return balance_update_events
