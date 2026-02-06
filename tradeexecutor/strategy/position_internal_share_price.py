"""Position internal share price state management functions.

This module provides functions to create and update share price state
incrementally on each trade execution.

See :py:class:`tradeexecutor.state.position_internal_share_price.SharePriceState`.
"""
import datetime

from tradeexecutor.state.position_internal_share_price import SharePriceState
from tradeexecutor.state.trade import TradeExecution


def create_share_price_state(
    trade: TradeExecution,
    initial_share_price: float = 1.0,
) -> SharePriceState:
    """Create initial share price state from first trade.

    Called when a position's first trade is executed to initialise
    the share price tracking state.

    :param trade:
        The first executed trade in the position.

    :param initial_share_price:
        Starting share price, typically 1.0.

    :return:
        New SharePriceState with initial values.
    """
    trade_value = trade.get_value()
    shares_minted = trade_value / initial_share_price

    return SharePriceState(
        current_share_price=initial_share_price,
        total_supply=shares_minted,
        cumulative_quantity=float(trade.executed_quantity),
        total_invested=trade_value,
        peak_total_supply=shares_minted,
        initial_share_price=initial_share_price,
        last_updated_at=trade.executed_at,
    )


def update_share_price_state(
    state: SharePriceState,
    trade: TradeExecution,
) -> SharePriceState:
    """Update share price state with a new trade.

    For buys: mint shares at current share price.
    For sells: update share price based on sell price, then burn proportional shares.

    :param state:
        Current share price state to update.

    :param trade:
        The newly executed trade.

    :return:
        New SharePriceState with updated values.
    """
    delta = float(trade.executed_quantity or 0)
    trade_value = trade.get_value()

    if delta > 0:
        # Buy: mint shares at current share price
        shares_to_mint = trade_value / state.current_share_price
        new_total_supply = state.total_supply + shares_to_mint

        return SharePriceState(
            current_share_price=state.current_share_price,  # Unchanged on buy
            total_supply=new_total_supply,
            cumulative_quantity=state.cumulative_quantity + delta,
            total_invested=state.total_invested + trade_value,
            peak_total_supply=max(state.peak_total_supply, new_total_supply),
            initial_share_price=state.initial_share_price,
            last_updated_at=trade.executed_at,
        )

    elif delta < 0:
        # Sell: update share price based on sell value, then burn proportional shares
        sell_quantity = abs(delta)

        if state.cumulative_quantity > 0:
            proportion_sold = sell_quantity / state.cumulative_quantity

            # Update share price based on current value at sell price
            total_assets_at_sell = state.cumulative_quantity * trade.executed_price
            new_share_price = (
                total_assets_at_sell / state.total_supply
                if state.total_supply > 0
                else state.current_share_price
            )

            # Burn proportional shares
            shares_to_burn = state.total_supply * proportion_sold
            new_total_supply = max(0.0, state.total_supply - shares_to_burn)

            return SharePriceState(
                current_share_price=new_share_price,
                total_supply=new_total_supply,
                cumulative_quantity=state.cumulative_quantity - sell_quantity,
                total_invested=state.total_invested,
                peak_total_supply=state.peak_total_supply,
                initial_share_price=state.initial_share_price,
                last_updated_at=trade.executed_at,
            )

    # No change for zero-quantity trades (e.g., failed/repaired)
    return state


def migrate_share_price_state(position: "TradingPosition") -> None:
    """Populate share_price_state for existing positions by replaying trades.

    For positions that were created before incremental share price tracking
    was added, this function rebuilds the state by replaying all successful
    trades.

    :param position:
        Position to migrate. Modified in place.
    """
    if position.share_price_state is not None:
        return  # Already has state

    if not (position.is_spot() or position.is_vault()):
        return  # Not applicable

    state = None
    for trade in position.trades.values():
        if trade.is_success():
            if state is None:
                state = create_share_price_state(trade)
            else:
                state = update_share_price_state(state, trade)

    position.share_price_state = state


def backfill_share_price_state(
    state: "State",
    store: "JSONFileStore | None" = None,
) -> int:
    """Backfill share_price_state for all positions in the portfolio.

    For positions created before incremental share price tracking was added,
    this function rebuilds the state by replaying all successful trades.

    Should be called after loading state to ensure all positions have
    share price tracking data.

    :param state:
        The trading state containing all positions.

    :param store:
        Optional store to sync state periodically during large migrations.
        If provided, state is saved every 100 positions.

    :return:
        Number of positions that were migrated.
    """
    import logging
    from itertools import chain

    logger = logging.getLogger(__name__)

    portfolio = state.portfolio
    migrated = 0

    # Iterate all positions (open, closed, frozen)
    all_positions = chain(
        portfolio.open_positions.values(),
        portfolio.closed_positions.values(),
        portfolio.frozen_positions.values(),
    )

    for position in all_positions:
        # Skip if already has state or not applicable
        if position.share_price_state is not None:
            continue

        if not (position.is_spot() or position.is_vault()):
            continue

        # Migrate this position
        migrate_share_price_state(position)
        migrated += 1

        # Periodic save for large portfolios
        if store and migrated % 100 == 0:
            logger.info("Migrated %d positions, saving state...", migrated)
            store.sync(state)

    if migrated > 0:
        logger.info("Share price state backfilled for %d positions", migrated)

    return migrated
