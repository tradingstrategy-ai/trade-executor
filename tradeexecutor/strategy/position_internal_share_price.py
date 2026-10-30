"""Position internal share price state management functions.

This module provides functions to create and update share price state
incrementally on each trade execution.

See :py:class:`tradeexecutor.state.position_internal_share_price.PositionInternalSharePriceState`.
"""
import dataclasses
import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.position_internal_share_price import PositionInternalSharePriceState
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import Percent, USDollarAmount

if TYPE_CHECKING:
    # TradingPosition calls this module, so it cannot be imported at runtime.
    from tradeexecutor.state.position import TradingPosition


def create_share_price_state(
    trade: TradeExecution,
    initial_share_price: float = 1.0,
) -> PositionInternalSharePriceState:
    """Create initial share price state from first trade.

    Called when a position's first trade is executed to initialise
    the share price tracking state.

    :param trade:
        The first executed trade in the position.

    :param initial_share_price:
        Starting share price, typically 1.0.

    :return:
        New PositionInternalSharePriceState with initial values.
    """
    trade_value = trade.get_value()
    shares_minted = trade_value / initial_share_price

    return PositionInternalSharePriceState(
        current_share_price=initial_share_price,
        total_supply=shares_minted,
        cumulative_quantity=float(trade.executed_quantity),
        total_invested=trade_value,
        peak_total_supply=shares_minted,
        initial_share_price=initial_share_price,
        last_updated_at=trade.executed_at,
        total_withdrawn=0.0,
    )


def _add_withdrawn(total_withdrawn: float | None, amount: float) -> float | None:
    """Add an outflow to a withdrawn total, keeping an unknown total unknown."""
    return None if total_withdrawn is None else total_withdrawn + amount


def create_share_price_state_for_exchange_account(
    total_value: float,
    valued_at: datetime.datetime,
    initial_share_price: float = 1.0,
) -> PositionInternalSharePriceState:
    """Create share price state for an exchange account position.

    Exchange account positions use placeholder trades (typically $0 or $1)
    that don't represent real capital. The first successful valuation from
    the exchange API establishes the actual initial capital.

    :param total_value:
        The total account value from the exchange API (the real initial capital).

    :param valued_at:
        Timestamp of the valuation.

    :param initial_share_price:
        Starting share price, typically 1.0.

    :return:
        New PositionInternalSharePriceState treating total_value as initial capital.
    """
    shares = total_value / initial_share_price

    return PositionInternalSharePriceState(
        current_share_price=initial_share_price,
        total_supply=shares,
        cumulative_quantity=total_value,
        total_invested=total_value,
        peak_total_supply=shares,
        initial_share_price=initial_share_price,
        last_updated_at=valued_at,
        total_withdrawn=0.0,
    )


def update_share_price_state(
    state: PositionInternalSharePriceState,
    trade: TradeExecution,
    remaining_quantity: Decimal | None = None,
) -> PositionInternalSharePriceState:
    """Update share price state with a new trade.

    For buys: mint shares at current share price.
    For sells: update share price based on sell price, then burn proportional shares.

    The state tracks the position value as a float while the position books
    Decimal quantities, so the two can differ in the last bits: withdrawing
    the whole 1.7 USDC of a position tracked at 1.7000000000000002 would
    leave 2e-16 shares, and a zero value over that dust reads as a share
    price of zero. When the caller passes the position's Decimal quantity
    after the trade and it is zero, the sell burns every share.

    :param state:
        Current share price state to update.

    :param trade:
        The newly executed trade.

    :param remaining_quantity:
        The position's quantity after the trade, from its Decimal books, if
        the caller has it. Zero marks a full exit.

    :return:
        New PositionInternalSharePriceState with updated values.
    """
    delta = float(trade.executed_quantity or 0)
    trade_value = trade.get_value()

    if delta > 0:
        if state.current_share_price <= 0:
            # The position was valued at zero while shares remained, a total
            # loss such as a liquidated exchange account. No price can be
            # minted at, so the worthless shares are burned and a new series
            # starts at the initial price. The return then covers the period
            # since the loss; the invested and withdrawn totals are kept, so
            # the USD profit still includes it.
            new_total_supply = trade_value / state.initial_share_price
            return PositionInternalSharePriceState(
                current_share_price=state.initial_share_price,
                total_supply=new_total_supply,
                cumulative_quantity=state.cumulative_quantity + delta,
                total_invested=state.total_invested + trade_value,
                peak_total_supply=max(state.peak_total_supply, new_total_supply),
                initial_share_price=state.initial_share_price,
                last_updated_at=trade.executed_at,
                total_withdrawn=state.total_withdrawn,
            )

        # Buy: mint shares at current share price
        shares_to_mint = trade_value / state.current_share_price
        new_total_supply = state.total_supply + shares_to_mint

        return PositionInternalSharePriceState(
            current_share_price=state.current_share_price,  # Unchanged on buy
            total_supply=new_total_supply,
            cumulative_quantity=state.cumulative_quantity + delta,
            total_invested=state.total_invested + trade_value,
            peak_total_supply=max(state.peak_total_supply, new_total_supply),
            initial_share_price=state.initial_share_price,
            last_updated_at=trade.executed_at,
            total_withdrawn=state.total_withdrawn,
        )

    elif delta < 0:
        # Sell: update share price based on sell value, then burn proportional shares
        sell_quantity = abs(delta)

        # The reserve received, which for an exchange-account withdrawal can
        # be less than the account debit.
        total_withdrawn = _add_withdrawn(state.total_withdrawn, abs(float(trade.executed_reserve or 0)))
        if state.cumulative_quantity <= 0:
            return dataclasses.replace(state, total_withdrawn=total_withdrawn, last_updated_at=trade.executed_at)

        proportion_sold = sell_quantity / state.cumulative_quantity
        full_exit = remaining_quantity is not None and remaining_quantity == 0

        # Update share price based on current value at sell price
        total_assets_at_sell = state.cumulative_quantity * trade.executed_price
        new_share_price = (
            total_assets_at_sell / state.total_supply
            if state.total_supply > 0
            else state.current_share_price
        )

        # Burn proportional shares
        shares_to_burn = state.total_supply * proportion_sold
        new_total_supply = 0.0 if full_exit else max(0.0, state.total_supply - shares_to_burn)

        return PositionInternalSharePriceState(
            current_share_price=new_share_price,
            total_supply=new_total_supply,
            cumulative_quantity=0.0 if full_exit else state.cumulative_quantity - sell_quantity,
            total_invested=state.total_invested,
            peak_total_supply=state.peak_total_supply,
            initial_share_price=state.initial_share_price,
            last_updated_at=trade.executed_at,
            total_withdrawn=total_withdrawn,
        )

    # No change for zero-quantity trades (e.g., failed/repaired)
    return state


def apply_capital_outflow(
    state: PositionInternalSharePriceState,
    amount: float,
    at: datetime.datetime,
    remaining_quantity: Decimal | None = None,
) -> PositionInternalSharePriceState:
    """Remove principal from a position without treating it as a loss.

    Used when capital leaves an exchange-account position for another
    account of the same strategy, for example Lighter master account funding
    a sub-account. The share price is marked to the current assets, then
    shares are burned in proportion to the amount removed, exactly like the
    sell branch of :py:func:`update_share_price_state` at a price of one.

    :param state:
        Current share price state.
    :param amount:
        Principal removed, in USD. Must be positive.
    :param at:
        Timestamp of the outflow.
    :param remaining_quantity:
        The position's quantity after the outflow, from its Decimal books, if
        the caller has it. Zero marks a full exit, which burns every share
        rather than leaving float residue, as in :py:func:`update_share_price_state`.
    :return:
        New share price state.
    """
    assert amount > 0, f"Capital outflow must be positive, got {amount}"
    total_withdrawn = _add_withdrawn(state.total_withdrawn, amount)
    if state.cumulative_quantity <= 0:
        return dataclasses.replace(state, total_withdrawn=total_withdrawn, last_updated_at=at)
    proportion = min(amount / state.cumulative_quantity, 1.0)
    full_exit = remaining_quantity is not None and remaining_quantity == 0
    new_share_price = state.cumulative_quantity / state.total_supply if state.total_supply > 0 else state.current_share_price
    new_total_supply = 0.0 if full_exit else max(0.0, state.total_supply - state.total_supply * proportion)
    return PositionInternalSharePriceState(
        current_share_price=new_share_price,
        total_supply=new_total_supply,
        cumulative_quantity=0.0 if full_exit else state.cumulative_quantity - amount,
        total_invested=state.total_invested,
        peak_total_supply=state.peak_total_supply,
        initial_share_price=state.initial_share_price,
        last_updated_at=at,
        total_withdrawn=total_withdrawn,
    )


def update_share_price_state_for_balance_update(
    state: PositionInternalSharePriceState,
    balance_update: BalanceUpdate,
) -> PositionInternalSharePriceState:
    """Update share price state from a balance update event.

    For exchange account positions, value changes arrive as balance updates
    rather than trades. The quantity diff changes total_assets while
    total_supply stays constant, naturally adjusting the share price.

    :param state:
        Current share price state to update.

    :param balance_update:
        The balance update event (from exchange API revaluation).

    :return:
        New PositionInternalSharePriceState with updated share price.
    """
    diff = float(balance_update.quantity)
    new_cumulative = state.cumulative_quantity + diff
    if balance_update.old_balance is not None and Decimal(balance_update.old_balance) + Decimal(balance_update.quantity) == 0:
        # The Decimal books are at zero, such as after a total loss, while the
        # float value can keep residue. Value the position at exactly zero,
        # so the share price is zero and the next deposit starts a new series.
        new_cumulative = 0.0

    # total_supply is unchanged — no shares minted or burned
    # share price moves as total_assets changes
    new_share_price = (
        new_cumulative / state.total_supply
        if state.total_supply > 0
        else state.current_share_price
    )

    return PositionInternalSharePriceState(
        current_share_price=new_share_price,
        total_supply=state.total_supply,
        cumulative_quantity=new_cumulative,
        total_invested=state.total_invested,
        peak_total_supply=state.peak_total_supply,
        initial_share_price=state.initial_share_price,
        last_updated_at=balance_update.block_mined_at,
        total_withdrawn=state.total_withdrawn,
    )


def calculate_flow_neutral_profit_usd(position: "TradingPosition") -> USDollarAmount | None:
    """Profit of an exchange-account position, independent of capital moved in and out.

    Deposits, withdrawals and transfers between the strategy's own exchange
    accounts are capital flows, not profit. The internal share price already
    gives a flow-neutral return; this is the matching dollar figure, from the
    cash-flow identity::

        profit = current value + capital withdrawn - capital invested

    ``total_invested`` and ``total_withdrawn`` come from the position's
    :py:class:`PositionInternalSharePriceState`, which counts every flow as it
    is booked, including transfers to another account that leave as balance
    updates rather than trades. Fees and gains or losses change the value, so
    they are profit.

    :param position:
        Position to measure.
    :return:
        Profit in dollars, or ``None`` when this is not an exchange-account
        position or the withdrawn total is unknown: a state saved before it
        was tracked, or rebuilt from history.
    """
    if not position.is_exchange_account():
        return None
    state = position.share_price_state
    if state is None or state.total_withdrawn is None:
        return None
    value = float(position.get_value() or 0.0) if position.is_open() else 0.0
    return value + state.total_withdrawn - state.total_invested


def calculate_closed_flow_neutral_return(position: "TradingPosition") -> Percent | None:
    """Money-weighted return of a closed exchange-account position, or ``None``.

    A closed position's return cannot be its share price: closing writes off a
    residual as a balance update, which takes the value of the remaining
    internal shares, and so the share price, to zero. The flow-neutral profit
    over the capital invested gives the return instead, as for closed
    Hypercore vault positions.

    :param position:
        Position to measure.
    :return:
        Return as a fraction, or ``None`` if the position is open or its
        flow-neutral profit is unknown.
    """
    if not position.is_closed():
        return None
    profit = calculate_flow_neutral_profit_usd(position)
    if profit is None:
        return None
    invested = position.share_price_state.total_invested
    return profit / invested if invested > 0 else 0.0


def migrate_share_price_state(position: "TradingPosition") -> None:
    """Populate share_price_state for existing positions by replaying events.

    For positions that were created before incremental share price tracking
    was added, this function rebuilds the state by replaying all successful
    trades.

    For exchange account positions, trades are skipped (they are placeholders).
    The first balance update establishes the initial capital, and subsequent
    balance updates adjust the share price.

    :param position:
        Position to migrate. Modified in place.
    """
    if position.share_price_state is not None:
        return  # Already has state

    if not (position.is_spot() or position.is_vault() or position.is_exchange_account()):
        return  # Not applicable

    if position.is_exchange_account():
        total_bought = position.get_total_bought_usd()

        sorted_bus = sorted(
            position.balance_updates.values(),
            key=lambda b: b.balance_update_id,
        )

        if total_bought > 1:
            # Real capital allocation — the trade amount is the true initial
            # investment. Replay all BUs as value changes on top of it.
            state = None
            for trade in position.trades.values():
                if trade.is_success():
                    if state is None:
                        state = create_share_price_state(trade)
                    else:
                        state = update_share_price_state(state, trade)

            if state is not None:
                for bu in sorted_bus:
                    state = update_share_price_state_for_balance_update(state, bu)

            position.share_price_state = state
        else:
            # Placeholder trade ($0 or $1) — skip trades, use the first BU
            # that brings the running balance above zero as initial capital.
            if not sorted_bus:
                return  # No balance updates yet, nothing to migrate

            state = None
            for bu in sorted_bus:
                if state is None:
                    running = float(bu.old_balance) + float(bu.quantity)
                    if running > 0:
                        state = create_share_price_state_for_exchange_account(
                            running, bu.block_mined_at,
                        )
                else:
                    state = update_share_price_state_for_balance_update(state, bu)

            position.share_price_state = state
    else:
        state = None
        for trade in position.trades.values():
            if trade.is_success():
                if state is None:
                    state = create_share_price_state(trade)
                else:
                    state = update_share_price_state(state, trade)

        position.share_price_state = state

    # A rebuilt state can miss capital that left as balance updates, such as
    # an exchange-account funding principal replayed as performance, and can
    # count a placeholder trade as invested. Keep its withdrawn total unknown
    # so it never reports a flow-neutral USD profit.
    if position.share_price_state is not None:
        position.share_price_state = dataclasses.replace(position.share_price_state, total_withdrawn=None)


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

        if not (position.is_spot() or position.is_vault() or position.is_exchange_account()):
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
