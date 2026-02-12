"""Helpers for managing exchange account positions in state.

Exchange account positions (Derive, Hyperliquid, etc.) are created directly
on the state object, bypassing PositionManager and the normal execution pipeline.

Trades are immediately spoofed (marked success) so they never reach routing
or execution.
"""

import datetime
from decimal import Decimal

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType


def open_exchange_account_position(
    state: State,
    strategy_cycle_at: datetime.datetime,
    pair: TradingPairIdentifier,
    reserve_currency: AssetIdentifier,
    reserve_amount: Decimal = Decimal(1),
    notes: str | None = None,
) -> list[TradeExecution]:
    """Open an exchange account position by creating a spoofed trade.

    This creates a trade directly on the state and immediately marks it
    as successfully executed. The trade never reaches routing or execution.

    Used in ``decide_trades()`` for exchange account strategies.
    The returned list should NOT be included in the trades returned
    by ``decide_trades()`` (return ``[]`` instead).

    :param state:
        Current strategy state.

    :param strategy_cycle_at:
        The timestamp of the current strategy cycle.

    :param pair:
        The exchange account trading pair identifier.

    :param reserve_currency:
        The reserve currency asset (e.g. USDC).

    :param reserve_amount:
        Nominal reserve amount for the position.
        Defaults to 1 USD as a placeholder.

    :param notes:
        Optional human-readable notes for the trade.

    :return:
        List containing the single spoofed trade.
        Do NOT return these from ``decide_trades()``.
    """
    assert pair.is_exchange_account(), f"Expected exchange account pair, got: {pair}"

    position, trade, created = state.create_trade(
        strategy_cycle_at=strategy_cycle_at,
        pair=pair,
        quantity=None,
        reserve=reserve_amount,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_currency,
        reserve_currency_price=1.0,
        notes=notes or f"Open exchange account position for {pair.get_exchange_account_protocol()}",
        pair_fee=0.0,
        lp_fees_estimated=0,
    )

    # Immediately spoof the trade as successfully executed
    trade.mark_success(
        executed_at=native_datetime_utc_now(),
        executed_price=1.0,
        executed_quantity=reserve_amount,
        executed_reserve=reserve_amount,
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    return [trade]
