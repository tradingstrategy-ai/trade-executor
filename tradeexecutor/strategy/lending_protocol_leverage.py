"""Lendindg protocol leveraged.

- Various helpers related to lending protocol leverage
"""

import datetime

from tradeexecutor.state.identifier import AssetWithTrackedValue, AssetType
from tradeexecutor.state.loan import Loan
from tradeexecutor.state.trade import TradeExecution


def create_short_loan(
    position: "tradeexeecutor.state.position.TradingPosition",
    trade: TradeExecution,
) -> Loan:
    """Create the loan data tracking for short position.

    - Check that the information looks correct for a short position.

    - Populates :py:class:`Loan` data structure.
    """

    assert trade.is_leverage_short()
    assert len(position.trades) == 1, "Can be only called when position is opening"

    pair = trade.pair

    assert pair.base.underlying, "Base token lacks underlying asset"
    assert pair.quote.underlying, "Quote token lacks underlying asset"

    assert pair.base.type == AssetType.borrowed, f"Trading pair base asset is not borrowed: {pair.base}"
    assert pair.quote.type == AssetType.collateral, f"Trading pair quote asset is not collateral: {pair.base}"

    # Extra checks when position is opened
    assert trade.planned_quantity < 0, f"Short position must open with  a sell with negative quantity, got: {trade.planned_quantity}"
    assert trade.planned_reserve > 0, f"Collateral must be positive: {trade.planned_reserve}"

    # vToken
    borrowed = AssetWithTrackedValue(
        asset=pair.base,
        last_usd_price=trade.planned_price,
        last_pricing_at=datetime.datetime.utcnow(),
        quantity=abs(trade.planned_quantity),
        created_strategy_cycle_at=trade.strategy_cycle_at,
    )

    # aToken
    collateral = AssetWithTrackedValue(
        asset=pair.quote,
        last_usd_price=trade.reserve_currency_exchange_rate,
        last_pricing_at=datetime.datetime.utcnow(),
        quantity=trade.planned_reserve,
    )

    loan = Loan(
        pair=trade.pair,
        collateral=collateral,
        borrowed=borrowed,
    )

    return loan


def plan_short_loan_update(
    loan: Loan,
    position: "tradeexeecutor.state.position.TradingPosition",
    trade: TradeExecution,
):
    """Update the loan data tracking for short position.

    - Check that the information looks correct for a short position.

    """
    assert trade.is_leverage_short()
    assert len(position.trades) > 1, "Can be only called when closing/reducing/increasing/position"

    if trade.is_reduce():
        loan.collateral.change_quantity_and_value(
            trade.planned_reserve,
            trade.reserve_currency_exchange_rate,
            trade.opened_at,
        )

        # In short position, positive value reduces the borrowed amount
        loan.borrowed.change_quantity_and_value(
            -trade.planned_quantity,
            trade.planned_price,
            trade.opened_at,
        )
    else:
        raise NotImplementedError()

    return loan
